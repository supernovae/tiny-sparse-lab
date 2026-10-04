"""Bounded, authenticated source-token denominator for a frozen corpus release.

The SQLite index and tokenizer batches live outside the immutable release. Source
labels overlap intentionally: a document is deduplicated within each domain.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from sparselab.corpus.progress import memory_bytes
from sparselab.corpus.release import _verification_operation
from sparselab.corpus.token_denominator_identity import _authenticate_inputs, _safe_path
from sparselab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    validate_tokenizer_batch_limits,
)
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.manifest import canonical_json, sha256_file

if TYPE_CHECKING:
    from sparselab.campaign.policy import CorpusReadinessPolicy

_VERSION = 1
_SELECTION = {
    "split": "train",
    "drop_reason": None,
    "deduplicate_by": ["domain", "content_sha256"],
    "add_special_tokens": False,
    "padding": False,
    "truncation": False,
}
_DOMAIN_FIELDS = (
    "eligible_documents",
    "distinct_documents",
    "source_bytes",
    "source_tokens",
)


def _operation(
    function: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with _verification_operation():
            return function(*args, **kwargs)

    return guarded


def _policy(path: Path) -> tuple[CorpusReadinessPolicy, str]:
    # Lazy: campaign.policy imports this module to share the streaming primitive.
    from sparselab.campaign.policy import CorpusReadinessPolicy
    from sparselab.experiments.plan import _ExactUniqueLoader

    raw = path.read_bytes()
    policy = CorpusReadinessPolicy.model_validate(
        yaml.load(raw, Loader=_ExactUniqueLoader)
    )
    if not policy.min_unique_train_tokens_by_domain and not (
        policy.passes is not None and policy.passes.basis == "tokens"
    ):
        raise ValueError("source-token measurement requires a token-domain policy")
    return policy, hashlib.sha256(raw).hexdigest()


def _domains(policy: CorpusReadinessPolicy) -> list[str]:
    requested = set(policy.min_unique_train_bytes_by_domain)
    requested.update(policy.min_unique_train_tokens_by_domain)
    if policy.passes is not None:
        requested.update(policy.passes.mixture)
    return sorted(requested)


def _scratch_dir() -> Path:
    from sparselab.workdir import resolve_work_dir

    return resolve_work_dir(None) / "scratch" / "source-token-denominator"


def _scan_documents(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _inventory(manifest: dict[str, Any], digest: str, size: int) -> None:
    document = manifest["files"]["documents.jsonl"]
    if document["sha256"] != digest or document["size"] != size:
        raise ValueError("streamed documents differ from release inventory")


def _implementation() -> tuple[str, str]:
    return sha256_file(Path(__file__)), sha256_file(
        Path(__file__).with_name("token_denominator_identity.py")
    )


def _scientific(receipt: dict[str, Any]) -> str:
    fields = (
        "measurement_version",
        "release_id",
        "release_manifest_sha256",
        "documents_sha256",
        "documents_size",
        "tokenizer_sha256",
        "tokenizer_manifest_sha256",
        "policy_sha256",
        "policy",
        "selection_definition",
        "implementation_sha256",
        "identity_sha256",
        "domains",
        "rows_scanned",
    )
    evidence = receipt["evidence"]
    scientific_evidence = (
        {
            key: value
            for key, value in evidence.items()
            if key not in {"release_evidence", "selection_evidence", "report_path"}
        }
        if evidence is not None
        else None
    )
    from sparselab.campaign.state import digest

    return digest(
        "source-token-denominator-v1",
        {
            **{name: receipt[name] for name in fields},
            "evidence": scientific_evidence,
        },
    )


def _measure_source_domains(
    release: Path,
    tokenizer: Path | None,
    policy: CorpusReadinessPolicy,
    *,
    batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    scratch: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    all_domains: bool = False,
    max_document_source_bytes: int | None = TOKENIZER_BATCH_SOURCE_BYTES,
) -> dict[str, Any]:
    """Stream and authenticate documents once; keep dedup keys on disk.

    Caller authenticates release and tokenizer provenance. ``all_domains``
    preserves mixed-readiness reporting; unlimited single-record encoding is
    only for that legacy path, never the public denominator.
    """
    batch_documents, batch_source_bytes = validate_tokenizer_batch_limits(
        batch_documents, batch_source_bytes
    )
    release = _safe_path(Path(release))
    document_path = _safe_path(release / "documents.jsonl")
    manifest_path = _safe_path(release / "manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    model = (
        load_tokenizer(_safe_path(Path(tokenizer))) if tokenizer is not None else None
    )
    if model is not None:
        model.no_padding()
        model.no_truncation()
    requested = _domains(policy)
    domains: dict[str, dict[str, int | None]] = {
        domain: dict.fromkeys(_DOMAIN_FIELDS, 0) for domain in requested
    }
    if model is None:
        for counts in domains.values():
            counts["source_tokens"] = None
    scratch_path = _safe_path(scratch or _scratch_dir())
    scratch_path.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    labels: list[list[str]] = []
    batch_bytes = 0
    batch_count = 0
    rows_scanned = 0
    document_size = 0
    digest = hashlib.sha256()

    def flush() -> None:
        nonlocal batch_bytes, batch_count
        if not texts:
            return
        assert model is not None
        encoded = (
            model.encode(texts[0], add_special_tokens=False)
            if len(texts) == 1
            else model.encode_batch(texts, add_special_tokens=False)
        )
        encodings = [encoded] if len(texts) == 1 else encoded
        for encoding, memberships in zip(encodings, labels, strict=True):
            count = len(encoding.ids)
            for domain in memberships:
                current = domains[domain]["source_tokens"]
                assert current is not None
                domains[domain]["source_tokens"] = current + count
        texts.clear()
        labels.clear()
        batch_bytes = 0
        batch_count += 1

    with tempfile.TemporaryDirectory(prefix="measurement-", dir=scratch_path) as temp:
        connection = sqlite3.connect(str(Path(temp) / "seen.sqlite"))
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute(
                "CREATE TABLE seen (domain TEXT NOT NULL, digest TEXT NOT NULL, "
                "PRIMARY KEY (domain, digest)) WITHOUT ROWID"
            )
            with document_path.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    document_size += len(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows_scanned += 1
                    if progress is not None and rows_scanned % 10_000 == 0:
                        progress({"rows_scanned": rows_scanned, "domains": domains})
                    if row["split"] != "train" or row["drop_reason"] is not None:
                        continue
                    applicable = (
                        set(row["domains"])
                        if all_domains
                        else set(row["domains"]).intersection(domains)
                    )
                    if not applicable:
                        continue
                    for domain in applicable:
                        if domain not in domains:
                            domains[domain] = {
                                "eligible_documents": 0,
                                "distinct_documents": 0,
                                "source_bytes": 0,
                                "source_tokens": 0 if model is not None else None,
                            }
                    text = row["text"]
                    length = len(text.encode("utf-8"))
                    if (
                        model is not None
                        and max_document_source_bytes is not None
                        and length > max_document_source_bytes
                    ):
                        raise ValueError("eligible source document exceeds 1 MiB")
                    new_domains: list[str] = []
                    for domain in sorted(applicable):
                        counts = domains[domain]
                        counts["eligible_documents"] += 1  # type: ignore[operator]
                        cursor = connection.execute(
                            "INSERT OR IGNORE INTO seen VALUES (?, ?)",
                            (domain, row["content_sha256"]),
                        )
                        if cursor.rowcount:
                            new_domains.append(domain)
                            counts["distinct_documents"] += 1  # type: ignore[operator]
                    if not new_domains:
                        continue
                    for domain in new_domains:
                        counts = domains[domain]
                        counts["source_bytes"] += length  # type: ignore[operator]
                    if model is not None:
                        if texts and (
                            len(texts) >= batch_documents
                            or batch_bytes + length > batch_source_bytes
                        ):
                            flush()
                        if length > batch_source_bytes:
                            if max_document_source_bytes is not None:
                                raise ValueError(
                                    "source document exceeds tokenizer batch byte limit"
                                )
                            count = len(
                                model.encode(text, add_special_tokens=False).ids
                            )
                            for domain in new_domains:
                                current = domains[domain]["source_tokens"]
                                assert current is not None
                                domains[domain]["source_tokens"] = current + count
                            batch_count += 1
                            continue
                        texts.append(text)
                        labels.append(new_domains)
                        batch_bytes += length
                        if (
                            len(texts) >= batch_documents
                            or batch_bytes >= batch_source_bytes
                        ):
                            flush()
            flush()
        finally:
            connection.close()
    documents_sha256 = digest.hexdigest()
    _inventory(manifest, documents_sha256, document_size)
    if progress is not None:
        progress({"rows_scanned": rows_scanned, "domains": domains})
    return {
        "domains": domains,
        "rows_scanned": rows_scanned,
        "documents_sha256": documents_sha256,
        "documents_size": document_size,
        "batch_count": batch_count,
    }


def _check_locations(
    release: Path, tokenizer: Path, policy_path: Path, output: Path
) -> tuple[Path, Path, Path, Path]:
    release, tokenizer, policy_path, output = (
        _safe_path(Path(item)) for item in (release, tokenizer, policy_path, output)
    )
    if output.suffix in {".partial", ".ready"}:
        raise ValueError("source-token receipt is not complete: reserved partial path")
    sources = (
        release,
        release / "manifest.json",
        release / "documents.jsonl",
        tokenizer,
        tokenizer.with_name("tokenizer_manifest.json"),
        policy_path,
    )
    if (
        output in sources
        or output.is_relative_to(release)
        or any(source.is_relative_to(output) for source in sources)
    ):
        raise ValueError("source and output paths conflict")
    return release, tokenizer, policy_path, output


@_operation
def read_source_token_receipt(
    path: Path, release: Path, tokenizer: Path, policy_path: Path
) -> dict[str, Any]:
    """Authenticate a COMPLETE result, including fresh streamed input identity."""
    release, tokenizer, policy_path, path = _check_locations(
        release, tokenizer, policy_path, path
    )
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if (
        not isinstance(receipt, dict)
        or raw != canonical_json(receipt) + b"\n"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("measurement_version") != _VERSION
    ):
        raise ValueError("source-token receipt is not complete or canonical")
    policy, policy_sha = _policy(policy_path)
    evidence = receipt.get("evidence")
    if evidence is not None and not isinstance(evidence, dict):
        raise ValueError("invalid evidence binding")
    manifest, bindings = _authenticate_inputs(
        release,
        tokenizer,
        evidence_commit=evidence.get("commit") if evidence else None,
        release_evidence=Path(evidence["release_evidence"]) if evidence else None,
        selection_evidence=Path(evidence["selection_evidence"]) if evidence else None,
    )
    implementation, identity = _implementation()
    expected = {
        "release_id": manifest["release_id"],
        "release_manifest_sha256": bindings["release_manifest_sha256"],
        "tokenizer_sha256": bindings["tokenizer_sha256"],
        "tokenizer_manifest_sha256": bindings["tokenizer_manifest_sha256"],
        "policy_sha256": policy_sha,
        "policy": policy.model_dump(mode="json"),
        "selection_definition": _SELECTION,
        "implementation_sha256": implementation,
        "identity_sha256": identity,
        "evidence": bindings["evidence"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "source-token receipt input or implementation binding mismatch"
        )
    digest, size = _scan_documents(_safe_path(release / "documents.jsonl"))
    _inventory(manifest, digest, size)
    if (receipt.get("documents_sha256"), receipt.get("documents_size")) != (
        digest,
        size,
    ):
        raise ValueError("source-token receipt documents changed")
    domain_counts = receipt.get("domains")
    if not isinstance(domain_counts, dict) or set(domain_counts) != set(
        _domains(policy)
    ):
        raise ValueError("source-token receipt domain inventory mismatch")
    for counts in domain_counts.values():
        if (
            not isinstance(counts, dict)
            or set(counts) != set(_DOMAIN_FIELDS)
            or any(
                type(counts[key]) is not int or counts[key] < 0
                for key in _DOMAIN_FIELDS
            )
            or counts["distinct_documents"] > counts["eligible_documents"]
        ):
            raise ValueError("invalid source-token receipt domain counts")
    if (
        type(receipt.get("rows_scanned")) is not int
        or receipt["rows_scanned"] < 0
        or receipt.get("scientific_sha256") != _scientific(receipt)
    ):
        raise ValueError("source-token receipt scientific identity mismatch")
    return receipt


@_operation
def measure_source_tokens(
    release: Path,
    tokenizer: Path,
    policy_path: Path,
    output: Path,
    *,
    evidence_commit: str | None = None,
    release_evidence: Path | None = None,
    selection_evidence: Path | None = None,
    batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
) -> dict[str, Any]:
    """Publish an exclusive, full-EOF result or leave an interrupted partial."""
    start = time.monotonic()
    batch_documents, batch_source_bytes = validate_tokenizer_batch_limits(
        batch_documents, batch_source_bytes
    )
    release, tokenizer, policy_path, output = _check_locations(
        release, tokenizer, policy_path, output
    )
    policy, policy_sha = _policy(policy_path)
    manifest, bindings = _authenticate_inputs(
        release,
        tokenizer,
        evidence_commit=evidence_commit,
        release_evidence=release_evidence,
        selection_evidence=selection_evidence,
    )
    if output.exists():
        existing = read_source_token_receipt(output, release, tokenizer, policy_path)
        if existing["evidence"] != bindings["evidence"]:
            raise ValueError("existing source-token receipt evidence mode mismatch")
        return existing
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    implementation, identity = _implementation()
    launch = uuid.uuid4().hex
    partial = _safe_path(output.with_name(f".{output.name}.{launch}.partial"))
    ready = _safe_path(output.with_name(f".{output.name}.{launch}.ready"))
    progress_path = _safe_path(
        output.with_name(f".{output.name}.{launch}.progress.jsonl")
    )
    last_report = start
    last_update: dict[str, Any] = {}

    def report(update: dict[str, Any], *, status: str = "RUNNING") -> None:
        nonlocal last_report, last_update
        last_update = update
        now = time.monotonic()
        if status == "RUNNING" and now - last_report < 10:
            return
        last_report = now
        totals = update.get("domains", {})
        message = {
            "launch": launch,
            "status": status,
            "rows_scanned": update.get("rows_scanned", 0),
            "eligible_documents": sum(
                item["eligible_documents"] for item in totals.values()
            ),
            "distinct_documents": sum(
                item["distinct_documents"] for item in totals.values()
            ),
            "source_bytes": sum(item["source_bytes"] for item in totals.values()),
            "source_tokens": sum(
                item["source_tokens"] or 0 for item in totals.values()
            ),
            "elapsed_seconds": now - start,
            "rows_per_second": update.get("rows_scanned", 0) / max(now - start, 1e-9),
            "peak_rss_kib": memory_bytes()[1] // 1024,
        }
        record = canonical_json(message) + b"\n"
        with progress_path.open("ab") as log:
            log.write(record)
            log.flush()
            os.fsync(log.fileno())
        print(record.decode().rstrip(), file=sys.stderr, flush=True)

    with partial.open("xb") as temporary:
        temporary.write(
            canonical_json(
                {
                    "measurement_version": _VERSION,
                    "status": "INTERRUPTED",
                    "launch": launch,
                }
            )
            + b"\n"
        )
        temporary.flush()
        os.fsync(temporary.fileno())
    report({}, status="STARTED")
    published_identity: tuple[int, int] | None = None
    try:
        measured = _measure_source_domains(
            release,
            tokenizer,
            policy,
            batch_documents=batch_documents,
            batch_source_bytes=batch_source_bytes,
            progress=report,
            scratch=output.parent / "scratch" / "source-token-denominator",
        )
        stable_files = {
            release / "manifest.json": bindings["release_manifest_sha256"],
            tokenizer: bindings["tokenizer_sha256"],
            tokenizer.with_name("tokenizer_manifest.json"): bindings[
                "tokenizer_manifest_sha256"
            ],
            policy_path: policy_sha,
            Path(__file__): implementation,
            Path(__file__).with_name("token_denominator_identity.py"): identity,
        }
        evidence = bindings["evidence"]
        if evidence is not None:
            for path_key, hash_key in (
                ("release_evidence", "release_evidence_sha256"),
                ("selection_evidence", "selection_evidence_sha256"),
                ("report_path", "report_sha256"),
            ):
                stable_files[Path(evidence[path_key])] = evidence[hash_key]
            stable_files[
                Path(evidence["release_evidence"]).with_name("crash-recovery.md")
            ] = evidence["cold_record_sha256"]
        if any(
            sha256_file(_safe_path(path)) != expected
            for path, expected in stable_files.items()
        ):
            raise ValueError("measurement input changed during source scan")
        receipt = {
            "measurement_version": _VERSION,
            "status": "COMPLETE",
            "release_id": manifest["release_id"],
            "release_manifest_sha256": bindings["release_manifest_sha256"],
            "documents_sha256": measured["documents_sha256"],
            "documents_size": measured["documents_size"],
            "tokenizer_sha256": bindings["tokenizer_sha256"],
            "tokenizer_manifest_sha256": bindings["tokenizer_manifest_sha256"],
            "policy_sha256": policy_sha,
            "policy": policy.model_dump(mode="json"),
            "selection_definition": _SELECTION,
            "implementation_sha256": implementation,
            "identity_sha256": identity,
            "domains": measured["domains"],
            "rows_scanned": measured["rows_scanned"],
            "evidence": bindings["evidence"],
            "policy_path": str(policy_path),
            "operational": {
                "launch": launch,
                "wall_seconds": time.monotonic() - start,
                "peak_rss_kib": memory_bytes()[1] // 1024,
                "batch_documents": batch_documents,
                "batch_source_bytes": batch_source_bytes,
                "batch_count": measured["batch_count"],
                "progress_path": str(progress_path),
            },
        }
        receipt["scientific_sha256"] = _scientific(receipt)
        # The interrupted marker is never rewritten as COMPLETE. A separate
        # fsynced sibling is linked exclusively into place after all checks.
        with ready.open("xb") as temporary:
            temporary.write(canonical_json(receipt) + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            ready_stat = os.fstat(temporary.fileno())
        published_identity = (ready_stat.st_dev, ready_stat.st_ino)
        os.link(ready, output)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        report(measured, status="COMPLETE")
        ready.unlink()
        partial.unlink()
        return receipt
    except BaseException:
        if published_identity is not None:
            try:
                final_stat = output.lstat()
            except FileNotFoundError:
                pass
            else:
                if (final_stat.st_dev, final_stat.st_ino) == published_identity:
                    output.unlink(missing_ok=True)
        ready.unlink(missing_ok=True)
        report(last_update, status="INTERRUPTED")
        raise
