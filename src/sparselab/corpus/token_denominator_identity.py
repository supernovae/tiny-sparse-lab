"""Authenticate source-token inputs; cold-record reuse is v5-only operator trust."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from sparselab.config.loading import load_tokenizer_config
from sparselab.corpus.release import _digest, _verification_operation, verify_release
from sparselab.data.tokenizer import (
    _verify_tokenizer_manifest,
    load_tokenizer,
    verify_tokenizer_artifact,
)
from sparselab.training.manifest import sha256_file

_REPO = Path(__file__).resolve().parents[3]
_APPROVED = "8d74147b6f932045c3f5dcffb451b9cb4360b0aa"
_PREFIX = "experiments/research/devmind-pretrain-v5/"
_COLD = _PREFIX + "crash-recovery.md"
_PRIMARY = _PREFIX + "primary-verification.json"
_SELECTION = _PREFIX + "tokenizer-bakeoff-verification.json"
_RELEASE_ID = "72577dc6898c12caa3e17a731375573b5207d3f58a90963e4581531f3f1bf27b"
_MANIFEST_SHA = "003da8f335899d68391497847aa32f03043110acf68b04e2a55e132e5843ffcb"
_DOCUMENT_SHA = "ab80e11d4fce9e814d3e8978068242fe28883c7022d366ba463d2a8d39f450b9"
_DOCUMENT_SIZE = 6_123_770_983
_TOKENIZER_SHA = "ad186b251ca712e5deebf4cad2eda968a287a964a958604170b785cc380e2b56"
_TOKENIZER_MANIFEST_SHA = (
    "da5b295c42133ef2c31114a4fd81602e144b0113d2ad81cbddd2c0c722665aa5"
)
_REPORT_SHA = "6927c81a734a266ada296ea7fdd9e138f1887678482ef73815068f6ab8bcb092"


def _safe_path(path: Path) -> Path:
    """Reject symlinks anywhere in an existing path, before canonicalizing it."""
    path = Path(path)
    if ".." in path.parts:
        raise ValueError(f"parent traversal is not allowed: {path}")
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked input or ancestor: {component}")
    return absolute.resolve()


def _dataset_for_tokenizer(release: Path, tokenizer: Path, metadata: dict):
    """Recover only the export configuration adjacent to a bound tokenizer."""
    if metadata.get("corpus_export") is None:
        return None
    export = _safe_path(tokenizer.parent.parent)
    config = load_tokenizer_config(_safe_path(export / "tokenizer.yaml"))
    dataset = config.dataset
    if (
        _safe_path(config.output_dir) != tokenizer.parent
        or dataset.corpus_release_path is None
        or _safe_path(dataset.corpus_release_path) != release
        or dataset.corpus_export_path is None
        or _safe_path(dataset.corpus_export_path) != export
        or dataset.source != metadata.get("source")
        or dataset.revision != metadata.get("revision")
        or config.vocab_size != metadata.get("vocab_size")
    ):
        raise ValueError("tokenizer export configuration differs from measured inputs")
    return dataset


def _git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(_REPO), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"committed cold evidence unavailable: {' '.join(args)}")
    return result.stdout


def _committed_blob(commit: str, relative: str) -> tuple[str, bytes]:
    blob = _git("rev-parse", "--verify", f"{commit}:{relative}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", blob):
        raise ValueError("invalid evidence blob identity")
    if _git("cat-file", "-t", blob).strip() != b"blob":
        raise ValueError("evidence path is not a tracked file")
    return blob, _git("cat-file", "blob", blob)


def _evidence_file(
    path: Path, relative: str, commit: str
) -> tuple[dict[str, Any], str, str]:
    path = _safe_path(path)
    if path != _REPO / relative:
        raise ValueError(f"evidence must be at tracked path: {relative}")
    blob, committed = _committed_blob(commit, relative)
    approved_blob, approved_bytes = _committed_blob(_APPROVED, relative)
    if (
        blob != approved_blob
        or committed != approved_bytes
        or path.read_bytes() != committed
    ):
        raise ValueError(f"evidence differs from reviewed committed record: {relative}")
    return json.loads(committed), hashlib.sha256(committed).hexdigest(), blob


def _pinned_inputs(
    release: Path,
    tokenizer: Path,
    commit: str,
    release_evidence: Path,
    selection_evidence: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse only the reviewed post-mount v5 cold proof, never an arbitrary SHA."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("evidence commit must be a full committed SHA")
    if _git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("evidence commit is not a commit object")
    _git("merge-base", "--is-ancestor", _APPROVED, commit)
    cold_blob, cold_bytes = _committed_blob(commit, _COLD)
    approved_cold_blob, approved_cold_bytes = _committed_blob(_APPROVED, _COLD)
    if (
        cold_blob != approved_cold_blob
        or cold_bytes != approved_cold_bytes
        or _safe_path(_REPO / _COLD).read_bytes() != cold_bytes
    ):
        raise ValueError("reviewed post-mount cold verification record changed")
    cold = cold_bytes.decode("utf-8")
    if not all(
        item in cold
        for item in (
            "### Survival and cold authentication",
            "verify_release(primary_release,",
            "493.71 seconds",
            _RELEASE_ID,
            _MANIFEST_SHA,
            _TOKENIZER_SHA,
            _TOKENIZER_MANIFEST_SHA,
            _REPORT_SHA,
        )
    ):
        raise ValueError("committed cold record lacks the approved verification")

    primary, primary_sha, primary_blob = _evidence_file(
        release_evidence, _PRIMARY, commit
    )
    selection, selection_sha, selection_blob = _evidence_file(
        selection_evidence, _SELECTION, commit
    )
    if (
        primary.get("status") != "VERIFIED_PRIMARY"
        or primary.get("release_id") != _RELEASE_ID
        or primary.get("release_path") != str(release)
        or primary.get("metadata", {}).get("release-manifest.json", {}).get("sha256")
        != _MANIFEST_SHA
        or selection.get("status") != "VERIFIED_V5_BAKEOFF_AND_SELECTED_TOKENIZER"
        or selection.get("release_id") != _RELEASE_ID
        or selection.get("report", {}).get("sha256") != _REPORT_SHA
        or selection.get("selected_tokenizer", {}).get("path") != str(tokenizer)
        or selection.get("selected_tokenizer", {}).get("sha256") != _TOKENIZER_SHA
        or selection.get("tokenizer_artifact", {}).get("path") != str(tokenizer)
        or selection.get("tokenizer_artifact", {}).get("sha256") != _TOKENIZER_SHA
        or selection.get("selected_vocab_size") != 32768
    ):
        raise ValueError("v5 evidence release or tokenizer binding mismatch")
    winner = next(
        (
            row
            for row in selection.get("candidates", [])
            if row.get("vocab_size") == 32768
        ),
        None,
    )
    if (
        winner is None
        or winner.get("tokenizer_sha256") != _TOKENIZER_SHA
        or winner.get("manifest_sha256") != _TOKENIZER_MANIFEST_SHA
    ):
        raise ValueError("selected winner evidence mismatch")

    manifest_path = _safe_path(release / "manifest.json")
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    if (
        release.name != _RELEASE_ID
        or manifest_sha != _MANIFEST_SHA
        or manifest.get("release_id") != _RELEASE_ID
        or _digest({key: val for key, val in manifest.items() if key != "release_id"})
        != _RELEASE_ID
        or manifest.get("files", {}).get("documents.jsonl")
        != {"sha256": _DOCUMENT_SHA, "size": _DOCUMENT_SIZE}
    ):
        raise ValueError("v5 manifest or document identity mismatch")

    report_path = _safe_path(Path(selection["report"]["path"]))
    if report_path != tokenizer.parent.parent.parent / "report.json":
        raise ValueError("bakeoff report path is not adjacent to winner")
    if sha256_file(report_path) != _REPORT_SHA:
        raise ValueError("bakeoff report changed")
    report = json.loads(report_path.read_bytes())
    report_winner = next(
        (row for row in report.get("candidates", []) if row.get("vocab_size") == 32768),
        None,
    )
    from sparselab.corpus.tokenizer_bakeoff import choose_candidate

    if (
        report.get("identity", {}).get("release_id") != _RELEASE_ID
        or report["identity"].get("release_path") != str(release)
        or report["identity"].get("declaration", {}).get("release_path") != str(release)
        or report.get("release_binding", {}).get("release_id") != _RELEASE_ID
        or report["release_binding"].get("release_manifest_sha256") != _MANIFEST_SHA
        or report.get("selected_vocab_size") != 32768
        or report.get("selected_tokenizer") != str(tokenizer)
        or report_winner is None
        or report_winner.get("tokenizer_sha256") != _TOKENIZER_SHA
        or report_winner.get("manifest_sha256") != _TOKENIZER_MANIFEST_SHA
        or report_winner.get("manifest_path")
        != str(tokenizer.with_name("tokenizer_manifest.json"))
        or choose_candidate(report["candidates"], 0.98) != 32768
    ):
        raise ValueError("selected report or release binding changed")

    tokenizer_path = _safe_path(tokenizer)
    tokenizer_manifest_path = _safe_path(tokenizer.with_name("tokenizer_manifest.json"))
    if (
        sha256_file(tokenizer_path) != _TOKENIZER_SHA
        or sha256_file(tokenizer_manifest_path) != _TOKENIZER_MANIFEST_SHA
    ):
        raise ValueError("selected tokenizer bytes changed")
    candidate = _verify_tokenizer_manifest(
        tokenizer_path,
        source="local_text",
        revision=json.loads(tokenizer_manifest_path.read_bytes())["revision"],
        vocab_size=32768,
    )
    if (
        candidate != report_winner["manifest"]
        or candidate.get("corpus_forge_bakeoff") != report["release_binding"]
        or candidate.get("sha256") != _TOKENIZER_SHA
    ):
        raise ValueError("winner manifest is not bound to the selected report")
    if load_tokenizer(tokenizer_path).get_vocab_size() != 32768:
        raise ValueError("winner tokenizer vocabulary mismatch")
    return manifest, {
        "release_manifest_sha256": manifest_sha,
        "tokenizer_sha256": _TOKENIZER_SHA,
        "tokenizer_manifest_sha256": _TOKENIZER_MANIFEST_SHA,
        "evidence": {
            "commit": commit,
            "release_evidence": str(release_evidence),
            "release_evidence_sha256": primary_sha,
            "release_evidence_blob_sha": primary_blob,
            "selection_evidence": str(selection_evidence),
            "selection_evidence_sha256": selection_sha,
            "selection_evidence_blob_sha": selection_blob,
            "cold_record_sha256": hashlib.sha256(cold_bytes).hexdigest(),
            "cold_record_blob_sha": cold_blob,
            "report_path": str(report_path),
            "report_sha256": _REPORT_SHA,
        },
    }


def _authenticate_inputs(
    release: Path,
    tokenizer: Path,
    *,
    evidence_commit: str | None = None,
    release_evidence: Path | None = None,
    selection_evidence: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fully verify by default; explicit committed cold reuse is limited to reviewed v5."""
    release = _safe_path(release)
    tokenizer = _safe_path(tokenizer)
    if any(
        value is not None
        for value in (evidence_commit, release_evidence, selection_evidence)
    ):
        if not all(
            value is not None
            for value in (evidence_commit, release_evidence, selection_evidence)
        ):
            raise ValueError(
                "evidence commit and both tracked evidence paths are required"
            )
        assert evidence_commit is not None
        assert release_evidence is not None and selection_evidence is not None
        return _pinned_inputs(
            release,
            tokenizer,
            evidence_commit,
            _safe_path(release_evidence),
            _safe_path(selection_evidence),
        )

    with _verification_operation():
        manifest = verify_release(release)
        tokenizer_manifest_path = _safe_path(
            tokenizer.with_name("tokenizer_manifest.json")
        )
        metadata = json.loads(tokenizer_manifest_path.read_bytes())
        if not isinstance(metadata, dict):
            raise TypeError("invalid tokenizer manifest")
        verified = verify_tokenizer_artifact(
            tokenizer,
            source=metadata["source"],
            revision=metadata.get("revision"),
            vocab_size=metadata["vocab_size"],
            dataset=_dataset_for_tokenizer(release, tokenizer, metadata),
        )
        if verified != metadata:
            raise ValueError("tokenizer verifier returned inconsistent manifest")
        tokenizer_sha = sha256_file(tokenizer)
        if verified["sha256"] != tokenizer_sha:
            raise ValueError("tokenizer artifact digest mismatch")
        return manifest, {
            "release_manifest_sha256": sha256_file(release / "manifest.json"),
            "tokenizer_sha256": tokenizer_sha,
            "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest_path),
            "evidence": None,
        }
