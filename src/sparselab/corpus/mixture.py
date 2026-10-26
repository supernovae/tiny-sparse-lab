"""Authenticated, immutable target-token mixture over a frozen corpus release.

The family inventory is JSONL with one row per kept document: document_id,
family_id, split, stratum and content_sha256. It is deliberately separate from
the release's source-family label, which identifies a source rather than a work.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sparselab.config.loading import load_tokenizer_config
from sparselab.config.models import DatasetConfig
from sparselab.corpus.acquisition import _sync_dir
from sparselab.corpus.mixture_index import MAX_RECEIPT_BYTES, MixtureIndex, rows
from sparselab.corpus.release import _verification_operation, verify_release
from sparselab.data.tokenizer import load_tokenizer, verify_tokenizer_artifact
from sparselab.engram.packs import _rename_noreplace
from sparselab.training.manifest import canonical_json, sha256_file


class MixtureDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    release_path: Path
    tokenizer_config: Path
    family_inventory: Path
    source_strata: dict[str, str]
    target_quotas: dict[str, int]
    seed: int
    max_exposures: Literal[1, 2]
    unique_train_content_floors: dict[str, int] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    order_policy: Literal["kml-card03-v1"] | None = None
    # Reuse an authenticated tokenizer fitted to an earlier frozen release.
    # Absent for legacy declarations, preserving their canonical identity.
    tokenizer_origin_release_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("max_exposures", mode="before")
    @classmethod
    def strict_exposure_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("max_exposures must be integer 1 or 2")
        return value

    @field_validator("unique_train_content_floors", mode="before")
    @classmethod
    def strict_content_floors(cls, value):
        if value is not None and (
            not isinstance(value, dict)
            or any(type(item) is not int for item in value.values())
        ):
            raise ValueError("unique TRAIN content floors must be integers")
        return value

    @model_validator(mode="after")
    def valid(self) -> MixtureDeclaration:
        if (
            not self.source_strata
            or not self.target_quotas
            or set(self.source_strata.values()) != set(self.target_quotas)
            or any(not key or not value for key, value in self.source_strata.items())
            or any(
                not key or type(value) is not int or value <= 0
                for key, value in self.target_quotas.items()
            )
        ):
            raise ValueError(
                "mixture requires positive quotas for exactly the mapped strata"
            )
        if self.unique_train_content_floors is not None and (
            set(self.unique_train_content_floors) != set(self.target_quotas)
            or any(
                type(value) is not int or value <= 0
                for value in self.unique_train_content_floors.values()
            )
        ):
            raise ValueError(
                "unique TRAIN content floors must cover the exact mixture strata"
            )
        return self


_rows = rows


def _source_rights(release: Path, source_strata: dict[str, str]) -> None:
    sources = json.loads((release / "sources.json").read_text(encoding="utf-8"))
    source_map = {row["id"]: row for row in sources}
    if set(source_map) != set(source_strata):
        raise ValueError(
            "source strata must cover exactly the verified release sources"
        )
    for row in sources:
        rights = row.get("rights_policy")
        if (
            row.get("explicit_training_restriction") not in {None, "none_found"}
            or not isinstance(rights, dict)
            or rights.get("training_eligibility")
            not in {"eligible", "eligible_with_obligations", "review_required"}
        ):
            raise ValueError("source lacks a reviewed training rights path")


def _tokenizer(spec: MixtureDeclaration, release: Path, release_id: str):
    config = load_tokenizer_config(spec.tokenizer_config)
    dataset = config.dataset
    origin = spec.tokenizer_origin_release_id or release_id
    if (
        dataset.source != "local_text"
        or dataset.corpus_release_path is None
        or dataset.corpus_release_path.resolve().name != origin
        or dataset.revision != origin
        or (
            spec.tokenizer_origin_release_id is None
            and dataset.corpus_release_path.resolve() != release
        )
    ):
        raise ValueError("tokenizer configuration is not bound to this release")
    path = config.output_dir / "tokenizer.json"
    manifest = verify_tokenizer_artifact(
        path,
        source=dataset.source,
        revision=dataset.revision,
        vocab_size=config.vocab_size,
        dataset=dataset,
    )
    return load_tokenizer(path), manifest["sha256"]


def _rank(
    seed: int, stratum: str, family: str, doc_id: str, order_policy: str | None = None
) -> str:
    if order_policy == "kml-card03-v1":
        return hashlib.sha256(
            f"kml-card03-v1 | {stratum} | {doc_id}".encode()
        ).hexdigest()
    return hashlib.sha256(canonical_json([seed, stratum, family, doc_id])).hexdigest()


def materialize_mixture(declaration: Path, output: Path) -> dict:
    """Publish exact token IDs only after all quotas and closure checks pass."""
    with _verification_operation():
        return _materialize_mixture(declaration, output)


def _materialize_mixture(declaration: Path, output: Path) -> dict:
    raw = declaration.read_bytes()
    spec = MixtureDeclaration.model_validate(yaml.safe_load(raw))
    release = spec.release_path.resolve()
    manifest = verify_release(release)
    if manifest["release_id"] != release.name:
        raise ValueError("release path does not match its verified identity")
    _source_rights(release, spec.source_strata)
    report = json.loads((release / "report.json").read_text(encoding="utf-8"))
    requested = report.get("requested_mixture")
    total_quota = sum(spec.target_quotas.values())
    if (
        not isinstance(requested, dict)
        or set(requested) != set(spec.target_quotas)
        or any(
            abs(float(requested[stratum]) - quota / total_quota) > 1e-8
            for stratum, quota in spec.target_quotas.items()
        )
    ):
        raise ValueError(
            "target quotas disagree with verified release requested mixture"
        )
    tokenizer, tokenizer_sha = _tokenizer(spec, release, manifest["release_id"])
    from sparselab.corpus.completed_preparation import local_scope

    completed = local_scope(manifest)
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("verified tokenizer lacks EOS token")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"mixture output already exists: {output}")
    if completed is not None:
        cache = _completed_mixture_cache(spec)
        bound = [
            output,
            spec.family_inventory.resolve(),
            spec.tokenizer_config.resolve(),
            load_tokenizer_config(spec.tokenizer_config).output_dir.resolve(),
        ]
        if (
            cache.proofs is None
            or not cache.writable
            or any(not p.is_relative_to(cache.root) for p in bound)
        ):
            raise ValueError(
                "completed mixture requires existing trusted work-root completion storage"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mixture-", dir=output.parent))
    try:
        with (
            tempfile.TemporaryDirectory(prefix=".index-", dir=staging) as scratch,
            MixtureIndex(Path(scratch), release / "documents.jsonl") as index,
        ):
            index.inventory(
                spec.family_inventory,
                spec.source_strata,
                lambda row: _rank(
                    spec.seed,
                    row["stratum"],
                    row["family_id"],
                    row["document_id"],
                    spec.order_policy,
                ),
            )
            index.select(release / "lm/train.lineage.jsonl")
            receipt = _schedule(
                spec,
                raw,
                release,
                manifest,
                tokenizer,
                tokenizer_sha,
                eos,
                index,
                staging,
            )
            index.verify_inputs()
        if completed is not None:
            receipt["completed_preparation"] = {
                "policy": completed["policy"],
                "verification_scope": "native executed mixture schedule and unique-content counts; no later schedule replay",
            }
        with (staging / "receipt.json").open("xb") as stream:
            stream.write(canonical_json(receipt) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_dir(staging)
        _rename_noreplace(staging, output)
        _sync_dir(output.parent)
        if completed is not None:
            _seal_completed_mixture(spec, raw, output, manifest, tokenizer_sha, receipt)
        return receipt
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _schedule(
    spec, raw, release, manifest, tokenizer, tokenizer_sha, eos, index, staging
):
    totals: Counter[str] = Counter()
    unique: Counter[str] = Counter()
    available_unique: Counter[str] = Counter()
    available_content: Counter[str] = Counter()
    eligible_families: Counter[str] = Counter()
    source_totals: Counter[str] = Counter()
    records = 0
    excluded: list[dict] = []
    excluded_bytes = 0

    def exclude(doc, reason):
        nonlocal excluded_bytes
        row = {"document_id": doc["document_id"], "reason": reason}
        excluded_bytes += len(canonical_json(row)) + 1
        if excluded_bytes > MAX_RECEIPT_BYTES // 2:
            raise ValueError("mixture exclusion receipt-byte cap exceeded")
        excluded.append(row)

    with (staging / "train.tokens.jsonl").open("wb") as stream:
        for stratum, quota in sorted(spec.target_quotas.items()):
            available = 0
            for doc, family in index.candidates(stratum):
                digest = doc["content_sha256"]
                if index.seen(digest):
                    exclude(doc, "duplicate_train_content")
                    continue
                ids = tokenizer.encode(doc["text"], add_special_tokens=False).ids
                if not ids:
                    exclude(doc, "zero_encoded_targets")
                    continue
                index.append(
                    {
                        "document_id": doc["document_id"],
                        "source_id": doc["source_id"],
                        "family_id": family,
                        "token_ids": ids + [eos],
                        "stratum": stratum,
                    }
                )
                available += len(ids) + 1
                available_content[stratum] += len(ids)
            if (
                spec.unique_train_content_floors is not None
                and available_content[stratum]
                < spec.unique_train_content_floors[stratum]
            ):
                raise ValueError(
                    f"unique TRAIN content supply below floor for {stratum}: "
                    f"{available_content[stratum]} available, "
                    f"{spec.unique_train_content_floors[stratum]} required (EOS excluded)"
                )
            if available * spec.max_exposures < quota:
                raise ValueError(
                    f"required quota impossible for {stratum}: {available} unique, "
                    f"{quota} required, max_exposures={spec.max_exposures}, "
                    f"shortfall={quota - available * spec.max_exposures}"
                )
            available_unique[stratum] = available
            eligible_families[stratum] = index.family_count(stratum)
            for exposure in range(1, spec.max_exposures + 1):
                for doc in index.encoded(stratum):
                    ids = doc["token_ids"]
                    remaining = quota - totals[stratum]
                    if remaining <= 0:
                        break
                    take = min(len(ids), remaining)
                    # Every segment has an explicit end boundary. A final
                    # truncated document uses its reserved EOS position.
                    targets = (
                        ids[:take] if take == len(ids) else ids[: take - 1] + [eos]
                    )
                    row = {
                        "document_id": doc["document_id"],
                        "family_id": doc["family_id"],
                        "source_id": doc["source_id"],
                        "stratum": stratum,
                        "exposure": exposure,
                        "content_token_start": 0,
                        "content_token_end": take - 1,
                        "token_ids": targets,
                    }
                    stream.write(canonical_json(row) + b"\n")
                    totals[stratum] += take
                    source_totals[doc["source_id"]] += take
                    if exposure == 1:
                        unique[stratum] += take
                    records += 1
                if totals[stratum] == quota:
                    break
            if totals[stratum] != quota:
                raise ValueError(f"unfilled quota for {stratum}")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "release_manifest_sha256": sha256_file(release / "manifest.json"),
        "tokenizer_sha256": tokenizer_sha,
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "family_inventory_sha256": sha256_file(spec.family_inventory),
        "seed": spec.seed,
        **({"order_policy": spec.order_policy} if spec.order_policy else {}),
        "requested_target_quotas": spec.target_quotas,
        "actual_target_tokens": dict(totals),
        "unique_target_positions": dict(unique),
        "available_unique_positions": dict(available_unique),
        "eligible_train_families": dict(eligible_families),
        "source_target_tokens": dict(source_totals),
        "max_exposures": spec.max_exposures,
        "target_semantics": "ordered encoded content IDs plus segment EOS; counts are scheduled IDs before any trainer shift or block-edge loss",
        "training_supervision_verified": False,
        "records": records,
        "excluded": excluded,
        "quota_shortfalls": {
            stratum: spec.target_quotas[stratum] - totals[stratum]
            for stratum in spec.target_quotas
        },
        "train_tokens_sha256": sha256_file(staging / "train.tokens.jsonl"),
        "train_tokens_bytes": (staging / "train.tokens.jsonl").stat().st_size,
    }
    if spec.unique_train_content_floors is not None:
        receipt["available_unique_content_tokens"] = dict(available_content)
        receipt["unique_train_content_floors"] = spec.unique_train_content_floors
    if len(canonical_json(receipt)) + 1 > MAX_RECEIPT_BYTES:
        raise ValueError("mixture receipt-byte cap exceeded")
    return receipt


def _completed_mixture_cache(spec):
    from sparselab.corpus.admission_inspection_cache import InspectionCache
    from sparselab.corpus.completed_preparation import corpus_work_root

    cache = InspectionCache(corpus_work_root(spec.release_path.resolve()))
    cache.authority = {
        **cache.authority,
        "mixture": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("mixture.py", "mixture_index.py")
        },
    }
    return cache


def _completed_mixture_request(spec, raw, output, manifest, tokenizer_sha):
    return {
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "release_id": manifest["release_id"],
        "tokenizer_sha256": tokenizer_sha,
        "release_path": str(spec.release_path.resolve()),
        "output": str(output.resolve()),
        "tokenizer_config": str(spec.tokenizer_config.resolve()),
        "family_inventory": str(spec.family_inventory.resolve()),
    }


def _seal_completed_mixture(spec, raw, output, manifest, tokenizer_sha, receipt):
    from sparselab.corpus.admission_inspection_cache import capture
    from sparselab.corpus.completed_preparation import (
        corpus_work_root,
        local_dependencies,
    )

    release = spec.release_path.resolve()
    cache = _completed_mixture_cache(spec)
    cache.publish(
        "completed-mixture",
        _completed_mixture_request(spec, raw, output, manifest, tokenizer_sha),
        receipt,
        capture(
            [
                output,
                spec.family_inventory,
                spec.tokenizer_config,
                load_tokenizer_config(spec.tokenizer_config).output_dir,
                release,
                *local_dependencies(manifest, work_root=corpus_work_root(release)),
            ]
        ),
        limit=MAX_RECEIPT_BYTES,
    )


def _verify_completed_mixture(declaration, output, receipt):
    raw = declaration.read_bytes()
    spec = MixtureDeclaration.model_validate(yaml.safe_load(raw))
    manifest = verify_release(spec.release_path.resolve())
    from sparselab.corpus.completed_preparation import local_scope

    if local_scope(manifest) is None:
        raise ValueError("completed mixture has no matching local preparation scope")
    _, tokenizer_sha = _tokenizer(
        spec, spec.release_path.resolve(), manifest["release_id"]
    )
    cached = _completed_mixture_cache(spec).load(
        "completed-mixture",
        _completed_mixture_request(spec, raw, output, manifest, tokenizer_sha),
        limit=MAX_RECEIPT_BYTES,
    )
    if cached is None or cached[0] != receipt:
        raise ValueError(
            "completed native mixture evidence unavailable; no automatic schedule replay"
        )
    return receipt


def verify_mixture(declaration: Path, output: Path) -> dict:
    """Verify the declared cold-replay or completed-local execution policy."""
    output = output.resolve()
    receipt_path = output / "receipt.json"
    with receipt_path.open("rb") as stream:
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("mixture receipt-byte cap exceeded")
    receipt = json.loads(raw)
    if (
        receipt.get("completed_preparation", {}).get("policy")
        == "completed_local_preparation"
    ):
        return _verify_completed_mixture(declaration, output, receipt)
    if sha256_file(output / "train.tokens.jsonl") != receipt["train_tokens_sha256"]:
        raise ValueError("mixture token export digest mismatch")
    with tempfile.TemporaryDirectory(
        prefix=".verify-mixture-", dir=output.parent
    ) as temp:
        replay = materialize_mixture(declaration, Path(temp) / "replay")
        if replay != receipt:
            raise ValueError("mixture receipt differs from authenticated replay")
    return receipt


def verify_mixture_dataset(
    dataset: DatasetConfig, tokenizer_path: Path, vocab_size: int
) -> dict:
    """Bind a token-ID consumer to the replayed mix and its fit tokenizer."""
    if (
        dataset.source != "local_token_mixture"
        or dataset.mixture_declaration_path is None
        or dataset.mixture_output_path is None
        or dataset.train_path is None
        or dataset.validation_path is None
    ):
        raise ValueError("complete local token-mixture dataset required")
    spec = MixtureDeclaration.model_validate(
        yaml.safe_load(dataset.mixture_declaration_path.read_bytes())
    )
    receipt = verify_mixture(
        dataset.mixture_declaration_path, dataset.mixture_output_path
    )
    fit = load_tokenizer_config(spec.tokenizer_config)
    release = spec.release_path.resolve()
    if (
        dataset.train_path.resolve()
        != (dataset.mixture_output_path / "train.tokens.jsonl").resolve()
        or dataset.validation_path.resolve()
        != (release / "lm/validation.jsonl").resolve()
        or dataset.revision != receipt["release_id"]
        or dataset.train_max_tokens != sum(receipt["actual_target_tokens"].values())
        or dataset.train_max_documents != receipt["records"]
        or tokenizer_path.resolve() != (fit.output_dir / "tokenizer.json").resolve()
        or vocab_size != fit.vocab_size
    ):
        raise ValueError("token mixture dataset or tokenizer identity mismatch")
    from sparselab.corpus.export import _licenses, _split_stats

    records, rendered_bytes = _split_stats(dataset.validation_path, "lm")
    if (
        dataset.license != _licenses(release)
        or records != dataset.validation_max_documents
        or dataset.validation_max_tokens != rendered_bytes + records + 1
    ):
        raise ValueError("token mixture validation split count or budget mismatch")
    return receipt
