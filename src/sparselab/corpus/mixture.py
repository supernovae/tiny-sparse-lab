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
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sparselab.config.loading import load_tokenizer_config
from sparselab.config.models import DatasetConfig
from sparselab.corpus.acquisition import _sync_dir
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
    max_exposures: Literal[2]
    order_policy: Literal["kml-card03-v1"] | None = None
    # Reuse an authenticated tokenizer fitted to an earlier frozen release.
    # Absent for legacy declarations, preserving their canonical identity.
    tokenizer_origin_release_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

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
        return self


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{number}") from error


def _inventory(
    path: Path, documents: dict[str, dict], source_strata: dict[str, str]
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    families: dict[str, str] = {}
    content_splits: dict[str, str] = {}
    content_strata: dict[str, str] = {}
    for row in _rows(path):
        if (
            not isinstance(row, dict)
            or set(row)
            != {"document_id", "family_id", "split", "stratum", "content_sha256"}
            or any(not isinstance(row[key], str) or not row[key] for key in row)
            or row["split"] not in {"train", "validation", "test"}
        ):
            raise ValueError("invalid family inventory row")
        doc_id = row["document_id"]
        doc = documents.get(doc_id)
        if doc is None or doc_id in result:
            raise ValueError("family inventory has unknown or duplicate document")
        if (
            doc["split"] != row["split"]
            or doc["content_sha256"] != row["content_sha256"]
            or source_strata.get(doc["source_id"]) != row["stratum"]
            or row["stratum"] not in doc["domains"]
        ):
            raise ValueError("family inventory disagrees with verified release")
        family = row["family_id"]
        if family in families and families[family] != row["split"]:
            raise ValueError("document family leaks across splits")
        families[family] = row["split"]
        content = row["content_sha256"]
        if content in content_splits and content_splits[content] != row["split"]:
            raise ValueError("identical content leaks across splits")
        content_splits[content] = row["split"]
        if content in content_strata and content_strata[content] != row["stratum"]:
            raise ValueError("identical content assigned to different strata")
        content_strata[content] = row["stratum"]
        result[doc_id] = row
    if set(result) != set(documents):
        raise ValueError("family inventory does not cover every kept document")
    return result


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
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("verified tokenizer lacks EOS token")
    documents = {
        doc["document_id"]: doc
        for doc in _rows(release / "documents.jsonl")
        if doc["drop_reason"] is None
    }
    inventory = _inventory(spec.family_inventory, documents, spec.source_strata)
    selected = {row["record_id"] for row in _rows(release / "lm/train.lineage.jsonl")}
    if selected != {
        doc_id for doc_id, doc in documents.items() if doc["split"] == "train"
    }:
        raise ValueError(
            "train LM view does not cover exactly eligible train documents"
        )
    for doc_id in selected:
        doc = documents[doc_id]
        if doc.get("rights", {}).get("training_eligibility") not in {
            "eligible",
            "eligible_with_obligations",
        }:
            raise ValueError("train document lacks admitted file or record rights")

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"mixture output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mixture-", dir=output.parent))
    try:
        totals: Counter[str] = Counter()
        unique: Counter[str] = Counter()
        available_unique: Counter[str] = Counter()
        eligible_families: Counter[str] = Counter()
        source_totals: Counter[str] = Counter()
        records = 0
        excluded: list[dict] = []
        with (staging / "train.tokens.jsonl").open("wb") as stream:
            for stratum, quota in sorted(spec.target_quotas.items()):
                candidates = sorted(
                    (
                        documents[doc_id]
                        for doc_id in selected
                        if inventory[doc_id]["stratum"] == stratum
                    ),
                    key=lambda doc: _rank(
                        spec.seed,
                        stratum,
                        inventory[doc["document_id"]]["family_id"],
                        doc["document_id"],
                        spec.order_policy,
                    ),
                )
                seen_content: set[str] = set()
                encoded: list[tuple[dict, list[int]]] = []
                available = 0
                for doc in candidates:
                    digest = doc["content_sha256"]
                    if digest in seen_content:
                        excluded.append(
                            {
                                "document_id": doc["document_id"],
                                "reason": "duplicate_train_content",
                            }
                        )
                        continue
                    seen_content.add(digest)
                    ids = tokenizer.encode(doc["text"], add_special_tokens=False).ids
                    if not ids:
                        excluded.append(
                            {
                                "document_id": doc["document_id"],
                                "reason": "zero_encoded_targets",
                            }
                        )
                        continue
                    encoded.append((doc, ids + [eos]))
                    available += len(ids) + 1
                if available * spec.max_exposures < quota:
                    raise ValueError(
                        f"required quota impossible for {stratum}: {available} unique, {quota} required"
                    )
                available_unique[stratum] = available
                eligible_families[stratum] = len(
                    {inventory[doc["document_id"]]["family_id"] for doc, _ in encoded}
                )
                for exposure in range(1, spec.max_exposures + 1):
                    for doc, ids in encoded:
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
                            "family_id": inventory[doc["document_id"]]["family_id"],
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
        with (staging / "receipt.json").open("xb") as stream:
            stream.write(canonical_json(receipt) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_dir(staging)
        _rename_noreplace(staging, output)
        _sync_dir(output.parent)
        return receipt
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_mixture(declaration: Path, output: Path) -> dict:
    """Cold replay catches changed inputs, schedule, rights and token arrays."""
    output = output.resolve()
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
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
