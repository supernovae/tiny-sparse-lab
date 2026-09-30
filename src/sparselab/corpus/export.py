"""Immutable training configs and integrity checks for frozen corpus exports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import yaml

from sparselab.config.loading import load_config
from sparselab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
from sparselab.training.manifest import canonical_json, sha256_file


def _release(path: Path) -> dict[str, object]:
    from sparselab.corpus.release import verify_release

    manifest = verify_release(path)
    if manifest.get("release_id") != path.name:
        raise ValueError("release directory does not match verified release ID")
    return manifest


def _licenses(release_dir: Path) -> str:
    sources = json.loads((release_dir / "sources.json").read_text(encoding="utf-8"))
    items = sources.values() if isinstance(sources, dict) else sources
    names: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            declaration = item.get("declaration", item)
            if (
                isinstance(declaration, dict)
                and declaration.get("license")
                and declaration.get("redistribution") != "rejected"
            ):
                names.add(str(declaration["license"]))
    if any(isinstance(item, dict) and "rights_policy" in item for item in items):
        with (release_dir / "documents.jsonl").open(encoding="utf-8") as handle:
            names.update(json.loads(line)["license"] for line in handle)
    if not names:
        raise ValueError("frozen release lacks explicit source licenses")
    return "; ".join(sorted(names))


def _split_stats(path: Path, view: str) -> tuple[int, int]:
    count = size = 0
    if view == "lm":
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid LM record at {path}:{number}") from error
                if (
                    not isinstance(record, dict)
                    or set(record) != {"text"}
                    or not isinstance(record["text"], str)
                    or not record["text"].strip()
                ):
                    raise ValueError(f"invalid LM record at {path}:{number}")
                count += 1
                size += len(record["text"].encode("utf-8"))
    else:
        from sparselab.data.conversations import iter_conversations

        for text in iter_conversations(path):
            count += 1
            size += len(text.encode("utf-8"))
    return count, size


def _write_json(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_yaml(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def export_release(
    release_dir: Path,
    view: str,
    base_run_config: Path,
    vocab_size: int,
    work_root: Path,
) -> Path:
    """Create two complete configs bound to one verified immutable release."""
    if view not in {"lm", "chat"} or vocab_size < 260:
        raise ValueError("view must be lm or chat and vocab_size must be >= 260")
    release_dir = Path(release_dir).resolve()
    manifest = _release(release_dir)
    base = load_config(Path(base_run_config))
    base_digest = sha256_file(Path(base_run_config))
    release_id = str(manifest["release_id"])
    identity = hashlib.sha256(
        canonical_json(
            {
                "release_id": release_id,
                "view": view,
                "base_config_sha256": base_digest,
                "vocab_size": vocab_size,
            }
        )
    ).hexdigest()
    destination = (
        Path(work_root).resolve()
        / "corpora"
        / str(manifest["corpus_id"])
        / "exports"
        / release_id
        / view
        / identity
    )
    train = release_dir / view / "train.jsonl"
    validation = release_dir / view / "validation.jsonl"
    counts = [_split_stats(path, view) for path in (train, validation)]
    if any(count == 0 for count, _ in counts):
        raise ValueError(f"{view} requires nonempty train and validation")
    if any(
        byte_count + count < base.training.seq_len + 1 for count, byte_count in counts
    ):
        raise ValueError(
            "selected split cannot supply a full next-token block at fixed seq_len"
        )
    source = "local_text" if view == "lm" else "local_chat"
    dataset = base.dataset.model_copy(
        update={
            "source": source,
            "revision": release_id,
            "license": _licenses(release_dir),
            "train_path": train,
            "validation_path": validation,
            "corpus_release_path": release_dir,
            "corpus_export_path": destination,
            "source_manifest_path": None,
            "allocation_manifest_path": None,
            "dataset_config": None,
            "cache_dir": destination / "prepared",
            "train_max_documents": counts[0][0],
            "validation_max_documents": counts[1][0],
            "train_max_tokens": counts[0][1] + counts[0][0] + 1,
            "validation_max_tokens": counts[1][1] + counts[1][0] + 1,
        }
    )
    run = base.model_copy(
        update={
            "model": base.model.model_copy(update={"vocab_size": vocab_size}),
            "dataset": dataset,
            "tokenizer": base.tokenizer.model_copy(
                update={"path": destination / "tokenizer" / "tokenizer.json"}
            ),
        }
    )
    tokenizer = {
        "schema_version": 1,
        "vocab_size": vocab_size,
        "min_frequency": 1,
        "max_documents": counts[0][0],
        "output_dir": str(destination / "tokenizer"),
        "dataset": dataset.model_dump(mode="json"),
    }
    run_payload = run.model_dump(mode="json")
    RunConfig.model_validate(run_payload)
    TokenizerTrainConfig.model_validate(tokenizer)
    tokenizer_bytes = yaml.safe_dump(tokenizer, sort_keys=True).encode("utf-8")
    run_bytes = yaml.safe_dump(run_payload, sort_keys=True).encode("utf-8")
    sidecar = {
        "schema_version": 1,
        "release_id": release_id,
        "view": view,
        "base_config_sha256": base_digest,
        "vocab_size": vocab_size,
        "release_manifest_sha256": sha256_file(release_dir / "manifest.json"),
        "report_sha256": sha256_file(release_dir / "report.json"),
        "license_report_sha256": sha256_file(release_dir / "license-report.json"),
        "tokenizer_config_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "run_config_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "splits": {
            split: {
                "path": f"{view}/{split}.jsonl",
                "sha256": sha256_file(path),
                "records": count,
                "rendered_bytes": size,
            }
            for split, path, (count, size) in zip(
                ("train", "validation"), (train, validation), counts, strict=True
            )
        },
    }
    if manifest["build_identity"]["release"]["schema_version"] == 2:
        rights = json.loads(
            (release_dir / "license-report.json").read_text(encoding="utf-8")
        )
        sidecar["publication_mode"] = rights["publication_mode"]
        sidecar["weight_license_status"] = rights["weight_license_status"]
    if destination.exists():
        if any(
            not (destination / filename).is_file()
            for filename in ("export.json", "run.yaml", "tokenizer.yaml")
        ):
            raise ValueError("incomplete immutable corpus export")
        if (
            json.loads((destination / "export.json").read_text(encoding="utf-8"))
            != sidecar
            or (destination / "run.yaml").read_bytes() != run_bytes
            or (destination / "tokenizer.yaml").read_bytes() != tokenizer_bytes
        ):
            raise ValueError("existing corpus export has changed")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".export-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        _write_yaml(staging / "tokenizer.yaml", tokenizer)
        _write_yaml(staging / "run.yaml", run_payload)
        _write_json(staging / "export.json", sidecar)
        if destination.exists():
            raise FileExistsError(
                f"corpus export appeared during publication: {destination}"
            )
        staging.rename(destination)
    return destination


def verify_release_export(dataset: DatasetConfig) -> dict[str, object]:
    """Verify the release, export request, and exact bytes selected by dataset."""
    if dataset.corpus_release_path is None or dataset.corpus_export_path is None:
        raise ValueError("dataset does not reference a frozen corpus export")
    release_dir = dataset.corpus_release_path
    export_dir = dataset.corpus_export_path
    manifest = _release(release_dir)
    export = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    if (
        not isinstance(export, dict)
        or export.get("schema_version") != 1
        or export.get("release_id") != manifest["release_id"]
        or dataset.revision != manifest["release_id"]
    ):
        raise ValueError("corpus export release identity mismatch")
    view = export.get("view")
    if view not in {"lm", "chat"} or dataset.source != (
        "local_text" if view == "lm" else "local_chat"
    ):
        raise ValueError("corpus export view mismatch")
    request = {
        "release_id": manifest["release_id"],
        "view": view,
        "base_config_sha256": export.get("base_config_sha256"),
        "vocab_size": export.get("vocab_size"),
    }
    if (
        not isinstance(request["base_config_sha256"], str)
        or len(request["base_config_sha256"]) != 64
        or not isinstance(request["vocab_size"], int)
        or request["vocab_size"] < 260
        or export_dir.name != hashlib.sha256(canonical_json(request)).hexdigest()
        or export_dir.parent
        != release_dir.parent.parent / "exports" / str(manifest["release_id"]) / view
        or dataset.license != _licenses(release_dir)
    ):
        raise ValueError("corpus export request or license mismatch")
    for name, key in (
        ("run.yaml", "run_config_sha256"),
        ("tokenizer.yaml", "tokenizer_config_sha256"),
    ):
        if sha256_file(export_dir / name) != export.get(key):
            raise ValueError(f"corpus export config changed: {name}")
    if (
        export.get("release_manifest_sha256")
        != sha256_file(release_dir / "manifest.json")
        or export.get("report_sha256") != sha256_file(release_dir / "report.json")
        or export.get("license_report_sha256")
        != sha256_file(release_dir / "license-report.json")
    ):
        raise ValueError("corpus export evidence mismatch")
    splits = export.get("splits")
    if not isinstance(splits, dict):
        raise TypeError("corpus export split inventory missing")
    for split, path in (
        ("train", dataset.train_path),
        ("validation", dataset.validation_path),
    ):
        entry = splits.get(split)
        expected = release_dir / str(view) / f"{split}.jsonl"
        if (
            not isinstance(entry, dict)
            or path is None
            or path.resolve() != expected.resolve()
            or entry.get("path") != f"{view}/{split}.jsonl"
            or entry.get("sha256") != sha256_file(expected)
        ):
            raise ValueError(f"corpus export {split} split digest or path mismatch")
        count = (
            dataset.train_max_documents
            if split == "train"
            else dataset.validation_max_documents
        )
        budget = (
            dataset.train_max_tokens
            if split == "train"
            else dataset.validation_max_tokens
        )
        if (
            entry.get("records") != count
            or not isinstance(entry.get("rendered_bytes"), int)
            or budget < entry["rendered_bytes"] + count
        ):
            raise ValueError(
                f"corpus export {split} budget cannot include every record"
            )
    return {
        "release_id": manifest["release_id"],
        "view": view,
        "vocab_size": export["vocab_size"],
        "train_sha256": splits["train"]["sha256"],
        "validation_sha256": splits["validation"]["sha256"],
        "report_sha256": export["report_sha256"],
        "license_report_sha256": export["license_report_sha256"],
        "export_sha256": sha256_file(export_dir / "export.json"),
    }
