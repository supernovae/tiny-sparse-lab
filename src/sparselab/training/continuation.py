"""Verified same-experiment continuation versus explicit weight promotion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sparselab.config.models import RunConfig
from sparselab.runtime import RuntimeInfo
from sparselab.training.checkpoints import CheckpointManager, TrainingSnapshot
from sparselab.training.manifest import (
    architecture_sha256,
    canonical_json,
    config_sha256,
    read_manifest,
    sha256_file,
)


@dataclass(frozen=True)
class Continuation:
    kind: Literal["FRESH", "RESUMED", "PROMOTED"] = "FRESH"
    snapshot: TrainingSnapshot | None = None
    source_run: Path | None = None
    parent_run_id: str | None = None
    parent_checkpoint_sha256: str | None = None
    decisions: tuple[dict[str, object], ...] = ()


def _resume_settings(config: RunConfig) -> str:
    settings = config.model_dump(mode="json")
    for field in ("name", "logging", "checkpoint"):
        settings.pop(field)
    settings["runtime"].pop("device_index")
    return config_sha256(settings)


def _verify_artifacts(root: Path, manifest: dict[str, Any], *, resume: bool) -> None:
    records = manifest.get("artifacts", [])
    names: set[str] = set()
    required = {"tokenizer.json"}
    if resume:
        required.update({"data/manifest.json", "data/train.npy", "data/validation.npy"})
    for record in records:
        name = record["relative_path"]
        if name in names:
            raise ValueError(f"duplicate run artifact: {name}")
        names.add(name)
        if (
            not resume
            and name != "tokenizer.json"
            and not name.startswith("portable_package")
        ):
            continue
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe run artifact: {name}")
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"symlinked run artifact: {name}")
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(root)
            or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"run artifact integrity failure: {name}")
    if not required <= names:
        raise ValueError(
            f"run lacks bound continuation artifacts: {sorted(required - names)}; "
            "use compatible weight promotion for historical incomplete state"
        )


def _package_identity(path: Path) -> dict[str, str]:
    if path.is_file():
        return {".": sha256_file(path)}
    if not path.is_dir():
        raise ValueError(f"portable package is missing: {path}")
    inventory = {}
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"symlinked portable package member: {member}")
        if member.is_file():
            inventory[str(member.relative_to(path))] = sha256_file(member)
    return inventory


def load_continuation(
    config: RunConfig,
    runtime: RuntimeInfo,
    current_source: dict[str, object],
    *,
    resume: Path | None,
    promote: Path | None,
    recover: Path | None,
    allow_runtime_drift: bool,
) -> Continuation:
    selected = resume or promote
    if selected is None and recover is None:
        return Continuation()
    if recover is not None:
        root = recover.resolve()
    else:
        assert selected is not None
        root = selected.parent.parent.resolve()
    manifest = read_manifest(root / "manifest.json")
    manifest_digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manager = CheckpointManager(root, manifest_sha256=manifest_digest)
    decisions: list[dict[str, object]] = []
    with manager.writer_lease():
        if recover is not None:
            recovery = manager.reconcile()
            if recovery.record is None:
                raise ValueError(
                    f"no valid checkpoint for recovery: {recovery.rejected}"
                )
            selected = root / "checkpoints" / recovery.record.relative_path
            decisions.append(
                {
                    "kind": "recovery",
                    "checkpoint_sha256": recovery.record.manifest_sha256,
                    "rejected": [list(report.errors) for report in recovery.rejected],
                    "reason": "explicit recovery from verified finalized generations",
                }
            )
        assert selected is not None
        snapshot = manager.load(
            selected, "promote" if promote is not None else "resume"
        )
        # Promotion verifies the source binding without loading its optimizer state.
        binding = manager.verify(
            selected, expected_manifest=manifest_digest, require_training_state=False
        )
        if not binding.valid:
            raise ValueError(f"checkpoint run binding failed: {binding.errors}")
        _verify_artifacts(root, manifest, resume=promote is None)
        source_config = RunConfig.model_validate(manifest["effective_config"])
        if architecture_sha256(
            source_config.model_dump(mode="json")
        ) != architecture_sha256(config.model_dump(mode="json")):
            raise ValueError(
                "continuation configuration differs in architecture semantics"
            )
        if promote is not None:
            if sha256_file(root / "tokenizer.json") != sha256_file(
                config.tokenizer.path
            ):
                raise ValueError("promotion requires matching tokenizer identity")
            if config.model.memory_package_path is not None and _package_identity(
                root / "portable_package"
            ) != _package_identity(config.model.memory_package_path):
                raise ValueError(
                    "promotion requires matching portable package identity"
                )
            return Continuation(
                "PROMOTED",
                snapshot,
                root,
                manifest["run_id"],
                snapshot.checkpoint_sha256,
                tuple(decisions),
            )
        saved_config = RunConfig.model_validate(snapshot.config)
        if config_sha256(snapshot.config) != config_sha256(
            source_config.model_dump(mode="json")
        ) or snapshot.source_identity_sha256 != manifest.get("source_identity", {}).get(
            "sha256"
        ):
            raise ValueError(
                "checkpoint scientific identity differs from its source run manifest"
            )
        if _resume_settings(saved_config) != _resume_settings(config):
            raise ValueError(
                "resume configuration differs from checkpoint; use promotion for changed scientific settings"
            )
        if snapshot.engine != runtime.engine or snapshot.backend != runtime.backend:
            raise ValueError("full resume requires the same engine and backend")
        drift: list[dict[str, object]] = []
        old_source = manifest.get("source_identity", {}).get("sha256")
        if old_source != current_source["sha256"]:
            drift.append(
                {
                    "field": "source_identity_sha256",
                    "requested": old_source,
                    "effective": current_source["sha256"],
                }
            )
        old_runtime = manifest["runtime"]
        for field in (
            "framework_version",
            "runtime_version",
            "driver_version",
            "device_name",
            "device_index",
            "physical_device_id",
            "os",
        ):
            previous, actual = old_runtime.get(field), getattr(runtime, field)
            if previous != actual:
                drift.append(
                    {
                        "field": f"runtime.{field}",
                        "requested": previous,
                        "effective": actual,
                    }
                )
        if drift and not allow_runtime_drift:
            raise ValueError(
                "source or runtime identity differs; --allow-runtime-drift is required "
                f"for best-effort same-backend resume: {drift}"
            )
        if drift:
            decisions.append(
                {
                    "kind": "runtime_drift",
                    "resume_level": "best_effort",
                    "changes": drift,
                    "reason": "explicit --allow-runtime-drift",
                }
            )
        return Continuation(
            "RESUMED",
            snapshot,
            root,
            manifest["run_id"],
            snapshot.checkpoint_sha256,
            tuple(decisions),
        )
