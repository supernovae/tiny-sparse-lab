"""Validated immutable checkpoint generations and legacy v1 readers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from sparselab.training.manifest import canonical_json, sha256_file

FORMAT_VERSION = 2
SHARD_BYTES = 256 * 1024 * 1024


def _canonical(value: object) -> bytes:
    return canonical_json(value)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _safe_member(directory: Path, name: object) -> Path | None:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        return None
    relative = Path(name)
    if ".." in relative.parts or relative.name != name.split("/")[-1]:
        return None
    member = directory / relative
    try:
        resolved_directory = directory.resolve(strict=True)
        resolved_member = member.resolve(strict=True)
    except OSError:
        return None
    if resolved_directory not in (resolved_member, *resolved_member.parents):
        return None
    current = directory
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    return member


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


@dataclass(frozen=True)
class CheckpointRecord:
    generation_id: int
    relative_path: str
    manifest_sha256: str
    step: int
    tokens_seen: int
    created_at: str
    bytes: int
    validation_loss: float | None
    engine: str
    backend: str
    verification_status: str = "verified"
    resume_level: str = "full"


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    errors: tuple[dict[str, str], ...]
    verified_files: tuple[dict[str, str], ...]
    resume_level: str


@dataclass(frozen=True)
class RecoveryResult:
    record: CheckpointRecord | None
    rejected: tuple[VerificationReport, ...]


@dataclass
class TrainingSnapshot:
    model: dict[str, torch.Tensor]
    optimizer: dict[str, object]
    schedule: dict[str, object]
    step: int
    tokens_seen: int
    cursor: tuple[int, int]
    config: dict[str, object]
    run_id: str
    rng: dict[str, object] | None = None
    scaler: dict[str, object] | None = None
    validation_loss: float | None = None
    cadence: dict[str, object] | None = None
    engine: str = "pytorch"
    backend: str = "cpu"


class CheckpointManager:
    """Owns immutable v2 generations rooted at a single run directory."""

    def __init__(self, run_dir: Path, *, manifest_sha256: str | None = None) -> None:
        self.run_dir = run_dir
        self.root = run_dir / "checkpoints"
        self.manifest_sha256 = manifest_sha256
        self.root.mkdir(parents=True, exist_ok=True)

    def _next_generation(self) -> int:
        generations = [
            int(part.name.rsplit("_", 1)[-1])
            for part in self.root.glob("step_*_gen_*")
            if part.is_dir() and part.name.rsplit("_", 1)[-1].isdigit()
        ]
        return max(generations, default=0) + 1

    def _save_weights(
        self, destination: Path, tensors: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        aliases: dict[str, str] = {}
        canonical: dict[str, torch.Tensor] = {}
        seen: dict[tuple[int, int, tuple[int, ...], str], str] = {}
        for name, tensor in sorted(tensors.items()):
            key = (
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                str(tensor.dtype),
            )
            if key in seen:
                aliases[name] = seen[key]
            else:
                seen[key] = name
                canonical[name] = tensor.detach().cpu().contiguous()
        shards: list[dict[str, object]] = []
        current: dict[str, torch.Tensor] = {}
        size = 0
        shard_number = 1

        def flush() -> None:
            nonlocal current, size, shard_number
            if not current:
                return
            filename = f"weights-{shard_number:05d}.safetensors"
            path = destination / filename
            save_file(current, path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            shards.append(
                {
                    "name": filename,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "tensors": sorted(current),
                }
            )
            current, size = {}, 0
            shard_number += 1

        for name, tensor in canonical.items():
            tensor_bytes = tensor.numel() * tensor.element_size()
            if current and size + tensor_bytes > SHARD_BYTES:
                flush()
            current[name] = tensor
            size += tensor_bytes
        flush()
        inventory = {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "trainable": True,
            }
            for name, tensor in canonical.items()
        }
        return {"tensors": inventory, "aliases": aliases}, shards

    def save(
        self, snapshot: TrainingSnapshot, validation_loss: float | None = None
    ) -> CheckpointRecord:
        generation_id = self._next_generation()
        name = f"step_{snapshot.step:08d}_gen_{generation_id:06d}"
        final = self.root / name
        if final.exists():
            raise FileExistsError(f"checkpoint generation already exists: {final}")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".step_{snapshot.step}.", dir=self.root)
        )
        try:
            weight_index, shards = self._save_weights(temporary, snapshot.model)
            native = {
                "format_version": FORMAT_VERSION,
                "optimizer": snapshot.optimizer,
                "schedule": snapshot.schedule,
                "step": snapshot.step,
                "tokens_seen": snapshot.tokens_seen,
                "cursor": snapshot.cursor,
                "config": snapshot.config,
                "run_id": snapshot.run_id,
                "rng": snapshot.rng,
                "scaler": snapshot.scaler,
                "cadence": snapshot.cadence,
                "engine": snapshot.engine,
                "backend": snapshot.backend,
                "state_codec": "pytorch_native",
                "state_codec_version": 1,
            }
            native_path = temporary / "training_state.pt"
            torch.save(native, native_path)
            with native_path.open("rb") as handle:
                os.fsync(handle.fileno())
            files = [
                {
                    "name": "training_state.pt",
                    "sha256": _sha256(native_path),
                    "bytes": native_path.stat().st_size,
                },
                *shards,
            ]
            content = {
                "format_version": FORMAT_VERSION,
                "generation_id": generation_id,
                "step": snapshot.step,
                "tokens_seen": snapshot.tokens_seen,
                "created_at": datetime.now(UTC).isoformat(),
                "validation_loss": validation_loss,
                "manifest_sha256": self.manifest_sha256,
                "engine": snapshot.engine,
                "backend": snapshot.backend,
                "resume_level": "full",
                "state_codec": "pytorch_native",
                "state_codec_version": 1,
                "files": files,
                "weights": weight_index,
            }
            digest = hashlib.sha256(_canonical(content)).hexdigest()
            manifest = {**content, "sha256": digest}
            _atomic_json(temporary / "manifest.json", manifest)
            report = self.verify(temporary)
            if not report.valid:
                raise ValueError(f"checkpoint verification failed: {report.errors}")
            temporary.replace(final)
            _fsync_directory(self.root)
            record = CheckpointRecord(
                generation_id,
                name,
                digest,
                snapshot.step,
                snapshot.tokens_seen,
                str(content["created_at"]),
                sum(item["bytes"] for item in files),
                validation_loss,
                snapshot.engine,
                snapshot.backend,
            )
            _atomic_json(
                self.root / "latest.json", {**asdict(record), "relative_path": name}
            )
            if validation_loss is not None and (
                not (self.root / "best.json").is_file()
                or self._is_best(validation_loss)
            ):
                _atomic_json(
                    self.root / "best.json", {**asdict(record), "relative_path": name}
                )
            return record
        except BaseException:
            if temporary.exists():
                for item in temporary.iterdir():
                    item.unlink()
                temporary.rmdir()
            raise

    def _is_best(self, loss: float) -> bool:
        try:
            current = json.loads((self.root / "best.json").read_text())
            old = current.get("validation_loss")
            return not isinstance(old, (int, float)) or loss < old
        except (OSError, json.JSONDecodeError):
            return True

    def _resolve(self, path: Path) -> Path:
        if path.name not in {"latest.json", "best.json"}:
            return path
        if path.is_symlink():
            raise ValueError("unsafe checkpoint pointer")
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise TypeError("invalid checkpoint pointer")
        relative = record.get("relative_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("invalid checkpoint pointer")
        directory = path.parent / relative
        if directory.parent != path.parent or directory.is_symlink():
            raise ValueError("unsafe checkpoint pointer")
        return directory

    def verify(
        self,
        path: Path,
        expected_manifest: str | None = None,
        require_training_state: bool = True,
    ) -> VerificationReport:
        errors: list[dict[str, str]] = []
        files: list[dict[str, str]] = []
        try:
            pointer: dict[str, object] | None = None
            if path.name in {"latest.json", "best.json"}:
                loaded = json.loads(path.read_text())
                if not isinstance(loaded, dict):
                    raise ValueError("invalid checkpoint pointer")
                pointer = loaded
            directory = self._resolve(path)
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError("unsafe checkpoint directory")
            manifest_path = _safe_member(directory, "manifest.json")
            if manifest_path is None:
                raise ValueError("unsafe checkpoint manifest path")
            raw = json.loads(manifest_path.read_text())
            if not isinstance(raw, dict):
                raise TypeError("invalid checkpoint manifest")
            digest = raw.pop("sha256", None)
            if (
                not isinstance(digest, str)
                or hashlib.sha256(_canonical(raw)).hexdigest() != digest
            ):
                errors.append({"field": "manifest", "reason": "hash mismatch"})
            if pointer is not None and pointer.get("manifest_sha256") != digest:
                errors.append(
                    {"field": "pointer", "reason": "manifest digest mismatch"}
                )
            if raw.get("format_version") != FORMAT_VERSION:
                errors.append({"field": "format_version", "reason": "unsupported"})
            if (
                expected_manifest is not None
                and raw.get("manifest_sha256") != expected_manifest
            ):
                errors.append({"field": "manifest_sha256", "reason": "mismatch"})
            listed = raw.get("files")
            if not isinstance(listed, list):
                errors.append({"field": "files", "reason": "invalid"})
                listed = []
            names: set[str] = set()
            safe_members: dict[str, Path] = {}
            for entry in listed:
                name = entry.get("name") if isinstance(entry, dict) else None
                member = _safe_member(directory, name)
                if (
                    member is None
                    or not isinstance(entry, dict)
                    or name in names
                    or not isinstance(entry.get("sha256"), str)
                    or not isinstance(entry.get("bytes"), int)
                ):
                    errors.append(
                        {"field": "files", "reason": "unsafe or invalid member"}
                    )
                    continue
                names.add(name)
                safe_members[name] = member
                if (
                    not member.is_file()
                    or member.stat().st_size != entry["bytes"]
                    or _sha256(member) != entry["sha256"]
                ):
                    errors.append({"field": name, "reason": "hash or size mismatch"})
                else:
                    files.append({"name": name, "sha256": entry["sha256"]})
            if require_training_state and "training_state.pt" not in names:
                errors.append({"field": "training_state.pt", "reason": "missing"})
            if ("state_codec" in raw or "state_codec_version" in raw) and (
                raw.get("state_codec") != "pytorch_native"
                or raw.get("state_codec_version") != 1
            ):
                errors.append(
                    {"field": "state_codec", "reason": "unsupported or unknown codec"}
                )
            weights = raw.get("weights", {})
            tensors = weights.get("tensors", {}) if isinstance(weights, dict) else {}
            if not isinstance(tensors, dict):
                errors.append({"field": "weights", "reason": "invalid inventory"})
                tensors = {}
            found: dict[str, torch.Tensor] = {}
            for name, member in safe_members.items():
                if name.endswith(".safetensors"):
                    try:
                        found.update(load_file(member, device="cpu"))
                    except (
                        SafetensorError,
                        OSError,
                        ValueError,
                        RuntimeError,
                    ) as error:
                        errors.append(
                            {"field": name, "reason": f"invalid safetensors: {error}"}
                        )
            if set(tensors) != set(found):
                errors.append(
                    {"field": "weights", "reason": "tensor inventory mismatch"}
                )
            for name, tensor in found.items():
                declared = tensors.get(name)
                if (
                    not isinstance(declared, dict)
                    or declared.get("shape") != list(tensor.shape)
                    or declared.get("dtype") != str(tensor.dtype)
                ):
                    errors.append(
                        {
                            "field": f"weights.{name}",
                            "reason": "shape or dtype mismatch",
                        }
                    )
            aliases = weights.get("aliases", {}) if isinstance(weights, dict) else {}
            if not isinstance(aliases, dict):
                errors.append({"field": "aliases", "reason": "invalid"})
                aliases = {}
            for alias, target in aliases.items():
                if (
                    not isinstance(alias, str)
                    or not isinstance(target, str)
                    or alias in tensors
                    or target not in tensors
                ):
                    errors.append({"field": "aliases", "reason": "invalid alias"})
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            errors.append({"field": "checkpoint", "reason": str(error)})
        return VerificationReport(
            not errors,
            tuple(errors),
            tuple(files),
            "full" if require_training_state else "weights_only",
        )

    def load(
        self, path: Path, mode: Literal["resume", "promote"] = "resume"
    ) -> TrainingSnapshot:
        report = self.verify(
            path,
            self.manifest_sha256 if mode == "resume" else None,
            require_training_state=mode == "resume",
        )
        if not report.valid:
            raise ValueError(f"invalid checkpoint: {report.errors}")
        directory = self._resolve(path)
        raw = json.loads((directory / "manifest.json").read_text())
        weights: dict[str, torch.Tensor] = {}
        for entry in raw["files"]:
            if entry["name"].endswith(".safetensors"):
                weights.update(load_file(directory / entry["name"], device="cpu"))
        for alias, target in raw["weights"]["aliases"].items():
            weights[alias] = weights[target]
        if mode == "promote":
            return TrainingSnapshot(
                weights,
                {},
                {},
                0,
                0,
                (0, 0),
                {},
                "",
                engine=raw["engine"],
                backend=raw["backend"],
            )
        native = torch.load(
            directory / "training_state.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        return TrainingSnapshot(
            weights,
            native["optimizer"],
            native["schedule"],
            native["step"],
            native["tokens_seen"],
            tuple(native["cursor"]),
            native["config"],
            native["run_id"],
            native.get("rng"),
            native.get("scaler"),
            raw.get("validation_loss"),
            native.get("cadence"),
            native.get("engine", "pytorch"),
            native.get("backend", "cpu"),
        )

    def latest_valid(self) -> RecoveryResult:
        rejected: list[VerificationReport] = []
        candidates = sorted(self.root.glob("step_*_gen_*"), reverse=True)
        for candidate in candidates:
            report = self.verify(candidate, self.manifest_sha256)
            if report.valid:
                raw = json.loads((candidate / "manifest.json").read_text())
                return RecoveryResult(
                    CheckpointRecord(
                        raw["generation_id"],
                        candidate.name,
                        raw["sha256"],
                        raw["step"],
                        raw["tokens_seen"],
                        raw["created_at"],
                        sum(item["bytes"] for item in raw["files"]),
                        raw.get("validation_loss"),
                        raw["engine"],
                        raw["backend"],
                    ),
                    tuple(rejected),
                )
            rejected.append(report)
        return RecoveryResult(None, tuple(rejected))


# Legacy v1 writer/reader stay for historical tests and artifacts. v1 is promotion-only.
def save_checkpoint(path: Path, state: dict[str, object]) -> None:
    state = {**state, "format_version": 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    record: dict[str, object] = {
        "filename": path.name,
        "format_version": 1,
        "step": state["step"],
        "sha256": _sha256(path),
    }
    _atomic_json(path.with_suffix(".json"), record)
    _atomic_json(path.parent / "latest.json", record)


def load_checkpoint(path: Path) -> dict[str, object]:
    manifest = path.with_suffix(".json")
    if not manifest.is_file():
        raise ValueError(f"checkpoint manifest missing: {manifest}")
    record = json.loads(manifest.read_text())
    if _sha256(path) != record.get("sha256"):
        raise ValueError("checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "optimizer", "step", "tokens_seen", "cursor", "config"}
    if (
        not isinstance(state, dict)
        or not required <= state.keys()
        or state.get("format_version") != 1
    ):
        raise ValueError("invalid checkpoint schema")
    return state
