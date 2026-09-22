"""Immutable manifest and artifact identity support."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sparselab.runtime import RuntimeInfo

MANIFEST_VERSION = 1
IDENTITY_VERSION = "run-identity-v2"
ARCHITECTURE_VERSION = "sparselab-decoder-v1"
_MACHINE_LOCAL_PATH_KEYS = frozenset(
    {
        "path",
        "cache_dir",
        "root_dir",
        "output_dir",
        "memory_package_path",
        "train_path",
        "validation_path",
    }
)


def canonical_json(value: object) -> bytes:
    """Encode only JSON-native data in a canonical, finite representation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _without_machine_local_paths(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_machine_local_paths(item)
            for key, item in value.items()
            if key not in _MACHINE_LOCAL_PATH_KEYS
        }
    if isinstance(value, list):
        return [_without_machine_local_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_without_machine_local_paths(item) for item in value]
    return value


def architecture_sha256(config: Mapping[str, Any]) -> str:
    """Digest model semantics, excluding the separately inventoried package location."""
    model = config.get("model", config)
    if not isinstance(model, Mapping):
        raise TypeError("architecture identity requires a mapping-valued model")
    attention = config.get("attention", {})
    if not isinstance(attention, Mapping):
        raise TypeError("architecture identity requires a mapping-valued attention")
    model_identity = dict(model)
    model_identity.pop("memory_package_path", None)
    return _digest(
        {
            "identity_version": IDENTITY_VERSION,
            "architecture_version": config.get(
                "architecture_version", ARCHITECTURE_VERSION
            ),
            "model": model_identity,
            "attention": dict(attention),
        }
    )


def config_sha256(config: Mapping[str, Any]) -> str:
    """Digest scientific configuration while excluding machine-local locations."""
    return _digest(
        {
            "identity_version": IDENTITY_VERSION,
            "config": _without_machine_local_paths(config),
        }
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity() -> dict[str, object]:
    root = Path(__file__).parents[1]
    inventory: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.suffix in {".py", ".json", ".yaml", ".yml", ".md"}
            and "__pycache__" not in path.parts
        ):
            inventory.append((str(path.relative_to(root)), sha256_file(path)))
    return {
        "algorithm": "package-path-sha256-v1",
        "files": inventory,
        "sha256": _digest(inventory),
    }


@dataclass(frozen=True)
class ArtifactIdentity:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    name: str
    runtime: RuntimeInfo
    requested_config: dict[str, object]
    effective_config: dict[str, object]
    architecture_sha256: str
    source_identity: dict[str, object]
    worker_id: str
    continuation_kind: str = "FRESH"
    experiment_id: str | None = None
    attempt_id: str | None = None
    parent_run_id: str | None = None
    checkpoint_sha256: str | None = None
    purpose: str = "training"
    manifest_version: int = MANIFEST_VERSION
    architecture_version: str = ARCHITECTURE_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str | None = None
    git_dirty: bool | None = None
    package_version: str | None = None
    python_version: str = field(default_factory=platform.python_version)
    artifacts: tuple[ArtifactIdentity, ...] = ()
    resource_decisions: tuple[dict[str, object], ...] = ()

    def payload(self) -> dict[str, object]:
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version: {self.manifest_version}")
        value = asdict(self)
        value["identity_version"] = IDENTITY_VERSION
        value["requested_config_sha256"] = config_sha256(self.requested_config)
        value["effective_config_sha256"] = config_sha256(self.effective_config)
        return value

    def digest(self) -> str:
        return _digest(self.payload())


def write_manifest(path: Path, manifest: RunManifest) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.payload()
    data["sha256"] = _digest(data)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json(data) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(data["sha256"])


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_current_identity(data: dict[str, Any]) -> None:
    requested, effective = data.get("requested_config"), data.get("effective_config")
    if not isinstance(requested, Mapping) or not isinstance(effective, Mapping):
        raise TypeError(
            "manifest identity requires requested and effective config mappings"
        )
    if data.get("requested_config_sha256") != config_sha256(requested):
        raise ValueError("manifest requested config hash mismatch")
    if data.get("effective_config_sha256") != config_sha256(effective):
        raise ValueError("manifest effective config hash mismatch")
    expected_architecture = architecture_sha256(effective)
    if data.get("architecture_sha256") != expected_architecture:
        raise ValueError("manifest architecture hash mismatch")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("manifest artifacts must be an array")
    names: set[str] = set()
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("relative_path"), str)
            or not artifact["relative_path"]
            or Path(artifact["relative_path"]).is_absolute()
            or ".." in Path(artifact["relative_path"]).parts
            or artifact["relative_path"] in names
            or not _is_sha256(artifact.get("sha256"))
            or isinstance(artifact.get("size_bytes"), bool)
            or not isinstance(artifact.get("size_bytes"), int)
            or artifact["size_bytes"] < 0
        ):
            raise ValueError("manifest artifact inventory is invalid")
        names.add(artifact["relative_path"])


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a supported, hash-verified manifest without reinterpreting v1 digests."""
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid manifest {path}: {error}") from error
    if not isinstance(data, dict):
        raise TypeError("manifest must be a JSON object")
    version = data.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version: {version!r}")
    actual = data.pop("sha256", None)
    if not isinstance(actual, str) or _digest(data) != actual:
        raise ValueError("manifest hash mismatch")
    identity_version = data.get("identity_version")
    if identity_version is not None and identity_version != IDENTITY_VERSION:
        raise ValueError(f"unsupported manifest identity version: {identity_version!r}")
    if identity_version == IDENTITY_VERSION:
        _validate_current_identity(data)
    return data
