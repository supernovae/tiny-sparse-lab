"""Immutable manifest and artifact identity support."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sparselab.runtime import RuntimeInfo


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode()


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
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml", ".md"} and "__pycache__" not in path.parts:
            inventory.append((str(path.relative_to(root)), sha256_file(path)))
    return {"algorithm": "package-path-sha256-v1", "files": inventory, "sha256": hashlib.sha256(canonical_json(inventory)).hexdigest()}


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
    manifest_version: int = 1
    architecture_version: str = "sparselab-decoder-v1"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str | None = None
    git_dirty: bool | None = None
    package_version: str | None = None
    python_version: str = field(default_factory=platform.python_version)
    artifacts: tuple[ArtifactIdentity, ...] = ()
    resource_decisions: tuple[dict[str, object], ...] = ()

    def payload(self) -> dict[str, object]:
        value = asdict(self)
        value["requested_config_sha256"] = hashlib.sha256(canonical_json(self.requested_config)).hexdigest()
        value["effective_config_sha256"] = hashlib.sha256(canonical_json(self.effective_config)).hexdigest()
        return value

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.payload())).hexdigest()


def write_manifest(path: Path, manifest: RunManifest) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.payload()
    data["sha256"] = hashlib.sha256(canonical_json(data)).hexdigest()
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json(data) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return str(data["sha256"])


def read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    actual = data.pop("sha256", None)
    if not isinstance(actual, str) or hashlib.sha256(canonical_json(data)).hexdigest() != actual:
        raise ValueError("manifest hash mismatch")
    return data
