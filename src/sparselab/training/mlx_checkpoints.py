"""Inspection and integrity verification for local MLX native checkpoints."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLXCheckpointReport:
    valid: bool
    metadata: dict[str, object]
    files: tuple[dict[str, object], ...]
    errors: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> MLXCheckpointReport:
    directory = path.parent if path.name == "state.json" else path
    errors: list[str] = []
    metadata: dict[str, object] = {}
    try:
        metadata = json.loads((directory / "state.json").read_text())
        if metadata.get("codec") != "mlx_native" or metadata.get("version") != 1:
            errors.append("unsupported MLX checkpoint codec")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid state metadata: {error}")
    files = []
    for name in ("weights.safetensors", "optimizer.safetensors"):
        file = directory / name
        if not file.is_file() or file.is_symlink():
            errors.append(f"missing checkpoint file: {name}")
            continue
        files.append({"name": name, "bytes": file.stat().st_size, "sha256": _sha256(file)})
    return MLXCheckpointReport(not errors, metadata, tuple(files), tuple(errors))
