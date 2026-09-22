from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

FORMAT_VERSION = 1


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_checkpoint(path: Path, state: dict[str, object]) -> None:
    state = {**state, "format_version": FORMAT_VERSION}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    temp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record: dict[str, object] = {
        "filename": path.name,
        "format_version": FORMAT_VERSION,
        "step": state["step"],
        "sha256": digest,
    }
    _atomic_json(path.with_suffix(".json"), record)
    _atomic_json(path.parent / "latest.json", record)


def load_checkpoint(path: Path) -> dict[str, object]:
    manifest = path.with_suffix(".json")
    if not manifest.is_file():
        raise ValueError(f"checkpoint manifest missing: {manifest}")
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        expected = record["sha256"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("invalid checkpoint manifest") from error
    if (
        not isinstance(expected, str)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise ValueError("checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format_version",
        "model",
        "optimizer",
        "step",
        "tokens_seen",
        "cursor",
        "config",
    }
    if (
        not isinstance(state, dict)
        or not required <= state.keys()
        or state["format_version"] != FORMAT_VERSION
    ):
        raise ValueError("invalid checkpoint schema")
    return state
