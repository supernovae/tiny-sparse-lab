from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def save_checkpoint(path: Path, state: dict[str, object]) -> None:
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    temp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = path.with_suffix(".json")
    manifest.write_text(
        json.dumps({"filename": path.name, "step": state["step"], "sha256": digest})
        + "\n"
    )
    (path.parent / "latest.json").write_text(manifest.read_text())


def load_checkpoint(path: Path) -> dict[str, object]:
    manifest = path.with_suffix(".json")
    if not manifest.is_file():
        raise ValueError(f"checkpoint manifest missing: {manifest}")
    expected = json.loads(manifest.read_text())["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "optimizer", "step", "tokens_seen", "cursor", "config"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("invalid checkpoint schema")
    return state
