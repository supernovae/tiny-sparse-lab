from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sparselab.training.checkpoints import load_checkpoint, save_checkpoint


def state() -> dict[str, object]:
    return {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {},
        "schedule": {"max_steps": 2},
        "step": 1,
        "tokens_seen": 4,
        "cursor": (0, 1),
        "config": {"schema_version": 1},
    }


def test_checkpoint_requires_matching_manifest_hash(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, state())
    loaded = load_checkpoint(path)
    assert loaded["step"] == 1
    assert loaded["schedule"] == {"max_steps": 2}
    record = json.loads(path.with_suffix(".json").read_text())
    latest = json.loads((path.parent / "latest.json").read_text())
    assert record["format_version"] == 1
    assert latest == record
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path)


def test_checkpoint_requires_manifest(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(state(), path)
    with pytest.raises(ValueError, match="manifest missing"):
        load_checkpoint(path)
