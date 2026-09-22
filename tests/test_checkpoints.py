from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sparselab.training.checkpoints import (
    CheckpointManager,
    TrainingSnapshot,
    load_checkpoint,
    save_checkpoint,
)


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


def test_pointer_digest_must_bind_selected_generation(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(
        TrainingSnapshot(
            model={"weight": torch.tensor([1.0])},
            optimizer={},
            schedule={"kind": "warmup_cosine_v1"},
            step=1,
            tokens_seen=4,
            cursor=(0, 1),
            config={},
            run_id="run",
        )
    )
    pointer = tmp_path / "checkpoints/latest.json"
    payload = json.loads(pointer.read_text())
    payload["manifest_sha256"] = "0" * 64
    pointer.write_text(json.dumps(payload))
    report = manager.verify(pointer)
    assert not report.valid
    assert any(error["field"] == "pointer" for error in report.errors)


def test_checkpoint_verification_rejects_tampered_weight_file(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    record = manager.save(
        TrainingSnapshot(
            model={"weight": torch.tensor([1.0])},
            optimizer={},
            schedule={"kind": "warmup_cosine_v1"},
            step=1,
            tokens_seen=4,
            cursor=(0, 1),
            config={},
            run_id="run",
        )
    )
    generation = tmp_path / "checkpoints" / record.relative_path
    weights = next(generation.glob("*.safetensors"))
    weights.write_bytes(weights.read_bytes() + b"tampered")
    assert not manager.verify(generation).valid


def test_checkpoint_allows_symlinked_ancestor_but_not_member(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    manager = CheckpointManager(alias / "run")
    record = manager.save(
        TrainingSnapshot(
            model={"weight": torch.tensor([1.0])},
            optimizer={},
            schedule={},
            step=1,
            tokens_seen=4,
            cursor=(0, 1),
            config={},
            run_id="run",
        )
    )
    generation = alias / "run/checkpoints" / record.relative_path
    loaded = manager.load(generation)
    assert torch.equal(loaded.model["weight"], torch.tensor([1.0]))
    weights = next(generation.glob("*.safetensors"))
    outside = real / "outside.safetensors"
    weights.replace(outside)
    weights.symlink_to(outside)
    assert not manager.verify(generation).valid
