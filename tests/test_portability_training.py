from __future__ import annotations

import json
from pathlib import Path

import torch

from sparselab.config.loading import load_config
from sparselab.model.transformer import DenseLM
from sparselab.research.portability import (
    PortabilityRun,
    apply_trainable_parameter_filter,
    initialize_recipient_backbone,
)
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import architecture_sha256, sha256_file
from sparselab.training.trainer import train


def _directory_descriptor(root: Path, directory: Path) -> dict[str, object]:
    return {
        "path": directory.relative_to(root).as_posix(),
        "files": [
            {
                "path": item.relative_to(directory).as_posix(),
                "sha256": sha256_file(item),
                "size_bytes": item.stat().st_size,
            }
            for item in sorted(directory.rglob("*"))
            if item.is_file()
        ],
    }


def test_recipient_checkpoint_load_preserves_backbone_and_freezes_memory(tmp_path: Path) -> None:
    base = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    prepared = base.model_copy(
        update={
            "logging": base.logging.model_copy(update={"root_dir": tmp_path / "runs"}),
            "dataset": base.dataset.model_copy(
                update={"cache_dir": tmp_path / "cache"}
            ),
        }
    )
    train(prepared, run_id="recipient-prep")
    prep_run = prepared.logging.root_dir / "recipient-prep"
    manager = CheckpointManager(prep_run)
    record = manager.reconcile().record
    assert record is not None
    checkpoint = prep_run / "checkpoints" / record.relative_path
    descriptor = _directory_descriptor(tmp_path, checkpoint)
    checkpoint_snapshot = manager.load(checkpoint, mode="promote")
    checkpoint_digest = checkpoint_snapshot.checkpoint_sha256

    selected = ("memory.output.weight", "memory.gate.weight")
    target = prepared.model_copy(
        update={
            "model": prepared.model.model_copy(
                update={
                    "memory": "ngram",
                    "memory_table_size": 257,
                    "memory_ngram_size": 4,
                    "memory_dim": 8,
                }
            ),
            "training": prepared.training.model_copy(
                update={
                    "trainable_parameters": selected,
                    "portability_manifest_path": tmp_path / "run.json",
                }
            ),
        }
    )
    portability = PortabilityRun(
        path=tmp_path / "run.json",
        root=tmp_path,
        payload={},
        backbone={
            "checkpoint": descriptor,
            "checkpoint_sha256": checkpoint_digest,
            "architecture_sha256": architecture_sha256(
                json.loads((prep_run / "manifest.json").read_text())["effective_config"]
            ),
            "tokenizer_sha256": sha256_file(target.tokenizer.path),
        },
        memory={"kind": "token", "artifact": None},
    )
    model = DenseLM(target.model, target.attention)
    snapshot = initialize_recipient_backbone(model, target, portability)
    assert snapshot is not None
    for name, tensor in snapshot.items():
        assert torch.equal(model.state_dict()[name].cpu(), tensor.cpu()), name

    parameters = apply_trainable_parameter_filter(model, target, portability)
    assert tuple(name for name, _ in parameters) == selected
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(selected)

