from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.engines.pytorch import PyTorchEngine
from sparselab.model.memory import ByteAddressMemory
from sparselab.data.tokenizer import train_tokenizer
from sparselab.model.transformer import DenseLM
from sparselab.research.portability import (
    PortabilityRun,
    apply_trainable_parameter_filter,
    initialize_recipient_backbone,
)
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import architecture_sha256, sha256_file
from sparselab.training.trainer import train

ROOT = Path(__file__).resolve().parents[1]


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


def test_recipient_checkpoint_load_preserves_backbone_and_freezes_memory(
    tmp_path: Path,
) -> None:
    tokenizer_config = load_tokenizer_config(ROOT / "configs/tokenizer_smoke.yaml")
    tokenizer_config = tokenizer_config.model_copy(
        update={
            "output_dir": tmp_path / "tokenizer",
            "dataset": tokenizer_config.dataset.model_copy(
                update={"cache_dir": tmp_path / "tokenizer-cache"}
            ),
        }
    )
    tokenizer_path = train_tokenizer(tokenizer_config)
    base = load_config(ROOT / "configs/runtime_smoke_cpu.yaml")
    base = base.model_copy(
        update={"tokenizer": base.tokenizer.model_copy(update={"path": tokenizer_path})}
    )
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


@pytest.mark.parametrize(
    ("max_steps", "expected_presentations"),
    ((4, 2), (8, 4)),
)
def test_learned_fact_exposure_threshold_tracks_budget(
    max_steps: int, expected_presentations: int
) -> None:
    model = torch.nn.Module()
    model.memory = ByteAddressMemory(2, 16, 2)
    parameter = model.memory.table.weight
    rows = (1, 2)
    initial_rows: dict[int, bytes] = {}
    with torch.no_grad():
        for row in rows:
            parameter[row].zero_()
            initial_rows[row] = bytes(
                parameter[row].contiguous().view(torch.uint8).numpy()
            )
            parameter[row].fill_(1.0)

    engine = PyTorchEngine()
    engine.config = SimpleNamespace(
        training=SimpleNamespace(max_steps=max_steps, micro_batch_size=1)
    )
    engine.model = model
    engine.portability_run = SimpleNamespace(
        payload={"coordinate": {"condition": "source-real"}}
    )
    engine.optimizer = torch.optim.AdamW([parameter])
    engine._portability_parameters = (("memory.table.weight", parameter),)
    engine._portability_update_history = [
        {
            "gradient_parameter_names": ["memory.table.weight"],
            "update_parameter_names": ["memory.table.weight"],
            "row_gradient_evidence": {},
        }
        for _ in range(max_steps)
    ]
    engine._learned_fact_targets = {1: 3, 2: 4}
    engine._learned_fact_ids_by_row = {1: "fact-a", 2: "fact-b"}
    engine._learned_initial_table_rows = initial_rows
    engine._learned_fact_exposures = {
        row: expected_presentations for row in rows
    }
    engine._learned_address_collision_targets = 7
    engine._learned_audit_parameter = parameter
    engine.optimizer.state[parameter]["sparselab_learned_audit_v1"] = {
        "applied_steps": torch.ones(max_steps, dtype=torch.bool),
        "gradient_seen": torch.ones(len(rows), dtype=torch.bool),
    }

    with patch.object(engine, "_source_memory_ablation", return_value={}):
        audit = engine._learned_byte_audit(completed=True)

    assert audit["expected_presentations_per_fact"] == expected_presentations
    assert audit["full_fact_exposure"] is True
    assert audit["address_collision_target_count"] == 7
