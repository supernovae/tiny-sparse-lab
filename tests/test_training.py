from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sparselab.config.migrate import migrate_v1
from sparselab.config.models import RunConfig, TokenizerTrainConfig
from sparselab.data.tokenizer import train_tokenizer
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.trainer import train


def equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            equal(first, second)
    else:
        assert left == right


def config(root: Path) -> RunConfig:
    tokenizer = train_tokenizer(
        TokenizerTrainConfig.model_validate(
            {
                "schema_version": 1,
                "vocab_size": 512,
                "min_frequency": 1,
                "max_documents": 200,
                "output_dir": root / "tokenizer",
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": root / "cache",
                    "train_max_documents": 200,
                    "validation_max_documents": 40,
                    "train_max_tokens": 8192,
                    "validation_max_tokens": 2048,
                    "synthetic_seed": 7,
                },
            }
        )
    )
    return RunConfig.model_validate(
        migrate_v1(
            {
                "schema_version": 1,
                "name": "test",
                "seed": 7,
                "device": "cpu",
                "model": {
                    "vocab_size": 512,
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "max_seq_len": 32,
                },
                "tokenizer": {"path": tokenizer},
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": root / "cache",
                    "train_max_documents": 200,
                    "validation_max_documents": 40,
                    "train_max_tokens": 8192,
                    "validation_max_tokens": 2048,
                    "synthetic_seed": 7,
                },
                "training": {
                    "batch_size": 2,
                    "seq_len": 16,
                    "max_steps": 12,
                    "max_tokens": 384,
                    "deterministic": True,
                },
                "optimizer": {
                    "learning_rate": 0.003,
                    "min_learning_rate": 0.0003,
                    "warmup_steps": 2,
                },
                "logging": {"root_dir": root / "runs", "checkpoint_every_steps": 6},
            }
        )
    )


def test_interrupted_resume_matches_uninterrupted(tmp_path: Path) -> None:
    full = config(tmp_path / "full")
    train(full, run_id="full")
    split = config(tmp_path / "split")
    train(split, run_id="part", stop_after_step=5)
    train(
        split,
        run_id="resumed",
        resume=split.logging.root_dir / "part/checkpoints/latest.json",
    )
    left = CheckpointManager(full.logging.root_dir / "full").load(
        full.logging.root_dir / "full/checkpoints/latest.json"
    )
    right = CheckpointManager(split.logging.root_dir / "resumed").load(
        split.logging.root_dir / "resumed/checkpoints/latest.json"
    )
    assert left.tokens_seen == right.tokens_seen == 384
    assert left.cursor == right.cursor
    equal(left.optimizer, right.optimizer)
    equal(left.model, right.model)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_mps_interrupted_checkpoint_resumes_locally(tmp_path: Path) -> None:
    original = config(tmp_path)
    mps = original.model_copy(
        update={"runtime": original.runtime.model_copy(update={"backend": "mps"})}
    )
    train(mps, run_id="part", stop_after_step=2)
    train(
        mps,
        run_id="resumed",
        resume=mps.logging.root_dir / "part/checkpoints/latest.json",
    )

    resumed = CheckpointManager(mps.logging.root_dir / "resumed").load(
        mps.logging.root_dir / "resumed/checkpoints/latest.json"
    )
    assert resumed.step == mps.training.max_steps
    assert resumed.backend == "mps"



def test_adafactor_state_offload_is_rejected(tmp_path: Path) -> None:
    payload = config(tmp_path).model_dump(mode="json")
    payload["optimizer"] = {
        "name": "adafactor",
        "peak": 0.003,
        "floor": 0.0003,
        "warmup_steps": 2,
        "weight_decay": 0.1,
        "state_offload": True,
    }

    with pytest.raises(ValueError, match="state_offload is deferred"):
        RunConfig.model_validate(payload)

def test_resume_rejects_model_configuration_mismatch(tmp_path: Path) -> None:
    original = config(tmp_path / "original")
    train(original, run_id="part", stop_after_step=5)
    incompatible = original.model_copy(
        update={"model": original.model.model_copy(update={"ffn_dim": 48})}
    )
    with pytest.raises(ValueError, match="configuration differs"):
        train(
            incompatible,
            run_id="rejected",
            resume=original.logging.root_dir / "part/checkpoints/latest.json",
        )
