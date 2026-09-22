from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from sparselab.config.models import RunConfig, TokenizerTrainConfig
from sparselab.data.tokenizer import train_tokenizer
from sparselab.training.mlx_checkpoints import inspect
from sparselab.training.trainer import train


def mlx_config(root: Path) -> RunConfig:
    tokenizer = train_tokenizer(
        TokenizerTrainConfig.model_validate(
            {
                "schema_version": 1,
                "vocab_size": 260,
                "min_frequency": 1,
                "max_documents": 20,
                "output_dir": root / "tokenizer",
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": root / "cache",
                    "train_max_documents": 20,
                    "validation_max_documents": 4,
                    "train_max_tokens": 512,
                    "validation_max_tokens": 128,
                    "synthetic_seed": 7,
                },
            }
        )
    )
    return RunConfig.model_validate(
        {
            "schema_version": 2,
            "name": "mlx-smoke",
            "seed": 7,
            "model": {
                "vocab_size": 260,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "max_seq_len": 16,
            },
            "tokenizer": {"path": tokenizer},
            "dataset": {
                "source": "synthetic",
                "cache_dir": root / "cache",
                "train_max_documents": 20,
                "validation_max_documents": 4,
                "train_max_tokens": 512,
                "validation_max_tokens": 128,
                "synthetic_seed": 7,
            },
            "training": {
                "seq_len": 16,
                "max_steps": 2,
                "max_tokens": 64,
                "micro_batch_size": 2,
            },
            "optimizer": {
                "name": "adamw",
                "peak": 0.003,
                "floor": 0.0003,
                "warmup_steps": 1,
            },
            "logging": {"root_dir": root / "runs"},
            "runtime": {"engine": "mlx", "backend": "metal"},
            "checkpoint": {"every_steps": 1},
        }
    )


def test_mlx_train_writes_verifiable_native_checkpoint(tmp_path: Path) -> None:
    config = mlx_config(tmp_path)

    run_id = train(config, run_id="mlx")

    checkpoint = config.logging.root_dir / run_id / "mlx_checkpoints" / "step_00000002"
    report = inspect(checkpoint)
    assert report.valid
    assert report.metadata["step"] == 2
    assert report.metadata["tokens"] == 64
