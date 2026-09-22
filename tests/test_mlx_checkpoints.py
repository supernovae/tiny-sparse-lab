from __future__ import annotations

import json
from pathlib import Path

from sparselab.training.mlx_checkpoints import inspect


def test_mlx_checkpoint_verification_reports_native_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_00000001"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text(
        json.dumps({"codec": "mlx_native", "version": 1, "step": 1})
    )
    (checkpoint / "weights.safetensors").write_bytes(b"weights")
    (checkpoint / "optimizer.safetensors").write_bytes(b"optimizer")

    report = inspect(checkpoint)

    assert report.valid
    assert report.metadata["step"] == 1
    assert {file["name"] for file in report.files} == {
        "weights.safetensors",
        "optimizer.safetensors",
    }


def test_mlx_checkpoint_verification_rejects_missing_optimizer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_00000001"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text(
        json.dumps({"codec": "mlx_native", "version": 1})
    )
    (checkpoint / "weights.safetensors").write_bytes(b"weights")

    report = inspect(checkpoint)

    assert not report.valid
    assert report.errors == ("missing checkpoint file: optimizer.safetensors",)
