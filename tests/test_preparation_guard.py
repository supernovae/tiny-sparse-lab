"""The Card 03 preparation guard refuses every model-update entry point."""

from __future__ import annotations

import pytest

from sparselab.staging import stage
from sparselab.training.pilot import main, run_pilot
from sparselab.training.trainer import _train_impl, train


def test_preparation_only_blocks_model_training_and_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("SPARSELAB_PREPARATION_ONLY", "1")
    for invoke in (
        lambda: train(None),
        lambda: _train_impl(None),
        lambda: stage(None, tmp_path / "stage"),
        lambda: run_pilot(tmp_path, "smoke"),
        main,
    ):
        with pytest.raises(RuntimeError, match="SPARSELAB_PREPARATION_ONLY"):
            invoke()
