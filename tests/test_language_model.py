from __future__ import annotations

import numpy as np
import pytest
import torch

from sparselab.data.packing import TokenBlockDataset
from sparselab.evaluation.language_model import evaluate


class UniformModel(torch.nn.Module):
    def forward(
        self, input_ids: torch.Tensor, *, byte_addresses: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.zeros((*input_ids.shape, 3), device=input_ids.device)


class RaisingModel(torch.nn.Module):
    def forward(
        self, input_ids: torch.Tensor, *, byte_addresses: torch.Tensor | None = None
    ) -> torch.Tensor:
        torch.rand(1)
        raise RuntimeError("evaluation failure")


def test_evaluation_counts_only_limited_held_out_targets() -> None:
    dataset = TokenBlockDataset(np.array([0, 1, 2, 1, 2, 0, 1], dtype=np.int32), 2)
    result = evaluate(
        UniformModel(), dataset, batch_size=2, max_batches=1, device=torch.device("cpu")
    )
    assert result["batches"] == 1
    assert result["valid_targets"] == 4
    assert result["loss"] == pytest.approx(torch.log(torch.tensor(3.0)).item())


def test_evaluation_restores_mode_and_rng_after_failure() -> None:
    dataset = TokenBlockDataset(np.array([0, 1, 2], dtype=np.int32), 2)
    model = RaisingModel()
    model.train()
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    with pytest.raises(RuntimeError, match="evaluation failure"):
        evaluate(
            model, dataset, batch_size=1, max_batches=1, device=torch.device("cpu")
        )
    assert model.training
    assert torch.equal(torch.rand(1), expected)
