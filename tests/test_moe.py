from __future__ import annotations

import torch

from sparselab.config.models import AttentionConfig, ModelConfig
from sparselab.model.inspection import inspect_model
from sparselab.model.moe import Top1MoE
from sparselab.model.transformer import DenseLM


def test_top_one_dispatch_preserves_token_order() -> None:
    moe = Top1MoE(hidden_dim=2, ffn_dim=4, num_experts=2)
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        for parameter in moe.experts[0].parameters():
            parameter.zero_()
        for parameter in moe.experts[1].parameters():
            parameter.zero_()
    values = torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [3.0, 0.0]]])
    output = moe(values)
    assert output.shape == values.shape
    assert moe.last_diagnostics is not None
    assert moe.last_diagnostics.counts.tolist() == [2, 1]
    assert torch.isclose(moe.last_diagnostics.fractions.sum(), torch.tensor(1.0))


def test_moe_decoder_runs_backward_and_reports_active_expert() -> None:
    config = ModelConfig(
        vocab_size=512,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=16,
        ffn="moe",
        num_experts=3,
    )
    model = DenseLM(config, AttentionConfig())
    logits = model(torch.randint(0, config.vocab_size, (2, 8)))
    logits.mean().backward()
    inspection = inspect_model(model)
    assert inspection["expert"] > 0
    assert inspection["ffn"] == 0
    assert inspection["active_per_token"] < inspection["total"]
    for block in model.blocks:
        assert isinstance(block.ffn, Top1MoE)
        assert block.ffn.last_diagnostics is not None
        assert int(block.ffn.last_diagnostics.counts.sum()) == 16
