from __future__ import annotations

import torch

from sparselab.model.attention.sparse import BlockSparseAttention


def test_block_sparse_attention_is_causal_and_reports_selection() -> None:
    attention = BlockSparseAttention(8, 2, 8, 10_000.0, block_size=2, selected_blocks=1)
    first = torch.randn(1, 6, 8)
    second = first.clone()
    second[:, 4:] = torch.randn_like(second[:, 4:])
    assert torch.allclose(attention(first)[:, :4], attention(second)[:, :4], atol=1e-6)
    diagnostics = attention.last_diagnostics
    assert diagnostics is not None
    assert 0 < int(diagnostics.selected_tokens) < int(diagnostics.available_tokens)
    assert 0 < float(diagnostics.selection_ratio) < 1
