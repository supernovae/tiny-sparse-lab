from __future__ import annotations

import torch

from sparselab.model.attention.dense import DenseAttention
from sparselab.model.attention.sparse import BlockSparseAttention, select_sparse_backend


def test_sparse_backend_dispatches_cpu_and_generic_devices() -> None:
    assert select_sparse_backend(torch.device("cpu")) == "cpu"
    assert select_sparse_backend(torch.device("cuda")) == "torch"


def test_block_sparse_attention_is_causal_and_reports_selection() -> None:
    attention = BlockSparseAttention(8, 2, 8, 10_000.0, block_size=2, selected_blocks=1)
    first = torch.randn(1, 6, 8)
    second = first.clone()
    second[:, 4:] = torch.randn_like(second[:, 4:])
    assert torch.allclose(attention(first)[:, :4], attention(second)[:, :4], atol=1e-6)
    diagnostics = attention.last_diagnostics
    assert attention.last_backend == "cpu"
    assert diagnostics is not None
    assert 0 < int(diagnostics.selected_tokens) < int(diagnostics.available_tokens)
    assert 0 < float(diagnostics.selection_ratio) < 1
    assert 0 < float(diagnostics.dense_teacher_mass) <= 1
    assert 0 < float(diagnostics.dense_teacher_topk_recall) <= 1


def test_sparse_matches_dense_when_every_causal_block_is_selected() -> None:
    dense = DenseAttention(8, 2, 8, 10_000.0)
    sparse = BlockSparseAttention(8, 2, 8, 10_000.0, block_size=2, selected_blocks=8)
    sparse.load_state_dict(dense.state_dict(), strict=True)
    values = torch.randn(2, 6, 8)
    assert torch.allclose(dense(values), sparse(values), atol=1e-6, rtol=1e-5)
    diagnostics = sparse.last_diagnostics
    assert diagnostics is not None
    assert torch.allclose(diagnostics.dense_teacher_mass, torch.ones(()))
    assert torch.allclose(diagnostics.dense_teacher_topk_recall, torch.ones(()))
