"""Reference causal block-selection attention without a dense QK matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from sparselab.model.rope import RoPE


@dataclass(frozen=True)
class SparseAttentionDiagnostics:
    available_tokens: Tensor
    selected_tokens: Tensor
    selection_ratio: Tensor
    selected_blocks: Tensor
    estimated_attention_flops: Tensor
    dense_teacher_mass: Tensor
    dense_teacher_topk_recall: Tensor


SparseBackend = Literal["cpu", "mps", "torch"]


def select_sparse_backend(device: torch.device) -> SparseBackend:
    """Choose a verified reference path without claiming an unavailable kernel."""
    if device.type == "cpu":
        return "cpu"
    if device.type == "mps":
        return "mps"
    return "torch"


class BlockSparseAttention(nn.Module):
    """Select causal K/V blocks from compressed keys, then attend to raw K/V.

    This deliberately loops over query positions for clarity.  It never forms
    a ``[B, H, T, T]`` score matrix: index scores use one mean key per causal
    block and final scores use only the selected original keys.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float,
        block_size: int,
        selected_blocks: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.block_size = block_size
        self.selected_blocks = selected_blocks
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_base)
        self.max_seq_len = max_seq_len
        self.last_diagnostics: SparseAttentionDiagnostics | None = None
        self.last_backend: SparseBackend | None = None

    def forward(self, x: Tensor) -> Tensor:
        batch, length, hidden = x.shape
        if length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")

        def heads(projection: nn.Linear) -> Tensor:
            return (
                projection(x)
                .view(batch, length, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        self.last_backend = select_sparse_backend(x.device)

        query, key, value = (
            self.rope(heads(self.q_proj)),
            self.rope(heads(self.k_proj)),
            heads(self.v_proj),
        )
        key_prefix = torch.cat(
            (torch.zeros_like(key[:, :, :1]), key.cumsum(dim=2)),
            dim=2,
        )
        result = torch.empty_like(query)
        selected_total = 0
        available_total = 0
        selected_block_ids: list[Tensor] = []
        dense_mass_total = torch.zeros((), device=x.device)
        dense_recall_total = torch.zeros((), device=x.device)
        for position in range(length):
            starts = torch.arange(0, position + 1, self.block_size, device=x.device)
            ends = (starts + self.block_size).clamp_max(position + 1)
            summaries = (
                key_prefix.index_select(2, ends) - key_prefix.index_select(2, starts)
            ) / (ends - starts).view(1, 1, -1, 1)
            block_count = starts.numel()
            index_scores = (query[:, :, position].unsqueeze(2) * summaries).sum(-1)
            chosen = index_scores.topk(
                min(self.selected_blocks, block_count), dim=-1
            ).indices
            selected_block_ids.append(chosen.detach())
            indices = torch.cat(
                [
                    torch.arange(
                        block * self.block_size,
                        min((block + 1) * self.block_size, position + 1),
                        device=x.device,
                    )
                    for block in range(block_count)
                ]
            )
            mask = torch.zeros_like(indices, dtype=torch.bool)
            for block in chosen.flatten().unique():
                mask |= (indices // self.block_size) == block
            indices = indices[mask]
            chosen_key, chosen_value = (
                key.index_select(2, indices),
                value.index_select(2, indices),
            )
            scores = (query[:, :, position].unsqueeze(2) * chosen_key).sum(
                -1
            ) / math.sqrt(self.head_dim)
            result[:, :, position] = (
                torch.softmax(scores, dim=-1).unsqueeze(-1).mul(chosen_value).sum(dim=2)
            )
            with torch.no_grad():
                dense_scores = (
                    query[:, :, position].unsqueeze(2) * key[:, :, : position + 1]
                ).sum(-1) / math.sqrt(self.head_dim)
                dense_probabilities = torch.softmax(dense_scores, dim=-1)
                dense_mass_total += (
                    dense_probabilities.index_select(2, indices).sum(dim=-1).mean()
                )
                dense_topk = dense_scores.topk(indices.numel(), dim=-1).indices
                dense_recall_total += torch.isin(dense_topk, indices).float().mean()
            selected_total += indices.numel()
            available_total += position + 1
        self.last_diagnostics = SparseAttentionDiagnostics(
            torch.tensor(available_total),
            torch.tensor(selected_total),
            torch.tensor(selected_total / available_total),
            torch.cat(selected_block_ids, dim=-1),
            torch.tensor(selected_total * self.head_dim),
            (dense_mass_total / length).detach(),
            (dense_recall_total / length).detach(),
        )
        return self.out_proj(
            result.transpose(1, 2).contiguous().view(batch, length, hidden)
        )
