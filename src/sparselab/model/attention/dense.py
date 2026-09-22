"""Bias-free dense causal self-attention over [batch, sequence, hidden]."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from sparselab.model.rope import RoPE


class DenseAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float,
        window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_base)
        causal_mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).triu(1)
        if window_size is not None:
            positions = torch.arange(max_seq_len)
            causal_mask |= positions[:, None] - positions[None, :] >= window_size
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, hidden = x.shape
        if length > self.causal_mask.shape[0]:
            raise ValueError("sequence length exceeds configured attention context")

        def heads(projection: nn.Linear) -> Tensor:
            return (
                projection(x)
                .view(batch, length, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        query, key, value = (
            self.rope(heads(self.q_proj)),
            self.rope(heads(self.k_proj)),
            heads(self.v_proj),
        )
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores.masked_fill_(self.causal_mask[:length, :length], float("-inf"))
        output = torch.softmax(scores, dim=-1) @ value
        return self.out_proj(
            output.transpose(1, 2).contiguous().view(batch, length, hidden)
        )
