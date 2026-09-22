"""Reference causal multi-head latent attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from sparselab.model.rope import RoPE


class LatentAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        latent_dim: int,
        max_seq_len: int,
        rope_base: float,
    ) -> None:
        super().__init__()
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.value_dim = latent_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_down = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_base)
        self.register_buffer(
            "causal_mask",
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        if length > self.causal_mask.shape[0]:
            raise ValueError("sequence length exceeds configured attention context")
        query = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        query = self.rope(query.transpose(1, 2))
        latent = self.kv_down(x)
        key = self.k_up(latent).view(batch, length, self.num_heads, self.head_dim)
        key = self.rope(key.transpose(1, 2))
        value = latent.view(batch, length, self.num_heads, self.value_dim).transpose(
            1, 2
        )
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores.masked_fill_(self.causal_mask[:length, :length], float("-inf"))
        output = torch.softmax(scores, dim=-1) @ value
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, -1))
