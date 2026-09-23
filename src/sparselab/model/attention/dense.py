"""Bias-free dense causal self-attention over [batch, sequence, hidden]."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from sparselab.model.cache import AttentionKVCache
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
        self.window_size = window_size
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

    def _heads(self, projection: nn.Linear, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return (
            projection(x)
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, length, hidden = x.shape
        if length > self.causal_mask.shape[0]:
            raise ValueError("sequence length exceeds configured attention context")
        query, key, value = (
            self.rope(self._heads(self.q_proj, x)),
            self.rope(self._heads(self.k_proj, x)),
            self._heads(self.v_proj, x),
        )
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores.masked_fill_(self.causal_mask[:length, :length], float("-inf"))
        output = torch.softmax(scores.float(), dim=-1).to(value.dtype) @ value
        return self.out_proj(
            output.transpose(1, 2).contiguous().view(batch, length, hidden)
        )

    def create_cache(
        self, batch: int, capacity: int, *, device: torch.device, dtype: torch.dtype
    ) -> AttentionKVCache:
        """Allocate one bounded inference request's projected K/V storage."""
        if capacity <= 0:
            raise ValueError("KV cache capacity must be positive")
        shape = (batch, self.num_heads, capacity, self.head_dim)
        return AttentionKVCache(
            key=torch.empty(shape, device=device, dtype=dtype),
            value=torch.empty(shape, device=device, dtype=dtype),
            length=0,
            position=0,
        )

    def forward_cached(
        self, x: Tensor, cache: AttentionKVCache
    ) -> tuple[Tensor, AttentionKVCache]:
        """Evaluate appended tokens using bounded request-local projected K/V."""
        batch, length, hidden = x.shape
        offset = cache.position
        prior_length = cache.length
        query = self.rope(self._heads(self.q_proj, x), position_offset=offset)
        key = self.rope(self._heads(self.k_proj, x), position_offset=offset)
        value = self._heads(self.v_proj, x)
        cache.append(key, value, position=offset + length)
        keys = cache.key[:, :, : cache.length]
        values = cache.value[:, :, : cache.length]
        query_positions = torch.arange(offset, offset + length, device=x.device)
        key_positions = torch.arange(
            offset - prior_length, offset + length, device=x.device
        )
        allowed = key_positions[None, :] <= query_positions[:, None]
        if self.window_size is not None:
            allowed &= (
                key_positions[None, :] > query_positions[:, None] - self.window_size
            )
        scores = query @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores.masked_fill_(~allowed[None, None], float("-inf"))
        output = torch.softmax(scores.float(), dim=-1).to(values.dtype) @ values
        return (
            self.out_proj(
                output.transpose(1, 2).contiguous().view(batch, length, hidden)
            ),
            cache,
        )
