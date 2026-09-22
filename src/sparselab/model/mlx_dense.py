"""Canonical bias-free dense decoder executed by optional Apple MLX."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from sparselab.config.models import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return (
            x
            * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
            * self.weight
        )


class SwiGLU(nn.Module):
    def __init__(self, dims: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dims, hidden, bias=False)
        self.up = nn.Linear(dims, hidden, bias=False)
        self.down = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, dims: int, heads: int, rope_base: float = 10000.0) -> None:
        super().__init__()
        self.heads, self.head_dim, self.rope_base = heads, dims // heads, rope_base
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape

        def split(layer):
            return mx.transpose(
                layer(x).reshape(batch, length, self.heads, self.head_dim), (0, 2, 1, 3)
            )

        q = mx.fast.rope(
            split(self.q_proj),
            self.head_dim,
            traditional=True,
            base=self.rope_base,
            scale=1.0,
            offset=0,
        )
        k = mx.fast.rope(
            split(self.k_proj),
            self.head_dim,
            traditional=True,
            base=self.rope_base,
            scale=1.0,
            offset=0,
        )
        v = split(self.v_proj)
        scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) / (self.head_dim**0.5)
        scores = scores + mx.where(mx.triu(mx.ones((length, length)), k=1), -1e9, 0.0)
        output = mx.matmul(mx.softmax(scores, axis=-1), v)
        return self.out_proj(
            mx.transpose(output, (0, 2, 1, 3)).reshape(batch, length, -1)
        )


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attention = Attention(config.hidden_dim, config.num_heads)
        self.norm2 = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_dim, config.ffn_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class MLXDenseLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = [DecoderBlock(config) for _ in range(config.num_layers)]
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight

    def __call__(self, input_ids: mx.array) -> mx.array:
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))
