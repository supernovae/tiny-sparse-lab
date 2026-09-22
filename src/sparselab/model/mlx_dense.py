"""Dense causal decoder executed by optional Apple MLX."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from sparselab.config.models import ModelConfig


class MLXDenseLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.decoder = nn.TransformerEncoder(
            config.num_layers,
            config.hidden_dim,
            config.num_heads,
            mlp_dims=config.ffn_dim,
            dropout=0.0,
            norm_first=True,
        )
        self.norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight

    def __call__(self, input_ids: mx.array) -> mx.array:
        length = input_ids.shape[1]
        mask = mx.where(mx.triu(mx.ones((length, length)), k=1), -1e9, 0.0)
        return self.output(self.norm(self.decoder(self.embedding(input_ids), mask)))
