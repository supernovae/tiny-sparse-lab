"""Explicit pre-norm dense decoder language model."""

from __future__ import annotations

from torch import Tensor, nn

from sparselab.config.models import AttentionConfig, ModelConfig
from sparselab.model.attention.dense import DenseAttention
from sparselab.model.attention.latent import LatentAttention
from sparselab.model.ffn import SwiGLU
from sparselab.model.memory import ByteAddressMemory, TokenNgramMemory
from sparselab.model.moe import Top1MoE
from sparselab.model.norm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, model: ModelConfig, attention: AttentionConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(model.hidden_dim, model.rms_norm_eps)
        self.attention: nn.Module = (
            LatentAttention(
                model.hidden_dim,
                model.num_heads,
                attention.latent_dim,
                model.max_seq_len,
                attention.rope_base,
            )
            if attention.kind == "mla"
            else DenseAttention(
                model.hidden_dim,
                model.num_heads,
                model.max_seq_len,
                attention.rope_base,
                attention.window_size,
            )
        )
        self.norm2 = RMSNorm(model.hidden_dim, model.rms_norm_eps)
        self.ffn: nn.Module = (
            SwiGLU(model.hidden_dim, model.ffn_dim)
            if model.ffn == "dense"
            else Top1MoE(model.hidden_dim, model.ffn_dim, model.num_experts)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class DenseLM(nn.Module):
    def __init__(self, model: ModelConfig, attention: AttentionConfig) -> None:
        super().__init__()
        self.config = model
        self.embedding = nn.Embedding(model.vocab_size, model.hidden_dim)
        self.blocks = nn.ModuleList(
            DecoderBlock(model, attention) for _ in range(model.num_layers)
        )
        self.norm = RMSNorm(model.hidden_dim, model.rms_norm_eps)
        self.memory: nn.Module | None = (
            None
            if model.memory == "none"
            else (
                TokenNgramMemory(
                    model.hidden_dim,
                    model.memory_table_size,
                    model.memory_ngram_size,
                    model.memory_dim,
                )
                if model.memory == "ngram"
                else ByteAddressMemory(
                    model.hidden_dim, model.memory_table_size, model.memory_dim
                )
            )
        )
        self.output = nn.Linear(model.hidden_dim, model.vocab_size, bias=False)
        self.apply(self._initialize)
        if model.tie_embeddings:
            self.output.weight = self.embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self, input_ids: Tensor, *, byte_addresses: Tensor | None = None
    ) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds model.max_seq_len")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if isinstance(self.memory, TokenNgramMemory):
            x = self.memory(x, input_ids)
        elif isinstance(self.memory, ByteAddressMemory):
            if byte_addresses is None:
                raise ValueError("byte memory requires prepared byte addresses")
            x = self.memory(x, byte_addresses)
        return self.output(x)
