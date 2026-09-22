"""Explicit pre-norm dense decoder language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sparselab.config.models import AttentionConfig, ModelConfig
from sparselab.model.attention.dense import DenseAttention
from sparselab.model.attention.latent import LatentAttention
from sparselab.model.attention.sparse import BlockSparseAttention
from sparselab.model.ffn import SwiGLU
from sparselab.model.memory import ByteAddressMemory, TokenNgramMemory
from sparselab.model.moe import TopKMoE
from sparselab.model.norm import RMSNorm
from sparselab.model.portable_engram import PortableEngramAdapter, load_portable_engram


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
            else BlockSparseAttention(
                model.hidden_dim,
                model.num_heads,
                model.max_seq_len,
                attention.rope_base,
                attention.block_size,
                attention.selected_blocks,
            )
            if attention.kind == "block_sparse"
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
            else TopKMoE(
                model.hidden_dim,
                model.ffn_dim,
                model.num_experts,
                model.experts_per_token,
                model.shared_expert,
                model.router_aux_loss_coefficient,
            )
        )

    def forward_with_aux(
        self, x: Tensor, *, valid_target_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        x = x + self.attention(self.norm1(x))
        if isinstance(self.ffn, TopKMoE):
            x = x + self.ffn(self.norm2(x), valid_target_mask=valid_target_mask)
            return x, self.ffn.auxiliary_loss
        return x + self.ffn(self.norm2(x)), x.new_zeros(())

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_aux(x)[0]


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
                    model.memory_ngram_orders,
                    model.memory_hash_heads,
                )
                if model.memory == "ngram"
                else (
                    ByteAddressMemory(
                        model.hidden_dim, model.memory_table_size, model.memory_dim
                    )
                    if model.memory == "byte"
                    else PortableEngramAdapter(
                        load_portable_engram(model.memory_package_path),
                        model.hidden_dim,
                    )
                )
            )
        )
        self.output = nn.Linear(model.hidden_dim, model.vocab_size, bias=False)
        self.apply(self._initialize)
        if model.tie_embeddings:
            self.output.weight = self.embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if (
            isinstance(module, (nn.Linear, nn.Embedding))
            and module.weight.requires_grad
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward_with_aux(
        self,
        input_ids: Tensor,
        *,
        byte_addresses: Tensor | None = None,
        valid_target_mask: Tensor | None = None,
        activation_checkpointing: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds model.max_seq_len")
        x = self.embedding(input_ids)
        auxiliary_loss = x.new_zeros(())
        for block in self.blocks:
            if activation_checkpointing and self.training and torch.is_grad_enabled():
                def run_block(hidden: Tensor, current: DecoderBlock = block) -> tuple[Tensor, Tensor]:
                    return current.forward_with_aux(
                        hidden, valid_target_mask=valid_target_mask
                    )

                x, block_aux = torch.utils.checkpoint.checkpoint(
                    run_block,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                x, block_aux = block.forward_with_aux(
                    x, valid_target_mask=valid_target_mask
                )
            auxiliary_loss = auxiliary_loss + block_aux
        x = self.norm(x)
        if isinstance(self.memory, TokenNgramMemory):
            x = self.memory(x, input_ids)
        elif isinstance(self.memory, (ByteAddressMemory, PortableEngramAdapter)):
            if byte_addresses is None:
                raise ValueError("byte-addressed memory requires prepared byte addresses")
            x = self.memory(x, byte_addresses)
        return self.output(x), auxiliary_loss

    def forward(
        self, input_ids: Tensor, *, byte_addresses: Tensor | None = None
    ) -> Tensor:
        return self.forward_with_aux(input_ids, byte_addresses=byte_addresses)[0]
