"""Explicit pre-norm dense decoder language model."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch
from torch import Tensor, nn

from sparselab.config.models import AttentionConfig, ModelConfig
from sparselab.model.attention.dense import DenseAttention
from sparselab.model.attention.latent import LatentAttention
from sparselab.model.attention.sparse import BlockSparseAttention
from sparselab.model.cache import (
    AttentionKVCache,
    GenerationCache,
    IncrementalCacheCapability,
)
from sparselab.model.ffn import SwiGLU
from sparselab.model.memory import ByteAddressMemory, TokenNgramMemory
from sparselab.model.moe import (
    TopKMoE,
    diagnostic_writes_enabled,
    suppress_diagnostic_writes,
)
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
        self,
        x: Tensor,
        *,
        valid_target_mask: Tensor | None = None,
        diagnostics: Literal["scalar", "full"] = "scalar",
    ) -> tuple[Tensor, Tensor]:
        attention_input = self.norm1(x)
        if isinstance(self.attention, BlockSparseAttention):
            attention = self.attention(
                attention_input,
                valid_target_mask=valid_target_mask,
                diagnostics=diagnostics,
            )
        else:
            attention = self.attention(attention_input)
        x = x + attention
        if isinstance(self.ffn, TopKMoE):
            ffn_output, auxiliary_loss = self.ffn.forward_with_aux(
                self.norm2(x),
                valid_target_mask=valid_target_mask,
                diagnostics=diagnostics,
            )
            return x + ffn_output, auxiliary_loss
        return x + self.ffn(self.norm2(x)), x.new_zeros(())

    def forward_cached(
        self, x: Tensor, cache: AttentionKVCache
    ) -> tuple[Tensor, AttentionKVCache]:
        """Inference-only incremental block evaluation without auxiliary diagnostics."""
        attention_input = self.norm1(x)
        if not isinstance(self.attention, (DenseAttention, LatentAttention)):
            raise TypeError("block-sparse attention requires prefix recomputation")
        attention, next_cache = self.attention.forward_cached(attention_input, cache)
        x = x + attention
        return x + self.ffn(self.norm2(x)), next_cache

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
                        load_portable_engram(
                            model.memory_package_path,
                            expected_shape=(model.memory_table_size, model.memory_dim),
                            expected_ngram_size=model.memory_ngram_size,
                        ),
                        model.hidden_dim,
                    )
                )
            )
        )
        self.output = nn.Linear(model.hidden_dim, model.vocab_size, bias=False)
        self.apply(self._initialize)
        if model.tie_embeddings:
            self.output.weight = self.embedding.weight
        self._forward_block_calls = 0
        self._cache_owner = object()
        self._recomputed_block_calls = 0

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if (
            isinstance(module, (nn.Linear, nn.Embedding))
            and module.weight.requires_grad
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _apply_memory(
        self,
        hidden: Tensor,
        token_ids: Tensor,
        byte_addresses: Tensor | None,
        *,
        incremental: bool = False,
    ) -> Tensor:
        if self.memory is None:
            return hidden
        if isinstance(self.memory, TokenNgramMemory):
            if incremental:
                return self.memory.forward_last(hidden, token_ids)
            return self.memory(hidden, token_ids)
        if isinstance(self.memory, (ByteAddressMemory, PortableEngramAdapter)):
            if byte_addresses is None:
                raise ValueError(
                    "byte-addressed memory requires prepared byte addresses"
                )
            return self.memory(hidden, byte_addresses)
        raise TypeError(f"unsupported memory module: {type(self.memory).__name__}")

    def forward_with_aux(
        self,
        input_ids: Tensor,
        *,
        byte_addresses: Tensor | None = None,
        valid_target_mask: Tensor | None = None,
        activation_checkpointing: bool = False,
        diagnostics: Literal["scalar", "full"] = "scalar",
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds model.max_seq_len")
        if diagnostics not in ("scalar", "full"):
            raise ValueError(f"unknown diagnostics policy: {diagnostics}")
        self._forward_block_calls = 0
        self._recomputed_block_calls = 0
        x = self.embedding(input_ids)
        if self.config.memory_injection == "embedding":
            x = self._apply_memory(x, input_ids, byte_addresses)
        auxiliary_loss = x.new_zeros(())
        for block in self.blocks:
            if activation_checkpointing and self.training and torch.is_grad_enabled():

                def run_block(
                    hidden: Tensor, current: DecoderBlock = block
                ) -> tuple[Tensor, Tensor]:
                    if diagnostic_writes_enabled():
                        self._forward_block_calls += 1
                    else:
                        self._recomputed_block_calls += 1
                    return current.forward_with_aux(
                        hidden,
                        valid_target_mask=valid_target_mask,
                        diagnostics=diagnostics,
                    )

                def checkpoint_context() -> tuple[object, object]:
                    return nullcontext(), suppress_diagnostic_writes()

                x, block_aux = torch.utils.checkpoint.checkpoint(
                    run_block,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    context_fn=checkpoint_context,
                )
            else:
                self._forward_block_calls += 1
                x, block_aux = block.forward_with_aux(
                    x, valid_target_mask=valid_target_mask, diagnostics=diagnostics
                )
            auxiliary_loss = auxiliary_loss + block_aux
        x = self.norm(x)
        if self.config.memory_injection == "final":
            x = self._apply_memory(x, input_ids, byte_addresses)
        return self.output(x), auxiliary_loss

    def architecture_metric_tensors(self) -> dict[str, Tensor]:
        """Detached scalar diagnostics from the most recent forward on this device.

        The engine deliberately materializes these only at its update boundary so
        accelerator accumulation does not synchronize once per microbatch.
        """
        result: dict[str, Tensor] = {}
        if self.memory is not None and self.memory.last_diagnostics is not None:
            diagnostic = self.memory.last_diagnostics
            for name in (
                "lookup_count",
                "unique_addresses",
                "collision_count",
                "bucket_reuse_rate",
                "table_utilization",
                "maximum_address_fraction",
                "gate_mean",
                "value_norm",
                "hidden_norm",
            ):
                result[f"engram/{name}"] = getattr(diagnostic, name).detach()
            result[f"engram/injection/{self.config.memory_injection}"] = (
                diagnostic.gate_mean.new_ones(()).detach()
            )
        for index, block in enumerate(self.blocks):
            if (
                isinstance(block.ffn, TopKMoE)
                and block.ffn.last_diagnostics is not None
            ):
                diagnostic = block.ffn.last_diagnostics
                for metric, field in (
                    ("router_entropy", "entropy"),
                    ("maximum_expert_fraction", "maximum_fraction"),
                    ("mean_topk_probability", "mean_topk_probability"),
                ):
                    result[f"moe/layer_{index}/{metric}"] = getattr(
                        diagnostic, field
                    ).detach()
            diagnostic = getattr(block.attention, "last_diagnostics", None)
            if diagnostic is not None:
                for metric, field in (
                    ("available_tokens", "available_tokens"),
                    ("selected_tokens", "selected_tokens"),
                    ("selection_ratio", "selection_ratio"),
                    ("estimated_flops", "estimated_attention_flops"),
                    ("dense_teacher_mass", "dense_teacher_mass"),
                    ("dense_teacher_topk_recall", "dense_teacher_topk_recall"),
                ):
                    value = getattr(diagnostic, field)
                    if value is not None:
                        result[f"attention/layer_{index}/{metric}"] = value.detach()
        return result

    @property
    def incremental_cache_capability(self) -> IncrementalCacheCapability:
        """Describe whether this architecture can decode from projected K/V state."""
        if any(
            isinstance(block.attention, BlockSparseAttention) for block in self.blocks
        ):
            return IncrementalCacheCapability(
                False,
                "block_sparse selection is prefix-dependent; generation recomputes "
                "the active prefix for exact reference semantics",
            )
        mode = (
            "mla"
            if any(
                isinstance(block.attention, LatentAttention) for block in self.blocks
            )
            else "sliding"
            if any(
                isinstance(block.attention, DenseAttention)
                and block.attention.window_size is not None
                for block in self.blocks
            )
            else "dense"
        )
        return IncrementalCacheCapability(True, backend=mode)

    @torch.no_grad()
    def forward_cached(
        self,
        input_ids: Tensor,
        *,
        cache_capacity: int | None = None,
        cache: GenerationCache | None = None,
        byte_addresses: Tensor | None = None,
    ) -> tuple[Tensor, GenerationCache]:
        """Return logits and bounded, owner-checked inference-only request state.

        A new request must state its total prompt-plus-generation capacity.
        Callers rebuild the cache at a context rollover so absolute RoPE positions
        retain the same reset semantics as full-prefix reference generation.
        """
        if self.training:
            raise RuntimeError(
                "forward_cached is inference-only; call model.eval() first"
            )
        capability = self.incremental_cache_capability
        if not capability.supported:
            raise RuntimeError(capability.reason)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if input_ids.device != self.embedding.weight.device:
            raise ValueError("input_ids device does not match model parameters")
        projection_dtype = (
            torch.get_autocast_dtype(input_ids.device.type)
            if torch.is_autocast_enabled(input_ids.device.type)
            else self.embedding.weight.dtype
        )
        if cache is None:
            if cache_capacity is None:
                raise ValueError(
                    "new cached requests require an explicit cache_capacity"
                )
            if type(cache_capacity) is not int:
                raise TypeError("cache_capacity must be an integer")
            if not 0 < cache_capacity <= self.config.max_seq_len:
                raise ValueError("cache_capacity must be within model.max_seq_len")
            if input_ids.shape[1] > cache_capacity:
                raise ValueError("input_ids exceed requested cache_capacity")
            layers = [
                block.attention.create_cache(
                    input_ids.shape[0],
                    cache_capacity,
                    device=input_ids.device,
                    dtype=projection_dtype,
                )
                for block in self.blocks
                if isinstance(block.attention, (DenseAttention, LatentAttention))
            ]
            cache = GenerationCache(
                layers=layers,
                input_ids=torch.empty(
                    (input_ids.shape[0], cache_capacity),
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                ),
                length=0,
                capacity=cache_capacity,
                mode=capability.backend or "dense",
                owner=self._cache_owner,
                batch_size=input_ids.shape[0],
                device=input_ids.device,
                input_dtype=input_ids.dtype,
            )
        else:
            if cache_capacity is not None:
                raise ValueError("cache_capacity is set only when creating a cache")
            if cache.owner is not self._cache_owner:
                raise ValueError(
                    "generation cache belongs to a different model instance"
                )
            if (
                cache.mode != capability.backend
                or len(cache.layers) != len(self.blocks)
                or cache.length < 0
                or cache.length > cache.capacity
            ):
                raise ValueError("generation cache is incompatible with this model")
            if cache.length + input_ids.shape[1] > cache.capacity:
                raise ValueError(
                    "generation cache capacity exceeded; rebuild the active context"
                )
            if (
                cache.input_ids.shape != (cache.batch_size, cache.capacity)
                or cache.input_ids.device != cache.device
                or cache.input_ids.dtype != cache.input_dtype
            ):
                raise ValueError("generation cache input-ID storage is invalid")
            if any(
                layer.length != cache.length
                or layer.position != cache.length
                or layer.capacity != cache.capacity
                for layer in cache.layers
            ):
                raise ValueError("generation cache layers are inconsistent")
            for block, layer in zip(self.blocks, cache.layers, strict=True):
                attention = block.attention
                if not isinstance(attention, (DenseAttention, LatentAttention)):
                    raise TypeError("generation cache contains a non-incremental layer")
                value_dim = (
                    attention.head_dim
                    if isinstance(attention, DenseAttention)
                    else attention.value_dim
                )
                if (
                    layer.key.shape
                    != (
                        cache.batch_size,
                        attention.num_heads,
                        cache.capacity,
                        attention.head_dim,
                    )
                    or layer.value.shape
                    != (
                        cache.batch_size,
                        attention.num_heads,
                        cache.capacity,
                        value_dim,
                    )
                    or layer.key.device != input_ids.device
                    or layer.value.device != input_ids.device
                    or layer.key.dtype != projection_dtype
                    or layer.value.dtype != projection_dtype
                    or layer.key.requires_grad
                    or layer.value.requires_grad
                    or layer.key.grad_fn is not None
                    or layer.value.grad_fn is not None
                ):
                    raise ValueError(
                        "generation cache layer shape, device, or dtype is invalid"
                    )
        if isinstance(self.memory, (ByteAddressMemory, PortableEngramAdapter)):
            if byte_addresses is None:
                raise ValueError(
                    "byte-addressed memory requires prepared byte addresses"
                )
            if (
                byte_addresses.shape != input_ids.shape
                or byte_addresses.device != input_ids.device
            ):
                raise ValueError(
                    "byte_addresses must match cached input IDs in shape and device"
                )
        cache.append_input_ids(input_ids)
        full_ids = cache.input_ids[:, : cache.length]
        x = self.embedding(input_ids)
        if self.config.memory_injection == "embedding":
            x = self._apply_memory(x, full_ids, byte_addresses, incremental=True)
        for block, layer_cache in zip(self.blocks, cache.layers, strict=True):
            x, _ = block.forward_cached(x, layer_cache)
        x = self.norm(x)
        if self.config.memory_injection == "final":
            x = self._apply_memory(x, full_ids, byte_addresses, incremental=True)
        return self.output(x), cache

    @property
    def recomputed_block_call_ratio(self) -> float:
        """Recomputed DecoderBlock calls divided by original forward block calls."""
        if not self._forward_block_calls:
            return 0.0
        return self._recomputed_block_calls / self._forward_block_calls

    def forward(
        self,
        input_ids: Tensor,
        *,
        byte_addresses: Tensor | None = None,
        diagnostics: Literal["scalar", "full"] = "scalar",
    ) -> Tensor:
        return self.forward_with_aux(
            input_ids, byte_addresses=byte_addresses, diagnostics=diagnostics
        )[0]
