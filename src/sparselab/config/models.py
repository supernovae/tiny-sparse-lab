"""Strict, immutable schemas for SparseLab training and tokenizer jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int | None = None
    ffn_dim: int
    max_seq_len: int
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    ffn: Literal["dense", "moe"] = "dense"
    num_experts: int = 1
    experts_per_token: int = 1
    shared_expert: bool = False
    router_aux_loss_coefficient: float = Field(default=0.0, ge=0)
    memory: Literal["none", "ngram", "byte", "portable"] = "none"
    memory_injection: Literal["final", "embedding"] = "final"
    memory_table_size: int = 0
    memory_ngram_size: int = 0
    memory_dim: int = 0
    memory_package_path: Path | None = None
    memory_ngram_orders: tuple[int, ...] = ()
    memory_hash_heads: int = Field(default=1, gt=0)
    semantic_memory_dim: int | None = Field(default=None, gt=0)

    @model_serializer(mode="wrap")
    def _serialize_legacy_final(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        result = handler(self)
        if self.memory_injection == "final":
            result.pop("memory_injection", None)
        if self.num_kv_heads is None:
            result.pop("num_kv_heads", None)
        return result

    @model_validator(mode="after")
    def validate_dimensions(self) -> ModelConfig:
        values = (
            self.vocab_size,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.ffn_dim,
            self.max_seq_len,
        )
        if any(value <= 0 for value in values):
            raise ValueError("model dimensions must be positive")
        if self.vocab_size < 260:
            raise ValueError("model.vocab_size must be at least 260")
        if self.hidden_dim % self.num_heads or (self.hidden_dim // self.num_heads) % 2:
            raise ValueError("model head dimension must be even and divide hidden_dim")
        if self.num_kv_heads is not None and (
            self.num_kv_heads <= 0
            or self.num_kv_heads > self.num_heads
            or self.num_heads % self.num_kv_heads
        ):
            raise ValueError(
                "model.num_kv_heads must be positive, no greater than num_heads, "
                "and divide num_heads"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError("model.rms_norm_eps must be positive")
        if self.ffn == "dense" and (
            self.num_experts != 1
            or self.experts_per_token != 1
            or self.shared_expert
            or self.router_aux_loss_coefficient
        ):
            raise ValueError("dense model.ffn does not accept MoE settings")
        if self.ffn == "moe" and (
            self.num_experts < 2 or not 1 <= self.experts_per_token <= self.num_experts
        ):
            raise ValueError("invalid MoE expert configuration")
        settings = (self.memory_table_size, self.memory_ngram_size, self.memory_dim)
        if self.memory == "none" and self.memory_injection != "final":
            raise ValueError("embedding memory injection requires enabled memory")
        if self.memory == "none" and (
            any(settings)
            or self.memory_package_path is not None
            or self.memory_ngram_orders
            or self.memory_hash_heads != 1
        ):
            raise ValueError("disabled model.memory requires zero memory settings")
        if self.memory != "none" and (
            self.memory_table_size <= 0
            or self.memory_ngram_size < 2
            or self.memory_dim <= 0
        ):
            raise ValueError(
                "enabled model.memory requires table, dimension, and ngram size"
            )
        if self.memory_ngram_orders and (
            any(order < 2 for order in self.memory_ngram_orders)
            or tuple(sorted(set(self.memory_ngram_orders))) != self.memory_ngram_orders
        ):
            raise ValueError(
                "model.memory_ngram_orders must be sorted unique values >= 2"
            )
        if self.memory == "byte" and self.memory_ngram_orders:
            raise ValueError("byte memory uses its configured raw-byte ngram size")
        if (self.memory == "portable") != (self.memory_package_path is not None):
            raise ValueError("portable memory requires exactly memory_package_path")
        return self


class TokenizerConfig(StrictModel):
    path: Path


class DatasetConfig(StrictModel):
    source: Literal[
        "tinystories",
        "synthetic",
        "instruction_reference",
        "chat_recall",
        "local_chat",
        "withheld_facts",
        "engram_recall",
        "fineweb_edu",
        "cosmopedia",
    ]
    revision: str | None = None
    dataset_config: str | None = None
    cache_dir: Path
    train_max_documents: int = Field(gt=0)
    validation_max_documents: int = Field(gt=0)
    train_max_tokens: int = Field(gt=0)
    validation_max_tokens: int = Field(gt=0)
    synthetic_seed: int = 42
    train_path: Path | None = None
    validation_path: Path | None = None
    license: str | None = None
    allocation_manifest_path: Path | None = None

    @model_validator(mode="after")
    def validate_source(self) -> DatasetConfig:
        if self.allocation_manifest_path is not None and self.source != "local_chat":
            raise ValueError(
                "dataset.allocation_manifest_path requires source=local_chat"
            )
        if (
            self.source in {"tinystories", "fineweb_edu", "cosmopedia"}
            and not self.revision
        ):
            raise ValueError("dataset.revision is required for remote datasets")
        if self.source in {"fineweb_edu", "cosmopedia"} and not self.dataset_config:
            raise ValueError("dataset.dataset_config is required for this source")
        if self.source == "local_chat":
            if self.train_path is None or self.validation_path is None:
                raise ValueError("local_chat requires train_path and validation_path")
            if self.train_path.resolve() == self.validation_path.resolve():
                raise ValueError(
                    "local_chat training and validation must be separate files"
                )
            if not self.license or not self.license.strip():
                raise ValueError(
                    "local_chat requires explicit dataset.license provenance"
                )
        elif any(
            value is not None
            for value in (self.train_path, self.validation_path, self.license)
        ):
            raise ValueError(
                "train_path, validation_path and license are only for local_chat"
            )
        return self


class ActivationCheckpointingConfig(StrictModel):
    enabled: bool = False
    strategy: Literal["transformer_block"] = "transformer_block"


class ActivationOffloadConfig(StrictModel):
    enabled: bool = False


class MemoryConfig(StrictModel):
    policy: Literal["fast", "balanced", "low_memory", "max_fit"] = "balanced"
    max_device_memory_fraction: float = Field(default=0.90, gt=0, le=1)
    budget_bytes: int | None = Field(default=None, gt=0)
    activation_checkpointing: ActivationCheckpointingConfig = (
        ActivationCheckpointingConfig()
    )
    activation_offload: ActivationOffloadConfig = ActivationOffloadConfig()
    allowed_sequence_lengths: tuple[int, ...] = ()
    allowed_optimizers: tuple[Literal["adamw", "adafactor"], ...] = ()

    @model_validator(mode="after")
    def validate_lengths(self) -> MemoryConfig:
        if any(value <= 0 for value in self.allowed_sequence_lengths):
            raise ValueError("runtime.memory.allowed_sequence_lengths must be positive")
        return self


class RuntimeConfig(StrictModel):
    engine: Literal["pytorch", "mlx"] = "pytorch"
    backend: Literal["auto", "cpu", "mps", "cuda", "rocm", "xpu", "metal"] = "auto"
    device_index: int = Field(default=0, ge=0)
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "fp32"
    memory: MemoryConfig = MemoryConfig()


class TrainingConfig(StrictModel):
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(default=1, gt=0)
    seq_len: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    neural_loss_weight: float = Field(default=1.0, ge=0, le=1)
    deterministic: bool = True


class AdamWConfig(StrictModel):
    name: Literal["adamw"] = "adamw"
    peak: float = Field(default=3e-4, gt=0)
    floor: float = Field(default=3e-5, ge=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.1, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0)
    state_offload: bool = False


class AdafactorConfig(StrictModel):
    name: Literal["adafactor"] = "adafactor"
    peak: float = Field(default=3e-4, gt=0)
    floor: float = Field(default=3e-5, ge=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.1, ge=0)
    beta2_decay: float = -0.8
    eps: tuple[float | None, float] = (None, 0.001)
    d: float = Field(default=1.0, gt=0)
    state_offload: bool = False


OptimizerConfig = Annotated[AdamWConfig | AdafactorConfig, Field(discriminator="name")]


class AttentionConfig(StrictModel):
    kind: Literal["dense", "sliding_window", "mla", "block_sparse"] = "dense"
    rope_base: float = Field(default=10000.0, gt=0)
    window_size: int | None = Field(default=None, gt=0)
    latent_dim: int | None = Field(default=None, gt=0)
    block_size: int | None = Field(default=None, gt=0)
    selected_blocks: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_kind(self) -> AttentionConfig:
        sparse = (
            self.window_size,
            self.latent_dim,
            self.block_size,
            self.selected_blocks,
        )
        if self.kind == "dense" and any(value is not None for value in sparse):
            raise ValueError("dense attention does not accept sparse settings")
        if self.kind == "sliding_window" and (
            self.window_size is None or any(value is not None for value in sparse[1:])
        ):
            raise ValueError("sliding_window attention requires only window_size")
        if self.kind == "mla" and (
            self.latent_dim is None
            or self.window_size is not None
            or self.block_size is not None
            or self.selected_blocks is not None
        ):
            raise ValueError("mla attention requires only latent_dim")
        if self.kind == "block_sparse" and (
            self.block_size is None
            or self.selected_blocks is None
            or self.window_size is not None
            or self.latent_dim is not None
        ):
            raise ValueError(
                "block_sparse attention requires block_size and selected_blocks"
            )
        return self


class EvaluationConfig(StrictModel):
    every_steps: int = Field(default=25, gt=0)
    max_batches: int = Field(default=16, gt=0)


class CheckpointConfig(StrictModel):
    every_steps: int | None = Field(default=50, gt=0)
    every_tokens: int | None = Field(default=None, gt=0)
    every_minutes: float | None = Field(default=None, gt=0)
    keep_periodic: bool = True

    @model_validator(mode="after")
    def validate_cadence(self) -> CheckpointConfig:
        if (
            self.every_steps is None
            and self.every_tokens is None
            and self.every_minutes is None
        ):
            raise ValueError("at least one checkpoint cadence is required")
        return self


class StagingConfig(StrictModel):
    smoke_steps: int = Field(default=2, gt=0)
    warmup_steps: int = Field(default=5, gt=0)


class LoggingConfig(StrictModel):
    root_dir: Path
    every_steps: int = Field(default=1, gt=0)
    architecture_diagnostics: Literal["scalar", "full"] = "scalar"


class RunConfig(StrictModel):
    schema_version: Literal[2]
    name: str = Field(min_length=1)
    seed: int
    runtime: RuntimeConfig = RuntimeConfig()
    model: ModelConfig
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    training: TrainingConfig
    optimizer: OptimizerConfig = AdamWConfig()
    attention: AttentionConfig = AttentionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    staging: StagingConfig = StagingConfig()
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_cross_section(self) -> RunConfig:
        if self.training.seq_len > self.model.max_seq_len:
            raise ValueError("training.seq_len cannot exceed model.max_seq_len")
        if (
            self.attention.kind == "mla"
            and self.attention.latent_dim % self.model.num_heads
        ):
            raise ValueError("attention.latent_dim must divide evenly across heads")
        if (
            self.model.num_kv_heads is not None
            and self.model.num_kv_heads != self.model.num_heads
            and (
                self.attention.kind in {"mla", "block_sparse"}
                or self.runtime.engine == "mlx"
            )
        ):
            raise ValueError(
                "grouped-query attention is supported only for PyTorch dense or sliding_window attention"
            )
        if self.optimizer.warmup_steps >= self.training.max_steps:
            raise ValueError(
                "optimizer.warmup_steps must be smaller than training.max_steps"
            )
        if self.optimizer.floor > self.optimizer.peak:
            raise ValueError("optimizer.floor cannot exceed optimizer.peak")
        if isinstance(self.optimizer, AdamWConfig) and any(
            not 0 <= beta < 1 for beta in self.optimizer.betas
        ):
            raise ValueError("optimizer.betas entries must be in [0, 1)")
        if self.optimizer.state_offload:
            raise ValueError("optimizer.state_offload is deferred and unsupported")
        if self.runtime.engine == "mlx" and self.runtime.backend != "metal":
            raise ValueError("MLX engine requires runtime.backend=metal")
        if self.runtime.engine == "pytorch" and self.runtime.backend == "metal":
            raise ValueError("PyTorch runtime does not use backend=metal")
        if (
            self.runtime.engine == "mlx"
            and self.model.num_kv_heads is not None
            and self.model.num_kv_heads != self.model.num_heads
        ):
            raise ValueError("MLX engine does not support grouped-query attention")
        if (
            self.model.semantic_memory_dim is not None
            and self.dataset.allocation_manifest_path is None
        ):
            raise ValueError(
                "model.semantic_memory_dim requires dataset.allocation_manifest_path"
            )
        if (
            self.dataset.allocation_manifest_path is not None
            and self.runtime.engine != "pytorch"
        ):
            raise ValueError(
                "allocation ownership currently requires the PyTorch engine"
            )
        return self


class TokenizerTrainConfig(StrictModel):
    schema_version: Literal[1]
    vocab_size: int = Field(ge=260)
    min_frequency: int = Field(default=2, gt=0)
    max_documents: int = Field(gt=0)
    output_dir: Path
    dataset: DatasetConfig
