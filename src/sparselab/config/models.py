"""Validated, dense-only configuration for a SparseLab run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
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
    memory_table_size: int = 0
    memory_ngram_size: int = 0
    memory_dim: int = 0
    memory_package_path: Path | None = None
    memory_ngram_orders: tuple[int, ...] = ()
    memory_hash_heads: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> ModelConfig:
        fields = (
            self.vocab_size,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.ffn_dim,
            self.max_seq_len,
        )
        if any(value <= 0 for value in fields):
            raise ValueError("model dimensions must be positive")
        if self.vocab_size < 260:
            raise ValueError("model.vocab_size must be at least 260")
        if self.hidden_dim % self.num_heads:
            raise ValueError("model.hidden_dim must divide evenly by model.num_heads")
        if (self.hidden_dim // self.num_heads) % 2:
            raise ValueError("model head dimension must be even for RoPE")
        if self.rms_norm_eps <= 0:
            raise ValueError("model.rms_norm_eps must be positive")
        if self.ffn == "dense" and self.num_experts != 1:
            raise ValueError("dense model.ffn requires model.num_experts to be 1")
        if self.ffn == "moe":
            if self.num_experts < 2:
                raise ValueError("MoE model.ffn requires at least two experts")
            if not 1 <= self.experts_per_token <= self.num_experts:
                raise ValueError(
                    "model.experts_per_token must be between one and model.num_experts"
                )
        elif (
            self.num_experts != 1
            or self.experts_per_token != 1
            or self.shared_expert
            or self.router_aux_loss_coefficient != 0
        ):
            raise ValueError("dense model.ffn does not accept MoE settings")
        memory_settings = (
            self.memory_table_size,
            self.memory_ngram_size,
            self.memory_dim,
        )
        if self.memory == "none" and (
            any(value != 0 for value in memory_settings)
            or self.memory_package_path is not None
            or self.memory_ngram_orders
            or self.memory_hash_heads != 1
        ):
            raise ValueError("disabled model.memory requires zero memory settings")
        if self.memory in {"ngram", "byte", "portable"} and (
            self.memory_table_size <= 0
            or self.memory_ngram_size < 2
            or self.memory_dim <= 0
        ):
            raise ValueError(
                "n-gram/byte model.memory requires positive table/dimension and ngram size >= 2"
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
        if self.memory == "portable" and self.memory_package_path is None:
            raise ValueError("portable memory requires model.memory_package_path")
        if self.memory != "portable" and self.memory_package_path is not None:
            raise ValueError("model.memory_package_path requires portable memory")
        return self


class TokenizerConfig(StrictModel):
    path: Path


class DatasetConfig(StrictModel):
    source: Literal["tinystories", "synthetic", "withheld_facts"]
    revision: str | None = None
    cache_dir: Path
    train_max_documents: int = Field(gt=0)
    validation_max_documents: int = Field(gt=0)
    train_max_tokens: int = Field(gt=0)
    validation_max_tokens: int = Field(gt=0)
    synthetic_seed: int = 42

    @model_validator(mode="after")
    def validate_source(self) -> DatasetConfig:
        if self.source == "tinystories" and not self.revision:
            raise ValueError("dataset.revision is required for TinyStories")
        return self


class TrainingConfig(StrictModel):
    batch_size: int = Field(gt=0)
    seq_len: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    deterministic: bool = False


class OptimizerConfig(StrictModel):
    learning_rate: float = Field(default=3e-4, gt=0)
    min_learning_rate: float = Field(default=3e-5, ge=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.1, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0)


class AttentionConfig(StrictModel):
    kind: Literal["dense", "sliding_window", "mla", "block_sparse"] = "dense"
    rope_base: float = Field(default=10000.0, gt=0)
    window_size: int | None = Field(default=None, gt=0)
    latent_dim: int | None = Field(default=None, gt=0)
    block_size: int | None = Field(default=None, gt=0)
    selected_blocks: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_kind(self) -> AttentionConfig:
        sparse_fields = (
            self.window_size,
            self.latent_dim,
            self.block_size,
            self.selected_blocks,
        )
        if self.kind == "dense" and any(value is not None for value in sparse_fields):
            raise ValueError("dense attention does not accept sparse settings")
        if self.kind == "sliding_window" and (
            self.window_size is None
            or any(value is not None for value in sparse_fields[1:])
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


class LoggingConfig(StrictModel):
    root_dir: Path
    every_steps: int = Field(default=1, gt=0)
    checkpoint_every_steps: int = Field(default=50, gt=0)


class RunConfig(StrictModel):
    schema_version: Literal[1]
    name: str = Field(min_length=1)
    seed: int
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    model: ModelConfig
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    training: TrainingConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    attention: AttentionConfig = AttentionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_cross_section(self) -> RunConfig:
        if self.training.seq_len > self.model.max_seq_len:
            raise ValueError("training.seq_len cannot exceed model.max_seq_len")
        if (
            self.attention.kind == "mla"
            and self.attention.latent_dim % self.model.num_heads != 0
        ):
            raise ValueError("attention.latent_dim must divide evenly across heads")
        if self.optimizer.warmup_steps >= self.training.max_steps:
            raise ValueError(
                "optimizer.warmup_steps must be smaller than training.max_steps"
            )
        if self.optimizer.min_learning_rate > self.optimizer.learning_rate:
            raise ValueError(
                "optimizer.min_learning_rate cannot exceed optimizer.learning_rate"
            )
        if any(not 0 <= beta < 1 for beta in self.optimizer.betas):
            raise ValueError("optimizer.betas entries must be in [0, 1)")
        return self


class TokenizerTrainConfig(StrictModel):
    schema_version: Literal[1]
    vocab_size: int = Field(ge=260)
    min_frequency: int = Field(default=2, gt=0)
    max_documents: int = Field(gt=0)
    output_dir: Path
    dataset: DatasetConfig

    @model_validator(mode="after")
    def train_split_only(self) -> TokenizerTrainConfig:
        if self.dataset.source == "tinystories" and not self.dataset.revision:
            raise ValueError("dataset.revision is required for TinyStories")
        return self
