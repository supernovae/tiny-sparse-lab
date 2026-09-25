from pathlib import Path

import pytest
import torch

from sparselab.config.loading import load_config
from sparselab.config.models import AdafactorConfig, ModelConfig
from sparselab.memory import parameter_inventory
from sparselab.model.inspection import (
    architecture_metrics,
    inspect_model,
    inspection_report,
    named_tensor_inventory,
)
from sparselab.model.transformer import DenseLM
from sparselab.training.manifest import config_sha256
from sparselab.training.metric_registry import metric_spec


@pytest.mark.parametrize(
    "preset", ["smoke_mla_cpu", "smoke_combined_cpu", "smoke_moe_cpu"]
)
def test_shape_inventory_matches_instantiated_architecture(preset):
    config = load_config(Path(f"configs/{preset}.yaml"))
    actual = inspect_model(DenseLM(config.model, config.attention))
    estimated = parameter_inventory(config)
    report = inspection_report(config)
    for key in (
        "total",
        "trainable",
        "active_per_token",
        "embedding",
        "attention",
        "ffn",
        "norm",
        "output_head",
        "expert",
        "routed_expert",
        "shared_expert",
        "router",
        "engram",
        "engram_table",
        "engram_adapter",
        "frozen",
    ):
        assert report[key] == actual[key]
    assert estimated.total == actual["total"]
    assert estimated.trainable == actual["trainable"]


def test_grouped_query_inventory_and_cache_use_kv_head_width() -> None:
    base = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    model = base.model.model_copy(
        update={"num_heads": 4, "num_kv_heads": 2, "hidden_dim": 16, "ffn_dim": 32}
    )
    config = base.model_copy(
        update={
            "model": model,
            "training": base.training.model_copy(update={"seq_len": 8}),
        }
    )
    instantiated = DenseLM(config.model, config.attention).eval()
    tensors = named_tensor_inventory(config.model, config.attention)

    assert tensors["blocks.0.attention.q_proj.weight"].shape == (16, 16)
    assert tensors["blocks.0.attention.k_proj.weight"].shape == (8, 16)
    assert tensors["blocks.0.attention.v_proj.weight"].shape == (8, 16)
    assert (
        inspection_report(config)["attention"]
        == inspect_model(instantiated)["attention"]
    )
    _, cache = instantiated.forward_cached(torch.tensor([[3, 7]]), cache_capacity=5)
    assert cache.layers[0].key.shape == (1, 2, 5, 4)
    assert cache.layers[0].value.shape == (1, 2, 5, 4)
    assert cache.allocated_bytes == 5 * (8 + config.model.num_layers * 2 * 2 * 4 * 4)


def test_omitted_kv_heads_preserve_legacy_model_serialization() -> None:
    model = ModelConfig(
        vocab_size=260,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=8,
    )
    base = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    explicit_none = base.model_copy(
        update={"model": base.model.model_copy(update={"num_kv_heads": None})}
    )

    assert "num_kv_heads" not in model.model_dump(mode="json")
    assert config_sha256(base.model_dump(mode="json")) == config_sha256(
        explicit_none.model_dump(mode="json")
    )


def test_engram_tables_count_as_storage_not_all_active_rows():
    config = load_config(Path("configs/capability_recall_ngram_cpu.yaml"))
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "memory_ngram_orders": (1, 2, 3),
                    "memory_hash_heads": 2,
                }
            )
        }
    )
    model = DenseLM(config.model, config.attention)
    measured = inspect_model(model)
    assert measured["engram"] == sum(p.numel() for p in model.memory.parameters())
    assert parameter_inventory(config).total == measured["total"]
    inactive_rows = 6 * (config.model.memory_table_size - 1) * config.model.memory_dim
    assert measured["total"] - measured["active_per_token"] == inactive_rows


@pytest.mark.parametrize("optimizer_name", ["adamw", "adafactor"])
def test_inventory_matches_actual_persistent_optimizer_storage(optimizer_name) -> None:
    config = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    if optimizer_name == "adafactor":
        config = config.model_copy(update={"optimizer": AdafactorConfig()})
    model = DenseLM(config.model, config.attention)
    optimizer_type = (
        torch.optim.Adafactor if optimizer_name == "adafactor" else torch.optim.AdamW
    )
    optimizer = optimizer_type(
        model.parameters(), lr=config.optimizer.peak, foreach=False
    )
    model(torch.arange(8).reshape(1, 8)).sum().backward()
    optimizer.step()
    actual_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    assert inspection_report(config)["optimizer_state_bytes"] == actual_bytes


def test_scalar_sparse_diagnostics_omit_unavailable_dense_teacher_metrics() -> None:
    config = load_config(Path("configs/smoke_sparse_cpu.yaml"))
    model = DenseLM(config.model, config.attention)
    model.forward_with_aux(
        torch.zeros((1, config.training.seq_len), dtype=torch.long),
        diagnostics="scalar",
    )
    metrics = architecture_metrics(model)
    assert not any("dense_teacher" in name for name in metrics)


def test_memory_injection_diagnostics_and_inspection() -> None:
    plain = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    plain_model = DenseLM(plain.model, plain.attention)
    plain_model(torch.tensor([[3, 7, 11]]))
    assert not any(
        name.startswith("engram/injection/")
        for name in architecture_metrics(plain_model)
    )
    assert inspect_model(plain_model)["memory_injection"] == "none"
    assert inspection_report(plain)["memory_injection"] == "none"

    enabled = load_config(Path("configs/context_study_dense_s17_b24k.yaml"))
    enabled = enabled.model_copy(
        update={
            "model": enabled.model.model_copy(
                update={
                    "memory": "ngram",
                    "memory_table_size": 31,
                    "memory_ngram_size": 3,
                    "memory_dim": 8,
                }
            )
        }
    )
    models = {
        placement: DenseLM(
            enabled.model.model_copy(update={"memory_injection": placement}),
            enabled.attention,
        )
        for placement in ("final", "embedding")
    }

    reports = {}
    for placement, model in models.items():
        config = enabled.model_copy(update={"model": model.config})
        model(torch.tensor([[3, 7, 11]]))
        metrics = architecture_metrics(model)
        indicators = {
            name: value
            for name, value in metrics.items()
            if name.startswith("engram/injection/")
        }
        expected = f"engram/injection/{placement}"
        assert set(indicators) == {expected}
        indicator = indicators[expected]
        assert indicator.shape == torch.Size([])
        assert indicator.device == model.memory.last_diagnostics.gate_mean.device
        assert not indicator.requires_grad
        assert indicator.item() == 1
        assert "engram/lookup_count" in metrics
        assert inspect_model(model)["memory_injection"] == placement
        report = inspection_report(config)
        assert report["memory_injection"] == placement

        reports[placement] = (inspect_model(model), report)

    final, embedding = reports["final"], reports["embedding"]
    for name in (
        "total",
        "trainable",
        "active_per_token",
        "engram",
        "optimizer_state_bytes",
        "estimated_checkpoint_bytes",
    ):
        assert final[0][name] == embedding[0][name]
        assert final[1][name] == embedding[1][name]
    final_parameters = models["final"].state_dict()
    embedding_parameters = models["embedding"].state_dict()
    assert final_parameters.keys() == embedding_parameters.keys()
    assert {name: tuple(value.shape) for name, value in final_parameters.items()} == {
        name: tuple(value.shape) for name, value in embedding_parameters.items()
    }
    for placement in ("final", "embedding"):
        spec = metric_spec(f"engram/injection/{placement}")
        assert spec is not None
        assert (spec.unit, spec.producer, spec.help_slug) == (
            "indicator",
            "model",
            "memory",
        )
