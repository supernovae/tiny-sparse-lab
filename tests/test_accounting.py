from pathlib import Path

import pytest
import torch

from sparselab.config.loading import load_config
from sparselab.config.models import AdafactorConfig
from sparselab.memory import parameter_inventory
from sparselab.model.inspection import (
    architecture_metrics,
    inspect_model,
    inspection_report,
)
from sparselab.model.transformer import DenseLM


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
