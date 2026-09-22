from pathlib import Path

import pytest

from sparselab.config.loading import load_config
from sparselab.memory import parameter_inventory
from sparselab.model.inspection import inspect_model, inspection_report
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
