from __future__ import annotations

from pathlib import Path

from sparselab.config.loading import load_config
from sparselab.memory import estimate_memory, parameter_inventory, plan_memory
from sparselab.model.inspection import inspection_report, named_tensor_inventory
from sparselab.runtime import RuntimeInfo


def _runtime(
    *,
    backend: str = "cpu",
    total: int | None = 16_000_000_000,
    available: int | None = 12_000_000_000,
    device_total: int | None = None,
    device_free: int | None = None,
    recommended: int | None = None,
    driver_allocated: int | None = None,
) -> RuntimeInfo:
    return RuntimeInfo(
        engine="pytorch",
        backend=backend,
        torch_device=backend,
        device_index=0,
        device_name="test",
        physical_device_id=None,
        framework_version="test",
        runtime_version=None,
        driver_version=None,
        os="test",
        system_total_bytes=total,
        system_available_bytes=available,
        device_total_bytes=device_total,
        device_free_bytes=device_free,
        device_recommended_bytes=recommended,
        measurement_source="test",
        measured_at="test",
        precision_capabilities=("fp32",),
        device_driver_allocated_bytes=driver_allocated,
    )


def test_shape_report_never_opens_portable_package() -> None:
    base = load_config(Path("configs/smoke_byte_memory_cpu.yaml"))
    config = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "memory": "portable",
                    "memory_package_path": Path("/definitely/unavailable.engram"),
                }
            )
        }
    )
    report = inspection_report(config)
    tensors = named_tensor_inventory(config.model, config.attention)
    assert report["frozen"] == config.model.memory_table_size * config.model.memory_dim
    assert tensors["memory.embedding.weight"].trainable is False
    assert tensors["output.weight"].alias_of == "embedding.weight"


def test_billion_scale_shape_inspection_has_no_model_state() -> None:
    base = load_config(Path("configs/smoke_mla_cpu.yaml"))
    config = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "vocab_size": 1_000_000,
                    "hidden_dim": 4096,
                    "num_heads": 32,
                    "ffn_dim": 16_384,
                    "num_layers": 48,
                }
            ),
            "attention": base.attention.model_copy(update={"latent_dim": 1024}),
        }
    )
    report = inspection_report(config)
    assert report["total"] > 1_000_000_000
    assert report["model_weight_bytes"] == report["total"] * 4


def test_artificial_budget_cannot_certify_a_physical_fit() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    budgeted = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={"budget_bytes": 10**18}
                    )
                }
            )
        }
    )
    estimate = estimate_memory(
        budgeted, _runtime(total=None, available=None), parameter_inventory(budgeted)
    )
    assert estimate.result == "UNKNOWN"
    assert estimate.capacity_ceiling_bytes == 10**18
    assert "cannot certify" in " ".join(estimate.assumptions)


def test_unified_memory_is_one_conservative_pool() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    estimate = estimate_memory(
        config,
        _runtime(
            backend="mps",
            total=64_000,
            available=4_000,
            recommended=10_000,
            driver_allocated=7_000,
        ),
        parameter_inventory(config),
    )
    assert estimate.capacity_ceiling_bytes == max(
        0, int(config.runtime.memory.max_device_memory_fraction * 10_000) - 7_000
    )
    assert estimate.result == "LIKELY_TO_EXCEED"
    unknown = estimate_memory(
        config,
        _runtime(backend="mps", available=4_000, recommended=10_000),
        parameter_inventory(config),
    )
    assert unknown.result == "UNKNOWN"


def test_proposal_is_complete_recomputed_and_preserves_effective_batch() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    constrained = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={"budget_bytes": 1, "policy": "balanced"}
                    )
                }
            )
        }
    )
    original = estimate_memory(
        constrained, _runtime(), parameter_inventory(constrained)
    )
    proposal = plan_memory(constrained, _runtime(), original)
    assert constrained.training.micro_batch_size == 4
    assert proposal.config["training"]["micro_batch_size"] < 4
    assert (
        proposal.config["training"]["micro_batch_size"]
        * proposal.config["training"]["gradient_accumulation"]
        == 4
    )
    candidate = constrained.__class__.model_validate(proposal.config)
    assert proposal.estimate == estimate_memory(
        candidate, _runtime(), parameter_inventory(candidate)
    )
    assert any(
        decision["requested"] == "activation_checkpointing"
        for decision in proposal.decisions
    )


def test_unsupported_offload_has_no_imaginary_saving() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    offload = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={
                            "budget_bytes": 1,
                            "activation_offload": config.runtime.memory.activation_offload.model_copy(
                                update={"enabled": True}
                            ),
                        }
                    )
                }
            )
        }
    )
    estimate = estimate_memory(offload, _runtime(), parameter_inventory(offload))
    proposal = plan_memory(offload, _runtime(), estimate)
    assert any("no saving" in decision["reason"] for decision in proposal.decisions)
    assert any(
        "activation offload is unsupported" in assumption
        for assumption in proposal.estimate.assumptions
    )
