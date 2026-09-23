from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sparselab import runtime


def _config(
    *,
    backend: str = "cpu",
    precision: str = "fp32",
    index: int = 0,
    checkpointing: bool = False,
):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            engine="pytorch",
            backend=backend,
            precision=precision,
            device_index=index,
            memory=SimpleNamespace(
                activation_checkpointing=SimpleNamespace(enabled=checkpointing),
                activation_offload=SimpleNamespace(enabled=False),
            ),
        ),
        optimizer=SimpleNamespace(name="adamw", state_offload=False),
    )


def _mlx_config(*, attention: str = "dense", checkpointing: bool = False):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            engine="mlx",
            backend="metal",
            precision="fp32",
            device_index=0,
            memory=SimpleNamespace(
                activation_checkpointing=SimpleNamespace(enabled=checkpointing),
                activation_offload=SimpleNamespace(enabled=False),
            ),
        ),
        optimizer=SimpleNamespace(name="adamw", state_offload=False),
        attention=SimpleNamespace(kind=attention),
    )


def _mlx_probe_result(*, features: list[str]) -> dict[str, object]:
    info = runtime.RuntimeInfo(
        engine="mlx",
        backend="metal",
        torch_device=None,
        device_index=0,
        device_name="Apple Metal",
        physical_device_id="apple-metal:test",
        framework_version="0.32.2",
        runtime_version="0.32.2",
        driver_version="test",
        os="test",
        system_total_bytes=1,
        system_available_bytes=1,
        device_total_bytes=None,
        device_free_bytes=None,
        device_recommended_bytes=None,
        measurement_source="mlx native",
        measured_at="now",
        precision_capabilities=("fp32",),
        tested_precisions=("fp32",),
        tested_features=tuple(features),
        validated_at="now",
    )
    return {
        "format_version": 1,
        "ok": True,
        "tested_precision": "fp32",
        "tested_features": features,
        "runtime": info.as_dict(),
    }


def test_cpu_probe_isolated_from_parent_rng() -> None:
    torch.manual_seed(918)
    before = torch.get_rng_state().clone()

    info = runtime.validate_runtime(_config())

    assert torch.equal(torch.get_rng_state(), before)
    assert info.backend == "cpu"
    assert info.tested_precisions == ("fp32",)
    assert info.tested_features == ("forward_backward_optimizer", "optimizer:adamw")
    assert info.validated_at is not None


def test_explicit_unavailable_backend_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_backend_available", lambda backend: backend == "cpu")

    with pytest.raises(ValueError, match="requested backend unavailable: mps"):
        runtime.validate_runtime(_config(backend="mps"))


def test_hip_inventory_does_not_claim_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime, "_backend_available", lambda backend: backend == "rocm"
    )
    monkeypatch.setattr(torch.version, "hip", "6.2")

    infos = {
        info.backend: info
        for info in runtime.discover_runtimes()
        if info.engine == "pytorch"
    }

    assert infos["rocm"].torch_device == "cuda:0"
    assert "PyTorch build targets the other CUDA/HIP API" in infos["cuda"].limitations
    assert infos["cuda"].precision_capabilities == ()


def test_cpu_fp16_is_rejected_before_probe() -> None:
    with pytest.raises(ValueError, match="fp16 is unsupported on CPU"):
        runtime.validate_runtime(_config(precision="fp16"))


def test_state_offload_stays_explicitly_unsupported() -> None:
    config = _config()
    config.optimizer.state_offload = True

    with pytest.raises(ValueError, match="state_offload is deferred"):
        runtime.validate_runtime(config)


def test_bf16_checkpoint_probe_reports_only_exercised_precision_and_features():
    requested = _config(precision="bf16", checkpointing=True)
    info = runtime.validate_runtime(requested)
    assert info.tested_precisions == ("bf16",)
    assert "activation_checkpointing" in info.tested_features
    assert requested.runtime.precision == "bf16"


def test_mlx_validation_uses_tiny_versioned_native_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sparselab.engines.mlx as mlx_engine

    captured: dict[str, object] = {}
    features = [
        "forward_backward_optimizer",
        "optimizer:adamw",
        "native_block_sparse_attention",
    ]
    monkeypatch.setattr(mlx_engine, "validate", lambda config: None)

    def probe(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return _mlx_probe_result(features=features)

    monkeypatch.setattr(runtime, "_probe_runtime", probe)
    info = runtime.validate_runtime(_mlx_config(attention="block_sparse"))

    assert captured == {
        "engine": "mlx",
        "backend": "metal",
        "device_index": 0,
        "precision": "fp32",
        "optimizer": "adamw",
        "checkpointing": False,
        "attention": "block_sparse",
    }
    assert info.tested_features == tuple(features)
    assert info.device_total_bytes is None
    assert info.device_recommended_bytes is None
