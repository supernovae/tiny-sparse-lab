"""Intel contracts on CPU, plus an explicit serial XPU training acceptance test."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparselab import runtime
from sparselab.config.loading import load_config


@pytest.mark.parametrize("memory_api", ["available", "absent", "error"])
def test_xpu_memory_reads_selected_device_and_retains_unavailability(
    monkeypatch, memory_api
):
    calls = []

    def free_memory(index):
        calls.append(index)
        if memory_api == "error":
            raise RuntimeError("driver query failed")
        return 3 * 1024**3, 6 * 1024**3

    xpu = SimpleNamespace(
        get_device_name=lambda index: "Intel Arc A380",
        get_device_properties=lambda index: SimpleNamespace(total_memory=6 * 1024**3),
    )
    if memory_api != "absent":
        xpu.mem_get_info = free_memory
    monkeypatch.setattr(torch, "xpu", xpu)
    monkeypatch.setattr(torch.version, "xpu", "2026.1", raising=False)
    monkeypatch.setattr(runtime, "_backend_available", lambda backend: backend == "xpu")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    passive = next(
        info for info in runtime.discover_runtimes() if info.backend == "xpu"
    )
    assert calls == []
    assert passive.device_free_bytes is None
    info = next(
        info
        for info in runtime.discover_runtimes(measurement_device=torch.device("xpu:1"))
        if info.backend == "xpu"
    )
    assert info.device_index == 1
    assert info.device_total_bytes == 6 * 1024**3
    assert info.runtime_version == "2026.1"
    if memory_api == "available":
        assert calls == [1]
        assert info.device_free_bytes == 3 * 1024**3
        assert info.measurement_source == "torch.xpu.mem_get_info"
    else:
        assert info.device_free_bytes is None
        assert any("free-memory" in item for item in info.limitations)


@pytest.mark.parametrize("available", [False, True])
def test_attention_rng_roundtrip_preserves_xpu_only_when_available(
    monkeypatch, available
):
    saved = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(
            is_available=lambda: available,
            get_rng_state_all=lambda: saved,
            set_rng_state_all=lambda value: restored.append(value),
        ),
    )
    state = runtime.capture_rng_state()
    assert ("xpu" in state) == available
    runtime.restore_rng_state(state)
    assert len(restored) == int(available)
    if available:
        assert restored[0] is saved
    # Historical CPU diagnostic state remains readable on an XPU host.
    state.pop("xpu", None)
    runtime.restore_rng_state(state)
    assert len(restored) == int(available)


@pytest.mark.parametrize("suffix,precision", [("", "fp32"), ("_bf16", "bf16")])
def test_xpu_smoke_uses_same_workload_as_rocm(suffix, precision):
    root = Path(__file__).resolve().parents[1]
    intel = load_config(root / f"configs/runtime_smoke_xpu{suffix}.yaml")
    amd = load_config(root / f"configs/runtime_smoke_rocm{suffix}.yaml")
    assert intel.runtime.backend == "xpu"
    assert intel.runtime.precision == precision
    assert intel.model == amd.model
    assert intel.training == amd.training
    assert intel.dataset == amd.dataset
    assert intel.training.max_steps == 20
    assert intel.training.max_tokens == 640


@pytest.mark.xpu
@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU is unavailable")
def test_xpu_training_checkpoint_resume_and_heldout_evidence(tmp_path):
    from test_training import config, equal

    from sparselab.evaluation.evidence import experiment_evidence
    from sparselab.runtime_profile import RuntimeProfile, authorize_profile
    from sparselab.training.checkpoints import CheckpointManager
    from sparselab.training.trainer import train

    selected = config(tmp_path)
    raw = selected.model_dump(mode="python")
    raw["runtime"]["backend"] = "xpu"
    raw["training"].update(max_steps=4, max_tokens=128)
    raw["runtime"]["memory"]["activation_checkpointing"]["enabled"] = True
    selected = type(selected).model_validate(raw)
    authorization = authorize_profile(
        RuntimeProfile(
            runtime_profile_version=1,
            id="intel-test",
            python=Path(sys.executable).absolute(),
            engine="pytorch",
            backend="xpu",
            device_index=0,
        ),
        selected,
    )
    train(selected, run_id="full", authorization=authorization)
    train(selected, run_id="part", stop_after_step=2, authorization=authorization)
    parent = selected.logging.root_dir / "part/checkpoints/latest.json"
    train(selected, run_id="resumed", resume=parent, authorization=authorization)
    snapshots = [
        CheckpointManager(selected.logging.root_dir / name).load(
            selected.logging.root_dir / name / "checkpoints/latest.json"
        )
        for name in ("full", "resumed")
    ]
    left, right = snapshots
    assert left.backend == right.backend == "xpu"
    assert left.step == right.step == 4
    assert left.tokens_seen == right.tokens_seen == 128
    equal(left.model, right.model)
    equal(left.optimizer, right.optimizer)
    equal(left.rng, right.rng)
    evidence = experiment_evidence(selected.logging.root_dir / "resumed")
    assert evidence["evidence_level"] == "checkpointed_held_out"
