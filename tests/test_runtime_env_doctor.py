"""The doctor must prove an optimizer update in the chosen interpreter."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab import runtime_env_doctor as doctor_module
from sparselab.runtime_profile import RuntimeProfile


def profile(**changes):
    value = {
        "runtime_profile_version": 1,
        "id": "doctor-cpu",
        "python": str(Path(sys.executable).absolute()),
        "engine": "pytorch",
        "backend": "cpu",
        "device_index": 0,
        "requirements": {},
    }
    value.update(changes)
    return RuntimeProfile.model_validate(value)


def test_tiny_config_is_typed_and_inert():
    config = doctor_module.tiny_config(profile())
    assert config.model.hidden_dim == 16
    assert config.model.num_layers == 1
    assert config.training.seq_len == config.model.max_seq_len == 16
    assert config.optimizer.name == "adamw"
    assert config.runtime.precision == "fp32"
    assert not config.runtime.memory.activation_checkpointing.enabled
    assert not config.runtime.memory.activation_offload.enabled
    assert config.optimizer.state_offload is False
    assert not config.dataset.cache_dir.exists()
    assert not config.logging.root_dir.exists()
    assert (
        doctor_module.tiny_config(
            profile(requirements={"bf16": True})
        ).runtime.precision
        == "bf16"
    )


def test_real_cpu_doctor_verifies_disposable_optimizer(tmp_path, monkeypatch):
    work = tmp_path / "scientific-work"
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(work))
    result = doctor_module.doctor(profile())
    assert result["status"] == "READY", result["reason"]
    assert result["probe"]["backend"] == "cpu"
    assert "fp32" in result["tested_runtime"]["tested_precisions"]
    assert "forward_backward_optimizer" in result["tested_runtime"]["tested_features"]
    assert result["reason"] is None
    assert not work.exists()


def test_source_mismatch_refuses_child_and_reports_quoted_repair(monkeypatch):
    selected = profile()
    monkeypatch.setattr(
        doctor_module,
        "probe_runtime_profile",
        lambda _: (_ for _ in ()).throw(
            ValueError(
                "profile python SparseLab package/source identity differs from this checkout"
            )
        ),
    )
    monkeypatch.setattr(
        doctor_module,
        "run_bounded",
        lambda *a, **kw: pytest.fail("child must not run"),
    )
    result = doctor_module.doctor(selected)
    assert result["status"] == "SOURCE_MISMATCH"
    expected = (
        f"uv pip install --python {shlex.quote(str(selected.python))} --no-deps "
        f"--editable {shlex.quote(str(Path(doctor_module.__file__).resolve().parents[2]))}"
    )
    assert expected in result["reason"]
    assert result["tested_runtime"] is None


def test_missing_framework_never_claims_ready(monkeypatch):
    monkeypatch.setattr(
        doctor_module,
        "probe_runtime_profile",
        lambda _: (_ for _ in ()).throw(ValueError("No module named 'torch'")),
    )
    monkeypatch.setattr(
        doctor_module,
        "run_bounded",
        lambda *a, **kw: pytest.fail("child must not run"),
    )
    result = doctor_module.doctor(profile())
    assert result["status"] == "UNAVAILABLE"
    assert result["tested_runtime"] is None


def test_failed_child_is_reported_without_optimizer_evidence(monkeypatch):
    selected = profile()
    probe = {"package_root": "/checkout", "source_sha256": "source"}
    monkeypatch.setattr(doctor_module, "probe_runtime_profile", lambda _: probe)
    monkeypatch.setattr(doctor_module, "source_identity", lambda: probe)
    answer = doctor_module._envelope(selected)
    answer.update(
        probe=probe, status="UNAVAILABLE", reason="requested backend unavailable: cpu"
    )
    monkeypatch.setattr(
        doctor_module,
        "run_bounded",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(answer).encode(), stderr=b""
        ),
    )
    result = doctor_module.doctor(selected)
    assert result["status"] == "UNAVAILABLE"
    assert result["tested_runtime"] is None


def test_child_cannot_claim_ready_without_optimizer_update(monkeypatch):
    import sparselab.runtime_profile as profiles
    from sparselab import runtime

    selected = profile()
    monkeypatch.setattr(
        doctor_module, "probe_runtime_profile", lambda _: {"backend": "cpu"}
    )
    monkeypatch.setattr(profiles, "authorize_profile", lambda *args: object())
    monkeypatch.setattr(
        runtime,
        "validate_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            engine="pytorch",
            backend="cpu",
            device_index=0,
            tested_precisions=("fp32",),
            tested_features=(),
            as_dict=lambda: {"tested_features": []},
        ),
    )
    result = doctor_module._child(selected)
    assert result["status"] == "ERROR"
    assert "optimizer update" in result["reason"]


def test_child_rejects_backend_requirement_mismatch(monkeypatch):
    import sparselab.runtime_profile as profiles
    from sparselab import runtime

    selected = profile(backend="rocm", requirements={"torch_hip": True, "bf16": True})
    monkeypatch.setattr(
        doctor_module,
        "probe_runtime_profile",
        lambda _: {"backend": "rocm", "torch_hip": None},
    )
    monkeypatch.setattr(
        profiles,
        "authorize_profile",
        lambda *args: (_ for _ in ()).throw(
            ValueError(
                "runtime profile torch_hip requirement does not match Torch build"
            )
        ),
    )
    monkeypatch.setattr(
        runtime,
        "validate_runtime",
        lambda *args, **kwargs: pytest.fail("unauthorized optimizer"),
    )
    result = doctor_module._child(selected)
    assert result["status"] == "UNAVAILABLE"
    assert result["tested_runtime"] is None


def test_mlx_child_uses_typed_metal_config_and_disposable_validation(monkeypatch):
    import sparselab.runtime_profile as profiles
    from sparselab import runtime

    selected = profile(engine="mlx", backend="metal", requirements={"bf16": True})
    monkeypatch.setattr(
        doctor_module, "probe_runtime_profile", lambda _: {"backend": "metal"}
    )
    observed = []

    def authorize(actual, config):
        assert actual == selected
        assert config.runtime.engine == "mlx"
        assert config.runtime.backend == "metal"
        assert config.runtime.precision == "bf16"
        observed.append("authorized")
        return object()

    def validate(config, *, authorization):
        assert config.runtime.engine == "mlx"
        assert authorization is not None
        observed.append("validated")
        return SimpleNamespace(
            engine="mlx",
            backend="metal",
            device_index=0,
            tested_precisions=("bf16",),
            tested_features=("forward_backward_optimizer",),
            as_dict=lambda: {
                "engine": "mlx",
                "backend": "metal",
                "device_index": 0,
                "tested_precisions": ["bf16"],
                "tested_features": ["forward_backward_optimizer"],
            },
        )

    monkeypatch.setattr(profiles, "authorize_profile", authorize)
    monkeypatch.setattr(runtime, "validate_runtime", validate)
    result = doctor_module._child(selected)
    assert result["status"] == "READY"
    assert observed == ["authorized", "validated"]
