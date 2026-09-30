from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from test_training import config as training_config

from sparselab import runtime
from sparselab.cli.main import main
from sparselab.runtime_profile import (
    RuntimeProfile,
    authorize_profile,
    load_runtime_profile,
    probe_runtime_profile,
)


def profile_value(**changes):
    return {
        "runtime_profile_version": 1,
        "id": "offline-cpu",
        "python": str(Path(sys.executable).absolute()),
        "engine": "pytorch",
        "backend": "cpu",
        "device_index": 0,
        "requirements": {},
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": True},
        {"runtime_profile_version": 2},
        {"runtime_profile_version": True},
        {"id": "../unsafe"},
        {"python": "relative/python"},
        {"python": "/nonexistent/runtime-prep-v1-python"},
        {"engine": "mlx", "backend": "cpu"},
        {"engine": "pytorch", "backend": "metal"},
        {"device_index": -1},
        {"device_index": True},
        {"requirements": {"unknown": True}},
        {"requirements": {"device_name_regex": "["}},
        {"requirements": {"torch_hip": True}},
        {"requirements": {"bf16": 1}},
    ],
)
def test_profile_rejects_unsafe_or_ambiguous_values(tmp_path, changes):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile_value(**changes)))
    with pytest.raises((ValueError, TypeError)):
        load_runtime_profile(path)


def test_profile_rejects_nonexecutable_file(tmp_path):
    python = tmp_path / "python"
    python.write_text("not an interpreter")
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile_value(python=str(python))))
    with pytest.raises(ValueError):
        load_runtime_profile(path)


def test_real_cpu_probe_and_authorization(tmp_path):
    profile = RuntimeProfile.model_validate(profile_value())
    result = probe_runtime_profile(profile)
    assert result["schema_version"] == 1
    authorization = authorize_profile(profile, training_config(tmp_path))
    info = runtime.validate_runtime(
        training_config(tmp_path), authorization=authorization
    )
    assert info.backend == "cpu"
    assert "forward_backward_optimizer" in info.tested_features


def test_mlx_namespace_package_probe_uses_loaded_core_identity(tmp_path, monkeypatch):
    from sparselab.runtime_identity_probe import probe

    namespace = ModuleType("mlx")
    namespace.__file__ = None
    namespace.__path__ = [str(tmp_path)]
    core = ModuleType("mlx.core")
    core.__file__ = str(tmp_path / "core.so")
    core.__version__ = "0.32.2"
    core.metal = SimpleNamespace(is_available=lambda: True)
    namespace.core = core
    monkeypatch.setitem(sys.modules, "mlx", namespace)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "torch", None)

    result = probe(
        {
            "id": "mlx-namespace",
            "engine": "mlx",
            "backend": "metal",
            "device_index": 0,
        }
    )
    assert result["framework_path"] == str(Path(core.__file__).resolve())
    assert result["framework_version"] == "0.32.2"
    assert result["available"] is True
    assert result["torch_path"] is None


@pytest.mark.parametrize("command", ["train", "stage", "run"])
def test_accelerator_cli_rejects_before_workdir(tmp_path, monkeypatch, command):
    config = training_config(tmp_path)
    value = config.model_dump(mode="json")
    value["runtime"]["backend"] = "cuda"
    path = tmp_path / "accelerator.yaml"
    path.write_text(yaml.safe_dump(value))
    work = tmp_path / "must-not-exist"
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(work))
    args = ["sparselab", command, str(path)]
    if command == "stage":
        args += ["--output", str(tmp_path / "stage")]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit, match="runtime-profile|registered"):
        main()
    assert not work.exists()
    assert not (tmp_path / "stage").exists()
    assert not config.logging.root_dir.exists()


@pytest.mark.parametrize("backend", ["cuda", "rocm", "mps", "xpu", "metal"])
def test_unregistered_acceleration_is_not_authorized(tmp_path, backend):
    config = training_config(tmp_path)
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "backend": backend,
                    "engine": "mlx" if backend == "metal" else "pytorch",
                }
            )
        }
    )
    with pytest.raises(ValueError, match="runtime-profile|registered"):
        runtime.validate_runtime(config)


def fake_runtime(monkeypatch, config, backend):
    import sparselab.runtime_identity_probe as identity
    import sparselab.runtime_profile as profiles

    engine = "mlx" if backend == "metal" else "pytorch"
    evidence = {
        "schema_version": 1,
        "profile_id": "offline-cpu",
        "engine": engine,
        "backend": backend,
        "device_index": 0,
        "python": str(Path(sys.executable).resolve()),
        "available": True,
        "device_count": 1,
        "device_name": "Fixture Accelerator",
        "device_names": ["Fixture Accelerator"],
        "torch_hip": "fixture-hip" if backend == "rocm" else None,
        "torch_cuda": "fixture-cuda" if backend == "cuda" else None,
        "bf16_supported": None,
        **identity.source_identity(),
    }
    monkeypatch.setattr(profiles, "_probe", lambda *args, **kwargs: dict(evidence))
    monkeypatch.setattr(identity, "probe", lambda request: dict(evidence))
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"engine": engine, "backend": backend}
            )
        }
    )
    return evidence, config


@pytest.mark.parametrize("backend", ["cuda", "rocm", "xpu", "mps", "metal"])
def test_matched_vendor_probe_authorizes_without_hardware(
    tmp_path, monkeypatch, backend
):
    from sparselab.runtime_profile import require_authorization

    _, config = fake_runtime(monkeypatch, training_config(tmp_path), backend)
    profile = RuntimeProfile.model_validate(
        profile_value(engine=config.runtime.engine, backend=backend)
    )
    authorization = authorize_profile(profile, config)
    require_authorization(config, authorization)


@pytest.mark.parametrize(
    ("backend", "requirements", "changed", "message"),
    [
        ("rocm", {"torch_hip": True}, {"torch_hip": None}, "CUDA/HIP|torch_hip"),
        ("cuda", {}, {"torch_hip": "fixture-hip"}, "CUDA/HIP"),
        ("xpu", {}, {"available": False}, "unavailable"),
        ("mps", {"device_name_regex": "Different"}, {}, "device name"),
    ],
)
def test_profile_capability_mismatch_rejected(
    tmp_path, monkeypatch, backend, requirements, changed, message
):
    evidence, config = fake_runtime(monkeypatch, training_config(tmp_path), backend)
    evidence.update(changed)
    profile = RuntimeProfile.model_validate(
        profile_value(backend=backend, requirements=requirements)
    )
    with pytest.raises(ValueError, match=message):
        authorize_profile(profile, config)


def test_bf16_requirement_needs_disposable_update(tmp_path, monkeypatch):
    _, config = fake_runtime(monkeypatch, training_config(tmp_path), "cuda")
    profile = RuntimeProfile.model_validate(
        profile_value(backend="cuda", requirements={"bf16": True})
    )
    calls = []

    def precision_update(requested, *, authorization):
        calls.append((requested.runtime.precision, authorization))
        return SimpleNamespace(tested_precisions=("fp32",))

    monkeypatch.setattr(runtime, "validate_runtime", precision_update)
    with pytest.raises(ValueError, match="BF16.*not verified"):
        authorize_profile(profile, config)
    assert [precision for precision, _ in calls] == ["bf16"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("python", "/nonexistent/python"),
        ("torch_version", "changed-framework"),
        ("torch_hip", "changed-runtime"),
        ("source_sha256", "0" * 64),
        ("device_name", "Changed Accelerator"),
    ],
)
def test_authorization_rejects_runtime_drift(tmp_path, monkeypatch, field, replacement):
    from sparselab.runtime_profile import require_authorization

    evidence, config = fake_runtime(monkeypatch, training_config(tmp_path), "rocm")
    profile = RuntimeProfile.model_validate(profile_value(backend="rocm"))
    authorization = authorize_profile(profile, config)
    evidence[field] = replacement
    with pytest.raises(ValueError, match="identity changed|source changed"):
        require_authorization(config, authorization)
