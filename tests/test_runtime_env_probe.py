"""Framework fixtures exercise inventory without requiring any accelerator library."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from sparselab import runtime_env_probe


@pytest.fixture
def mocked_imports(monkeypatch):
    original_import = importlib.import_module
    modules = {}

    def import_module(name):
        if name in modules:
            value = modules[name]
            if isinstance(value, Exception):
                raise value
            return value
        if name in ("torch", "mlx.core", "sparselab"):
            missing = name.split(".")[0]
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return original_import(name)

    monkeypatch.setattr(runtime_env_probe.importlib, "import_module", import_module)
    return modules


def device_api(available, names=(), error=None):
    def is_available():
        if error:
            raise error
        return available

    return SimpleNamespace(
        is_available=is_available,
        device_count=lambda: len(names),
        get_device_name=lambda i: names[i],
    )


def torch_fixture(*, hip=None, cuda=None, available=False, names=()):
    return SimpleNamespace(
        __version__="2.14.0",
        __file__="/test/torch/__init__.py",
        version=SimpleNamespace(hip=hip, cuda=cuda),
        cuda=device_api(available, names),
        xpu=device_api(False),
        backends=SimpleNamespace(mps=device_api(False)),
    )


def test_missing_frameworks_keep_interpreter_visible(mocked_imports):
    result = runtime_env_probe.probe()
    assert result["python"] == sys.executable
    assert result["prefix"] == sys.prefix
    assert result["python_version"]
    assert result["sparse_lab_import"]["success"] is False
    assert result["source_sha256"] is None
    assert result["torch"]["installed"] is False
    assert result["mlx"]["installed"] is False
    assert result["backends"] == []
    assert all(not item["available"] for item in result["devices"].values())


def test_false_rocm_wheel_does_not_claim_rocm(mocked_imports):
    mocked_imports["torch"] = torch_fixture(hip="10.0.0", available=False)
    result = runtime_env_probe.probe()
    assert result["torch"]["hip"] == "10.0.0"
    assert result["devices"]["rocm"] == {"available": False, "count": 0, "names": []}
    assert result["backends"] == ["cpu"]


@pytest.mark.parametrize(
    "hip,cuda,expected,excluded",
    [
        ("10.0.0", None, "rocm", "cuda"),
        (None, "13.0", "cuda", "rocm"),
    ],
)
def test_gpu_builds_remain_separate(mocked_imports, hip, cuda, expected, excluded):
    mocked_imports["torch"] = torch_fixture(
        hip=hip, cuda=cuda, available=True, names=[f"GPU {i}" for i in range(40)]
    )
    result = runtime_env_probe.probe()
    assert result["devices"][expected]["count"] == 40
    assert len(result["devices"][expected]["names"]) == 32
    assert result["devices"][excluded]["available"] is False
    assert result["backends"] == ["cpu", expected]


def test_query_error_is_independent_of_xpu_and_mps(mocked_imports):
    torch = torch_fixture(hip="10.0", available=False)
    torch.cuda = device_api(False, error=RuntimeError("driver failed"))
    torch.xpu = device_api(True, ["Intel GPU"])
    torch.backends.mps = device_api(True)
    mocked_imports["torch"] = torch
    result = runtime_env_probe.probe()
    assert "driver failed" in result["devices"]["rocm"]["error"]
    assert result["backends"] == ["cpu", "xpu", "mps"]
    assert result["devices"]["xpu"]["names"] == ["Intel GPU"]


def test_mlx_only_metal_availability(mocked_imports):
    mocked_imports["mlx.core"] = SimpleNamespace(
        __version__="0.32.2", metal=SimpleNamespace(is_available=lambda: True)
    )
    result = runtime_env_probe.probe()
    assert result["mlx"] == {"installed": True, "version": "0.32.2", "available": True}
    assert result["devices"]["metal"]["count"] == 1
    assert result["backends"] == ["metal"]


def test_source_failure_does_not_suppress_frameworks(mocked_imports, monkeypatch):
    mocked_imports["sparselab"] = ModuleType("sparselab")
    identity = ModuleType("sparselab.runtime_identity_probe")

    def broken_identity():
        raise RuntimeError("source inaccessible")

    identity.source_identity = broken_identity
    monkeypatch.setitem(sys.modules, "sparselab.runtime_identity_probe", identity)
    mocked_imports["torch"] = torch_fixture()
    result = runtime_env_probe.probe()
    assert result["sparse_lab_import"] == {"success": True, "error": None}
    assert "source inaccessible" in result["source_error"]
    assert result["source_sha256"] is None
    assert result["backends"] == ["cpu"]


def test_framework_import_failure_is_reported_independently(mocked_imports):
    mocked_imports["torch"] = RuntimeError("torch broken")
    mocked_imports["mlx.core"] = RuntimeError("mlx broken")
    result = runtime_env_probe.probe()
    assert "torch broken" in result["torch"]["error"]
    assert "mlx broken" in result["mlx"]["error"]
    assert result["backends"] == []


def test_script_runs_without_site_packages(tmp_path):
    script = tmp_path / "runtime_env_probe.py"
    script.write_bytes(Path(runtime_env_probe.__file__).read_bytes())
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    parsed = json.loads(result.stdout)
    assert parsed["sparse_lab_import"]["success"] is False
    assert parsed["source_sha256"] is None
    assert parsed["torch"]["installed"] is False
    assert parsed["backends"] == []
