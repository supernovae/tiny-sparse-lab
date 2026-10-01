"""Provisioning contracts without installing vendor wheels or mutating the checkout."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab import runtime_env_provision as provision_module
from sparselab.runtime_environments import RuntimeEntry, read_registry, register_runtime
from sparselab.runtime_identity_probe import source_identity


@pytest.fixture
def provision_fixture(tmp_path, monkeypatch):
    root = tmp_path / "data" / "sparselab" / "runtimes"
    registry = tmp_path / "config" / "sparselab" / "runtimes.yaml"
    monkeypatch.setenv("SPARSELAB_RUNTIME_DIR", str(root))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(registry.parent.parent))
    monkeypatch.setenv("UV_EXTRAS", "cpu")
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://cpu.invalid/simple")
    monkeypatch.setenv("UV_INDEX_URL", "https://cpu.invalid/simple")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://cpu.invalid/simple")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "cpu-venv"))
    monkeypatch.setattr(
        provision_module,
        "preflight_runtime_root",
        lambda path: {
            "root": str(path),
            "free_bytes": 30 * 1024**3,
            "free_inodes": 200_000,
        },
    )
    monkeypatch.setattr(
        provision_module,
        "_validate_python",
        lambda path: {
            "version": "3.14.4",
            "major": 3,
            "minor": 14,
            "base_executable": str(path),
        },
    )
    monkeypatch.setattr(provision_module, "_source_commit", lambda checkout: "a" * 40)
    monkeypatch.setattr(
        provision_module,
        "inspect_python",
        lambda python: {
            "python": str(python),
            "status": "READY",
            "source_sha256": source_identity()["source_sha256"],
            "torch": {
                "installed": True,
                "version": "2.13.0+rocm10.0.0",
                "path": str(
                    root / "gpu/lib/python3.14/site-packages/torch/__init__.py"
                ),
                "hip": "10.0.0",
                "cuda": None,
            },
            "backends": ["rocm"],
            "devices": {"rocm": {"count": 1, "names": ["AMD Radeon RX 7900 XTX"]}},
        },
    )
    probe = {
        "backend": "rocm",
        "torch_hip": "10.0.0",
        "device_name": "AMD Radeon RX 7900 XTX",
        **source_identity(),
    }
    monkeypatch.setattr(
        provision_module, "probe_runtime_profile", lambda profile: probe
    )
    doctor = {
        "runtime_env_doctor_version": 1,
        "status": "READY",
        "probe": probe,
        "tested_runtime": {
            "engine": "pytorch",
            "backend": "rocm",
            "device_index": 0,
            "tested_precisions": ["bf16"],
            "tested_features": ["forward_backward_optimizer"],
        },
        "reason": None,
    }
    monkeypatch.setattr(
        provision_module,
        "doctor",
        lambda profile: {
            **doctor,
            "id": profile.id,
            "profile": profile.model_dump(mode="json"),
        },
    )
    commands = []
    exports = ["numpy==2.4.2\npyyaml==6.0.3\n"]
    inventory = [
        {"name": "torch", "version": "2.13.0+rocm10.0.0"},
        {"name": "sparselab", "version": "0.1.0"},
    ]

    def fake_uv(command, *, environment, checkout, log):
        command = [str(part) for part in command]
        commands.append((command, dict(environment), Path(checkout), Path(log)))
        if command[:2] == ["uv", "venv"]:
            path = Path(command[-1]) / "bin" / "python"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o700)
        elif command[:2] == ["uv", "export"]:
            Path(command[command.index("--output-file") + 1]).write_text(exports[0])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("mock uv completed\n")

    monkeypatch.setattr(provision_module, "_run_uv", fake_uv)

    def fake_bounded(command, **kwargs):
        if command[1:3] == ["-I", "-c"]:
            assert str(root / "gpu/bin/python") == command[0]
            return SimpleNamespace(
                returncode=0, stdout=b"SparseLab CLI import complete\n", stderr=b""
            )
        assert command[:3] == ["uv", "pip", "list"]
        assert command[command.index("--python") + 1] == str(root / "gpu/bin/python")
        assert command[-2:] == ["--format", "json"]
        assert Path(kwargs["env"]["UV_CACHE_DIR"]).is_relative_to(root)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(inventory).encode(), stderr=b""
        )

    monkeypatch.setattr(provision_module, "run_bounded", fake_bounded)
    return SimpleNamespace(
        root=root,
        registry=registry,
        commands=commands,
        exports=exports,
        inventory=inventory,
        probe=probe,
        doctor=doctor,
    )


def _provision(identifier="gpu", **kwargs):
    return provision_module.provision(identifier, recipe="rocm-gfx1100-v1", **kwargs)


def _commands(fixture):
    return [item[0] for item in fixture.commands]


def test_success_receipt_registers_only_verified_vendor_environment(provision_fixture):
    fixture = provision_fixture
    result = _provision()
    assert result["status"] == "READY"
    assert result["runtime_provision_version"] == 1
    assert result["id"] == "gpu"
    assert result["recipe"] == "rocm-gfx1100-v1"
    target = fixture.root / "gpu"
    assert Path(result["environment_path"]) == target
    receipt_path = Path(result["receipt_path"])
    assert receipt_path == target / "provision-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["runtime_provision_version"] == 1
    assert receipt["id"] == "gpu"
    assert receipt["recipe"] == "rocm-gfx1100-v1"
    assert receipt["recipe_version"] == 1
    recipe_file = provision_module.get_recipe("rocm-gfx1100-v1").requirements_file
    assert (
        receipt["recipe_sha256"] == hashlib.sha256(recipe_file.read_bytes()).hexdigest()
    )
    assert receipt["source_sha256"] == source_identity()["source_sha256"]
    assert receipt["source_commit"] == "a" * 40
    assert receipt["package_inventory"] == fixture.inventory
    assert receipt["indexes"] == list(
        provision_module.get_recipe("rocm-gfx1100-v1").indexes
    )
    assert receipt["profile"]["backend"] == "rocm"
    assert receipt["profile"]["requirements"]["torch_hip"] is True
    assert receipt["probe"] == fixture.probe
    assert receipt["tested_runtime"] == fixture.doctor["tested_runtime"]
    registered = read_registry(fixture.registry).runtimes["gpu"]
    assert registered.python == target / "bin/python"
    assert registered.backend == "rocm"
    assert registered.requirements.bf16 is True
    commands = _commands(fixture)
    venv, export = commands[:2]
    assert venv[:2] == ["uv", "venv"]
    assert "--no-project" in venv and "--no-python-downloads" in venv
    assert "--python" in venv and venv[-1] == str(target)
    assert export[:2] == ["uv", "export"]
    for option in ("--locked", "--no-default-groups", "--no-dev", "--no-emit-project"):
        assert option in export
    assert "--extra" not in export and "--all-extras" not in export
    installs = [c for c in commands if c[:3] == ["uv", "pip", "install"]]
    assert len(installs) == 3
    assert installs[0][-2:] == ["-r", str(target / "common-requirements.txt")]
    assert installs[1][-2:] == ["-r", str(recipe_file)]
    assert all(index in recipe_file.read_text() for index in receipt["indexes"])
    assert all(
        "--no-config" in command for command in commands if command[:2] == ["uv", "pip"]
    )
    assert any(
        c[:3] == ["uv", "pip", "install"] and "--no-deps" in c and "--editable" in c
        for c in commands
    )
    editable = next(c for c in commands if "--editable" in c)
    assert editable[editable.index("--editable") + 1] == str(
        Path(provision_module.__file__).resolve().parents[2]
    )
    assert all(
        "pytorch-cpu" not in " ".join(c) and "cpu.invalid" not in " ".join(c)
        for c in commands
    )
    for _, environment, _, _ in fixture.commands:
        assert not any(
            key in environment
            for key in (
                "UV_EXTRAS",
                "UV_EXTRA_INDEX_URL",
                "UV_INDEX_URL",
                "UV_DEFAULT_INDEX",
                "UV_PROJECT_ENVIRONMENT",
            )
        )
        assert environment["UV_PYTHON_DOWNLOADS"] == "never"
        assert Path(environment["UV_CACHE_DIR"]).is_relative_to(fixture.root)
        assert Path(environment["TMPDIR"]).is_relative_to(fixture.root)
        assert "cpu.invalid" not in str(environment)


def test_failed_install_keeps_partial_bytes_failure_record_and_no_registration(
    provision_fixture, monkeypatch
):
    fixture = provision_fixture
    original = provision_module._run_uv

    def fail_vendor(command, *, environment, checkout, log):
        original(command, environment=environment, checkout=checkout, log=log)
        if "rocm-gfx1100.txt" in " ".join(map(str, command)):
            raise ValueError("vendor download rejected")

    monkeypatch.setattr(provision_module, "_run_uv", fail_vendor)
    result = _provision()
    assert result["runtime_provision_version"] == 1
    assert result["status"] == "ERROR"
    assert "vendor download rejected" in result["reason"]
    assert Path(result["environment_path"]) == fixture.root / "gpu"
    assert (fixture.root / "gpu/bin/python").exists()
    assert Path(result["failure_record"]).is_file()
    assert "vendor download rejected" in Path(result["failure_record"]).read_text()
    assert "gpu" not in read_registry(fixture.registry).runtimes
    assert not (fixture.root / "gpu/provision-receipt.json").exists()


@pytest.mark.parametrize("conflict", ["registered", "target"])
def test_existing_id_or_target_refuses_before_uv_without_registry_change(
    provision_fixture, conflict
):
    fixture = provision_fixture
    fixture.registry.parent.mkdir(parents=True, exist_ok=True)
    if conflict == "registered":
        register_runtime(
            "gpu",
            RuntimeEntry.model_validate(
                {
                    "python": str(Path(sys.executable).absolute()),
                    "engine": "pytorch",
                    "backend": "cpu",
                    "device_index": 0,
                }
            ),
            registry_path=fixture.registry,
        )
    else:
        fixture.root.mkdir(parents=True)
        (fixture.root / "gpu").mkdir()
        (fixture.root / "gpu" / "owned-data").write_text("preserve me")
        fixture.registry.write_text("runtime_registry_version: 1\nruntimes: {}\n")
    before = hashlib.sha256(fixture.registry.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already|exist"):
        _provision()
    assert hashlib.sha256(fixture.registry.read_bytes()).hexdigest() == before
    assert not fixture.commands
    if conflict == "target":
        assert (fixture.root / "gpu/owned-data").read_text() == "preserve me"


def test_preflight_rejection_is_read_only_and_never_invokes_uv(
    provision_fixture, monkeypatch
):
    fixture = provision_fixture

    def refuse(root):
        assert root == fixture.root
        assert not root.exists()
        raise ValueError("runtime root requires a local writable filesystem")

    monkeypatch.setattr(provision_module, "preflight_runtime_root", refuse)
    with pytest.raises(ValueError, match="local writable filesystem"):
        _provision()
    assert not fixture.root.exists()
    assert not fixture.registry.exists()
    assert not fixture.commands


@pytest.mark.parametrize(
    "export",
    [
        "torch==2.14.0+cpu\n",
        "--index-url https://download.pytorch.org/whl/cpu\nnumpy==2.4.2\n",
        "numpy==2.4.2\n--extra-index-url https://download.pytorch.org/whl/cpu\n",
        "pytorch-cpu==2.14.0\n",
    ],
)
def test_common_export_rejects_torch_or_cpu_index_before_install(
    provision_fixture, export
):
    fixture = provision_fixture
    fixture.exports[0] = export
    result = _provision()
    assert result["status"] == "ERROR"
    assert "gpu" not in read_registry(fixture.registry).runtimes
    assert (fixture.root / "gpu/common-requirements.txt").read_text() == export
    assert not any(
        command[:3] == ["uv", "pip", "install"] for command in _commands(fixture)
    )


def test_editable_source_cannot_replace_vendor_torch(provision_fixture, monkeypatch):
    fixture = provision_fixture
    original = provision_module.inspect_python
    observations = []

    def changed(python):
        item = original(python)
        observations.append(item)
        if len(observations) >= 2:
            return {
                **item,
                "torch": {**item["torch"], "version": "2.14.0+cpu", "hip": None},
            }
        return item

    monkeypatch.setattr(provision_module, "inspect_python", changed)
    result = _provision()
    assert result["status"] == "ERROR"
    assert len(observations) >= 2
    assert "gpu" not in read_registry(fixture.registry).runtimes
    assert Path(result["failure_record"]).is_file()
    assert not (fixture.root / "gpu/provision-receipt.json").exists()


@pytest.mark.parametrize(
    "doctor_change",
    [
        {"status": "UNAVAILABLE", "reason": "optimizer failed"},
        {"tested_runtime": None, "reason": "optimizer unverified"},
    ],
)
def test_publication_requires_verified_bf16_optimizer(
    provision_fixture, monkeypatch, doctor_change
):
    fixture = provision_fixture
    base_doctor = provision_module.doctor
    monkeypatch.setattr(
        provision_module,
        "doctor",
        lambda profile: {**base_doctor(profile), **doctor_change},
    )
    result = _provision()
    assert result["status"] == "ERROR"
    assert "gpu" not in read_registry(fixture.registry).runtimes
    assert Path(result["failure_record"]).is_file()
    assert not (fixture.root / "gpu/provision-receipt.json").exists()
