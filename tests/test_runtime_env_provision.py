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
    # Vendor installation is mocked, including the recipe's required host.
    monkeypatch.setattr(provision_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(provision_module.platform, "machine", lambda: "x86_64")
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
        lambda profile, *, precision=None: {
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


@pytest.mark.parametrize(
    "system,machine", [("Darwin", "arm64"), ("Darwin", "x86_64"), ("Linux", "aarch64")]
)
def test_unsupported_host_refuses_before_uv_or_publication(
    provision_fixture, monkeypatch, system, machine
):
    monkeypatch.setattr(provision_module.platform, "system", lambda: system)
    monkeypatch.setattr(provision_module.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Linux x86_64"):
        _provision()
    assert not provision_fixture.commands
    assert not provision_fixture.root.exists()
    assert not provision_fixture.registry.exists()


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


@pytest.mark.parametrize("executable", [True, False])
def test_intel_provision_requires_executable_xpu_and_registers_fp32(
    provision_fixture, monkeypatch, executable
):
    fixture = provision_fixture
    observed = provision_module.inspect_python(fixture.root / "gpu/bin/python")
    observed["torch"].update(version="2.14.1+xpu", hip=None, cuda=None)
    observed["backends"] = ["cpu", "xpu"] if executable else ["cpu"]
    observed["devices"] = {
        "xpu": {"count": int(executable), "names": ["Intel Arc A380"]}
    }
    monkeypatch.setattr(provision_module, "inspect_python", lambda python: observed)
    fixture.probe.update(backend="xpu", torch_hip=None, device_name="Intel Arc A380")
    fixture.doctor["tested_runtime"].update(backend="xpu", tested_precisions=["fp32"])
    result = provision_module.provision("gpu", recipe="xpu-arc-v1")
    if executable:
        assert result["status"] == "READY"
        registered = read_registry(fixture.registry).runtimes["gpu"]
        assert registered.backend == "xpu"
        assert registered.requirements.bf16 is None
        receipt = json.loads(Path(result["receipt_path"]).read_text())
        assert receipt["torch"]["version"] == "2.14.1+xpu"
        assert receipt["tested_runtime"]["tested_precisions"] == ["fp32"]
        assert receipt["recipe"] == "xpu-arc-v1"
    else:
        assert result["status"] == "ERROR"
        assert "not executable xpu" in result["reason"]
        assert not fixture.registry.exists()
        assert Path(result["failure_record"]).is_file()


def _seed_unrelated(fixture):
    entry = RuntimeEntry.model_validate(
        {
            "python": str(Path(sys.executable).absolute()),
            "engine": "pytorch",
            "backend": "cpu",
            "device_index": 0,
        }
    )
    register_runtime("other", entry, registry_path=fixture.registry)
    return entry


def _intent(fixture):
    return json.loads((fixture.root / "gpu/provision-registration.json").read_text())


def _assert_retained(fixture, entry):
    intent = _intent(fixture)
    assert intent["status"] == "REGISTRATION_PENDING"
    assert intent["registration_state"] == "unknown"
    assert intent["registry_path"] == str(fixture.registry)
    assert intent["profile"]["python"] == str(fixture.root / "gpu/bin/python")
    assert intent["tested_runtime"] == fixture.doctor["tested_runtime"]
    assert intent["package_inventory"] == fixture.inventory
    assert len(list(Path(intent["logs"]).glob("*.log"))) == 5
    assert "Do not rerun provision" in intent["recovery"]
    assert read_registry(fixture.registry).runtimes["other"] == entry
    return intent


@pytest.mark.parametrize("mode", ["conflict", "refusal", "write", "directory"])
def test_registration_failure_never_publishes_ready(
    provision_fixture, monkeypatch, mode
):
    import os

    from sparselab import runtime_environments as registry_module

    fixture = provision_fixture
    other = _seed_unrelated(fixture)
    before = fixture.registry.read_bytes()
    original_register = provision_module.register_runtime
    original_fsync = os.fsync

    def register(identifier, entry, **kwargs):
        _assert_retained(fixture, other)
        assert not (fixture.root / "gpu/provision-receipt.json").exists()
        if mode == "conflict":
            original_register(identifier, other, **kwargs)
        elif mode == "refusal":
            raise ValueError("injected registry refusal")
        return original_register(identifier, entry, **kwargs)

    def fsync(fd):
        if (
            mode == "directory"
            and os.fstat(fd).st_ino == fixture.registry.parent.stat().st_ino
        ):
            raise OSError("injected registry directory durability failure")
        return original_fsync(fd)

    def replace(source, destination):
        raise OSError("injected registry write failure")

    monkeypatch.setattr(provision_module, "register_runtime", register)
    if mode == "write":
        monkeypatch.setattr(registry_module.os, "replace", replace)
    monkeypatch.setattr(os, "fsync", fsync)
    result = _provision()
    assert result["status"] == "ERROR"
    assert result["registration_state"] == "unknown"
    assert json.loads(Path(result["failure_record"]).read_text()) == {
        key: value for key, value in result.items() if key != "failure_record"
    }
    assert not (fixture.root / "gpu/provision-receipt.json").exists()
    _assert_retained(fixture, other)
    entries = read_registry(fixture.registry).runtimes
    if mode == "conflict":
        assert entries["gpu"] == other
    elif mode == "directory":
        assert entries["gpu"].python == fixture.root / "gpu/bin/python"
    else:
        assert fixture.registry.read_bytes() == before


@pytest.mark.parametrize("mode", ["partial", "link", "directory"])
def test_ready_publication_failure_preserves_registered_entry_and_recovery(
    provision_fixture, monkeypatch, mode
):
    import os

    fixture = provision_fixture
    other = _seed_unrelated(fixture)
    original_record = provision_module._record
    original_fsync = os.fsync
    ready = fixture.root / "gpu/provision-receipt.json"

    def record(path, value):
        if path.name == "provision-ready-staged.json":
            assert "gpu" in read_registry(fixture.registry).runtimes
            if mode == "partial":
                path.write_text('{"status":')
                raise OSError("injected partial receipt write")
        return original_record(path, value)

    def link(source, destination):
        raise OSError("injected receipt link failure")

    def fsync(fd):
        if (
            mode == "directory"
            and ready.exists()
            and os.fstat(fd).st_ino == ready.parent.stat().st_ino
        ):
            raise OSError("injected receipt directory durability failure")
        return original_fsync(fd)

    monkeypatch.setattr(provision_module, "_record", record)
    if mode == "link":
        monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(os, "fsync", fsync)
    result = _provision()
    assert result["status"] == "ERROR"
    assert result["registration_state"] == "registered"
    assert not ready.exists()
    assert (ready.parent / "provision-ready-staged.json").exists()
    assert (
        read_registry(fixture.registry).runtimes["gpu"].python
        == ready.parent / "bin/python"
    )
    _assert_retained(fixture, other)
    before = fixture.registry.read_bytes()
    with pytest.raises(ValueError, match="already registered"):
        _provision()
    assert fixture.registry.read_bytes() == before


@pytest.mark.parametrize("after_registration", [False, True])
def test_process_interruption_retains_intent_without_claiming_ready(
    provision_fixture, monkeypatch, after_registration
):
    fixture = provision_fixture
    other = _seed_unrelated(fixture)
    original_register = provision_module.register_runtime

    class Interrupted(BaseException):
        pass

    def register(identifier, entry, **kwargs):
        _assert_retained(fixture, other)
        if after_registration:
            original_register(identifier, entry, **kwargs)
        raise Interrupted

    monkeypatch.setattr(provision_module, "register_runtime", register)
    with pytest.raises(Interrupted):
        _provision()
    assert not (fixture.root / "gpu/provision-receipt.json").exists()
    assert not (fixture.root / "gpu/provision-failure.json").exists()
    _assert_retained(fixture, other)
    assert ("gpu" in read_registry(fixture.registry).runtimes) == after_registration


def test_intent_durability_failure_prevents_registration(
    provision_fixture, monkeypatch
):
    fixture = provision_fixture
    other = _seed_unrelated(fixture)
    original = provision_module._record

    def record(path, value):
        if path.name == "provision-registration.json":
            raise OSError("injected intent durability failure")
        return original(path, value)

    def register(*args, **kwargs):
        pytest.fail("registration must not start before durable intent")

    monkeypatch.setattr(provision_module, "_record", record)
    monkeypatch.setattr(provision_module, "register_runtime", register)
    result = _provision()
    assert result["status"] == "ERROR"
    assert read_registry(fixture.registry).runtimes == {"other": other}
    assert not (fixture.root / "gpu/provision-receipt.json").exists()


def test_success_publishes_ready_only_after_registration(
    provision_fixture, monkeypatch
):
    fixture = provision_fixture
    other = _seed_unrelated(fixture)
    original = provision_module.register_runtime

    def register(identifier, entry, **kwargs):
        _assert_retained(fixture, other)
        assert not (fixture.root / "gpu/provision-receipt.json").exists()
        return original(identifier, entry, **kwargs)

    monkeypatch.setattr(provision_module, "register_runtime", register)
    result = _provision()
    assert result["status"] == "READY"
    receipt = Path(result["receipt_path"])
    assert json.loads(receipt.read_text())["status"] == "READY"
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert not (receipt.parent / "provision-failure.json").exists()
    _assert_retained(fixture, other)
