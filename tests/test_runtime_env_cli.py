"""Consumer-facing runtime environment CLI behavior with host-local registry isolation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sparselab import runtime_env_cli as cli
from sparselab.runtime_env_inventory import canonical_python
from sparselab.runtime_environments import RuntimeEntry, read_registry, register_runtime


@pytest.fixture
def registry_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("SPARSELAB_RUNTIME_DIR", str(tmp_path / "runtimes"))
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "science"))
    return tmp_path / "config/sparselab/runtimes.yaml"


def invoke(capsys: pytest.CaptureFixture[str], *args: str) -> dict:
    cli.main([*args, "--json"])
    return json.loads(capsys.readouterr().out)


def registered_cpu(path: Path, identifier: str = "cpu-local") -> None:
    register_runtime(
        identifier,
        RuntimeEntry(
            python=Path(sys.executable).absolute(),
            engine="pytorch",
            backend="cpu",
            device_index=0,
        ),
        registry_path=path,
    )


def observed(*, backends: list[str] | None = None, status: str = "READY") -> dict:
    return {
        "ids": [],
        "origins": ["active"],
        "python": str(canonical_python(Path(sys.executable).absolute())),
        "prefix": sys.prefix,
        "python_version": sys.version.split()[0],
        "sparse_lab_import": {"success": True, "error": None},
        "source_sha256": "test-source",
        "torch": {
            "installed": True,
            "version": "test",
            "path": "/torch",
            "hip": None,
            "cuda": None,
        },
        "mlx": {"installed": False, "version": None, "available": False},
        "devices": {"rocm": {"available": False, "count": 0}},
        "backends": ["cpu"] if backends is None else backends,
        "status": status,
        "reason": None if status == "READY" else status,
    }


def inventory_for(environment: dict) -> dict:
    return {
        "runtime_env_inventory_version": 1,
        "hardware": [],
        "environments": [environment],
        "truncated": False,
    }


def test_real_cpu_register_observe_doctor_and_unregister_preserves_interpreter(
    registry_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    python = Path(sys.executable).absolute()
    result = invoke(
        capsys, "register", "cpu-local", "--python", str(python), "--backend", "cpu"
    )
    assert result["status"] == "READY"
    assert result["id"] == "cpu-local"
    assert result["profile"]["backend"] == "cpu"
    assert "forward_backward_optimizer" in result["tested_runtime"]["tested_features"]
    assert read_registry(registry_path).runtimes["cpu-local"].python == python

    listed = invoke(capsys, "list")
    assert listed["runtime_registry_version"] == 1
    assert len(listed["runtimes"]) == 1
    assert listed["runtimes"][0]["id"] == "cpu-local"
    assert listed["runtimes"][0]["status"] == "READY"
    shown = invoke(capsys, "show", "cpu-local")
    assert shown["id"] == "cpu-local"
    assert shown["backend"] == "cpu"
    assert shown["status"] == "READY"
    profile = invoke(capsys, "profile", "cpu-local")
    assert profile["runtime_profile_version"] == 1
    assert profile["id"] == "cpu-local"
    assert profile["python"] == str(python)
    assert profile["backend"] == "cpu"
    result = invoke(capsys, "doctor", "cpu-local")
    assert result["runtime_env_doctor_version"] == 1
    assert result["status"] == "READY"
    assert "forward_backward_optimizer" in result["tested_runtime"]["tested_features"]
    assert result["profile"] == profile
    assert result["probe"]["backend"] == "cpu"

    invoke(capsys, "unregister", "cpu-local")
    assert not read_registry(registry_path).runtimes
    assert python.is_file() and python.stat().st_size > 0


def test_duplicate_and_invalid_id_leave_published_bytes_unchanged(
    registry_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registered_cpu(registry_path)
    before = registry_path.read_bytes()
    for identifier in ("cpu-local", "../outside"):
        with pytest.raises(SystemExit) as error:
            invoke(
                capsys,
                "register",
                identifier,
                "--python",
                str(Path(sys.executable).absolute()),
                "--backend",
                "cpu",
            )
        assert error.value.code != 0
        capsys.readouterr()
        assert registry_path.read_bytes() == before
    assert read_registry(registry_path).runtimes.keys() == {"cpu-local"}


@pytest.mark.parametrize(
    ("scenario", "environment"),
    [
        ("source drift", observed(status="SOURCE_MISMATCH")),
        (
            "missing Torch",
            {
                **observed(status="UNAVAILABLE"),
                "torch": {"installed": False},
                "backends": [],
            },
        ),
        ("CPU is not ROCm", observed()),
    ],
)
def test_failed_registration_never_publishes(
    scenario: str,
    environment: dict,
    registry_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_cpu(registry_path)
    before = registry_path.read_bytes()
    monkeypatch.setattr(cli, "inventory", lambda **kwargs: inventory_for(environment))
    monkeypatch.setattr(cli, "inspect_python", lambda *args, **kwargs: environment)
    if scenario == "CPU is not ROCm":
        backend = "rocm"
    else:
        backend = "cpu"
    with pytest.raises(SystemExit) as error:
        invoke(
            capsys,
            "register",
            "rejected",
            "--python",
            str(Path(sys.executable).absolute()),
            "--backend",
            backend,
        )
    assert error.value.code != 0
    assert registry_path.read_bytes() == before
    assert "rejected" not in read_registry(registry_path).runtimes


def test_doctor_refusal_does_not_publish(
    registry_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_cpu(registry_path)
    before = registry_path.read_bytes()
    monkeypatch.setattr(cli, "inspect_python", lambda *args, **kwargs: observed())
    monkeypatch.setattr(
        cli,
        "doctor",
        lambda profile: {
            "runtime_env_doctor_version": 1,
            "id": profile.id,
            "status": "UNAVAILABLE",
            "profile": profile.model_dump(mode="json"),
            "probe": {"backend": "cpu"},
            "tested_runtime": None,
            "reason": "optimizer update failed",
        },
    )
    with pytest.raises(SystemExit) as error:
        invoke(
            capsys,
            "register",
            "unverified",
            "--python",
            str(Path(sys.executable).absolute()),
            "--backend",
            "cpu",
        )
    assert error.value.code != 0
    assert registry_path.read_bytes() == before
    assert "unverified" not in read_registry(registry_path).runtimes


def test_inference_requires_unambiguous_executable_backend(
    registry_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = observed(backends=["cpu", "cuda", "rocm"])
    monkeypatch.setattr(cli, "inventory", lambda **kwargs: inventory_for(environment))
    monkeypatch.setattr(cli, "inspect_python", lambda *args, **kwargs: environment)
    with pytest.raises(SystemExit) as error:
        invoke(
            capsys,
            "register",
            "ambiguous",
            "--python",
            str(Path(sys.executable).absolute()),
        )
    assert error.value.code != 0
    assert not registry_path.exists()


def test_read_only_commands_never_publish_or_install(
    registry_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_cpu(registry_path)
    before = registry_path.read_bytes()
    environment = observed()
    inventory_result = inventory_for(environment)
    monkeypatch.setattr(cli, "inventory", lambda **kwargs: inventory_result)
    monkeypatch.setattr(cli, "inspect_python", lambda *args, **kwargs: environment)
    monkeypatch.setattr(
        cli,
        "doctor",
        lambda profile: {
            "runtime_env_doctor_version": 1,
            "id": profile.id,
            "status": "READY",
            "profile": profile.model_dump(mode="json"),
            "probe": {"backend": "cpu"},
            "tested_runtime": {"tested_features": ["forward_backward_optimizer"]},
            "reason": None,
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only runtime command tried to publish registry")

    monkeypatch.setattr(cli, "register_runtime", forbidden)
    monkeypatch.setattr(cli, "unregister_runtime", forbidden)
    run = subprocess.run

    def reject_installer(command, *args, **kwargs):
        if Path(str(command[0])).name == "uv":
            raise AssertionError("read-only runtime command tried to install/sync")
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", reject_installer)
    assert invoke(capsys, "list")["runtimes"][0]["status"] == "READY"
    assert invoke(capsys, "discover") == inventory_result
    assert invoke(capsys, "show", "cpu-local")["status"] == "READY"
    assert invoke(capsys, "profile", "cpu-local")["backend"] == "cpu"
    assert invoke(capsys, "doctor", "cpu-local")["status"] == "READY"
    assert registry_path.read_bytes() == before


def test_profile_resolution_never_inspects_inventory_or_runs_doctor(
    registry_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_cpu(registry_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("profile resolution must not probe runtime")

    monkeypatch.setattr(cli, "inventory", forbidden)
    monkeypatch.setattr(cli, "inspect_python", forbidden)
    monkeypatch.setattr(cli, "doctor", forbidden)
    result = invoke(capsys, "profile", "cpu-local")
    assert result["id"] == "cpu-local"
    assert result["python"] == str(Path(sys.executable).absolute())


def test_missing_registered_interpreter_remains_visible_after_candidate_cap(
    registry_path, tmp_path, capsys, monkeypatch
):
    missing = RuntimeEntry(
        python=tmp_path / "missing/bin/python",
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    register_runtime("missing", missing, registry_path=registry_path)
    monkeypatch.setattr(
        cli, "inventory", lambda **kwargs: {"environments": [], "truncated": True}
    )
    listed = invoke(capsys, "list")
    assert listed["runtimes"][0]["status"] == "NOT_PROVISIONED"
    with pytest.raises(SystemExit) as error:
        cli.main(["doctor", "missing", "--json"])
    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["runtime_env_doctor_version"] == 1
    assert result["status"] == "NOT_PROVISIONED"
    assert result["tested_runtime"] is None


def test_registration_profile_source_mismatch_returns_explicit_repair(
    registry_path, capsys, monkeypatch
):
    from sparselab import runtime_env_doctor

    monkeypatch.setattr(cli, "inspect_python", lambda *args: observed())

    def cloned_source(profile):
        raise ValueError(
            "profile python SparseLab package/source identity differs from this checkout"
        )

    monkeypatch.setattr(runtime_env_doctor, "probe_runtime_profile", cloned_source)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "register",
                "clone",
                "--python",
                str(Path(sys.executable).absolute()),
                "--backend",
                "cpu",
                "--json",
            ]
        )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "SOURCE_MISMATCH"
    assert (
        runtime_env_doctor.source_repair_command(Path(sys.executable).absolute())
        in result["reason"]
    )
    assert not registry_path.exists()
