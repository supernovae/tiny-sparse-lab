from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sparselab.hosted import bootstrap, bootstrap_worker, cli
from sparselab.hosted.models import HostedTarget


def _request(root: Path, runtime_root: Path, **overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "root": str(root),
        "runtime_root": str(runtime_root),
        "source_bytes": 1,
        "source_commit": "a" * 40,
        "source_sha256": "b" * 64,
        "recipe": "cuda-cu126-v1",
        "instance_id": "c" * 64,
    }
    request.update(overrides)
    return request


def test_remote_preflight_failure_preserves_cause_and_creates_no_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    program = bootstrap.preflight_program(
        **_request(tmp_path / ".." / "unsafe", runtime), timeout=30
    )
    monkeypatch.setattr(
        cli, "_ssh_command", lambda target, source: [sys.executable, "-c", source]
    )
    with pytest.raises(ValueError, match="hosted bootstrap failed: ValueError"):
        cli._remote_json(
            HostedTarget(ssh="native-fixture"), program, 30, scratch=tmp_path
        )
    assert not runtime.exists()


def _linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap_worker.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap_worker.platform, "machine", lambda: "x86_64")


def _storage(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "device": 1,
        "free_bytes": 100 * 1024**3,
        "free_inodes": 200_000,
    }


def test_preflight_refuses_unsupported_platform_before_creating_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bootstrap_worker.platform, "system", lambda: "Darwin")
    root = tmp_path / "root"
    runtime = tmp_path / "runtime"

    with pytest.raises(ValueError):
        bootstrap_worker.preflight(_request(root, runtime))

    assert not root.exists()
    assert not runtime.exists()


def test_preflight_refuses_insufficient_capacity_before_creating_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux(monkeypatch)
    root = tmp_path / "root"
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(
        bootstrap_worker,
        "local_storage",
        lambda path: {**_storage(path), "free_bytes": 0, "free_inodes": 0},
    )

    with pytest.raises(ValueError):
        bootstrap_worker.preflight(_request(root, runtime))

    assert not root.exists()
    assert not runtime.exists()


@pytest.mark.parametrize("filesystem", ["fuse.drivefs", "nfs"])
def test_local_storage_refuses_nonlocal_mount_without_creating_task_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filesystem: str
) -> None:
    root = tmp_path / "uncreated-root"
    original = Path.read_text

    def mountinfo(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/self/mountinfo"):
            return f"1 0 0:1 / / rw - {filesystem} {filesystem} rw\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mountinfo)

    with pytest.raises(ValueError):
        bootstrap_worker.local_storage(root)

    assert not root.exists()


def test_preflight_refuses_symlinked_root_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux(monkeypatch)
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "root"
    root.symlink_to(target, target_is_directory=True)
    runtime = tmp_path / "runtime"

    with pytest.raises(ValueError):
        bootstrap_worker.preflight(_request(root, runtime))

    assert list(target.iterdir()) == []
    assert not runtime.exists()


def test_preflight_refuses_nonprivate_existing_root_before_creating_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux(monkeypatch)
    root = tmp_path / "root"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    runtime = tmp_path / "runtime"

    with pytest.raises(ValueError):
        bootstrap_worker.preflight(_request(root, runtime))

    assert root.stat().st_mode & 0o077
    assert not runtime.exists()


def test_preflight_does_not_overwrite_partial_enrolled_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux(monkeypatch)
    root = tmp_path / "root"
    runtime = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    (root / "hosted-enrollment.json").write_text(
        json.dumps(
            {"enrollment_version": 1, "instance_id": "c" * 64, "boot_id": "boot"}
        )
    )
    unexpected = root / "untrusted-state"
    unexpected.write_text("preserve me")
    monkeypatch.setattr(bootstrap_worker, "local_storage", _storage)
    original = Path.read_text

    def boot_id(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/sys/kernel/random/boot_id"):
            return "boot"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boot_id)

    with pytest.raises(ValueError):
        bootstrap_worker.preflight(_request(root, runtime))

    assert unexpected.read_text() == "preserve me"
    assert not runtime.exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("recipe", "other-recipe"),
        ("source_sha256", "d" * 64),
        ("instance_id", "e" * 64),
    ],
)
def test_existing_setup_identity_mismatch_refuses_before_doctor_or_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, bad_value: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    request = _request(root, tmp_path / "runtime")
    saved_request = {
        key: value
        for key, value in request.items()
        if key in {"root", "runtime_root", "source_commit", "source_sha256", "recipe"}
    }
    result = {"status": "READY", "instance_id": request["instance_id"]}
    if field == "instance_id":
        result["instance_id"] = bad_value
    else:
        saved_request[field] = bad_value
    (root / "hosted-setup.json").write_text(
        json.dumps({"request": saved_request, "boot_id": "boot", "result": result})
    )
    original = Path.read_text

    def boot_id(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/sys/kernel/random/boot_id"):
            return "boot"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boot_id)

    def no_subprocess(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("reuse mismatch reached doctor/provision subprocess")

    monkeypatch.setattr(bootstrap_worker, "run", no_subprocess)

    with pytest.raises(ValueError):
        bootstrap_worker.verified_existing(request)


def test_corrupt_existing_recipe_receipt_refuses_before_live_doctor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    source = root / "source" / ("a" * 40)
    environment = root / "environment"
    worker = root / "work" / "worker"
    (source / "requirements").mkdir(parents=True)
    environment.mkdir(parents=True)
    worker.mkdir(parents=True)
    (source / "requirements" / "cuda-cu126.txt").write_text("expected recipe")
    request = _request(root, tmp_path / "runtime")
    receipt = {
        "recipe_sha256": "0" * 64,
        "tested_runtime": {},
        "source_sha256": request["source_sha256"],
        "profile": {},
    }
    (environment / "provision-receipt.json").write_text(json.dumps(receipt))
    (worker / "hosted-instance.json").write_text(
        json.dumps({"boot_id": "boot", "instance_id": request["instance_id"]})
    )
    result = {
        "status": "READY",
        "instance_id": request["instance_id"],
        "source_root": str(source),
        "worker_root": str(worker),
        "environment_path": str(environment),
        "runtime_id": "hosted-cuda",
        "receipt": receipt,
    }
    (root / "hosted-setup.json").write_text(
        json.dumps(
            {
                "request": {
                    key: value
                    for key, value in request.items()
                    if key
                    in {
                        "root",
                        "runtime_root",
                        "source_commit",
                        "source_sha256",
                        "recipe",
                    }
                },
                "boot_id": "boot",
                "result": result,
            }
        )
    )
    original = Path.read_text

    def boot_id(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/sys/kernel/random/boot_id"):
            return "boot"
        return original(path, *args, **kwargs)

    commands: list[list[str]] = []

    def git_only(command: list[str], **kwargs: object) -> bytes:
        commands.append(command)
        if command[-2:] == ["rev-parse", "HEAD"]:
            return ("a" * 40).encode()
        if command[-3:] == ["status", "--porcelain", "--untracked-files=normal"]:
            return b""
        raise AssertionError(
            f"corrupt receipt reached unexpected subprocess: {command}"
        )

    monkeypatch.setattr(Path, "read_text", boot_id)
    monkeypatch.setattr(bootstrap_worker, "run", git_only)

    with pytest.raises(ValueError):
        bootstrap_worker.verified_existing(request)

    assert len(commands) == 2


def test_setup_rejects_corrupt_source_before_git_or_environment_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    scratch = root / "scratch"
    worker = root / "work" / "worker"
    scratch.mkdir(parents=True)
    worker.mkdir(parents=True)
    bundle = scratch / "source.bundle"
    bundle.write_bytes(b"corrupt")
    request = _request(root, tmp_path / "runtime", source_bundle=str(bundle))

    def no_subprocess(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("corrupt source reached git or provisioning")

    monkeypatch.setattr(bootstrap_worker, "run", no_subprocess)

    with pytest.raises(ValueError):
        bootstrap_worker.setup(request)

    assert not (root / "source").exists()
    assert not (root / "hosted-setup.json").exists()
    assert bundle.read_bytes() == b"corrupt"


def test_expired_bootstrap_deadline_refuses_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparselab.runtime_env_subprocess import run_bounded

    monkeypatch.setattr(bootstrap_worker, "run_bounded", run_bounded, raising=False)
    monkeypatch.setattr(bootstrap_worker, "_deadline", 0.0)

    with pytest.raises(TimeoutError):
        bootstrap_worker.run(["unreachable"])
