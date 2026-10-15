"""Read-only runner inspection using retained receipts and tiny child fixtures."""

import fcntl
import json
import os
import subprocess
import sys

import pytest
from test_runner_control import run, setup

from sparselab import runner_control as runner

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux flock")


def status(options):
    return runner.runner_status(
        **{key: options[key] for key in ("lock_path", "state_dir", "blocked_path")}
    )


def retained(root):
    result = {}
    for path in root.rglob("*"):
        info = path.lstat()
        result[str(path.relative_to(root))] = (
            info.st_mode,
            info.st_ino,
            info.st_mtime_ns,
            os.readlink(path)
            if path.is_symlink()
            else path.read_bytes()
            if path.is_file()
            else None,
        )
    return result


def forbid_writes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("read-only status reached initialization or launch")

    monkeypatch.setattr(runner, "_write", forbidden)
    monkeypatch.setattr(runner, "_prior_runs", forbidden)
    monkeypatch.setattr(runner, "monitor_command", forbidden)


def test_absent_state_and_lock_are_not_initialized(tmp_path, monkeypatch):
    forbid_writes(monkeypatch)
    options = {
        "lock_path": tmp_path / "lock",
        "state_dir": tmp_path / "state",
        "blocked_path": tmp_path / "blocked",
    }
    before = retained(tmp_path)
    result = status(options)
    assert result["status"] == "UNRESOLVED"
    assert result["launch_authorized"] is False
    assert result["history_verified"] is False
    assert result["problem"]["type"] == "FileNotFoundError"
    assert retained(tmp_path) == before


def test_empty_state_is_read_only_and_does_not_create_contract(tmp_path, monkeypatch):
    options = setup(tmp_path, max_wall_seconds=3)
    options["lock_path"].write_text("retained lock metadata")
    before = retained(tmp_path)
    forbid_writes(monkeypatch)
    result = status(options)
    assert result["status"] == "UNINITIALIZED"
    assert result["launch_authorized"] is False
    assert retained(tmp_path) == before


def test_verified_history_does_not_write_or_truncate(tmp_path, monkeypatch):
    options = setup(tmp_path, max_wall_seconds=3)
    run(options)
    options["lock_path"].write_text("retained lock metadata")
    before = retained(tmp_path)
    opened = []
    original_open = os.open

    def read_only_open(path, flags, *args, **kwargs):
        opened.append(flags)
        assert not flags & (os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_RDWR)
        return original_open(path, flags, *args, **kwargs)

    forbid_writes(monkeypatch)
    monkeypatch.setattr(os, "open", read_only_open)
    result = status(options)
    assert result["status"] == "VERIFIED_HISTORY"
    assert result["verified_runs"] == 1
    assert result["history_verified"] is True
    assert result["launch_authorized"] is False
    assert opened
    assert retained(tmp_path) == before
    # The read's shared probe must be released before returning.
    with options["lock_path"].open("r+b") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_busy_status_and_launch_refusal_do_not_infer_or_reclaim_owner(tmp_path):
    options = setup(tmp_path, max_wall_seconds=3)
    options["lock_path"].write_text("retained advisory metadata")
    before = retained(tmp_path)
    with options["lock_path"].open("r+b") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = status(options)
        assert result["status"] == "BUSY"
        assert result["launch_authorized"] is False
        assert "owner identity is unknown" in result["problem"]["message"]
        with pytest.raises(
            runner.RunnerPreflightRefused, match="runner lock busy"
        ) as caught:
            run(options)
        assert caught.value.error_type == "BlockingIOError"
        assert caught.value.__cause__.filename == str(options["lock_path"])
    assert retained(tmp_path) == before


@pytest.mark.parametrize("kind", ["regular", "dangling", "directory"])
def test_blocked_markers_are_retained_without_interpretation(
    tmp_path, monkeypatch, kind
):
    options = setup(tmp_path, max_wall_seconds=3)
    blocked = options["blocked_path"]
    if kind == "regular":
        blocked.write_text("not necessarily JSON")
    elif kind == "dangling":
        blocked.symlink_to(tmp_path / "missing")
    else:
        blocked.mkdir()
    before = retained(tmp_path)
    forbid_writes(monkeypatch)
    assert status(options)["status"] == "BLOCKED"
    assert retained(tmp_path) == before


@pytest.mark.parametrize(
    "damage", ["contract", "identity", "pending", "completion", "monitor", "owned"]
)
def test_history_errors_are_typed_actionable_and_preserve_bytes(
    tmp_path, monkeypatch, damage
):
    options = setup(tmp_path, max_wall_seconds=3)
    run(options)
    run_dir = next(path for path in options["state_dir"].iterdir() if path.is_dir())
    if damage == "contract":
        path = options["state_dir"] / "contract.json"
        path.write_text("{")
    elif damage == "identity":
        path = options["state_dir"] / "contract.json"
        contract = json.loads(path.read_text())
        contract["lock_inode"] += 1
        path.write_text(json.dumps(contract))
    elif damage in {"pending", "completion"}:
        path = run_dir / f"{damage}.json"
        path.unlink()
    elif damage == "monitor":
        path = options["log_dir"] / "completion.json"
        path.write_text("{")
    else:
        path = options["log_dir"] / "owned-completion.json"
        path.unlink()
    before = retained(tmp_path)
    with monkeypatch.context() as patch:
        forbid_writes(patch)
        result = status(options)
    assert result["status"] == "UNRESOLVED"
    assert result["problem"]["type"] == "RunnerHistoryError"
    assert result["problem"]["path"] == str(path)
    assert "inspect retained evidence" in result["problem"]["message"]
    assert result["launch_authorized"] is False
    assert not result["history_verified"]
    options["log_dir"] = tmp_path / "never-launched"
    with pytest.raises(runner.RunnerPreflightRefused) as caught:
        run(options)
    assert isinstance(caught.value.__cause__, runner.RunnerHistoryError)
    assert caught.value.__cause__.path == path
    assert retained(tmp_path) == before


def test_status_cli_does_not_create_global_work_root(tmp_path):
    options = setup(tmp_path, max_wall_seconds=3)
    options["lock_path"].touch(mode=0o600)
    workspace = tmp_path / "must-not-create"
    before = retained(tmp_path)
    command = [
        sys.executable,
        "-B",
        "-m",
        "sparselab",
        "--work-dir",
        str(workspace),
        "runner-status",
        "--json",
    ]
    for name in ("lock", "state", "blocked"):
        key = "state_dir" if name == "state" else f"{name}_path"
        command.extend([f"--runner-{name}", str(options[key])])
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "UNINITIALIZED"
    assert not workspace.exists()
    assert retained(tmp_path) == before


@pytest.mark.parametrize("field", ["lock_path", "state_dir"])
def test_unsafe_runner_path_is_not_followed_or_repaired(tmp_path, monkeypatch, field):
    options = setup(tmp_path, max_wall_seconds=3)
    options["lock_path"].touch(mode=0o600)
    path = options[field]
    original = path.with_name(path.name + "-retained")
    path.rename(original)
    path.symlink_to(original, target_is_directory=field == "state_dir")
    before = retained(tmp_path)
    forbid_writes(monkeypatch)
    result = status(options)
    assert result["status"] == "UNRESOLVED"
    assert "unsafe runner state" in result["problem"]["message"]
    assert result["launch_authorized"] is False
    assert retained(tmp_path) == before
