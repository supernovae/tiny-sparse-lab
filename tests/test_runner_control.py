"""Tiny CPU processes only; subprocess waits have strict deadlines."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from sparselab import operational_monitor as monitor
from sparselab import owned_process
from sparselab import runner_control as runner

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux flock/subreaper")


def setup(tmp: Path, **fields: object) -> dict:
    policy = monitor.MonitorPolicy(
        monitor_policy_version=1,
        interval_seconds=0.02,
        termination_grace_seconds=0.05,
        **fields,
    )
    (tmp / "state").mkdir(mode=0o700)
    path = tmp / "policy.json"
    path.write_text(policy.model_dump_json())
    return {
        "policy": policy,
        "policy_path": path,
        "workspace": tmp,
        "log_dir": tmp / "logs",
        "lock_path": tmp / "lock",
        "state_dir": tmp / "state",
        "blocked_path": tmp / "blocked",
    }


def run(options: dict, code: str = "pass") -> monitor.MonitorCompletion:
    return runner.run_controlled([sys.executable, "-c", code], **options)


def child(options: dict, code: str, *, hook: str = "") -> subprocess.Popen:
    args = {key: str(value) for key, value in options.items() if key != "policy"}
    script = (
        "from pathlib import Path; "
        "from sparselab.runner_control import run_controlled; "
        "from sparselab.operational_monitor import load_monitor_policy; "
        "import sys; "
        f"exec({hook!r}); "
        f"opts={{k:Path(v) for k,v in {args!r}.items()}}; "
        "opts['policy']=load_monitor_policy(opts['policy_path']); "
        f"run_controlled([sys.executable,'-c',{code!r}], **opts)"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def await_file(path: Path, process: subprocess.Popen) -> None:
    end = time.monotonic() + 4
    while not path.exists() and time.monotonic() < end and process.poll() is None:
        time.sleep(0.01)
    assert path.exists(), process.communicate(timeout=2)


@pytest.mark.parametrize("rc", [0, 7])
def test_success_failure_and_stable_identity(tmp_path: Path, rc: int) -> None:
    options = setup(tmp_path)
    first = run(options, f"raise SystemExit({rc})")
    assert first.returncode == rc
    assert first.status == ("FAILED" if rc else "COMPLETE")
    inode = options["lock_path"].stat().st_ino
    owned = json.loads((options["log_dir"] / "owned-completion.json").read_text())
    assert owned["root"] == first.launch.root.model_dump()
    options["log_dir"] = tmp_path / "logs2"
    assert run(options).status == "COMPLETE"
    assert options["lock_path"].stat().st_ino == inode
    assert not options["blocked_path"].exists()


def test_independent_contender_cannot_launch(tmp_path: Path) -> None:
    options = setup(tmp_path, max_wall_seconds=3)
    ready = tmp_path / "work-ready"
    release = tmp_path / "release"
    code = (
        f"from pathlib import Path; Path({str(ready)!r}).touch(); import time\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(0.01)"
    )
    first = child(options, code)
    try:
        await_file(ready, first)
        second_options = {**options, "log_dir": tmp_path / "second"}
        second = child(second_options, "raise AssertionError('must not launch')")
        _, err = second.communicate(timeout=4)
        assert second.returncode != 0
        assert b"BlockingIOError" in err
        assert not second_options["log_dir"].exists()
        release.touch()
        assert first.wait(timeout=5) == 0
    finally:
        if first.poll() is None:
            first.terminate()
        first.communicate(timeout=5)


@pytest.mark.parametrize("post", [False, True])
def test_blocked_before_and_after_lock(tmp_path: Path, monkeypatch, post: bool) -> None:
    options = setup(tmp_path)
    marker = options["blocked_path"]
    if post:
        flock = runner.fcntl.flock

        def block_after(fd, flags):
            flock(fd, flags)
            marker.touch()

        monkeypatch.setattr(runner.fcntl, "flock", block_after)
    else:
        marker.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="runner blocked"):
        run(options)
    assert not options["log_dir"].exists()
    assert os.path.lexists(marker)


def test_timeout_cleans_detached_owned_descendant(tmp_path: Path) -> None:
    options = setup(tmp_path, max_wall_seconds=2)
    pidfile = tmp_path / "worker-pid"
    worker = f"import os,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(10)"
    code = f"import sys,subprocess,time; subprocess.Popen([sys.executable,'-c',{worker!r}],start_new_session=True); time.sleep(10)"
    process = child(options, code)
    try:
        assert process.wait(timeout=6) == 0  # API returns the VIOLATED receipt.
        receipt = monitor.MonitorCompletion.model_validate_json(
            (options["log_dir"] / "completion.json").read_bytes()
        )
        assert receipt.status == "VIOLATED"
        assert "max_wall_seconds" in receipt.violations
        pid = int(pidfile.read_text())
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )
        assert not options["blocked_path"].exists()
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=5)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
def test_interrupted_owner_never_automatically_recovers(
    tmp_path: Path, signum: int
) -> None:
    options = setup(tmp_path, max_wall_seconds=3)
    ready = tmp_path / "ready"
    process = child(
        options,
        f"from pathlib import Path; Path({str(ready)!r}).touch(); import time; time.sleep(10)",
    )
    try:
        await_file(ready, process)
        assert process.poll() is None, "owner exited before the interruption"
        process.send_signal(signum)
        _, stderr = process.communicate(timeout=5)
        if signum == signal.SIGTERM:
            assert options["blocked_path"].exists(), (process.returncode, stderr)
        # SIGKILL cannot write a marker: retained pending remains authoritative.
        options["blocked_path"].unlink(missing_ok=True)
        options["log_dir"] = tmp_path / "new-logs"
        with pytest.raises((ValueError, FileNotFoundError)):
            run(options)
        assert not options["log_dir"].exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize("damage", ["missing", "foreign", "monitor", "owned", "lock"])
def test_stale_or_incomplete_receipts_refuse_takeover(
    tmp_path: Path, damage: str
) -> None:
    options = setup(tmp_path)
    run(options)
    receipt = next(options["state_dir"].glob("*/completion.json"))
    if damage == "missing":
        receipt.unlink()
    elif damage == "foreign":
        wire = json.loads(receipt.read_text())
        wire["root"]["create_time"] += 1
        receipt.write_text(json.dumps(wire))
    elif damage == "monitor":
        (options["log_dir"] / "completion.json").write_text("{}")
    elif damage == "owned":
        (options["log_dir"] / "owned-completion.json").unlink()
    else:
        options["lock_path"].rename(tmp_path / "old-lock")
        options["lock_path"].touch()
    options["log_dir"] = tmp_path / "new-logs"
    with pytest.raises((ValueError, FileNotFoundError)):
        run(options)
    assert not options["log_dir"].exists()
    assert not options["blocked_path"].exists()


@pytest.mark.parametrize("stage", ["pending", "completion", "blocked"])
def test_durable_state_error_never_certifies_completion(
    tmp_path: Path, monkeypatch, stage: str
) -> None:
    options = setup(tmp_path)
    write = runner._write

    def fail(path, record):
        if path.name == f"{stage}.json" or (
            stage == "blocked" and path == options["blocked_path"]
        ):
            raise OSError("injected durable storage failure")
        write(path, record)

    monkeypatch.setattr(runner, "_write", fail)
    if stage == "blocked":
        monkeypatch.setattr(
            runner,
            "monitor_command",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cleanup uncertain")),
        )
    with pytest.raises(OSError, match="durable storage"):
        run(options)
    assert not list(options["state_dir"].glob("*/completion.json"))
    if stage == "pending":
        assert not options["log_dir"].exists()
    monkeypatch.setattr(runner, "_write", write)
    options["blocked_path"].unlink(missing_ok=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        run({**options, "log_dir": tmp_path / "next"})


def test_cleanup_failure_blocks_before_lock_release(
    tmp_path: Path, monkeypatch
) -> None:
    options = setup(tmp_path)

    def uncertain(*args, **kwargs):
        assert list(options["state_dir"].glob("*/pending.json"))
        raise RuntimeError("owned cleanup deadline exhausted")

    monkeypatch.setattr(runner, "monitor_command", uncertain)
    close = runner.os.close
    seen = []

    def check_close(fd):
        # os.close also closes directory sync fds; identify the lock inode.
        if os.fstat(fd).st_ino == options["lock_path"].stat().st_ino:
            seen.append(options["blocked_path"].exists())
        close(fd)

    monkeypatch.setattr(runner.os, "close", check_close)
    with pytest.raises(RuntimeError, match="deadline"):
        run(options)
    assert seen == [True]
    assert not list(options["state_dir"].glob("*/completion.json"))


def test_helper_post_kill_loop_is_bounded(tmp_path: Path, monkeypatch) -> None:
    # A harmless exited direct child plus a simulated unsignalable owned process.
    monkeypatch.setattr(
        owned_process.ctypes,
        "CDLL",
        lambda *a, **k: type("Libc", (), {"prctl": lambda *a: 0})(),
    )
    root = monitor._identity(psutil.Process())
    fake = type("Owned", (), {"pid": root.pid + 1000000})()
    clock = iter(range(100))
    monkeypatch.setattr(owned_process.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(owned_process.time, "sleep", lambda _: None)

    def owned(_root, known):
        known[fake.pid] = monitor.ProcessIdentity(pid=fake.pid, create_time=1)
        return [fake]

    signals = []
    monkeypatch.setattr(owned_process, "_owned", owned)
    monkeypatch.setattr(
        owned_process, "_signal_owned", lambda _targets, sig: signals.append(sig)
    )
    path = tmp_path / "owned.json"
    assert (
        owned_process.supervise(
            [sys.executable, "-c", "pass"], completion=path, deadline=2, grace=0
        )
        == 1
    )
    receipt = json.loads(path.read_text())
    assert receipt["living_descendants"] == 1
    assert receipt["reason"] == "owned cleanup deadline exhausted"
    assert signal.SIGKILL in signals


def test_foreign_owned_completion_is_rejected(tmp_path: Path) -> None:
    root = monitor._identity(psutil.Process())
    receipt = tmp_path / "owned.json"
    receipt.write_text(
        json.dumps(
            {
                "format": "sparselab-owned-completion-v1",
                "root": {"pid": root.pid, "create_time": root.create_time + 1},
                "living_descendants": 0,
                "returncode": 0,
            }
        )
    )
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=2)
    with pytest.raises(RuntimeError, match="could not be verified"):
        monitor._stop_enhanced_supervisor(process, root, receipt, 0)


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "writable", "state"])
def test_unsafe_paths_refuse_launch(tmp_path: Path, unsafe: str) -> None:
    options = setup(tmp_path)
    lock = options["lock_path"]
    if unsafe == "symlink":
        lock.symlink_to(tmp_path / "missing")
    elif unsafe == "hardlink":
        lock.touch()
        os.link(lock, tmp_path / "alias")
    elif unsafe == "writable":
        lock.touch(mode=0o666)
        lock.chmod(0o666)
    else:
        options["state_dir"].rmdir()
        options["state_dir"].symlink_to(tmp_path)
    with pytest.raises((ValueError, OSError)):
        run(options)
    assert not options["log_dir"].exists()


@pytest.mark.parametrize("number", [float("inf"), float("nan")])
def test_cleanup_grace_must_be_finite(number):
    with pytest.raises(ValueError):
        monitor.MonitorPolicy(
            monitor_policy_version=1, termination_grace_seconds=number
        )


def test_blocked_marker_stat_error_is_not_absence(tmp_path, monkeypatch):
    options = setup(tmp_path)
    original = Path.lstat

    def unreadable(path, *args, **kwargs):
        if path == options["blocked_path"]:
            raise PermissionError("marker stat unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", unreadable)
    with pytest.raises(PermissionError, match="marker stat unavailable"):
        run(options)
    assert not options["log_dir"].exists()


@pytest.mark.parametrize("damage", ["missing", "foreign"])
def test_unverified_owned_completion_blocks_current_attempt(
    tmp_path, monkeypatch, damage
):
    options = setup(tmp_path, max_wall_seconds=2)
    loads = monitor.json.loads

    def damaged_receipt(data, *args, **kwargs):
        record = loads(data, *args, **kwargs)
        if (
            isinstance(record, dict)
            and record.get("format") == "sparselab-owned-completion-v1"
        ):
            if damage == "missing":
                raise FileNotFoundError("owned completion unavailable")
            record["root"]["create_time"] += 1
        return record

    monkeypatch.setattr(monitor.json, "loads", damaged_receipt)
    with pytest.raises(RuntimeError, match="unverified|could not be verified"):
        run(options)
    assert options["blocked_path"].exists()
    assert not list(options["state_dir"].glob("*/completion.json"))


def test_missing_state_directory_is_not_automatically_recreated(tmp_path):
    options = setup(tmp_path)
    run(options)
    # Preserve the evidence elsewhere while simulating loss of the agreed path.
    options["state_dir"].rename(tmp_path / "retained-state")
    options["log_dir"] = tmp_path / "new-logs"
    with pytest.raises(
        FileNotFoundError, match=r"runner state directory missing:.*docs/workspaces.md"
    ):
        run(options)
    assert not options["state_dir"].exists()
    assert not options["log_dir"].exists()
    assert not options["blocked_path"].exists()


@pytest.mark.parametrize("alias", ["symlink", "dotdot"])
def test_two_clean_launches_with_noncanonical_log_parent(tmp_path, alias):
    options = setup(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    if alias == "symlink":
        parent = tmp_path / "link"
        parent.symlink_to(real, target_is_directory=True)
    else:
        parent = real / ".." / "real"
    options["log_dir"] = parent / "logs1"
    assert run(options).status == "COMPLETE"
    options["log_dir"] = parent / "logs2"
    assert run(options).status == "COMPLETE"
    for pending_path in options["state_dir"].glob("*/pending.json"):
        pending = json.loads(pending_path.read_text())
        assert Path(pending["log_dir"]) == Path(pending["log_dir"]).resolve()
    assert not options["blocked_path"].exists()


def test_signal_before_pending_records_blocker_and_restores_handlers(
    tmp_path, monkeypatch
):
    options = setup(tmp_path)
    write = runner._write
    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }

    def interrupt_pending(path, record):
        if path.name == "pending.json":
            os.kill(os.getpid(), signal.SIGTERM)
        write(path, record)

    monkeypatch.setattr(runner, "_write", interrupt_pending)
    with pytest.raises(InterruptedError, match="runner received signal"):
        run(options)
    assert options["blocked_path"].exists()
    assert not options["log_dir"].exists()
    assert not list(options["state_dir"].glob("*/pending.json"))
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_second_signal_during_teardown_does_not_abort_owned_cleanup(tmp_path):
    options = setup(tmp_path, max_wall_seconds=3)
    ready = tmp_path / "ready"
    hook = """import os, signal
from sparselab import operational_monitor as monitor
original_stop = monitor._stop_enhanced_supervisor
def second_signal(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGTERM)
    return original_stop(*args, **kwargs)
monitor._stop_enhanced_supervisor = second_signal
"""
    process = child(
        options,
        f"from pathlib import Path; Path({str(ready)!r}).touch(); import time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)",
        hook=hook,
    )
    try:
        await_file(ready, process)
        assert process.poll() is None
        process.terminate()
        _, stderr = process.communicate(timeout=6)
        assert process.returncode != 0, stderr
        owned = json.loads((options["log_dir"] / "owned-completion.json").read_text())
        assert owned["living_descendants"] == 0
        assert options["blocked_path"].exists()
        assert not list(options["state_dir"].glob("*/completion.json"))
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_mistyped_state_refuses_without_sticky_blocker(tmp_path):
    options = setup(tmp_path)
    provisioned = options["state_dir"]
    options["state_dir"] = tmp_path / "typo-state"
    with pytest.raises(FileNotFoundError, match="confirm no outstanding work"):
        run(options)
    assert not options["blocked_path"].exists()
    assert not options["state_dir"].exists()
    assert not options["log_dir"].exists()
    assert not list(provisioned.iterdir())


def test_preflight_storage_error_does_not_create_attempt_or_blocker(
    tmp_path, monkeypatch
):
    options = setup(tmp_path)

    def fail_sync(_path):
        raise OSError("preflight sync failed")

    monkeypatch.setattr(runner, "_sync_dir", fail_sync)
    with pytest.raises(OSError, match="preflight sync failed"):
        run(options)
    assert not options["blocked_path"].exists()
    assert not list(options["state_dir"].iterdir())
    assert not options["log_dir"].exists()


@pytest.mark.parametrize("exists", [False, True])
def test_final_log_symlink_is_refused_without_writing_target(tmp_path, exists):
    options = setup(tmp_path)
    target = tmp_path / "target"
    if exists:
        target.mkdir()
        (target / "sentinel").write_text("retain")
    options["log_dir"].symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        run(options)
    if exists:
        assert [path.name for path in target.iterdir()] == ["sentinel"]
        assert (target / "sentinel").read_text() == "retain"
    else:
        assert not target.exists()
    # A durable pending attempt exists; this is not a host preflight refusal.
    assert list(options["state_dir"].glob("*/pending.json"))
    assert options["blocked_path"].exists()


def test_signal_during_preflight_restores_handlers_without_new_blocker(
    tmp_path, monkeypatch
):
    options = setup(tmp_path)
    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }

    def interrupt(*args):
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(runner, "_prior_runs", interrupt)
    with pytest.raises(InterruptedError, match="runner received signal"):
        run(options)
    assert not options["blocked_path"].exists()
    assert not list(options["state_dir"].iterdir())
    assert not options["log_dir"].exists()
    assert {sig: signal.getsignal(sig) for sig in previous} == previous
