"""LAB64: bounded, real-helper startup cancellation; no model or host work."""

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
from sparselab import runner_control as runner

SITES = (
    "before_ready",
    "spawn_return",
    "identity",
    "ready_wait",
    "ready_read",
    "launch_before",
    "launch_after",
    "receipt_unavailable",
)


def _living(identity):
    try:
        process = psutil.Process(identity.pid)
        return (
            monitor._identity(process) == identity
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def _exercise(tmp: Path, site: str) -> None:
    os.umask(0o002)  # Isolated test owner; helper interruption inherits this mask.
    policy = monitor.MonitorPolicy(
        monitor_policy_version=1,
        interval_seconds=0.02,
        termination_grace_seconds=0.05,
        max_wall_seconds=3,
    )
    state = tmp / "state"
    state.mkdir(mode=0o700)
    policy_path = tmp / "policy.json"
    policy_path.write_text(policy.model_dump_json())
    logs = tmp / "logs"
    blocked = tmp / "blocked"
    sentinel = tmp / "child.json"
    release = tmp / "release-helper"
    options = {
        "policy": policy,
        "policy_path": policy_path,
        "workspace": tmp,
        "log_dir": logs,
        "lock_path": tmp / "lock",
        "state_dir": state,
        "blocked_path": blocked,
    }
    code = (
        "import json,os,time,psutil; from pathlib import Path; "
        f"p=Path({str(sentinel)!r}); q=p.with_suffix('.pending'); "
        "q.write_text(json.dumps(dict(pid=os.getpid(), "
        "create_time=psutil.Process().create_time()))); q.replace(p); time.sleep(2)"
    )
    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    original_popen = subprocess.Popen
    original_identity = monitor._identity
    helper = None
    identities = {}
    delivered = []
    patch = pytest.MonkeyPatch()

    def interrupt():
        # Restore injection hooks before delivery or any cleanup reads.
        patch.undo()
        if site != "before_ready":
            deadline = time.monotonic() + 2
            while not sentinel.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            assert sentinel.exists(), "test child never became ready"
            worker = monitor.ProcessIdentity.model_validate_json(sentinel.read_text())
            identities[worker.pid] = worker
        assert all(_living(identity) for identity in identities.values())
        delivered.append(site)
        if site == "receipt_unavailable":
            original_read = Path.read_text

            def fail_receipt(path, *args, **kwargs):
                if path == logs / "owned-completion.json":
                    patch.undo()
                    raise OSError("test receipt unavailable")
                return original_read(path, *args, **kwargs)

            patch.setattr(Path, "read_text", fail_receipt)
        os.kill(os.getpid(), signal.SIGTERM)
        if site == "before_ready":
            # This is reached only when cancellation was deferred. The real helper
            # cannot install handlers or publish readiness until we open this gate.
            assert not (logs / "owned-ready.json").exists()
            release.touch()

    def spawn(*args, **kwargs):
        nonlocal helper
        if site == "before_ready":
            argv = args[0]
            assert argv[1:3] == ["-m", "sparselab.owned_process"]
            gate = (
                "import runpy,sys,time; from pathlib import Path; "
                "end=time.monotonic()+2\n"
                f"while not Path({str(release)!r}).exists() and time.monotonic()<end: time.sleep(0.005)\n"
                "sys.argv=['sparselab.owned_process',*sys.argv[1:]]; "
                "runpy.run_module('sparselab.owned_process',run_name='__main__')"
            )
            args = ([argv[0], "-c", gate, *argv[3:]], *args[1:])
        helper = original_popen(*args, **kwargs)
        root = original_identity(psutil.Process(helper.pid))
        identities[root.pid] = root
        if site in ("spawn_return", "before_ready"):
            interrupt()
        return helper

    patch.setattr(subprocess, "Popen", spawn)
    if site == "identity":

        def identify(process):
            if helper is not None and process.pid == helper.pid:
                interrupt()
            return original_identity(process)

        patch.setattr(monitor, "_identity", identify)
    elif site in ("ready_wait", "ready_read"):
        method = "exists" if site == "ready_wait" else "read_text"
        original = getattr(Path, method)

        def readiness(path, *args, **kwargs):
            if path == logs / "owned-ready.json":
                interrupt()
            return original(path, *args, **kwargs)

        patch.setattr(Path, method, readiness)
    elif site.startswith("launch_") or site == "receipt_unavailable":
        original_record = monitor._record

        def record(stream, value):
            if isinstance(value, monitor.MonitorLaunch):
                if site == "launch_after":
                    original_record(stream, value)
                interrupt()
            return original_record(stream, value)

        patch.setattr(monitor, "_record", record)
    try:
        error_type = (
            RuntimeError if site == "receipt_unavailable" else runner.RunnerInterrupted
        )
        error_match = (
            "shutdown could not be verified"
            if site == "receipt_unavailable"
            else "runner received signal"
        )
        with pytest.raises(error_type, match=error_match):
            runner.run_controlled([sys.executable, "-c", code], **options)
        # Inspect immediately, before the test's unconditional fallback cleanup.
        live = [identity.pid for identity in identities.values() if _living(identity)]
        assert not live, (
            f"cancellation returned with live attributable processes: {live}"
        )
        assert delivered == [site]
        owned = json.loads((logs / "owned-completion.json").read_text())
        assert owned["root"] == identities[helper.pid].model_dump()
        assert owned["living_descendants"] == 0
        assert owned["reason"] in (
            f"supervisor received signal {signal.SIGTERM}",
            *(["stopped before command launch"] if site == "before_ready" else []),
        )
        assert owned["returncode"] == helper.returncode
        if site != "before_ready":
            assert len(owned["observed_descendants"]) >= 1
        assert {sig: signal.getsignal(sig) for sig in previous} == previous
        assert blocked.exists()
        assert error_match in json.loads(blocked.read_text())["reason"]
        pending = list(state.glob("*/pending.json"))
        assert len(pending) == 1
        assert not list(state.glob("*/completion.json"))
        assert not (logs / "completion.json").exists()
        assert logs.stat().st_mode & 0o777 == 0o700
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in logs.iterdir())
        retained = {path: path.read_bytes() for path in state.rglob("*.json")}
        with pytest.raises(ValueError, match="runner blocked"):
            runner.run_controlled([sys.executable, "-c", "pass"], **options)
        # Test-local marker removal proves historical pending alone refuses launch.
        blocked.unlink()
        options["log_dir"] = tmp / "refused-logs"
        with pytest.raises(runner.RunnerPreflightRefused) as refusal:
            runner.run_controlled([sys.executable, "-c", "pass"], **options)
        assert refusal.value.error_type == "RunnerHistoryError"
        assert refusal.value.__cause__.__cause__.filename == str(
            pending[0].with_name("completion.json")
        )
        assert not options["log_dir"].exists()
        assert {path: path.read_bytes() for path in state.rglob("*.json")} == retained
    finally:
        patch.undo()
        # Applies even when the original implementation leaks at the assertion.
        if helper is not None:
            release.touch()
            monitor._signal_owned({helper.pid: identities[helper.pid]}, signal.SIGTERM)
            try:
                helper.wait(timeout=5)
            finally:
                monitor._signal_owned(identities, signal.SIGKILL)
                helper.wait(timeout=2)
        assert not any(_living(identity) for identity in identities.values())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux owned helper/subreaper")
@pytest.mark.parametrize("site", SITES)
def test_startup_cancellation_cleans_before_propagation(tmp_path, site):
    # An isolated owner gives each injection a strict outer timeout. The helper's
    # three-second deadline and child's two-second lifetime are independent bounds.
    owner = subprocess.Popen(
        [sys.executable, __file__, str(tmp_path), site],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    root = monitor._identity(psutil.Process(owner.pid))
    known = {root.pid: root}
    try:
        stdout, stderr = owner.communicate(timeout=12)
        assert owner.returncode == 0, (stdout + stderr).decode()
    finally:
        if owner.poll() is None:
            # Discover only descendants of this still-owned test process.
            monitor._owned(root, known)
            monitor._signal_owned(known, signal.SIGTERM)
            try:
                owner.communicate(timeout=5)
            finally:
                monitor._signal_owned(known, signal.SIGKILL)
                owner.communicate(timeout=2)


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]), sys.argv[2])
