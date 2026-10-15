"""Real lock contention, bounded tiny children, no runtime provisioning."""

import fcntl
import subprocess
import sys

import pytest


@pytest.mark.parametrize("kind", ["campaign", "registry"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_contended_lock_reports_wait_without_reclaiming(tmp_path, kind, interrupt):
    lock = tmp_path / (".cleanup.lock" if kind == "campaign" else "runtimes.lock")
    lock.write_text("retained lock bytes")
    lock.chmod(0o600)
    before = (lock.stat().st_ino, lock.read_bytes())
    entered = tmp_path / "entered"
    program = """
import signal, sys
from pathlib import Path
signal.alarm(8)
kind, root, entered = sys.argv[1:]
if kind == 'campaign':
    from sparselab.workspace_cleanup import campaign_lock
    context = campaign_lock(Path(root))
else:
    from sparselab.runtime_environments import _locked
    context = _locked(Path(root) / 'runtimes.yaml')
with context:
    Path(entered).write_text('entered')
"""
    with lock.open("r+b") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", program, kind, str(tmp_path), str(entered)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # The child's alarm bounds this read even if the diagnostic regresses.
            line = child.stderr.readline()
            assert "Waiting for" in line and str(lock) in line
            assert "Owner identity is unknown" in line
            assert "normal blocking wait continues" in line
            assert not entered.exists()
            assert child.poll() is None
            assert (lock.stat().st_ino, lock.read_bytes()) == before
            if interrupt:
                child.terminate()
                child.communicate(timeout=10)
                assert child.returncode != 0
                assert not entered.exists()
                with lock.open("r+b") as contender, pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(holder, fcntl.LOCK_UN)
            if not interrupt:
                stdout, stderr = child.communicate(timeout=10)
                assert child.returncode == 0, stderr
                assert stdout == stderr == ""
                assert entered.read_text() == "entered"
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
    assert (lock.stat().st_ino, lock.read_bytes()) == before


@pytest.mark.parametrize("kind", ["campaign", "registry"])
def test_uncontended_lock_is_quiet(tmp_path, caplog, kind):
    from sparselab.runtime_environments import _locked
    from sparselab.workspace_cleanup import campaign_lock

    context = (
        campaign_lock(tmp_path)
        if kind == "campaign"
        else _locked(tmp_path / "runtimes.yaml")
    )
    with context:
        pass
    assert not caplog.records
