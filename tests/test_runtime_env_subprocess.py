import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

from sparselab.runtime_env_subprocess import run_bounded


def test_bounded_stdout_and_stderr_are_independent():
    result = run_bounded(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err',file=sys.stderr)",
        ],
        timeout=5,
    )
    assert result.returncode == 0
    assert result.stdout == b"out\n"
    assert result.stderr == b"err\n"


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_oversized_output_is_rejected(stream):
    with pytest.raises(ValueError, match="byte limit"):
        run_bounded(
            [sys.executable, "-c", f"import sys; sys.{stream}.write('x'*65536)"],
            timeout=5,
            output_limit=1024,
        )


def test_deadline_terminates_candidate():
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable, "-c", "import time; time.sleep(3)"], timeout=0.1)


class _UnreapedFixtureProcess(subprocess.Popen):
    """Observe exit status without releasing the owned leader's PID/PGID.

    Only teardown reaps these children. Keeping the zombie reserves the group
    through killpg even if production wait/poll observes a completed leader.
    This uses POSIX waitid/WNOWAIT, rather than a check-to-signal identity test.
    """

    _reaped = False
    wait_calls = 0

    def poll(self):
        if self.returncode is None:
            status = os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if status is not None:
                self.returncode = (
                    status.si_status
                    if status.si_code == os.CLD_EXITED
                    else -status.si_status
                )
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        deadline = time.monotonic() + timeout if timeout is not None else None
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.005)
        return self.returncode

    def terminate_and_reap(self):
        if self._reaped:
            return  # Never signal the numeric PGID after releasing our child.
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.returncode = None
        super().wait(timeout=2)
        self._reaped = True


@pytest.fixture
def owned_processes(monkeypatch):
    """Clean owned groups on failures, retaining their PID until the signal."""
    processes = []

    def capture(*args, **kwargs):
        process = _UnreapedFixtureProcess(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)
    try:
        yield processes
    finally:
        for process in processes:
            process.terminate_and_reap()


def test_fixture_reserves_leader_until_cleanup_and_never_signals_after_reaping(
    monkeypatch, owned_processes
):
    process = subprocess.Popen(
        [sys.executable, "-S", "-c", "pass"], start_new_session=True
    )
    assert process.wait(timeout=1) == 0
    # Exit-status observation has not consumed the kernel's child status.
    assert (
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT).si_pid == process.pid
    )
    signaled = []
    original = os.killpg

    def kill_reserved_group(pid, sig):
        assert not process._reaped
        assert os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT).si_pid == pid
        signaled.append(pid)
        original(pid, sig)

    monkeypatch.setattr(os, "killpg", kill_reserved_group)
    process.terminate_and_reap()
    process.terminate_and_reap()
    assert signaled == [process.pid]
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)


def test_production_cleanup_reaps_real_leader(monkeypatch):
    # The regression fixture deliberately defers reaping; check real wait here.
    processes = []
    original = subprocess.Popen

    def capture(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)
    try:
        with pytest.raises(ValueError, match="byte limit"):
            run_bounded(
                [sys.executable, "-S", "-c", "print('x' * 2048)"],
                output_limit=1024,
                timeout=2,
            )
        with pytest.raises(ChildProcessError):
            os.waitpid(processes[0].pid, os.WNOHANG)
    finally:
        for process in processes:
            process.wait(timeout=2)


def exited_parent_command(pid_path, stream, failure):
    # The descendant emits only after reparenting, proving the leader exited.
    # Both interpreters skip site startup; the descendant has a 3-second fuse.
    child = f"""
import os, time
from pathlib import Path
parent = os.getppid()
Path({str(pid_path)!r}).write_text(str(os.getpid()))
os.close({2 if stream == "stdout" else 1})
deadline = time.monotonic() + 3
while os.getppid() == parent and time.monotonic() < deadline:
    time.sleep(0.005)
os.write({1 if stream == "stdout" else 2}, b'ok')
{f"os.write({1 if stream == 'stdout' else 2}, b'x' * 2048)" if failure == "limit" else ""}
time.sleep(max(0, deadline - time.monotonic()))
"""
    return [
        sys.executable,
        "-S",
        "-c",
        f"""
import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, '-S', '-c', {child!r}])
deadline = time.monotonic() + 1
while not Path({str(pid_path)!r}).exists() and time.monotonic() < deadline:
    time.sleep(0.005)
""",
    ]


@pytest.mark.parametrize("stream", ["stdout", "stderr", "file"])
@pytest.mark.parametrize("failure", ["timeout", "limit"])
def test_exited_parent_descendant_is_terminated(
    tmp_path, owned_processes, stream, failure
):
    pid_path = tmp_path / "descendant.pid"
    stdout_path = tmp_path / "stdout" if stream == "file" else None
    command = exited_parent_command(
        pid_path, "stdout" if stream == "file" else stream, failure
    )
    error = subprocess.TimeoutExpired if failure == "timeout" else ValueError
    with pytest.raises(error) as caught:
        run_bounded(
            command,
            timeout=0.5 if failure == "timeout" else 2,
            output_limit=1024,
            stdout_path=stdout_path,
            stdout_limit=1024 if stdout_path else None,
        )
    if failure == "timeout":
        assert caught.value.cmd == command
        assert caught.value.timeout == 0.5
    else:
        assert str(caught.value) == "subprocess output exceeds byte limit"
    assert owned_processes[0].returncode == 0  # exited; harness defers reaping
    pid = int(pid_path.read_text())
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.005)
    else:
        pytest.fail(f"owned descendant {pid} still running")
    assert owned_processes[0].wait_calls == 1
    if stdout_path is not None:
        assert not stdout_path.exists()


def test_stdout_destination_collision_preserves_existing_file(
    tmp_path, owned_processes
):
    path = tmp_path / "stdout"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        run_bounded(
            [sys.executable, "-S", "-c", "import time; time.sleep(3)"],
            timeout=0.5,
            stdout_path=path,
            stdout_limit=1024,
        )
    assert path.read_bytes() == b"existing"
    assert owned_processes[0].returncode is not None


@pytest.mark.parametrize("exit_code", [0, 7])
def test_completed_file_output_and_exit_status_are_retained(
    tmp_path, monkeypatch, exit_code
):
    monkeypatch.setattr(
        os,
        "killpg",
        lambda *args: pytest.fail("completed execution must not kill its group"),
    )
    path = tmp_path / "stdout"
    result = run_bounded(
        [
            sys.executable,
            "-S",
            "-c",
            f"import sys; print('out'); print('err', file=sys.stderr); sys.exit({exit_code})",
        ],
        timeout=2,
        stdout_path=path,
        stdout_limit=1024,
    )
    assert result.returncode == exit_code
    assert result.stdout == b""
    assert result.stderr == b"err\n"
    assert path.read_bytes() == b"out\n"
