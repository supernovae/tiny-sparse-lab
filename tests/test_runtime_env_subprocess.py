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


@pytest.fixture
def owned_processes(monkeypatch):
    """Always clean fixture groups, even when the production cleanup regresses."""
    processes = []
    original = subprocess.Popen

    def capture(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)
    try:
        yield processes
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
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
    assert owned_processes[0].returncode == 0  # leader reaped, already exited
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
