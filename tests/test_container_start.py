"""Exercise the real shell entrypoint with bounded fake executables, no container."""

import json
import os
import selectors
import signal
import subprocess
import sys
from pathlib import Path

import pytest

START = Path(__file__).resolve().parents[1] / "container/start.sh"
pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux container entrypoint")


@pytest.fixture
def environment(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Directory provisioning is separately exercised by the image smoke. Avoid
    # writing the image-only /state path in these host dispatch tests.
    mkdir = bin_dir / "mkdir"
    mkdir.write_text("#!/bin/sh\nexit 0\n")
    mkdir.chmod(0o700)
    for name in ("sparselab", "python", "python3", "streamlit", "compare"):
        executable = bin_dir / name
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json,os,signal,sys\n"
            "print(json.dumps({'argv':sys.argv,'pid':os.getpid()}),flush=True)\n"
            "if os.environ.get('WAIT'): signal.pause()\n"
            "sys.exit(int(os.environ.get('EXIT','0')))\n"
        )
        executable.chmod(0o700)
    env = {**os.environ, "PATH": str(bin_dir) + ":/usr/bin:/bin"}
    for name in (
        "HOME",
        "TMPDIR",
        "SPARSELAB_WORK_DIR",
        "SPARSELAB_WORKER_LEASE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
    ):
        env[name] = str(tmp_path / name)
    env.pop("WAIT", None)
    env.pop("EXIT", None)
    return env


@pytest.mark.parametrize(
    "argv,executable,tail",
    [
        ([], "sparselab", ["--help"]),
        (["--help"], "sparselab", ["--help"]),
        (
            ["--work-dir=/state/a b", "report", "--help"],
            "sparselab",
            ["--work-dir=/state/a b", "report", "--help"],
        ),
        (
            ["--work-dir", "/state/a b", "report", "--help"],
            "sparselab",
            ["--work-dir", "/state/a b", "report", "--help"],
        ),
        (
            ["compare", "a b;$HOME$(touch nope)"],
            "sparselab",
            ["compare", "a b;$HOME$(touch nope)"],
        ),
        (
            ["eval", "a b;$HOME$(touch nope)"],
            "sparselab",
            ["eval", "a b;$HOME$(touch nope)"],
        ),
        (
            ["report", "a b;$HOME$(touch nope)"],
            "sparselab",
            ["report", "a b;$HOME$(touch nope)"],
        ),
        (["sparselab", "report", "--help"], "sparselab", ["report", "--help"]),
        (["python", "-c", "literal text"], "python", ["-c", "literal text"]),
        (["python3", "-c", "literal text"], "python3", ["-c", "literal text"]),
        (["streamlit", "run", "a b.py"], "streamlit", ["run", "a b.py"]),
        (["not-a-command"], "sparselab", ["not-a-command"]),
    ],
)
def test_dispatch_preserves_argv_and_exit(environment, argv, executable, tail):
    result = subprocess.run(
        ["/bin/sh", str(START), *argv],
        env={**environment, "EXIT": "23"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 23, result.stderr
    record = json.loads(result.stdout)
    assert Path(record["argv"][0]).name == executable
    assert record["argv"][1:] == tail


@pytest.mark.parametrize("shell", ["/bin/sh", "sh", "bash"])
def test_explicit_shell(environment, shell):
    result = subprocess.run(
        ["/bin/sh", str(START), shell, "-c", "exit 19"],
        env=environment,
        timeout=5,
        check=False,
    )
    assert result.returncode == 19


def test_exec_preserves_pid_and_signal(environment):
    with subprocess.Popen(
        ["/bin/sh", str(START), "report", "fixture"],
        env={**environment, "WAIT": "1"},
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(process.stdout, selectors.EVENT_READ)
                assert ready.select(timeout=5), "fake child did not start"
            assert json.loads(process.stdout.readline())["pid"] == process.pid
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=5) == -signal.SIGTERM
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize("relative", [False, True])
def test_other_executable_requires_explicit_path(environment, relative):
    bin_dir = Path(environment["PATH"].split(":")[0])
    executable = "./compare" if relative else str(bin_dir / "compare")
    tail = ["literal space;$HOME$(touch nope)"]
    result = subprocess.run(
        ["/bin/sh", str(START), executable, *tail],
        cwd=bin_dir,
        env={**environment, "EXIT": "17"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 17, result.stderr
    record = json.loads(result.stdout)
    assert Path(record["argv"][0]).name == "compare"
    assert record["argv"][1:] == tail
