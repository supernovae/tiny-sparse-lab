import subprocess
import sys

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
        run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.1)
