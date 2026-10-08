"""Bound one Linux operational measurement without inherited-pipe EOF waits."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from sparselab.operational_monitor import _identity, _owned, _signal_owned


def run(command: list[str], output: Path, *, seconds: float = 5) -> int:
    if sys.platform != "linux" or not 0 < seconds <= 5:
        raise ValueError("measurement requires Linux and a deadline in (0, 5] seconds")
    # This dedicated helper adopts orphaned grandchildren, including setsid
    # children whose leader exits before the first process-tree observation.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot establish measurement subreaper")
    root = _identity(psutil.Process())
    known = {root.pid: root}
    process = None

    def interrupted(signum, frame):
        raise RuntimeError(f"measurement interrupted by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        with output.open("wb") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream)
            deadline = time.monotonic() + seconds
            while True:
                # Reuse native identity-based ownership and PID-reuse protection.
                rc = process.poll()
                # Reap the leader BEFORE observing descendants: otherwise it can
                # exit between the snapshot and poll, hiding newly adopted orphans.
                owned = [p for p in _owned(root, known) if p.pid != root.pid]
                if time.monotonic() >= deadline:
                    raise TimeoutError("measurement deadline exceeded")
                if output.stat().st_size > 65536:
                    raise RuntimeError("measurement output exceeds 64 KiB")
                if rc is not None and not owned:
                    return rc
                time.sleep(0.02)
    finally:
        # No communicate(), pipe drain, or unbounded wait, even on exceptions.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        until = time.monotonic() + 0.5
        while True:
            try:
                _owned(root, known)
            finally:
                _signal_owned(
                    {pid: ident for pid, ident in known.items() if pid != root.pid},
                    signal.SIGKILL,
                )
                if process is not None:
                    process.poll()
            # Reap adopted children only in this dedicated supervisor process.
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    pid = 0
                if pid == 0:
                    break
                if process is not None and pid == process.pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
            if not [p for p in _owned(root, known) if p.pid != root.pid]:
                break
            if time.monotonic() >= until:
                break
            time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["vram", "bytes", "inodes", "validate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["stage", "train", "evaluate"], required=True)
    args = parser.parse_args()
    checkout = Path(os.environ["KML_CHECKOUT"])
    kind = os.environ.get("KML_MEASUREMENT_KIND", "card04")
    validators = {
        "card04": "validate-profile-phase.py",
        "card05": "validate-real-data-phase.py",
        "card05_full": "validate-full-tranche-phase.py",
    }
    if kind not in validators:
        raise ValueError("unknown bounded measurement kind")
    uv = ["uv", "run", "--locked", "--no-sync", "python"]
    commands = {
        "vram": uv
        + [
            str(checkout / "tools/kernel-memory-lab/read-vram-bytes.py"),
            "--expected-uuid",
            os.environ["KML_EXPECTED_GPU_UUID"],
        ],
        "validate": uv
        + [
            str(checkout / "tools/kernel-memory-lab" / validators[kind]),
            args.phase,
        ],
        "bytes": ["du", "-sbx", os.environ["KML_TASK_ROOT"]],
        "inodes": [
            "bash",
            "-o",
            "pipefail",
            "-c",
            'find "$KML_TASK_ROOT" -xdev -printf "\\0" | wc -c',
        ],
    }
    return run(commands[args.kind], args.output)


if __name__ == "__main__":
    raise SystemExit(main())
