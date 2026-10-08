"""Supervise one bounded Card 05 command and all of its owned descendants.

The command may include GNU timeout, the native monitor, uv and a worker, each
with distinct leaders or process groups. This Linux subreaper tracks process
identities through early parent exits and never signals an unrelated process.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from sparselab.operational_monitor import _identity, _owned, _signal_owned

TERM_GRACE_SECONDS = 1.0
POLL_SECONDS = 0.02


def _alive(root, known: dict) -> list[psutil.Process]:
    return [process for process in _owned(root, known) if process.pid != root.pid]


def _owner_alive(identity) -> bool:
    try:
        process = psutil.Process(identity.pid)
        return process.status() != psutil.STATUS_ZOMBIE and _identity(process) == identity
    except psutil.NoSuchProcess:
        return False


def supervise(
    command: list[str], *, deadline_ns: int, stop_file: Path, owner_pid: int, completion: Path
) -> int:
    if sys.platform != "linux" or not command or deadline_ns <= time.time_ns():
        raise ValueError("bounded Linux phase command and future deadline required")
    owner = _identity(psutil.Process(owner_pid))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot establish phase subreaper")
    root = _identity(psutil.Process())
    known = {root.pid: root}
    requested_signal = 0

    def interrupted(signum: int, frame: object) -> None:
        nonlocal requested_signal
        requested_signal = signum

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    process: subprocess.Popen[bytes] | None = None
    reason = ""
    returncode = 1
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        known[process.pid] = _identity(psutil.Process(process.pid))
        while True:
            # Reap the command leader first so an already detached grandchild
            # is adopted by this subreaper before ownership is sampled.
            rc = process.poll()
            living = _alive(root, known)
            if requested_signal:
                reason = f"supervisor received signal {requested_signal}"
                break
            if stop_file.exists():
                reason = "launcher requested stop"
                break
            if not _owner_alive(owner):
                reason = "launcher owner exited"
                break
            if time.time_ns() >= deadline_ns:
                reason = "phase deadline reached"
                break
            if rc is not None:
                if living:
                    reason = "command exited with live descendants"
                    break
                returncode = rc
                reason = "completed" if rc == 0 else f"command exited {rc}"
                break
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        reason = f"supervisor exception: {type(error).__name__}: {error}"
        raise
    finally:
        # TERM once, then KILL after a fixed grace period. Continue discovering
        # adopted orphans until every known owned identity has exited. A process
        # stuck in uninterruptible I/O cannot be declared clean while it lives.
        if process is not None:
            process.poll()
            if reason == "completed" and _alive(root, known):
                reason = "command completed with late owned descendants"
                returncode = 1
            grace_end = time.monotonic() + TERM_GRACE_SECONDS
            term_sent = False
            while True:
                process.poll()
                living = _alive(root, known)
                if not living:
                    break
                targets = {p.pid: known[p.pid] for p in living}
                if not term_sent:
                    _signal_owned(targets, signal.SIGTERM)
                    term_sent = True
                elif time.monotonic() >= grace_end:
                    _signal_owned(targets, signal.SIGKILL)
                time.sleep(POLL_SECONDS)
            process.poll()
            # Adopted zombies are no longer running, but reap them before exit.
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
            if _alive(root, known):
                raise RuntimeError("owned descendants remain after shutdown")
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        result = {
            "format": "kml-card05-owned-phase-completion-v1",
            "reason": reason,
            "returncode": returncode,
            "owner_pid": owner_pid,
            "observed_descendants": sorted(pid for pid in known if pid != root.pid),
            "living_descendants": 0,
        }
        with completion.open("x") as stream:
            json.dump(result, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-ns", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return supervise(
        command,
        deadline_ns=args.deadline_ns,
        stop_file=args.stop_file,
        owner_pid=args.owner_pid,
        completion=args.completion,
    )


if __name__ == "__main__":
    raise SystemExit(main())
