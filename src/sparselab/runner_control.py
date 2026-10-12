"""Opt-in, fail-closed host ownership around the native monitor (Linux only)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

from sparselab.operational_monitor import (
    MonitorCompletion,
    MonitorPolicy,
    monitor_command,
)


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, record: dict[str, Any]) -> None:
    """Exclusive durable records; partial writes remain authoritative blockers."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)


def _safe(path: Path, *, directory: bool = False) -> os.stat_result:
    info = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not kind(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o022
        or (not directory and info.st_nlink != 1)
    ):
        raise ValueError(f"unsafe runner state: {path}")
    return info


def _read(path: Path) -> dict[str, Any]:
    _safe(path)
    result = json.loads(path.read_bytes())
    if not isinstance(result, dict):
        raise TypeError(f"invalid runner record: {path}")
    return result


def _unblocked(path: Path) -> None:
    # A dangling symlink, malformed or unreadable marker still blocks launch.
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise ValueError(f"runner blocked: {path}")


def _prior_runs(state: Path, contract: dict[str, Any]) -> None:
    entries = list(state.iterdir())
    config = state / "contract.json"
    if not entries:
        _write(config, contract)
    elif _read(config) != contract:
        raise ValueError("runner contract or stable lock identity changed")
    for run in state.iterdir():
        if run == config:
            continue
        _safe(run, directory=True)
        pending = _read(run / "pending.json")
        complete = _read(run / "completion.json")
        if (
            pending.get("run_id") != run.name
            or pending.get("contract") != contract
            or complete.get("pending_sha256")
            != hashlib.sha256((run / "pending.json").read_bytes()).hexdigest()
            or complete.get("run_id") != run.name
            or complete.get("cleanup_verified") is not True
        ):
            raise ValueError(f"incomplete or foreign runner receipt: {run}")
        receipt_path = Path(pending["log_dir"]) / "completion.json"
        _safe(receipt_path)
        receipt_bytes = receipt_path.read_bytes()
        receipt = MonitorCompletion.model_validate_json(receipt_bytes)
        if (
            hashlib.sha256(receipt_bytes).hexdigest() != complete["monitor_sha256"]
            or receipt.launch.root.model_dump() != complete["root"]
            or receipt.launch.command != pending["command"]
            or receipt.launch.policy.model_dump(mode="json") != pending["policy"]
            or receipt.launch.completion_path != str(receipt_path)
            or receipt.launch.owned_source_sha256 is None
            or receipt.status == "MONITOR_ERROR"
        ):
            raise ValueError(f"unverified prior monitor receipt: {run}")
        owned_path = receipt_path.parent / "owned-completion.json"
        _safe(owned_path)
        owned_bytes = owned_path.read_bytes()
        owned = json.loads(owned_bytes)
        if (
            hashlib.sha256(owned_bytes).hexdigest() != complete["owned_sha256"]
            or owned["root"] != complete["root"]
            or owned["living_descendants"] != 0
            or owned["returncode"] != receipt.returncode
        ):
            raise ValueError(f"unverified prior owned receipt: {run}")


def run_controlled(
    command: list[str],
    policy: MonitorPolicy,
    *,
    lock_path: Path,
    state_dir: Path,
    blocked_path: Path,
    **monitor_options: Any,
) -> MonitorCompletion:
    """One flock owner; never reclaim incomplete state or remove a blocked marker."""
    if sys.platform != "linux":
        raise ValueError("runner control requires Linux")
    # Resolve parents, never the final component: symlinks must remain visible.
    lock = lock_path.parent.resolve(strict=True) / lock_path.name
    state = state_dir.parent.resolve(strict=True) / state_dir.name
    blocked = blocked_path.parent.resolve(strict=True) / blocked_path.name
    if len({lock, state, blocked}) != 3 or (
        state in lock.parents or state in blocked.parents
    ):
        raise ValueError("runner lock, state and blocked paths must be separate")
    _safe(lock.parent, directory=True)
    _safe(state.parent, directory=True)
    _safe(blocked.parent, directory=True)
    _unblocked(blocked)
    log_dir = Path(monitor_options["log_dir"])
    monitor_options = {
        **monitor_options,
        "log_dir": log_dir.parent.resolve(strict=True) / log_dir.name,
    }
    run_id = uuid.uuid4().hex
    stopping = False
    started = False

    def interrupted(signum: int, _frame: object) -> None:
        nonlocal stopping
        if stopping:
            return  # Repeated signals must not interrupt owned cleanup or persistence.
        stopping = True
        raise InterruptedError(f"runner received signal {signum}")

    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        identity = _safe(lock)
        opened = os.fstat(fd)
        if (identity.st_dev, identity.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("runner lock identity changed")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = {
            sig: signal.getsignal(sig)
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        }
        try:
            for sig in previous:
                signal.signal(sig, interrupted)
            _unblocked(blocked)
            identity = _safe(lock)
            if (identity.st_dev, identity.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("runner lock identity changed")
            os.fsync(fd)
            _sync_dir(lock.parent)
            # A missing directory may mean lost evidence, never a recovery request.
            try:
                _safe(state, directory=True)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"runner state directory missing: {state}; confirm no outstanding "
                    "work before provisioning it (docs/workspaces.md)"
                ) from error
            _sync_dir(state.parent)
            contract = {
                "format": "sparselab-runner-v1",
                "lock": str(lock),
                "lock_device": opened.st_dev,
                "lock_inode": opened.st_ino,
                "blocked": str(blocked),
            }
            _prior_runs(state, contract)
            run = state / run_id
            run.mkdir(mode=0o700)
            started = True
            _sync_dir(state)
            pending = {
                "run_id": run_id,
                "contract": contract,
                "command": command,
                "policy": policy.model_dump(mode="json"),
                "log_dir": str(monitor_options["log_dir"]),
                "owner_pid": os.getpid(),
            }
            # Durable before the monitor can spawn anything, including its helper.
            _write(run / "pending.json", pending)
            result = monitor_command(
                command, policy, owned_supervisor=True, **monitor_options
            )
            if result.status == "MONITOR_ERROR":
                raise RuntimeError("runner cleanup or monitoring unverified")
            info = _safe(lock)
            if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
                raise RuntimeError("runner lock replaced during command")
            receipt = Path(result.launch.completion_path).read_bytes()
            if MonitorCompletion.model_validate_json(receipt) != result:
                raise RuntimeError("runner monitor completion mismatch")
            # Seal directory entries as well as the monitor's fsynced file bytes.
            _sync_dir(Path(result.launch.completion_path).parent)
            _sync_dir(Path(result.launch.completion_path).parent.parent)
            owned_bytes = (
                Path(result.launch.completion_path).parent / "owned-completion.json"
            ).read_bytes()
            _write(
                run / "completion.json",
                {
                    "run_id": run_id,
                    "pending_sha256": hashlib.sha256(
                        (run / "pending.json").read_bytes()
                    ).hexdigest(),
                    "root": result.launch.root.model_dump(),
                    "monitor_sha256": hashlib.sha256(receipt).hexdigest(),
                    "owned_sha256": hashlib.sha256(owned_bytes).hexdigest(),
                    "cleanup_verified": True,
                },
            )
            return result
        except BaseException as error:
            stopping = True
            # Preflight refusals create no attempt and need no new host blocker.
            # After mkdir, retained incomplete state blocks even if this write fails.
            if started and not os.path.lexists(blocked):
                _write(blocked, {"run_id": run_id, "reason": str(error)})
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    finally:
        os.close(fd)  # No lock-keeper process; pending state survives SIGKILL.
