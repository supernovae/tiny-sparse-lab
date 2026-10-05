"""Operation-scoped Linux process monitoring; no scientific identity or retry semantics."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Literal

import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class MonitorPolicy(BaseModel):
    """Independent operational guard policy (not a ResourceEnvelope or run config)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    monitor_policy_version: Literal[1]
    max_tree_rss_bytes: int | None = Field(default=None, strict=True, gt=0)
    max_tree_swap_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_host_available_ram_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_host_free_swap_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_disk_free_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_disk_free_inodes: int | None = Field(default=None, strict=True, ge=0)
    min_projected_disk_free_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_projected_disk_free_inodes: int | None = Field(default=None, strict=True, ge=0)
    interval_seconds: float = Field(default=1.0, gt=0, le=2)
    termination_grace_seconds: float = Field(default=5.0, ge=0)

    @field_validator("monitor_policy_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("monitor policy version must be integer 1")
        return value


class ProcessIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    pid: int
    create_time: float


class MonitorSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    monitor_record_version: Literal[1] = 1
    kind: Literal["sample"] = "sample"
    elapsed_seconds: float
    processes: list[ProcessIdentity]
    tree_rss_bytes: int | None
    tree_swap_bytes: int | None
    host_available_ram_bytes: int | None
    host_free_swap_bytes: int | None
    host_used_swap_bytes: int | None
    disk_free_bytes: int | None
    disk_free_inodes: int | None
    projected_disk_free_bytes: int | None
    projected_disk_free_inodes: int | None
    violations: list[str]


class MonitorLaunch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    monitor_record_version: Literal[1] = 1
    kind: Literal["launch"] = "launch"
    command: list[str]
    root: ProcessIdentity
    policy: MonitorPolicy
    workspace: str
    cwd: str
    reserved_bytes: int
    reserved_inodes: int
    source_sha256: str
    policy_sha256: str
    source_path: str
    policy_path: str
    stdout_path: str
    stderr_path: str
    events_path: str
    completion_path: str


class MonitorCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    monitor_record_version: Literal[1] = 1
    kind: Literal["completion"] = "completion"
    launch: MonitorLaunch
    returncode: int | None
    status: Literal["COMPLETE", "FAILED", "VIOLATED", "MONITOR_ERROR"]
    violations: list[str]
    peak_tree_rss_bytes: int | None
    peak_tree_swap_bytes: int | None
    min_host_available_ram_bytes: int | None
    min_host_free_swap_bytes: int | None
    min_disk_free_bytes: int | None
    min_disk_free_inodes: int | None
    min_projected_disk_free_bytes: int | None
    min_projected_disk_free_inodes: int | None
    elapsed_seconds: float
    sample_count: int


def _parse_policy(raw: bytes, path: Path) -> MonitorPolicy:
    try:
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise TypeError("policy must be a YAML mapping")
        return MonitorPolicy.model_validate(data)
    except (yaml.YAMLError, ValidationError, TypeError) as error:
        raise ValueError(f"invalid monitor policy {path}: {error}") from error


def load_monitor_policy(path: Path) -> MonitorPolicy:
    try:
        return _parse_policy(path.read_bytes(), path)
    except OSError as error:
        raise ValueError(f"cannot read monitor policy {path}: {error}") from error


def _identity(process: psutil.Process) -> ProcessIdentity:
    return ProcessIdentity(pid=process.pid, create_time=process.create_time())


def _owned(
    root: ProcessIdentity, known: dict[int, ProcessIdentity]
) -> list[psutil.Process]:
    """Discover living descendants by parent linkage; retain observed orphans."""
    snapshot: dict[int, tuple[psutil.Process, ProcessIdentity, int]] = {}
    for process in psutil.process_iter(["ppid"]):
        try:
            identity = _identity(process)
            if process.pid != root.pid and process.status() == psutil.STATUS_ZOMBIE:
                continue
            snapshot[process.pid] = (process, identity, process.ppid())
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            if process.pid in known:
                raise RuntimeError(
                    f"owned process attribution unavailable: {process.pid}"
                ) from error
            continue
    root_entry = snapshot.get(root.pid)
    if root_entry is not None and root_entry[1] != root:
        raise RuntimeError("root PID was reused")
    reused = {
        pid
        for pid, entry in snapshot.items()
        if pid in known and known[pid] != entry[1]
    }
    discovered = {pid for pid, entry in snapshot.items() if known.get(pid) == entry[1]}
    if root_entry is not None:
        discovered.add(root.pid)
    changed = True
    while changed:
        before = len(discovered)
        for pid, (_, identity, ppid) in snapshot.items():
            if pid not in reused and ppid in discovered and pid not in discovered:
                discovered.add(pid)
                known[pid] = identity
        changed = len(discovered) != before
    for pid in reused:
        # Never adopt a recycled PID, even if it is now another descendant.
        del known[pid]
    return [snapshot[pid][0] for pid in sorted(discovered)]


def _swap_bytes(process: psutil.Process) -> int:
    try:
        swap = process.memory_full_info().swap
        if swap is not None:
            return int(swap)
    except psutil.NoSuchProcess:
        raise
    except AttributeError, psutil.Error, OSError:
        pass
    try:
        for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
            if line.startswith("VmSwap:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError(
            f"swap attribution unavailable for PID {process.pid}"
        ) from error
    raise RuntimeError(f"swap attribution unavailable for PID {process.pid}")


def _storage(workspace: Path) -> tuple[int, int]:
    current = workspace.absolute()
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"workspace ancestor unavailable: {workspace}")
        current = current.parent
    if not current.is_dir():
        raise RuntimeError(f"workspace ancestor not a directory: {current}")
    stats = os.statvfs(current)
    return stats.f_bavail * stats.f_frsize, stats.f_favail


def sample(
    root: ProcessIdentity,
    known: dict[int, ProcessIdentity],
    policy: MonitorPolicy,
    workspace: Path,
    *,
    reserved_bytes: int = 0,
    reserved_inodes: int = 0,
    elapsed_seconds: float = 0,
) -> MonitorSample:
    """Read one snapshot; missing requested metrics violate rather than pass."""
    if reserved_bytes < 0 or reserved_inodes < 0:
        raise ValueError("reservations must be nonnegative")
    processes = _owned(root, known)
    identities = []
    violations: list[str] = []
    rss = 0
    swap = 0
    rss_valid = swap_valid = True
    for process in processes:
        try:
            expected = known[process.pid]
            if _identity(process) != expected:
                raise RuntimeError(f"PID reused during sample: {process.pid}")
        except psutil.NoSuchProcess:
            continue
        except psutil.Error, OSError, RuntimeError:
            rss_valid = swap_valid = False
            continue
        current_rss = current_swap = None
        try:
            current_rss = int(process.memory_info().rss)
            if current_rss < 0:
                raise RuntimeError("negative owned process RSS")
        except psutil.NoSuchProcess:
            continue
        except psutil.Error, OSError, RuntimeError:
            current_rss = None
        try:
            current_swap = _swap_bytes(process)
            if current_swap < 0:
                raise RuntimeError("negative owned process swap")
        except psutil.NoSuchProcess:
            continue
        except psutil.Error, OSError, RuntimeError:
            current_swap = None
        try:
            if _identity(process) != expected:
                raise RuntimeError(f"PID reused during sample: {process.pid}")
        except psutil.NoSuchProcess:
            continue
        except psutil.Error, OSError, RuntimeError:
            rss_valid = swap_valid = False
            continue
        identities.append(expected)
        if current_rss is None:
            rss_valid = False
        else:
            rss += current_rss
        if current_swap is None:
            swap_valid = False
        else:
            swap += current_swap
    try:
        available = int(psutil.virtual_memory().available)
    except psutil.Error, OSError, AttributeError, ValueError:
        available = None
    try:
        swap_info = psutil.swap_memory()
        free_swap, used_swap = int(swap_info.free), int(swap_info.used)
    except psutil.Error, OSError, AttributeError, ValueError:
        free_swap = used_swap = None
    try:
        disk, inodes = _storage(workspace)
    except OSError, RuntimeError:
        disk = inodes = None
    projected_disk = disk - reserved_bytes if disk is not None else None
    projected_inodes = inodes - reserved_inodes if inodes is not None else None
    for name, actual, limit, direction in (
        (
            "max_tree_rss_bytes",
            rss if rss_valid else None,
            policy.max_tree_rss_bytes,
            "max",
        ),
        (
            "max_tree_swap_bytes",
            swap if swap_valid else None,
            policy.max_tree_swap_bytes,
            "max",
        ),
        (
            "min_host_available_ram_bytes",
            available,
            policy.min_host_available_ram_bytes,
            "min",
        ),
        ("min_host_free_swap_bytes", free_swap, policy.min_host_free_swap_bytes, "min"),
        ("min_disk_free_bytes", disk, policy.min_disk_free_bytes, "min"),
        ("min_disk_free_inodes", inodes, policy.min_disk_free_inodes, "min"),
        (
            "min_projected_disk_free_bytes",
            projected_disk,
            policy.min_projected_disk_free_bytes,
            "min",
        ),
        (
            "min_projected_disk_free_inodes",
            projected_inodes,
            policy.min_projected_disk_free_inodes,
            "min",
        ),
    ):
        if limit is not None and (
            actual is None or (actual > limit if direction == "max" else actual < limit)
        ):
            violations.append(name)
    return MonitorSample(
        elapsed_seconds=elapsed_seconds,
        processes=identities,
        tree_rss_bytes=rss if rss_valid else None,
        tree_swap_bytes=swap if swap_valid else None,
        host_available_ram_bytes=available,
        host_free_swap_bytes=free_swap,
        host_used_swap_bytes=used_swap,
        disk_free_bytes=disk,
        disk_free_inodes=inodes,
        projected_disk_free_bytes=projected_disk,
        projected_disk_free_inodes=projected_inodes,
        violations=violations,
    )


def _signal_owned(known: dict[int, ProcessIdentity], signum: int) -> None:
    """Signal attributed processes, using race-free pidfds where available.

    On other hosts psutil checks process creation time again before signalling;
    those hosts do not provide Linux's atomic pidfd signalling guarantee.
    """
    for identity in list(known.values()):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            try:
                process = psutil.Process(identity.pid)
                if _identity(process) == identity:
                    process.send_signal(signum)
            except psutil.NoSuchProcess:
                pass
            continue
        try:
            fd = os.pidfd_open(identity.pid)
        except ProcessLookupError:
            continue
        try:
            try:
                if _identity(psutil.Process(identity.pid)) == identity:
                    signal.pidfd_send_signal(fd, signum)
            except psutil.NoSuchProcess, ProcessLookupError:
                pass
        finally:
            os.close(fd)


def _record(stream: object, record: BaseModel) -> None:
    stream.write(record.model_dump_json() + "\n")  # type: ignore[attr-defined]
    stream.flush()  # type: ignore[attr-defined]
    os.fsync(stream.fileno())  # type: ignore[attr-defined]


def monitor_command(
    command: list[str],
    policy: MonitorPolicy,
    *,
    workspace: Path,
    log_dir: Path,
    policy_path: Path,
    reserved_bytes: int = 0,
    reserved_inodes: int = 0,
    cwd: Path | None = None,
) -> MonitorCompletion:
    """Launch once; exclusively create a log directory and persist each observation."""
    if not command or reserved_bytes < 0 or reserved_inodes < 0:
        raise ValueError("command required and reservations must be nonnegative")
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError("workspace must be an existing directory")
    working_directory = Path.cwd() if cwd is None else cwd
    if not working_directory.is_dir():
        raise ValueError("monitor command cwd must be an existing directory")
    policy_bytes = policy_path.read_bytes()
    if _parse_policy(policy_bytes, policy_path) != policy:
        raise ValueError("monitor policy differs from policy_path bytes")
    log_dir.mkdir(parents=True, exist_ok=False)
    events = log_dir / "events.jsonl"
    completion_path = log_dir / "completion.json"
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    stdout_path = log_dir / "command.stdout.log"
    stderr_path = log_dir / "command.stderr.log"
    start = time.monotonic()
    with (
        events.open("x", encoding="utf-8") as stream,
        stdout_path.open("xb") as command_stdout,
        stderr_path.open("xb") as command_stderr,
    ):
        process = subprocess.Popen(
            command, cwd=working_directory, stdout=command_stdout, stderr=command_stderr
        )
        root = _identity(psutil.Process(process.pid))
        known = {root.pid: root}
        launch = MonitorLaunch(
            command=command,
            root=root,
            policy=policy,
            workspace=str(workspace.resolve()),
            cwd=str(working_directory.resolve()),
            reserved_bytes=reserved_bytes,
            reserved_inodes=reserved_inodes,
            source_sha256=source_sha,
            policy_sha256=policy_sha,
            source_path=str(Path(__file__).resolve()),
            policy_path=str(policy_path.resolve()),
            stdout_path=str(stdout_path.resolve()),
            stderr_path=str(stderr_path.resolve()),
            events_path=str(events.resolve()),
            completion_path=str(completion_path.resolve()),
        )
        _record(stream, launch)
        minima: dict[str, int | None] = dict.fromkeys(
            (
                "host_available_ram_bytes",
                "host_free_swap_bytes",
                "disk_free_bytes",
                "disk_free_inodes",
                "projected_disk_free_bytes",
                "projected_disk_free_inodes",
            )
        )
        peak_rss: int | None = None
        peak_swap: int | None = None
        sample_count = 0
        violations: list[str] = []
        monitor_error = False
        terminating_at: float | None = None
        try:
            while True:
                try:
                    observation = sample(
                        root,
                        known,
                        policy,
                        workspace,
                        reserved_bytes=reserved_bytes,
                        reserved_inodes=reserved_inodes,
                        elapsed_seconds=time.monotonic() - start,
                    )
                    sample_count += 1
                    if observation.tree_rss_bytes is not None:
                        peak_rss = (
                            observation.tree_rss_bytes
                            if peak_rss is None
                            else max(peak_rss, observation.tree_rss_bytes)
                        )
                    if observation.tree_swap_bytes is not None:
                        peak_swap = (
                            observation.tree_swap_bytes
                            if peak_swap is None
                            else max(peak_swap, observation.tree_swap_bytes)
                        )
                    for field, prior in minima.items():
                        value = getattr(observation, field)
                        if value is not None:
                            minima[field] = (
                                value if prior is None else min(prior, value)
                            )
                    _record(stream, observation)
                    violations.extend(
                        v for v in observation.violations if v not in violations
                    )
                except (RuntimeError, OSError, psutil.Error) as error:
                    monitor_error = True
                    violations.append(f"monitor attribution: {error}")
                if violations and terminating_at is None:
                    terminating_at = time.monotonic()
                    _signal_owned(known, signal.SIGTERM)
                if (
                    terminating_at is not None
                    and time.monotonic() - terminating_at
                    >= policy.termination_grace_seconds
                ):
                    _signal_owned(known, signal.SIGKILL)
                # Continue until root AND all identified children exit; no group signals.
                if process.poll() is not None:
                    try:
                        if not _owned(root, known):
                            break
                    except RuntimeError as error:
                        monitor_error = True
                        violations.append(f"monitor attribution: {error}")
                        # A reused root can no longer own newly discovered children.
                        # Keep only already identified, living children for shutdown.
                        known.pop(root.pid, None)
                        if not any(
                            psutil.pid_exists(pid)
                            and _identity(psutil.Process(pid)) == identity
                            for pid, identity in known.items()
                        ):
                            break
                time.sleep(policy.interval_seconds)
        except BaseException:
            _signal_owned(known, signal.SIGKILL)
            raise
        rc = process.wait()
        result = MonitorCompletion(
            launch=launch,
            returncode=rc,
            status=(
                "MONITOR_ERROR"
                if monitor_error
                else "VIOLATED"
                if violations
                else "COMPLETE"
                if rc == 0
                else "FAILED"
            ),
            violations=violations,
            peak_tree_rss_bytes=peak_rss,
            peak_tree_swap_bytes=peak_swap,
            min_host_available_ram_bytes=minima["host_available_ram_bytes"],
            min_host_free_swap_bytes=minima["host_free_swap_bytes"],
            min_disk_free_bytes=minima["disk_free_bytes"],
            min_disk_free_inodes=minima["disk_free_inodes"],
            min_projected_disk_free_bytes=minima["projected_disk_free_bytes"],
            min_projected_disk_free_inodes=minima["projected_disk_free_inodes"],
            elapsed_seconds=time.monotonic() - start,
            sample_count=sample_count,
        )
        for output in (command_stdout, command_stderr):
            output.flush()
            os.fsync(output.fileno())
        _record(stream, result)
        with completion_path.open("x", encoding="utf-8") as output:
            output.write(result.model_dump_json(indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        return result
