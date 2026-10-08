"""Bound provider bootstrap time, then measure owned KML CI execution roots."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

LIMIT_RSS = 8 * 1024**3
LIMIT_DISK = 4 * 1024**3
LIMIT_INODES = 50_000
LIMIT_SECONDS = 1_800
SHUTDOWN_RESERVE_SECONDS = 15
POLL_SECONDS = 0.1
GRACE_SECONDS = 1.0
SAMPLE_SECONDS = 15.0
MIN_FREE_BYTES = 8 * 1024**3
MIN_FREE_INODES = 100_000


class StorageRootMissing(RuntimeError):
    """A declared storage root disappeared during a bounded sample."""


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def capacity_observation(workspace: Path, temp_root: Path, *, phase: str) -> dict:
    """Observe free capacity; do not infer exact writes from free-space deltas."""
    filesystems = []
    for name, path in (("checkout", workspace), ("runner_temp", temp_root)):
        info = os.statvfs(path)
        filesystems.append(
            {
                "name": name,
                "path": str(path),
                "free_bytes_observed": info.f_bavail * info.f_frsize,
                "free_inodes_observed": info.f_favail,
            }
        )
    provider_paths = {
        "runner_tool_cache": os.environ.get("RUNNER_TOOL_CACHE"),
        "runner_temp": os.environ.get("RUNNER_TEMP"),
        "provider_home": os.environ.get("HOME"),
        "uv_python_install_dir": os.environ.get("UV_PYTHON_INSTALL_DIR"),
        "checkout": str(workspace),
    }
    tools = {}
    for name in ("python3", "uv"):
        executable = shutil.which(name)
        if executable is None:
            tools[name] = {"path": None, "version": None}
            continue
        version = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        tools[name] = {
            "path": executable,
            "version": (version.stdout or version.stderr).strip(),
        }
    return {
        "phase": phase,
        "kind": "capacity_observation_not_added_storage_accounting",
        "observed_at_ns": time.time_ns(),
        "filesystems": filesystems,
        "provider_paths": provider_paths,
        "tools": tools,
        "runner_os": os.environ.get("RUNNER_OS"),
        "image_version": os.environ.get("ImageVersion"),
    }


def require_capacity(observation: dict) -> None:
    for entry in observation["filesystems"]:
        if (
            entry["free_bytes_observed"] < MIN_FREE_BYTES
            or entry["free_inodes_observed"] < MIN_FREE_INODES
        ):
            raise RuntimeError(
                f"insufficient free capacity before lab execution: {entry['name']}"
            )


TESTS = {
    "fast": (
        "tests/test_research_lint.py",
        "tests/test_corpus_identity.py",
        "tests/test_config.py",
        "tests/test_conversations.py",
        "tests/test_corpus_acquisition.py",
        "tests/test_corpus_pipeline.py",
        "tests/test_hf_auth.py",
        "tests/test_attempt_budget.py",
        "tests/test_operational_monitor.py",
        "tests/test_kml_card04_profile_tools.py",
    ),
    "archive": (
        "tests/test_research_lint.py",
        "tests/test_declaration_preflight.py",
        "tests/test_corpus_pipeline.py",
        "tests/test_corpus_identity.py",
        "tests/test_archive.py",
    ),
    "serving": (
        "tests/test_serving.py",
        "tests/test_inference_debug.py",
        "tests/test_generation_result.py",
        "tests/test_runtime_id_selection.py",
        "tests/test_snapshot_inference.py",
        "tests/test_snapshot_artifact_bindings.py",
        "tests/test_snapshot_authority.py",
    ),
}
COLLECTIONS = {
    "fast": (250, "bf9e611a699e121028c72431901d063543f950c7f22ac77d9895513e1fffea01"),
    "archive": (
        112,
        "f36da4a4976067bd70cad049835bc550a4a4ead6913a4608e187b2da03e739e7",
    ),
    "serving": (
        215,
        "597f80b8ae0fa3c03e7d07da7f21b5a5b3fc44cef1486ff8279e66702b592cb2",
    ),
}


def verify_collection(output: str, job: str) -> None:
    nodes = [
        line
        for line in output.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    digest = hashlib.sha256(("\n".join(nodes) + "\n").encode()).hexdigest()
    if (len(nodes), digest) != COLLECTIONS[job]:
        raise RuntimeError(
            f"CI {job} node collection differs from reviewed fixture set: {len(nodes)} {digest}"
        )


def verify_frozen_decoding(workspace: Path) -> None:
    frozen = json.loads(
        (
            workspace / "experiments/research/dense-lm-decoding-v1/preregistration.json"
        ).read_text()
    )
    relative = "tests/test_generation.py"
    actual = hashlib.sha256((workspace / relative).read_bytes()).hexdigest()
    if actual != frozen["frozen_sha256"][relative]:
        raise RuntimeError("frozen decoding test digest changed")


@contextmanager
def sample_timer(seconds: float):
    """Interrupt a blocked local traversal as well as a slow iteration."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("CI storage sampling requires the main thread")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("CI storage sampling cannot replace an occupied timer")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_number, _frame):
        raise TimeoutError("CI storage sample deadline exhausted")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
    except BaseException:
        signal.signal(signal.SIGALRM, previous)
        raise
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sample_tree(
    root: Path, *, seconds: float = SAMPLE_SECONDS, progress: dict | None = None
) -> tuple[int, int]:
    if seconds <= 0:
        raise TimeoutError("CI storage sample deadline exhausted")
    with sample_timer(seconds):
        return _sample_tree(root, seconds=seconds, progress=progress)


def _sample_tree(
    root: Path, *, seconds: float, progress: dict | None
) -> tuple[int, int]:
    """Count a live tree in one pass, skipping only entries already removed."""
    deadline = time.monotonic() + seconds
    stack: list[Path | os.DirEntry] = [root]
    root_path = os.fspath(root)
    seen: set[tuple[int, int]] = set()
    total = entries = 0
    if progress is not None:
        progress.update(root=str(root), last_path=str(root), vanished_entries=0)
    while stack:
        if time.monotonic() >= deadline:
            raise TimeoutError("CI storage sample deadline exhausted")
        item = stack.pop()
        path = item.path if isinstance(item, os.DirEntry) else os.fspath(item)
        if progress is not None:
            progress["last_path"] = str(path)
        try:
            info = (
                item.stat(follow_symlinks=False)
                if isinstance(item, os.DirEntry)
                else item.lstat()
            )
        except FileNotFoundError:
            if path == root_path:
                raise StorageRootMissing(
                    f"CI storage root disappeared: {root}"
                ) from None
            if progress is not None:
                progress["vanished_entries"] += 1
            continue
        entries += 1
        if path == root_path and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"CI storage root is not a directory: {root}")
        identity = (info.st_dev, info.st_ino)
        if not stat.S_ISREG(info.st_mode) or identity not in seen:
            total += info.st_size
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            seen.add(identity)
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(path) as listing:
                    for child in listing:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("CI storage sample deadline exhausted")
                        stack.append(child)
            except FileNotFoundError:
                if path == root_path:
                    raise StorageRootMissing(
                        f"CI storage root disappeared: {root}"
                    ) from None
                if progress is not None:
                    progress["vanished_entries"] += 1
    return total, entries


def sample_optional_root(
    path: Path,
    *,
    required: bool,
    seconds: float = SAMPLE_SECONDS,
    progress: dict | None = None,
) -> tuple[int, int]:
    try:
        return sample_tree(path, seconds=seconds, progress=progress)
    except StorageRootMissing:
        if required:
            raise
        return 0, 0


def sample_declared_root(
    path: Path,
    *,
    phase: str,
    deadline: float,
    required: bool,
    progress: dict,
) -> tuple[int, int]:
    progress.clear()
    progress.update(
        root=str(path),
        phase=phase,
        last_path=str(path),
        started_ns=time.monotonic_ns(),
        vanished_entries=0,
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("CI storage sample deadline exhausted")
    result = sample_optional_root(
        path, required=required, seconds=remaining, progress=progress
    )
    progress["elapsed_seconds"] = (time.monotonic_ns() - progress["started_ns"]) / 1e9
    return result


def sample_failure(progress: dict, error: BaseException) -> dict[str, object]:
    started = progress.get("started_ns")
    return {
        "root": progress.get("root"),
        "phase": progress.get("phase"),
        "last_inspected_path": progress.get("last_path"),
        "elapsed_seconds": (
            (time.monotonic_ns() - started) / 1e9 if started is not None else 0.0
        ),
        "exception": f"{type(error).__name__}: {error}",
        "vanished_entries": progress.get("vanished_entries", 0),
    }


def sweep_deadline(workflow_deadline_ns: int) -> float:
    remaining = (workflow_deadline_ns - time.time_ns()) / 1e9 - SHUTDOWN_RESERVE_SECONDS
    return time.monotonic() + min(SAMPLE_SECONDS, max(0.0, remaining))


def process_rows() -> dict[int, tuple[int, int, int, str, str, str]]:
    result = subprocess.run(
        ["ps", "-ww", "-axo", "pid=,ppid=,pgid=,rss=,stat=,lstart=,command="],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 10:
            raise RuntimeError("incomplete process inventory")
        pid, ppid, pgid, rss = map(int, parts[:4])
        rows[pid] = (
            ppid,
            pgid,
            rss * 1024,
            parts[4],
            " ".join(parts[5:10]),
            " ".join(parts[10:]),
        )
    if os.getpid() not in rows:
        raise RuntimeError("CI supervisor missing from process inventory")
    return rows


def owned(rows: dict[int, tuple[int, int, int, str, str, str]], group: int) -> set[int]:
    selected = {pid for pid, row in rows.items() if row[1] == group}
    prior = set()
    while selected != prior:
        prior = selected.copy()
        selected.update(pid for pid, row in rows.items() if row[0] in selected)
    return {pid for pid in selected if not rows[pid][3].startswith("Z")}


def descendants(rows: dict[int, tuple], ancestor: int) -> set[int]:
    selected = {ancestor}
    prior = set()
    while selected != prior:
        prior = selected.copy()
        selected.update(pid for pid, row in rows.items() if row[0] in selected)
    return selected


def _runner_worker(rows: dict[int, tuple]) -> tuple[int, str]:
    current = os.getpid()
    while current in rows and current > 1:
        if "Runner.Worker" in rows[current][5]:
            return current, rows[current][4]
        current = rows[current][0]
    raise RuntimeError("hosted runner worker identity is unavailable")


def _watcher_identity(root: Path) -> tuple[int, str]:
    record = json.loads((root / "watch-pid.json").read_text())
    pid, birth = record["pid"], record["birth"]
    rows = process_rows()
    if pid not in rows or rows[pid][4] != birth or rows[pid][3].startswith("Z"):
        raise RuntimeError("CI resource watcher is not alive")
    if (root / "watch-violation.json").exists():
        raise RuntimeError("CI resource watcher recorded a violation")
    return pid, birth


def hosted_paths(root: Path) -> tuple[Path, Path]:
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    temp_root = Path(os.environ["RUNNER_TEMP"]).resolve()
    resolved_root = root.resolve()
    if (
        not root.is_absolute()
        or root.parent.resolve() != temp_root
        or resolved_root.is_relative_to(workspace)
        or workspace.is_relative_to(resolved_root)
    ):
        raise ValueError(
            "CI job root must be a direct child of runner temp, disjoint from checkout"
        )
    return workspace, temp_root


def checkout_identity(workspace: Path, commit: str) -> None:
    if len(commit) != 40 or int(commit, 16) < 0:
        raise ValueError("exact 40-character commit SHA required")
    if os.environ.get("KML_CI_EXPECTED_SHA") != commit:
        raise RuntimeError("workflow expected SHA differs from requested checkout")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    if head != commit:
        raise RuntimeError("checkout identity differs from approved SHA")


def start_watcher(root: Path, commit: str) -> None:
    workspace, temp_root = hosted_paths(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "startup.json",
        {
            "state": "started",
            "phase": "provider-bootstrap",
            "commit": commit,
            "workspace": str(workspace),
            "runner_temp": str(temp_root),
        },
    )
    progress: dict = {
        "root": str(root),
        "phase": "provider-bootstrap",
        "last_path": str(root),
        "started_ns": time.monotonic_ns(),
        "vanished_entries": 0,
    }
    process = None
    try:
        checkout_identity(workspace, commit)
        deadline_ns = hosted_deadline_ns()
        if time.time_ns() >= deadline_ns - SHUTDOWN_RESERVE_SECONDS * 1_000_000_000:
            raise TimeoutError("aggregate deadline exhausted in hosted queue")
        worker_pid, worker_birth = _runner_worker(process_rows())
        write_json(
            root / "provider-before.json",
            capacity_observation(workspace, temp_root, phase="before-provider-setup"),
        )
        write_json(
            root / "watch-config.json",
            {
                "worker_pid": worker_pid,
                "worker_birth": worker_birth,
                "workspace": str(workspace),
                "job_root": str(root),
                "commit": commit,
                "deadline_ns": deadline_ns,
            },
        )
        progress["phase"] = "startup-watcher-launch"
        with (
            (root / "watch.stdout.log").open("xb") as stdout,
            (root / "watch.stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "--watch-loop",
                    "--root",
                    str(root),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        rows = process_rows()
        if process.pid not in rows:
            raise RuntimeError("watcher exited during start")
        write_json(
            root / "watch-pid.json", {"pid": process.pid, "birth": rows[process.pid][4]}
        )
        ready_deadline = time.monotonic() + 5
        while time.monotonic() < ready_deadline:
            if (root / "watch-ready.json").exists():
                _watcher_identity(root)
                return
            if process.poll() is not None:
                break
            time.sleep(0.05)
        raise RuntimeError("resource watcher failed to become ready")
    except BaseException as error:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired) as terminate_error:
                error.add_note(f"watcher TERM cleanup failed: {terminate_error}")
                try:
                    process.kill()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                    error.add_note(f"watcher cleanup failed: {cleanup_error}")
        try:
            write_json(root / "startup-failure.json", sample_failure(progress, error))
        except OSError as receipt_error:
            error.add_note(f"startup receipt failed: {receipt_error}")
        raise


def enter_lab(root: Path, commit: str) -> None:
    workspace, temp_root = hosted_paths(root)
    progress: dict = {
        "root": str(root),
        "phase": "provider-postflight",
        "last_path": str(root),
        "started_ns": time.monotonic_ns(),
        "vanished_entries": 0,
    }
    try:
        _watcher_identity(root)
        checkout_identity(workspace, commit)
        config = json.loads((root / "watch-config.json").read_text())
        if config["commit"] != commit or config["workspace"] != str(workspace):
            raise RuntimeError("provider postflight binding changed")
        if hosted_deadline_ns() != config["deadline_ns"]:
            raise RuntimeError("workflow deadline binding changed")
        if (
            time.time_ns()
            >= config["deadline_ns"] - SHUTDOWN_RESERVE_SECONDS * 1_000_000_000
        ):
            raise TimeoutError("aggregate deadline exhausted during provider setup")
        observation = capacity_observation(
            workspace, temp_root, phase="after-provider-setup"
        )
        write_json(root / "provider-after.json", observation)
        if observation["tools"]["uv"]["path"] is None:
            raise RuntimeError("provider setup did not supply uv")
        require_capacity(observation)
        deadline = sweep_deadline(config["deadline_ns"])
        checkout_bytes, checkout_inodes = sample_declared_root(
            workspace,
            phase="prelab-checkout",
            deadline=deadline,
            required=True,
            progress=progress,
        )
        job_bytes, job_inodes = sample_declared_root(
            root,
            phase="prelab-job-root",
            deadline=deadline,
            required=True,
            progress=progress,
        )
        total_bytes = checkout_bytes + job_bytes
        total_inodes = checkout_inodes + job_inodes
        write_json(
            root / "prelab-storage.json",
            {
                "roots": [str(workspace), str(root)],
                "live_bytes": total_bytes,
                "live_inodes": total_inodes,
                "max_bytes": LIMIT_DISK,
                "max_inodes": LIMIT_INODES,
            },
        )
        if total_bytes > LIMIT_DISK or total_inodes > LIMIT_INODES:
            raise RuntimeError("owned-root storage cap exceeded before lab execution")
        _watcher_identity(root)
        if (
            time.time_ns()
            >= config["deadline_ns"] - SHUTDOWN_RESERVE_SECONDS * 1_000_000_000
        ):
            raise TimeoutError("aggregate deadline exhausted before lab execution")
        write_json(root / "lab-enabled.json", {"commit": commit})
        ready_deadline = time.monotonic() + SAMPLE_SECONDS + 5
        while time.monotonic() < ready_deadline:
            _watcher_identity(root)
            if (root / "lab-active.json").exists():
                return
            time.sleep(0.05)
        raise RuntimeError("resource watcher did not confirm lab phase")
    except BaseException as error:
        try:
            write_json(root / "bootstrap-failure.json", sample_failure(progress, error))
        except OSError as receipt_error:
            error.add_note(f"bootstrap failure receipt failed: {receipt_error}")
        raise


def _terminate_worker_descendants(worker_pid: int, birth: str) -> dict:
    """Bounded TERM-to-KILL of this runner job, excluding the worker and watcher."""
    errors: list[str] = []
    rows = process_rows()
    if worker_pid not in rows or rows[worker_pid][4] != birth:
        return {
            "verified": False,
            "errors": ["runner worker identity changed"],
            "living": None,
        }
    targets: dict[int, str] = {}

    def inventory() -> tuple[dict, list[int]]:
        current = process_rows()
        if worker_pid not in current or current[worker_pid][4] != birth:
            raise RuntimeError("runner worker identity changed during cleanup")
        for pid in descendants(current, worker_pid) - {worker_pid, os.getpid()}:
            if not current[pid][3].startswith("Z"):
                targets.setdefault(pid, current[pid][4])
        living = [
            pid
            for pid, started in targets.items()
            if pid in current
            and current[pid][4] == started
            and not current[pid][3].startswith("Z")
        ]
        return current, living

    for sig in (signal.SIGTERM, signal.SIGKILL):
        _, living = inventory()
        for pid in living:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"{pid}: {type(error).__name__}: {error}")
        if sig == signal.SIGTERM:
            time.sleep(GRACE_SECONDS)
    deadline = time.monotonic() + 5
    while True:
        _, living = inventory()
        if not living or time.monotonic() >= deadline:
            break
        time.sleep(POLL_SECONDS)
    return {"verified": not living and not errors, "errors": errors, "living": living}


def watch_loop(root: Path) -> None:
    config = json.loads((root / "watch-config.json").read_text())
    worker_pid, birth = config["worker_pid"], config["worker_birth"]
    workspace = Path(config["workspace"])
    job_root = Path(config["job_root"])
    deadline_ns = config["deadline_ns"]
    peak_rss = peak_disk = peak_inodes = 0
    sample_count = 0
    sample_seconds = max_sample_seconds = 0.0
    progress: dict = {}
    write_json(root / "watch-ready.json", {"pid": os.getpid(), "phase": "bootstrap"})
    reason = "completed"
    phase = "bootstrap"
    cleanup: dict | None = None
    failure: BaseException | None = None
    receipt_error: BaseException | None = None
    try:
        while not (root / "watch-stop").exists():
            progress.clear()
            progress.update(
                root=None,
                phase=f"watch-{phase}-process-inventory",
                last_path=None,
                started_ns=time.monotonic_ns(),
            )
            rows = process_rows()
            if worker_pid not in rows or rows[worker_pid][4] != birth:
                raise RuntimeError("hosted runner worker identity changed")
            if time.time_ns() >= deadline_ns - SHUTDOWN_RESERVE_SECONDS * 1_000_000_000:
                raise TimeoutError("aggregate shutdown reserve reached")
            if (root / "lab-enabled.json").exists():
                enabled = json.loads((root / "lab-enabled.json").read_text())
                if enabled.get("commit") != config["commit"]:
                    raise RuntimeError("lab phase identity differs from bootstrap")
                phase = "lab"
            if phase == "bootstrap":
                time.sleep(POLL_SECONDS)
                continue
            owned_pids = descendants(rows, worker_pid) - {os.getpid()}
            rss = sum(
                rows[pid][2] for pid in owned_pids if not rows[pid][3].startswith("Z")
            )
            sweep_started = time.monotonic()
            storage_deadline = sweep_deadline(deadline_ns)
            checkout_bytes, checkout_inodes = sample_declared_root(
                workspace,
                phase="watch-checkout",
                deadline=storage_deadline,
                required=True,
                progress=progress,
            )
            job_bytes, job_inodes = sample_declared_root(
                job_root,
                phase="watch-job-root",
                deadline=storage_deadline,
                required=True,
                progress=progress,
            )
            disk, inodes = checkout_bytes + job_bytes, checkout_inodes + job_inodes
            duration = time.monotonic() - sweep_started
            sample_count += 1
            sample_seconds += duration
            max_sample_seconds = max(max_sample_seconds, duration)
            peak_rss, peak_disk, peak_inodes = (
                max(peak_rss, rss),
                max(peak_disk, disk),
                max(peak_inodes, inodes),
            )
            if rss > LIMIT_RSS or disk > LIMIT_DISK or inodes > LIMIT_INODES:
                raise RuntimeError(f"lab cap: rss={rss} disk={disk} inodes={inodes}")
            if not (root / "lab-active.json").exists():
                write_json(
                    root / "lab-active.json",
                    {
                        "commit": config["commit"],
                        "roots": [str(workspace), str(job_root)],
                        "first_rss_bytes": rss,
                        "first_live_bytes": disk,
                        "first_live_inodes": inodes,
                    },
                )
            time.sleep(POLL_SECONDS)
    except BaseException as error:  # noqa: BLE001 - retain primary watcher fault
        failure = error
        reason = f"{type(error).__name__}: {error}"
        try:
            write_json(
                root / "watch-violation.json",
                {"reason": reason, **sample_failure(progress, error)},
            )
        except OSError as receipt_error:
            error.add_note(f"watch violation receipt failed: {receipt_error}")
        try:
            cleanup = _terminate_worker_descendants(worker_pid, birth)
        except BaseException as cleanup_error:  # noqa: BLE001 - retain cleanup uncertainty
            cleanup = {
                "verified": False,
                "errors": [f"{type(cleanup_error).__name__}: {cleanup_error}"],
                "living": None,
            }
            error.add_note(f"watcher cleanup failed: {cleanup_error}")
    finally:
        try:
            write_json(
                root / "watch-receipt.json",
                {
                    "reason": reason,
                    "phase": phase,
                    "peak_rss_bytes": peak_rss,
                    "peak_disk_bytes": peak_disk,
                    "peak_inodes": peak_inodes,
                    "sample_count": sample_count,
                    "sample_seconds": sample_seconds,
                    "max_sample_seconds": max_sample_seconds,
                    "cleanup": cleanup,
                },
            )
        except BaseException as error:  # noqa: BLE001 - preserve the watcher fault
            receipt_error = error
    if failure is not None:
        if receipt_error is not None:
            failure.add_note(f"watch receipt failed: {receipt_error}")
        raise failure
    if receipt_error is not None:
        raise receipt_error


def stop_watcher(root: Path) -> None:
    if (root / "startup-failure.json").exists() or not (
        root / "watch-pid.json"
    ).exists():
        state = (
            "startup-failed"
            if (root / "startup-failure.json").exists()
            else "startup-incomplete"
        )
        if root.is_dir():
            try:
                (root / "finalization.json").write_text(
                    json.dumps({"state": state, "watcher_started": False}) + "\n"
                )
            except OSError as error:
                print(
                    f"could not retain finalization receipt: {error}", file=sys.stderr
                )
        print(f"CI watcher finalization: {state}; original startup result retained")
        return
    record = json.loads((root / "watch-pid.json").read_text())
    pid, birth = record["pid"], record["birth"]
    rows = process_rows()
    alive = pid in rows and rows[pid][4] == birth and not rows[pid][3].startswith("Z")
    if alive:
        (root / "watch-stop").write_text("stop\n")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (root / "watch-receipt.json").exists():
            receipt = json.loads((root / "watch-receipt.json").read_text())
            state = (
                "watcher-failed"
                if receipt["reason"] != "completed"
                else "bootstrap-failed"
                if (root / "bootstrap-failure.json").exists()
                else "completed"
            )
            write_json(
                root / "finalization.json",
                {
                    "state": state,
                    "watcher_started": True,
                    "watcher_reason": receipt["reason"],
                    "cleanup": receipt.get("cleanup"),
                },
            )
            if state == "watcher-failed":
                print(
                    f"CI watcher finalization: {receipt['reason']}; original failure retained"
                )
                raise RuntimeError(f"CI resource watcher failed: {receipt['reason']}")
            return
        rows = process_rows()
        if pid not in rows or rows[pid][4] != birth or rows[pid][3].startswith("Z"):
            break
        time.sleep(0.1)
    write_json(
        root / "finalization.json",
        {
            "state": "watcher-unverified",
            "watcher_started": True,
            "watcher_reason": "completion receipt missing",
        },
    )
    raise RuntimeError("CI watcher did not produce a completion receipt")


def stop_owned(
    process: subprocess.Popen[bytes], known: dict[int, str], *, terminate: bool
) -> None:
    """TERM then KILL the dedicated session and observed detached descendants."""
    if terminate:
        end = time.monotonic() + GRACE_SECONDS
        for sig in (signal.SIGTERM, signal.SIGKILL):
            rows = process_rows()
            if any(row[1] == process.pid for row in rows.values()):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
            for pid, birth in known.items():
                if pid != process.pid and pid in rows and rows[pid][4] == birth:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
            if sig == signal.SIGTERM:
                while time.monotonic() < end:
                    rows = process_rows()
                    living = owned(rows, process.pid) | {
                        pid
                        for pid, birth in known.items()
                        if pid in rows
                        and rows[pid][4] == birth
                        and not rows[pid][3].startswith("Z")
                    }
                    if not living:
                        break
                    time.sleep(POLL_SECONDS)
    process.wait(timeout=5)
    exit_deadline = time.monotonic() + 5
    while True:
        for pid in known:
            if pid != process.pid:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        rows = process_rows()
        # A stopped child can remain as a zombie until reaped. Keep waiting
        # for its PID to disappear; a zero-survivor receipt needs that proof.
        living = owned(rows, process.pid) | {
            pid for pid, birth in known.items() if pid in rows and rows[pid][4] == birth
        }
        if not living or time.monotonic() >= exit_deadline:
            break
        time.sleep(POLL_SECONDS)
    if living:
        raise RuntimeError(f"CI command retained descendants: {sorted(living)}")


def hosted_deadline_ns() -> int:
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    token = os.environ["KML_CI_GITHUB_TOKEN"]
    endpoint = (
        f"{os.environ['GITHUB_API_URL']}/repos/{repository}/actions/runs/{run_id}"
    )
    request = urllib.request.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        document = json.load(response)
    if (
        document.get("id") != int(run_id)
        or document.get("head_sha") != os.environ["KML_CI_EXPECTED_SHA"]
    ):
        raise RuntimeError("hosted workflow identity differs from pinned commit")
    created = dt.datetime.fromisoformat(document["created_at"])
    return int(created.timestamp() * 1e9) + LIMIT_SECONDS * 1_000_000_000


def run_phase(
    command: list[str],
    *,
    cwd: Path,
    root: Path,
    workspace: Path,
    environment: dict[str, str],
    deadline_ns: int,
    name: str,
    watcher_root: Path | None = None,
) -> int:
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot establish CI subreaper")
    if time.time_ns() >= deadline_ns - SHUTDOWN_RESERVE_SECONDS * 1_000_000_000:
        raise TimeoutError("aggregate CI shutdown reserve reached before phase")
    stdout_path, stderr_path = root / f"{name}.stdout.log", root / f"{name}.stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        known: dict[int, str] = {}
        peak_rss = peak_disk = peak_inodes = 0
        reason = ""
        progress: dict = {}
        sampling_failure: dict | None = None
        phase_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        receipt_error: BaseException | None = None
        try:
            while True:
                if watcher_root is not None:
                    _watcher_identity(watcher_root)
                rows = process_rows()
                living = owned(rows, process.pid) | {
                    pid
                    for pid, birth in known.items()
                    if pid in rows
                    and rows[pid][4] == birth
                    and not rows[pid][3].startswith("Z")
                }
                for pid in living:
                    known.setdefault(pid, rows[pid][4])
                rss = sum(rows[pid][2] for pid in living)
                storage_deadline = sweep_deadline(deadline_ns)
                a_bytes, a_inodes = sample_declared_root(
                    root,
                    phase=f"{name}-output",
                    deadline=storage_deadline,
                    required=True,
                    progress=progress,
                )
                b_bytes, b_inodes = sample_declared_root(
                    workspace,
                    phase=f"{name}-workspace",
                    deadline=storage_deadline,
                    required=True,
                    progress=progress,
                )
                disk, inodes = a_bytes + b_bytes, a_inodes + b_inodes
                peak_rss = max(peak_rss, rss)
                peak_disk = max(peak_disk, disk)
                peak_inodes = max(peak_inodes, inodes)
                if rss > LIMIT_RSS or disk > LIMIT_DISK or inodes > LIMIT_INODES:
                    reason = f"resource cap: rss={rss} disk={disk} inodes={inodes}"
                    break
                if (
                    time.time_ns()
                    >= deadline_ns - SHUTDOWN_RESERVE_SECONDS * 1_000_000_000
                ):
                    reason = "aggregate 30-minute shutdown reserve reached"
                    break
                rc = process.poll()
                if rc is not None:
                    # The storage sweep can outlast the process. The earlier
                    # inventory is only a peak sample, not exit evidence.
                    exit_rows = process_rows()
                    lingering = owned(exit_rows, process.pid) | {
                        pid
                        for pid, birth in known.items()
                        if pid in exit_rows
                        and exit_rows[pid][4] == birth
                        and not exit_rows[pid][3].startswith("Z")
                    }
                    if lingering:
                        reason = "leader exited with owned descendants"
                        break
                    reason = "completed" if rc == 0 else f"command exited {rc}"
                    break
                time.sleep(POLL_SECONDS)
        except BaseException as error:  # noqa: BLE001 - retain even interrupts for cleanup
            phase_error = error
            reason = f"monitor failure: {type(error).__name__}: {error}"
            sampling_failure = sample_failure(progress, error)
        finally:
            try:
                stop_owned(process, known, terminate=reason != "completed")
            except BaseException as error:  # noqa: BLE001 - receipt must record cleanup failure
                cleanup_error = error
            receipt = {
                "phase": name,
                "command": command,
                "reason": (
                    f"cleanup failure: {type(cleanup_error).__name__}: {cleanup_error}"
                    if reason == "completed" and cleanup_error is not None
                    else reason
                ),
                "returncode": process.returncode,
                "peak_rss_bytes": peak_rss,
                "peak_disk_bytes": peak_disk,
                "peak_inodes": peak_inodes,
                "observed_pids": sorted(known),
                "living_descendants": 0 if cleanup_error is None else None,
                "cleanup_verified": cleanup_error is None,
                "cleanup_error": (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                    if cleanup_error is not None
                    else None
                ),
                "sampling_failure": sampling_failure,
            }
            try:
                (root / f"{name}.monitor.json").write_text(
                    json.dumps(receipt, sort_keys=True) + "\n"
                )
            except BaseException as error:  # noqa: BLE001 - preserve the original failure
                receipt_error = error
        if phase_error is not None:
            if cleanup_error is not None:
                phase_error.add_note(
                    f"cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
            if receipt_error is not None:
                phase_error.add_note(
                    f"receipt write also failed: {type(receipt_error).__name__}: {receipt_error}"
                )
            raise phase_error
        if cleanup_error is not None:
            if receipt_error is not None:
                cleanup_error.add_note(
                    f"receipt write also failed: {type(receipt_error).__name__}: {receipt_error}"
                )
            raise cleanup_error
        if receipt_error is not None:
            raise receipt_error
    if reason != "completed":
        raise RuntimeError(f"{name}: {reason}; see {stdout_path} and {stderr_path}")
    return process.returncode or 0


def owned_environment(root: Path) -> dict[str, str]:
    """Route controllable lab subprocess writes into the measured job root."""
    paths = {
        "UV_CACHE_DIR": root / "cache",
        "UV_PROJECT_ENVIRONMENT": root / "venv",
        "UV_PYTHON_INSTALL_DIR": root / "python",
        "TMPDIR": root / "tmp",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
        "HOME": root / "home",
        "XDG_CACHE_HOME": root / "cache",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "PYTHONPYCACHEPREFIX": root / "pycache",
        "PIP_CACHE_DIR": root / "cache/pip",
        "HF_HOME": root / "cache/hf",
        "TORCH_HOME": root / "cache/torch",
        "MPLCONFIGDIR": root / "cache/mpl",
        "SPARSELAB_WORK_DIR": root / "work",
    }
    for name, path in paths.items():
        if name != "UV_PROJECT_ENVIRONMENT":
            path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({name: str(path) for name, path in paths.items()})
    environment.update(
        UV_PYTHON_DOWNLOADS="never",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    return environment


def run_job(job: str, commit: str, root: Path, deadline_ns: int) -> None:
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    if not root.is_absolute() or root.is_relative_to(workspace):
        raise ValueError("CI output root must be absolute and outside checkout")
    if len(commit) != 40 or int(commit, 16) < 0:
        raise ValueError("exact 40-character commit SHA required")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    if head != commit:
        raise RuntimeError("checkout does not match expected CI guard commit")
    verify_frozen_decoding(workspace)
    hosted = "GITHUB_RUN_ID" in os.environ
    if hosted:
        _watcher_identity(root)
        active = json.loads((root / "lab-active.json").read_text())
        if active["commit"] != commit or active["roots"] != [str(workspace), str(root)]:
            raise RuntimeError("lab watcher is not bound to this checkout and job root")
    else:
        root.mkdir(parents=True, exist_ok=False)
    environment = owned_environment(root)
    (root / "storage-baseline.json").write_text(
        json.dumps(
            {"root": str(root), "workspace": str(workspace), "charged_from_zero": True}
        )
        + "\n"
    )
    run_phase(
        ["uv", "sync", "--locked", "--extra", "cpu", "--dev"],
        cwd=workspace,
        root=root,
        workspace=workspace,
        environment=environment,
        deadline_ns=deadline_ns,
        name="setup",
        watcher_root=root if hosted else None,
    )
    kind = "zero" if job in {"lint", "fast"} else job
    venv_python = root / "venv/bin/python"
    run_phase(
        [
            str(venv_python),
            str(Path(__file__).with_name("init_budget.py")),
            "--root",
            str(root),
            "--kind",
            kind,
            "--commit",
            commit,
            "--deadline-ns",
            str(deadline_ns),
        ],
        cwd=workspace,
        root=root,
        workspace=workspace,
        environment=environment,
        deadline_ns=deadline_ns,
        name="init",
        watcher_root=root if hosted else None,
    )
    guard_env = {**environment, **json.loads((root / "init.stdout.log").read_text())}
    guard_env["PYTHONPATH"] = (
        str(Path(__file__).parent) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    if job != "lint":
        run_phase(
            [
                "uv",
                "run",
                "--locked",
                "--no-sync",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                *TESTS[job],
            ],
            cwd=workspace,
            root=root,
            workspace=workspace,
            environment={**guard_env, "KML_CI_PREFLIGHT_ONLY": "1"},
            deadline_ns=deadline_ns,
            name="collection",
            watcher_root=root if hosted else None,
        )
        verify_collection((root / "collection.stdout.log").read_text(), job)
    if job == "lint":
        commands = [
            ["uv", "run", "--locked", "--no-sync", "ruff", "check", "."],
            ["uv", "run", "--locked", "--no-sync", "ruff", "format", "--check", "."],
        ]
    else:
        commands = []
        if job in {"fast", "archive"}:
            commands.append(
                [
                    "uv",
                    "run",
                    "--locked",
                    "--no-sync",
                    "python",
                    "-m",
                    "sparselab.research.lint",
                    "--json",
                ]
            )
        commands.append(
            [
                "uv",
                "run",
                "--locked",
                "--no-sync",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(root / "pytest"),
                *TESTS[job],
            ]
        )
    for number, command in enumerate(commands, 1):
        run_phase(
            command,
            cwd=workspace,
            root=root,
            workspace=workspace,
            environment=guard_env,
            deadline_ns=deadline_ns,
            name=f"check-{number}",
            watcher_root=root if hosted else None,
        )
    run_phase(
        [
            str(venv_python),
            str(Path(__file__).with_name("finalize.py")),
            "--root",
            str(root),
            "--kind",
            kind,
            "--require-complete",
        ],
        cwd=workspace,
        root=root,
        workspace=workspace,
        environment=environment,
        deadline_ns=deadline_ns,
        name="final",
        watcher_root=root if hosted else None,
    )
    (root / "final-counters.json").write_text((root / "final.stdout.log").read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("lint", "fast", "archive", "serving"))
    parser.add_argument("--expected-sha")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--test-deadline-ns", type=int)
    parser.add_argument("--start-watch", action="store_true")
    parser.add_argument("--enter-lab", action="store_true")
    parser.add_argument("--watch-loop", action="store_true")
    parser.add_argument("--stop-watch", action="store_true")
    args = parser.parse_args()
    if args.watch_loop:
        watch_loop(args.root)
        return 0
    if args.start_watch:
        if args.expected_sha is None:
            parser.error("start-watch requires --expected-sha")
        start_watcher(args.root, args.expected_sha)
        return 0
    if args.enter_lab:
        if args.expected_sha is None:
            parser.error("enter-lab requires --expected-sha")
        enter_lab(args.root, args.expected_sha)
        return 0
    if args.stop_watch:
        stop_watcher(args.root)
        return 0
    if args.job is None or args.expected_sha is None:
        parser.error("job execution requires --job and --expected-sha")
    deadline_ns = args.test_deadline_ns or hosted_deadline_ns()
    try:
        run_job(args.job, args.expected_sha, args.root, deadline_ns)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(
            f"CI qualification failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        for path in sorted(args.root.glob("*.stderr.log")):
            print(
                f"{path}: {path.read_text(errors='replace')[-16000:]}", file=sys.stderr
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
