"""Run one reviewed KML CI job under a deadline and whole-job resource sampler.

This bootstrap uses only the standard library, so package installation is inside
the same cap as collection, fixtures, tests, logs and child processes.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
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


class StorageRootMissing(RuntimeError):
    """A declared storage root disappeared during a bounded sample."""


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


def start_watcher(root: Path, commit: str) -> None:
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
        ).strip()
        != commit
    ):
        raise RuntimeError("watcher checkout identity differs from approved SHA")
    deadline_ns = hosted_deadline_ns()
    if time.time_ns() >= deadline_ns:
        raise TimeoutError("aggregate deadline exhausted in hosted queue")
    worker_pid, worker_birth = _runner_worker(process_rows())
    root.mkdir(parents=True, exist_ok=False)
    extra_paths = [Path(os.environ.get("RUNNER_TEMP", str(root))).resolve()]
    if os.environ.get("GITHUB_ACTIONS") == "true":
        if os.environ.get("RUNNER_TOOL_CACHE"):
            extra_paths.append(Path(os.environ["RUNNER_TOOL_CACHE"]).resolve())
        extra_paths.append(Path.home() / ".cache")
    extra_paths = list(dict.fromkeys(extra_paths))
    (root / "startup.json").write_text(
        json.dumps(
            {
                "state": "started",
                "phase": "startup-ambient-baseline",
                "commit": commit,
                "workspace": str(workspace),
                "ambient_roots": [str(path) for path in extra_paths],
                "sample_deadline_seconds": SAMPLE_SECONDS,
                "workflow_deadline_ns": deadline_ns,
            },
            sort_keys=True,
        )
        + "\n"
    )
    progress: dict = {}
    baselines = []
    process = None
    try:
        extras = []
        baseline_deadline = sweep_deadline(deadline_ns)
        for path in extra_paths:
            baseline = sample_declared_root(
                path,
                phase="startup-ambient-baseline",
                deadline=baseline_deadline,
                required=False,
                progress=progress,
            )
            baselines.append(
                {
                    "root": str(path),
                    "bytes": baseline[0],
                    "inodes": baseline[1],
                    "elapsed_seconds": progress["elapsed_seconds"],
                    "vanished_entries": progress["vanished_entries"],
                }
            )
            extras.append(
                {
                    "path": str(path),
                    "bytes": baseline[0],
                    "inodes": baseline[1],
                    "required": baseline[1] > 0,
                }
            )
        (root / "startup-baselines.json").write_text(
            json.dumps(baselines, sort_keys=True) + "\n"
        )
        progress.update(
            root=str(root),
            phase="startup-watcher-launch",
            last_path=str(root),
            started_ns=time.monotonic_ns(),
        )
        (root / "watch-config.json").write_text(
            json.dumps(
                {
                    "worker_pid": worker_pid,
                    "worker_birth": worker_birth,
                    "workspace": str(workspace),
                    "deadline_ns": deadline_ns,
                    "extra_storage": extras,
                },
                sort_keys=True,
            )
            + "\n"
        )
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
        (root / "watch-pid.json").write_text(
            json.dumps({"pid": process.pid, "birth": rows[process.pid][4]}) + "\n"
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
                print(
                    f"watcher TERM cleanup failed: {terminate_error}", file=sys.stderr
                )
                try:
                    process.kill()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                    print(f"watcher cleanup failed: {cleanup_error}", file=sys.stderr)
        try:
            (root / "startup-failure.json").write_text(
                json.dumps(sample_failure(progress, error), sort_keys=True) + "\n"
            )
        except OSError as receipt_error:
            print(
                f"could not retain startup failure receipt: {receipt_error}",
                file=sys.stderr,
            )
        raise


def watch_loop(root: Path) -> None:
    config = json.loads((root / "watch-config.json").read_text())
    worker_pid, birth = config["worker_pid"], config["worker_birth"]
    workspace = Path(config["workspace"])
    deadline_ns = config["deadline_ns"]
    peak_rss = peak_disk = peak_inodes = 0
    sample_count = 0
    sample_seconds = max_sample_seconds = 0.0
    progress: dict = {}
    (root / "watch-ready.json").write_text(json.dumps({"pid": os.getpid()}) + "\n")
    reason = "completed"
    try:
        while not (root / "watch-stop").exists():
            progress.clear()
            progress.update(
                root=None,
                phase="watch-process-inventory",
                last_path=None,
                started_ns=time.monotonic_ns(),
            )
            rows = process_rows()
            if worker_pid not in rows or rows[worker_pid][4] != birth:
                raise RuntimeError("hosted runner worker identity changed")
            owned_pids = descendants(rows, worker_pid) - {os.getpid()}
            rss = sum(
                rows[pid][2] for pid in owned_pids if not rows[pid][3].startswith("Z")
            )
            sweep_started = time.monotonic()
            storage_deadline = sweep_deadline(deadline_ns)
            disk, inodes = sample_declared_root(
                workspace,
                phase="watch-workspace",
                deadline=storage_deadline,
                required=True,
                progress=progress,
            )
            for extra in config["extra_storage"]:
                path = Path(extra["path"])
                current = sample_declared_root(
                    path,
                    phase="watch-ambient",
                    deadline=storage_deadline,
                    required=extra["required"],
                    progress=progress,
                )
                disk += max(0, current[0] - extra["bytes"])
                inodes += max(0, current[1] - extra["inodes"])
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
                raise RuntimeError(
                    f"whole-job cap: rss={rss} disk={disk} inodes={inodes}"
                )
            if time.time_ns() >= deadline_ns - SHUTDOWN_RESERVE_SECONDS * 1_000_000_000:
                raise TimeoutError("aggregate shutdown reserve reached")
            time.sleep(POLL_SECONDS)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        reason = f"{type(error).__name__}: {error}"
        (root / "watch-violation.json").write_text(
            json.dumps(
                {"reason": reason, **sample_failure(progress, error)}, sort_keys=True
            )
            + "\n"
        )
        rows = process_rows()
        if worker_pid in rows and rows[worker_pid][4] == birth:
            targets = {
                pid
                for pid in descendants(rows, worker_pid) - {worker_pid, os.getpid()}
                if "run_job.py" not in rows[pid][5]
            }
            identities = {pid: rows[pid][4] for pid in targets}
            for sig in (signal.SIGTERM, signal.SIGKILL):
                current = process_rows()
                for pid, started in identities.items():
                    if pid in current and current[pid][4] == started:
                        try:
                            os.kill(pid, sig)
                        except ProcessLookupError:
                            pass
                if sig == signal.SIGTERM:
                    time.sleep(GRACE_SECONDS)
    finally:
        (root / "watch-receipt.json").write_text(
            json.dumps(
                {
                    "reason": reason,
                    "peak_rss_bytes": peak_rss,
                    "peak_disk_bytes": peak_disk,
                    "peak_inodes": peak_inodes,
                    "sample_count": sample_count,
                    "sample_seconds": sample_seconds,
                    "max_sample_seconds": max_sample_seconds,
                },
                sort_keys=True,
            )
            + "\n"
        )


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
    pid, birth = _watcher_identity(root)
    (root / "watch-stop").write_text("stop\n")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (root / "watch-receipt.json").exists():
            receipt = json.loads((root / "watch-receipt.json").read_text())
            if receipt["reason"] != "completed":
                raise RuntimeError(f"CI watcher failed: {receipt['reason']}")
            return
        rows = process_rows()
        if pid not in rows or rows[pid][4] != birth:
            break
        time.sleep(0.1)
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
                    lingering = living | {
                        pid
                        for pid, birth in known.items()
                        if pid in rows
                        and rows[pid][4] == birth
                        and not rows[pid][3].startswith("Z")
                    }
                    if lingering:
                        reason = "leader exited with owned descendants"
                        break
                    reason = "completed" if rc == 0 else f"command exited {rc}"
                    break
                time.sleep(POLL_SECONDS)
        except BaseException as error:
            reason = f"monitor failure: {type(error).__name__}: {error}"
            sampling_failure = sample_failure(progress, error)
            raise
        finally:
            stop_owned(process, known, terminate=reason != "completed")
            receipt = {
                "phase": name,
                "command": command,
                "reason": reason,
                "returncode": process.returncode,
                "peak_rss_bytes": peak_rss,
                "peak_disk_bytes": peak_disk,
                "peak_inodes": peak_inodes,
                "observed_pids": sorted(known),
                "living_descendants": 0,
                "sampling_failure": sampling_failure,
            }
            (root / f"{name}.monitor.json").write_text(
                json.dumps(receipt, sort_keys=True) + "\n"
            )
    if reason != "completed":
        raise RuntimeError(f"{name}: {reason}; see {stdout_path} and {stderr_path}")
    return process.returncode or 0


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
    else:
        root.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "cache", "work", "pycache"):
        (root / name).mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        UV_CACHE_DIR=str(root / "cache"),
        UV_PROJECT_ENVIRONMENT=str(root / "venv"),
        TMPDIR=str(root / "tmp"),
        XDG_CACHE_HOME=str(root / "cache"),
        PYTHONPYCACHEPREFIX=str(root / "pycache"),
        SPARSELAB_WORK_DIR=str(root / "work"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
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
