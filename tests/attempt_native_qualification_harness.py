"""Test-local enclosing CPU allocation; run only under the approved C05-Q1 scope."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from sparselab.operational_monitor import (
    capture_workspace_baseline,
    load_monitor_policy,
    monitor_command,
    sample_workspace_tree,
)

GIB = 1024**3
ROOT_RSS_HEADROOM = 512 * 1024**2
SUBTREE_RSS_CAP = 4 * GIB - ROOT_RSS_HEADROOM
TOTAL_SECONDS = 600


def _save(path: Path, value: dict) -> None:
    with path.open("x") as output:
        json.dump(value, output, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _worker(root: Path) -> int:
    checkout = Path(__file__).resolve().parent.parent
    precheck = [
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(root / "pytest-precheck"),
        "tests/test_attempt_campaign_guard.py",
        "tests/test_attempt_native_receipts.py",
        "tests/test_attempt_contract_cli.py",
    ]
    integration = [
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "pytest",
        "-vv",
        "--tb=long",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(root / "pytest-integration"),
        "tests/test_attempt_native_integration.py::test_bounded_cpu_final_mask_and_ledger",
    ]
    for name, command in (("precheck", precheck), ("integration", integration)):
        result = subprocess.run(command, cwd=checkout, check=False)
        _save(
            root / f"{name}-command.json",
            {"argv": command, "returncode": result.returncode},
        )
        if result.returncode != 0:
            return result.returncode
    return 0


def _deadline(_signum: int, _frame: object) -> None:
    raise TimeoutError("600-second aggregate deadline approaching; stop allocation")


def _run(root: Path) -> int:
    started = time.monotonic()
    old_handler = signal.signal(signal.SIGALRM, _deadline)
    # Leave five seconds for the native supervisor's TERM-to-KILL and receipts.
    signal.setitimer(signal.ITIMER_REAL, 595)
    try:
        root = root.absolute()
        if root.exists():
            raise FileExistsError("qualification root must be unused")
        root.parent.mkdir(parents=True, exist_ok=True)
        stats = os.statvfs(root.parent)
        if stats.f_bavail * stats.f_frsize < 2 * GIB or stats.f_favail < 2000:
            raise RuntimeError("insufficient free-space or inode safety margin")
        root.mkdir(mode=0o700)
        baseline_path = root / "baseline.json"
        baseline = capture_workspace_baseline(root, baseline_path)
        policy_path = root / "outer-policy.json"
        remaining = TOTAL_SECONDS - (time.monotonic() - started)
        if remaining <= 25:
            raise TimeoutError("setup exhausted the aggregate allowance")
        _save(
            policy_path,
            {
                "monitor_policy_version": 1,
                "max_tree_rss_bytes": SUBTREE_RSS_CAP,
                "max_added_workspace_bytes": GIB,
                "max_added_workspace_inodes": 1000,
                "max_wall_seconds": min(575, remaining - 20),
                "termination_grace_seconds": 3,
                "interval_seconds": 0.2,
            },
        )
        (root / "tmp").mkdir()
        os.environ.update(
            KML_QUAL_ROOT=str(root),
            SPARSELAB_WORK_DIR=str(root),
            TMPDIR=str(root / "tmp"),
            XDG_CACHE_HOME=str(root / "xdg-cache"),
            UV_CACHE_DIR=str(root / "uv-cache"),
            UV_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            CUDA_VISIBLE_DEVICES="",
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
            WANDB_MODE="offline",
        )
        if psutil.Process().memory_info().rss >= ROOT_RSS_HEADROOM:
            raise RuntimeError("harness RSS exceeds reserved headroom")
        result = monitor_command(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--root",
                str(root),
            ],
            load_monitor_policy(policy_path),
            workspace=root,
            log_dir=root / "outer-monitor",
            policy_path=policy_path,
            baseline_path=baseline_path,
            cwd=Path(__file__).resolve().parent.parent,
        )
        owned = json.loads(
            (root / "outer-monitor" / "owned-completion.json").read_text()
        )
        elapsed = time.monotonic() - started
        summary = {
            "format": "c05-native-cpu-allocation-v1",
            "status": result.status,
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "baseline_sha256": baseline.sha256,
            "peak_subtree_rss_bytes": result.peak_tree_rss_bytes,
            "peak_added_workspace_bytes": result.peak_added_workspace_bytes,
            "peak_added_workspace_inodes": result.peak_added_workspace_inodes,
            "living_descendants": owned["living_descendants"],
            "violations": result.violations,
        }
        _save(root / "outer-allocation-summary.json", summary)
        final_bytes, final_inodes = sample_workspace_tree(root)
        if (
            result.status != "COMPLETE"
            or result.returncode != 0
            or owned["living_descendants"] != 0
            or elapsed >= TOTAL_SECONDS
            or result.peak_tree_rss_bytes is None
            or result.peak_tree_rss_bytes > SUBTREE_RSS_CAP
            or final_bytes - baseline.apparent_bytes > GIB
            or final_inodes - baseline.inodes > 1000
            or psutil.Process().memory_info().rss >= ROOT_RSS_HEADROOM
        ):
            raise RuntimeError(
                "aggregate CPU qualification failed; inspect retained receipts"
            )
        print(json.dumps(summary, sort_keys=True))
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    return _worker(args.root) if args.worker else _run(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
