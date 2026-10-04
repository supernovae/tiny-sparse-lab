"""CLI adapter for the independent operational monitor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparselab.operational_monitor import load_monitor_policy, monitor_command


def register_monitor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `sparselab monitor` on the top-level CLI's subparsers."""
    parser = subparsers.add_parser(
        "monitor", help="Guard one owned command and record operational evidence"
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--reserve-inodes", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "monitored_command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...]"
    )
    parser.set_defaults(handler=_handle)


def execute_monitor(args: argparse.Namespace) -> int:
    """Print final typed completion and propagate command/guard failure to CLI exit."""
    command = list(args.monitored_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("monitor requires -- COMMAND [ARG ...]")
    result = monitor_command(
        command,
        load_monitor_policy(args.policy),
        workspace=args.workspace,
        log_dir=args.log_dir,
        policy_path=args.policy,
        cwd=args.cwd,
        reserved_bytes=args.reserve_bytes,
        reserved_inodes=args.reserve_inodes,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return (
        result.returncode
        if result.status in {"COMPLETE", "FAILED"}
        and result.returncode is not None
        and result.returncode >= 0
        else 1
    )


def _handle(args: argparse.Namespace) -> None:
    raise SystemExit(execute_monitor(args))
