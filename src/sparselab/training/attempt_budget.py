"""Persistent, conservative update and wall-time limits for bounded test attempts.

Reservations charge the full declared maximum before work starts. Failed or
interrupted attempts keep that charge, so retries cannot regain uncertain updates.
This does not replace native run counters or checkpoint verification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


class AttemptBudgetError(RuntimeError):
    """The shared attempt budget is missing, invalid, or exhausted."""


class AttemptBudget:
    def __init__(self, path: Path) -> None:
        if not path.is_absolute() or path.is_symlink():
            raise AttemptBudgetError("budget path must be absolute and not a symlink")
        self.path = path

    @classmethod
    def create(
        cls, path: Path, *, max_updates: int, max_wall_seconds: float
    ) -> AttemptBudget:
        if type(max_updates) is not int or max_updates <= 0:
            raise ValueError("max_updates must be a positive integer")
        if (
            isinstance(max_wall_seconds, bool)
            or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds)
            or max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be positive and finite")
        budget = cls(path)
        if not path.parent.is_dir():
            raise AttemptBudgetError("budget parent directory must already exist")
        started_ns = time.time_ns()
        deadline_ns = started_ns + int(max_wall_seconds * 1_000_000_000)
        if deadline_ns <= started_ns or deadline_ns > 2**63 - 1:
            raise ValueError("max_wall_seconds is outside the ledger clock range")
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
        with budget._connect() as connection:
            connection.execute(
                "CREATE TABLE budget ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), "
                "version INTEGER NOT NULL, max_updates INTEGER NOT NULL, "
                "used_updates INTEGER NOT NULL, started_ns INTEGER NOT NULL, "
                "deadline_ns INTEGER NOT NULL, last_checked_ns INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE reservations ("
                "id INTEGER PRIMARY KEY, label TEXT NOT NULL, "
                "updates INTEGER NOT NULL, reserved_ns INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT INTO budget VALUES (1, 1, ?, 0, ?, ?, ?)",
                (max_updates, started_ns, deadline_ns, started_ns),
            )
        return budget

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file() or self.path.is_symlink():
            raise AttemptBudgetError("budget ledger is missing or is a symlink")
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @staticmethod
    def _row(connection: sqlite3.Connection) -> tuple[int, int, int, int, int]:
        try:
            row = connection.execute(
                "SELECT version, max_updates, used_updates, deadline_ns, "
                "last_checked_ns FROM budget WHERE id = 1"
            ).fetchone()
        except sqlite3.DatabaseError as error:
            raise AttemptBudgetError("invalid budget ledger") from error
        if (
            row is None
            or len(row) != 5
            or any(type(item) is not int for item in row)
            or row[0] != 1
            or row[1] <= 0
            or not 0 <= row[2] <= row[1]
            or row[3] <= row[4]
        ):
            raise AttemptBudgetError("invalid or expired budget ledger")
        charged = connection.execute(
            "SELECT COALESCE(SUM(updates), 0) FROM reservations"
        ).fetchone()
        if charged is None or charged[0] != row[2]:
            raise AttemptBudgetError("reservation total disagrees with budget ledger")
        return row

    def _check(self, connection: sqlite3.Connection) -> tuple[int, int, int]:
        _, maximum, used, deadline_ns, last_checked_ns = self._row(connection)
        now_ns = time.time_ns()
        if now_ns < last_checked_ns:
            raise AttemptBudgetError("clock moved backwards; budget fails closed")
        if now_ns >= deadline_ns:
            raise AttemptBudgetError("shared wall-time limit reached")
        connection.execute(
            "UPDATE budget SET last_checked_ns = ? WHERE id = 1", (now_ns,)
        )
        return maximum, used, deadline_ns - now_ns

    def remaining_seconds(self) -> float:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _, _, remaining_ns = self._check(connection)
        return remaining_ns / 1_000_000_000

    def reserve(self, label: str, updates: int) -> int:
        if not label.strip() or type(updates) is not int or updates < 0:
            raise ValueError("reservation requires a label and nonnegative updates")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            maximum, used, _ = self._check(connection)
            if used + updates > maximum:
                raise AttemptBudgetError(
                    f"shared update limit: {used} charged + {updates} requested "
                    f"> {maximum} approved"
                )
            connection.execute(
                "UPDATE budget SET used_updates = ? WHERE id = 1", (used + updates,)
            )
            connection.execute(
                "INSERT INTO reservations (label, updates, reserved_ns) "
                "VALUES (?, ?, ?)",
                (label, updates, time.time_ns()),
            )
        return maximum - used - updates

    def status(self) -> dict[str, object]:
        with self._connect() as connection:
            _, maximum, used, deadline_ns, _ = self._row(connection)
            started_ns = connection.execute(
                "SELECT started_ns FROM budget WHERE id = 1"
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT label, updates FROM reservations ORDER BY id"
            ).fetchall()
        return {
            "max_updates": maximum,
            "charged_updates": used,
            "remaining_updates": maximum - used,
            "max_wall_seconds": (deadline_ns - started_ns) / 1e9,
            "started_at_utc": datetime.fromtimestamp(started_ns / 1e9, UTC).isoformat(),
            "deadline_utc": datetime.fromtimestamp(deadline_ns / 1e9, UTC).isoformat(),
            "remaining_wall_seconds": max(0.0, (deadline_ns - time.time_ns()) / 1e9),
            "reservations": [
                {"label": label, "updates": updates} for label, updates in rows
            ],
        }

    def run(self, command: list[str], *, reserve_updates: int = 0) -> int:
        if not command:
            raise ValueError("command is required")
        self.reserve("command: " + " ".join(command), reserve_updates)
        self.remaining_seconds()
        environment = os.environ.copy()
        environment["SPARSELAB_ATTEMPT_BUDGET_LEDGER"] = str(self.path)
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        try:
            return process.wait(timeout=self.remaining_seconds())
        except subprocess.TimeoutExpired, AttemptBudgetError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise AttemptBudgetError("shared wall-time limit reached") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--path", type=Path, required=True)
    initialize.add_argument("--max-updates", type=int, required=True)
    initialize.add_argument("--max-wall-seconds", type=float, required=True)
    inspect = commands.add_parser("status")
    inspect.add_argument("--path", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--path", type=Path, required=True)
    execute.add_argument("--reserve-updates", type=int, default=0)
    execute.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            budget = AttemptBudget.create(
                args.path,
                max_updates=args.max_updates,
                max_wall_seconds=args.max_wall_seconds,
            )
        else:
            budget = AttemptBudget(args.path)
        if args.action == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            return budget.run(command, reserve_updates=args.reserve_updates)
        print(json.dumps(budget.status(), sort_keys=True))
        return 0
    except (
        AttemptBudgetError,
        OSError,
        sqlite3.DatabaseError,
        ValueError,
    ) as error:
        print(f"attempt budget: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
