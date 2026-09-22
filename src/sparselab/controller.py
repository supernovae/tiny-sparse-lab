"""Controller-local durable queue for independent worker attempts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class QueueAttempt:
    attempt_id: str
    experiment_id: str
    run_id: str
    worker_id: str | None
    status: str


class Controller:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "controller.sqlite3"
        with self._connect() as c:
            c.executescript(
                "CREATE TABLE IF NOT EXISTS experiments(experiment_id TEXT PRIMARY KEY,specification_json TEXT NOT NULL,status TEXT NOT NULL,submitted_at TEXT NOT NULL); CREATE TABLE IF NOT EXISTS attempts(attempt_id TEXT PRIMARY KEY,experiment_id TEXT NOT NULL,run_id TEXT UNIQUE NOT NULL,worker_id TEXT,status TEXT NOT NULL,receipt_json TEXT); CREATE TABLE IF NOT EXISTS workers(worker_id TEXT PRIMARY KEY,record_json TEXT NOT NULL);"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def register_worker(self, worker_id: str, record: object) -> None:
        with self._connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO workers VALUES(?,?)",
                (worker_id, json.dumps(record, default=str, sort_keys=True)),
            )

    def submit(
        self, specification: object, *, worker_id: str | None = None
    ) -> QueueAttempt:
        experiment_id, attempt_id, run_id = (
            uuid.uuid4().hex,
            uuid.uuid4().hex,
            uuid.uuid4().hex,
        )
        with self._connect() as c:
            c.execute(
                "INSERT INTO experiments VALUES(?,?,?,?)",
                (
                    experiment_id,
                    json.dumps(specification, default=str, sort_keys=True),
                    "QUEUED",
                    datetime.now(UTC).isoformat(),
                ),
            )
            c.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,NULL)",
                (attempt_id, experiment_id, run_id, worker_id, "QUEUED"),
            )
        return QueueAttempt(attempt_id, experiment_id, run_id, worker_id, "QUEUED")

    def attempts(self) -> list[QueueAttempt]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT attempt_id,experiment_id,run_id,worker_id,status FROM attempts ORDER BY rowid"
            ).fetchall()
        return [QueueAttempt(*row) for row in rows]

    def assign(self, attempt_id: str, worker_id: str) -> QueueAttempt:
        with self._connect() as c:
            row = c.execute(
                "SELECT experiment_id,run_id,status FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None or row[2] != "QUEUED":
                raise ValueError("attempt is not queueable")
            c.execute(
                "UPDATE attempts SET worker_id=?,status=? WHERE attempt_id=?",
                (worker_id, "ASSIGNED", attempt_id),
            )
        return QueueAttempt(attempt_id, row[0], row[1], worker_id, "ASSIGNED")

    def submit_matrix(
        self, config: dict[str, object], matrix_path: Path, *, max_runs: int = 1000
    ) -> list[QueueAttempt]:
        from sparselab.experiments.matrix import apply_patch, expand

        attempts = []
        for coordinate, patch in expand(matrix_path, max_runs):
            attempts.append(
                self.submit(
                    {"matrix_coordinate": coordinate, "config": apply_patch(config, patch)}
                )
            )
        return attempts
