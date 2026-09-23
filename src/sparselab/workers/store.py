"""Durable controller queue projection backed by the local ExperimentStore."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sparselab.training.metrics import ExperimentStore, _canonical_json

QUEUE_STATES = frozenset(
    {
        "QUEUED",
        "ASSIGNED",
        "RUNNING",
        "CANCEL_REQUESTED",
        "CANCELLED",
        "COMPLETE",
        "FAILED",
        "INTERRUPTED",
        "UNKNOWN",
    }
)
ACTIVE_QUEUE_STATES = frozenset({"ASSIGNED", "RUNNING", "CANCEL_REQUESTED"})
RECONCILABLE_QUEUE_STATES = ACTIVE_QUEUE_STATES | frozenset({"UNKNOWN"})
TERMINAL_QUEUE_STATES = frozenset({"CANCELLED", "COMPLETE", "FAILED", "INTERRUPTED"})
TERMINAL_RECEIPT_STATES = frozenset({"COMPLETE", "FAILED", "INTERRUPTED", "UNKNOWN"})


class ControllerStore:
    """Queue authority co-located with the controller's experiment projection.

    Queue writes are intentionally not replication events. Remote workers retain
    their own outbox authority; this database only records controller decisions
    and imports their immutable records through :class:`ExperimentStore`.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.metrics = ExperimentStore(root)
        self.path = self.metrics.path

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.metrics._connect() as con:
            try:
                con.execute("BEGIN IMMEDIATE")
                yield con
                con.commit()
            except BaseException:
                con.rollback()
                raise

    @staticmethod
    def _json(value: object) -> str:
        return _canonical_json(value)

    def save_worker(self, worker_id: str, record: Mapping[str, object]) -> None:
        with self.transaction() as con:
            con.execute(
                "INSERT INTO workers(worker_id,record_json) VALUES(?,?) "
                "ON CONFLICT(worker_id) DO UPDATE SET record_json=excluded.record_json",
                (worker_id, self._json(record)),
            )

    def worker_records(self) -> list[dict[str, object]]:
        with self.metrics._connect() as con:
            rows = con.execute(
                "SELECT record_json FROM workers ORDER BY worker_id"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def enqueue_many(self, entries: list[Mapping[str, object]]) -> None:
        """Atomically reserve every submission after all bundles are prepared."""
        with self.transaction() as con:
            for entry in entries:
                experiment_id = str(entry["experiment_id"])
                attempt_id = str(entry["attempt_id"])
                run_id = str(entry["run_id"])
                spec = entry["spec"]
                if not isinstance(spec, Mapping):
                    raise TypeError("experiment specification must be an object")
                con.execute(
                    "INSERT INTO experiments(experiment_id,specification_json,status,submitted_at) "
                    "VALUES(?,?,?,datetime('now'))",
                    (experiment_id, self._json(spec), "QUEUED"),
                )
                con.execute(
                    "INSERT INTO attempts("
                    "attempt_id,experiment_id,run_id,worker_id,status,receipt_json,"
                    "terminal_receipt_json,queued_reason,ingestion_status,ingestion_error"
                    ") VALUES(?,?,?,NULL,'QUEUED',NULL,NULL,NULL,'PENDING',NULL)",
                    (attempt_id, experiment_id, run_id),
                )

    def attempts(self, statuses: frozenset[str] | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT a.attempt_id,a.experiment_id,a.run_id,a.worker_id,a.status,"
            "a.receipt_json,a.terminal_receipt_json,a.queued_reason,"
            "a.ingestion_status,a.ingestion_error,e.specification_json,e.submitted_at "
            "FROM attempts a JOIN experiments e ON e.experiment_id=a.experiment_id"
        )
        params: tuple[object, ...] = ()
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE a.status IN ({placeholders})"
            params = tuple(sorted(statuses))
        query += " ORDER BY e.submitted_at,a.rowid"
        with self.metrics._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [
            {
                "attempt_id": row[0],
                "experiment_id": row[1],
                "run_id": row[2],
                "worker_id": row[3],
                "status": row[4],
                "receipt": json.loads(row[5]) if row[5] else None,
                "terminal_receipt": json.loads(row[6]) if row[6] else None,
                "queued_reason": row[7],
                "ingestion_status": row[8],
                "ingestion_error": row[9],
                "spec": json.loads(row[10]),
                "submitted_at": row[11],
            }
            for row in rows
        ]

    def attempt_by_run(self, run_id: str) -> dict[str, Any] | None:
        return next((row for row in self.attempts() if row["run_id"] == run_id), None)

    def assign(self, attempt_id: str, worker_id: str) -> bool:
        with self.transaction() as con:
            updated = con.execute(
                "UPDATE attempts SET worker_id=?,status='ASSIGNED',queued_reason=NULL "
                "WHERE attempt_id=? AND status='QUEUED'",
                (worker_id, attempt_id),
            ).rowcount
            if updated:
                con.execute(
                    "UPDATE experiments SET status='ASSIGNED' WHERE experiment_id="
                    "(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                    (attempt_id,),
                )
            return updated == 1

    def set_queued_reason(self, attempt_id: str, reason: str | None) -> None:
        with self.transaction() as con:
            con.execute(
                "UPDATE attempts SET queued_reason=? WHERE attempt_id=? AND status='QUEUED'",
                (reason, attempt_id),
            )

    def set_receipt(
        self, attempt_id: str, receipt: Mapping[str, object], status: str
    ) -> None:
        """Persist worker evidence without allowing stale nonterminal regressions."""
        if status not in QUEUE_STATES:
            raise ValueError(f"invalid queue status: {status}")
        encoded = self._json(receipt)
        receipt_state = receipt.get("state")
        if receipt_state is not None and receipt_state not in {
            "PREPARED",
            "RUNNING",
            "COMPLETE",
            "FAILED",
            "INTERRUPTED",
            "UNKNOWN",
        }:
            raise ValueError("invalid attempt receipt state")
        terminal = receipt_state in TERMINAL_RECEIPT_STATES
        with self.transaction() as con:
            current = con.execute(
                "SELECT status,terminal_receipt_json FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"unknown attempt: {attempt_id}")
            if current[1] is not None:
                # Worker terminal receipts are monotonic.  A delayed response
                # cannot revise the first durable terminal evidence.
                return
            if current[0] == "CANCEL_REQUESTED" and not terminal:
                status = "CANCEL_REQUESTED"
            con.execute(
                "UPDATE attempts SET receipt_json=?,terminal_receipt_json="
                "CASE WHEN ? THEN ? ELSE terminal_receipt_json END,"
                "status=?,ingestion_status=CASE WHEN ? THEN 'PENDING' "
                "ELSE ingestion_status END,ingestion_error=CASE WHEN ? THEN NULL "
                "ELSE ingestion_error END WHERE attempt_id=?",
                (encoded, terminal, encoded, status, terminal, terminal, attempt_id),
            )
            con.execute(
                "UPDATE experiments SET status=? WHERE experiment_id="
                "(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                (status, attempt_id),
            )

    def mark_ingestion_complete(self, attempt_id: str) -> None:
        with self.transaction() as con:
            con.execute(
                "UPDATE attempts SET ingestion_status='COMPLETE',ingestion_error=NULL "
                "WHERE attempt_id=? AND terminal_receipt_json IS NOT NULL",
                (attempt_id,),
            )

    def mark_ingestion_not_required(self, attempt_id: str) -> None:
        with self.transaction() as con:
            con.execute(
                "UPDATE attempts SET ingestion_status='NOT_REQUIRED',ingestion_error=NULL "
                "WHERE attempt_id=? AND terminal_receipt_json IS NOT NULL",
                (attempt_id,),
            )

    def mark_ingestion_error(self, attempt_id: str, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        with self.transaction() as con:
            con.execute(
                "UPDATE attempts SET ingestion_status='ERROR',ingestion_error=? "
                "WHERE attempt_id=? AND terminal_receipt_json IS NOT NULL",
                (message, attempt_id),
            )

    def pending_ingestion(self) -> list[dict[str, Any]]:
        return [
            attempt
            for attempt in self.attempts()
            if attempt["terminal_receipt"] is not None
            and attempt["ingestion_status"] in {"PENDING", "ERROR"}
        ]

    def mark_unknown_if_nonterminal(self, attempt_id: str, reason: str) -> None:
        """Record controller reachability without replacing worker receipt evidence."""
        with self.transaction() as con:
            row = con.execute(
                "SELECT terminal_receipt_json FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown attempt: {attempt_id}")
            if row[0] is not None:
                return
            con.execute(
                "UPDATE attempts SET status=CASE WHEN status='CANCEL_REQUESTED' "
                "THEN status ELSE 'UNKNOWN' END,queued_reason=? WHERE attempt_id=?",
                (reason, attempt_id),
            )
            con.execute(
                "UPDATE experiments SET status=(SELECT status FROM attempts WHERE attempt_id=?) "
                "WHERE experiment_id=(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                (attempt_id, attempt_id),
            )

    def request_cancel(self, run_id: str) -> dict[str, Any]:
        with self.transaction() as con:
            row = con.execute(
                "SELECT attempt_id,status,terminal_receipt_json FROM attempts WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            attempt_id, status, terminal_receipt = row
            if status == "QUEUED":
                target = "CANCELLED"
                con.execute(
                    "UPDATE attempts SET ingestion_status='NOT_REQUIRED',ingestion_error=NULL "
                    "WHERE attempt_id=?",
                    (attempt_id,),
                )
            elif status in TERMINAL_QUEUE_STATES or terminal_receipt is not None:
                target = status
            else:
                target = "CANCEL_REQUESTED"
            con.execute(
                "UPDATE attempts SET status=? WHERE attempt_id=?", (target, attempt_id)
            )
            con.execute(
                "UPDATE experiments SET status=? WHERE experiment_id="
                "(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                (target, attempt_id),
            )
            return {"attempt_id": attempt_id, "status": target}

    def import_records(self, records: list[Mapping[str, object] | str | bytes]) -> None:
        self.metrics.import_records(records)

    def imported_sequence(self, origin_id: str) -> int:
        """Return the committed contiguous replication watermark for one origin."""
        with self.metrics._connect() as con:
            return int(
                con.execute(
                    "SELECT COALESCE(MAX(sequence),0) FROM ingested_records WHERE origin_id=?",
                    (origin_id,),
                ).fetchone()[0]
            )
