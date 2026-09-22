"""Short-lived, read-only SQLite queries for the local dashboard."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    status: str
    parent_run_id: str | None
    updated_at: str
    config: dict[str, object]


def _connect(root: Path) -> sqlite3.Connection:
    database = root / "experiments.sqlite3"
    return sqlite3.connect(f"file:{database}?mode=ro", uri=True)


def runs(root: Path) -> list[RunRecord]:
    with _connect(root) as connection:
        rows = connection.execute(
            "SELECT run_id,name,status,parent_run_id,updated_at,config_json FROM runs ORDER BY updated_at DESC"
        ).fetchall()
    return [RunRecord(*row[:5], json.loads(row[5])) for row in rows]


def metrics(root: Path, run_ids: list[str]) -> list[dict[str, object]]:
    if not run_ids:
        return []
    marks = ",".join("?" * len(run_ids))
    with _connect(root) as connection:
        rows = connection.execute(
            f"SELECT run_id,step,tokens_seen,wall_time,name,value FROM metrics WHERE run_id IN ({marks}) ORDER BY step",
            run_ids,
        ).fetchall()
    return [
        dict(
            zip(
                ("run_id", "step", "tokens_seen", "wall_time", "name", "value"),
                row,
                strict=True,
            )
        )
        for row in rows
    ]


def events(root: Path, run_ids: list[str]) -> list[dict[str, object]]:
    if not run_ids:
        return []
    marks = ",".join("?" * len(run_ids))
    with _connect(root) as connection:
        rows = connection.execute(
            f"SELECT run_id,step,tokens_seen,wall_time,kind,payload_json FROM events WHERE run_id IN ({marks}) ORDER BY id",
            run_ids,
        ).fetchall()
    return [
        dict(
            zip(
                ("run_id", "step", "tokens_seen", "wall_time", "kind", "payload_json"),
                row,
                strict=True,
            )
        )
        for row in rows
    ]
