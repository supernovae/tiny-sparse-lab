from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping
from pathlib import Path


class ExperimentStore:
    def __init__(self, root_dir: Path) -> None:
        root_dir.mkdir(parents=True, exist_ok=True)
        self.path = root_dir / "experiments.sqlite3"
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA foreign_keys=ON")
            con.executescript(
                "CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,name TEXT NOT NULL,status TEXT NOT NULL,parent_run_id TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,config_json TEXT NOT NULL,metadata_json TEXT NOT NULL,latest_checkpoint TEXT); CREATE TABLE IF NOT EXISTS metrics(run_id TEXT NOT NULL,step INTEGER NOT NULL,tokens_seen INTEGER NOT NULL,wall_time REAL NOT NULL,name TEXT NOT NULL,value REAL NOT NULL,PRIMARY KEY(run_id,step,name)); CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,run_id TEXT NOT NULL,step INTEGER NOT NULL,tokens_seen INTEGER NOT NULL,wall_time REAL NOT NULL,kind TEXT NOT NULL,payload_json TEXT NOT NULL);"
            )
            con.execute("PRAGMA user_version=1")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=5)
        con.execute("PRAGMA busy_timeout=5000")
        return con

    def create_run(
        self,
        run_id: str,
        config: object,
        metadata: object,
        parent_run_id: str | None = None,
    ) -> None:
        with self._connect() as c:
            c.execute(
                "INSERT INTO runs VALUES(?,?,'running',?,datetime('now'),datetime('now'),?,?,NULL)",
                (
                    run_id,
                    getattr(config, "name", run_id),
                    parent_run_id,
                    json.dumps(config, default=str, sort_keys=True),
                    json.dumps(metadata, default=str, sort_keys=True),
                ),
            )

    def log_metrics(
        self,
        run_id: str,
        step: int,
        tokens_seen: int,
        wall_time: float,
        values: Mapping[str, float],
    ) -> None:
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError("nonfinite metric")
        with self._connect() as c:
            c.executemany(
                "INSERT INTO metrics VALUES(?,?,?,?,?,?)",
                [
                    (run_id, step, tokens_seen, wall_time, k, v)
                    for k, v in values.items()
                ],
            )
            c.execute(
                "UPDATE runs SET updated_at=datetime('now') WHERE run_id=?", (run_id,)
            )

    def log_event(
        self,
        run_id: str,
        step: int,
        tokens_seen: int,
        wall_time: float,
        kind: str,
        payload: object,
    ) -> None:
        with self._connect() as c:
            c.execute(
                "INSERT INTO events(run_id,step,tokens_seen,wall_time,kind,payload_json) VALUES(?,?,?,?,?,?)",
                (run_id, step, tokens_seen, wall_time, kind, json.dumps(payload)),
            )

    def finish_run(
        self, run_id: str, status: str, checkpoint: str | None = None
    ) -> None:
        with self._connect() as c:
            c.execute(
                "UPDATE runs SET status=?,latest_checkpoint=?,updated_at=datetime('now') WHERE run_id=?",
                (status, checkpoint, run_id),
            )
