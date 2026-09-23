from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4
ENVELOPE_SCHEMA_VERSION = 1
MAX_ENVELOPE_BYTES = 65_536


def _canonical_json(value: object) -> str:
    """Encode persisted protocol data without implementation-dependent whitespace."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(value: str | bytes) -> object:
    try:
        parsed = json.loads(value, object_pairs_hook=_reject_duplicate_keys)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid canonical JSON") from error
    try:
        _canonical_json(parsed)
    except (TypeError, ValueError) as error:
        raise ValueError("nonfinite or noncanonical JSON value") from error
    return parsed


def _envelope_digest(envelope: Mapping[str, object]) -> str:
    body = {key: value for key, value in envelope.items() if key != "sha256"}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _execute_ddl(con: sqlite3.Connection, script: str) -> None:
    """Execute our fixed DDL without executescript's implicit transaction commit."""
    for statement in script.split(";"):
        if statement.strip():
            con.execute(statement)


class ExperimentStore:
    """The local durable experiment projection and its append-only replication outbox."""

    def __init__(self, root_dir: Path) -> None:
        root_dir.mkdir(parents=True, exist_ok=True)
        self._existed_at_open = (root_dir / "experiments.sqlite3").exists()
        self.path = root_dir / "experiments.sqlite3"
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=5)
        try:
            con.execute("PRAGMA busy_timeout=5000")
            con.execute("PRAGMA foreign_keys=ON")
            with con:
                yield con
        finally:
            con.close()

    def _initialize(self) -> None:
        with self._connect() as con:
            version = int(con.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported experiment-store schema version: {version}"
                )
            con.execute("PRAGMA journal_mode=WAL")
            integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError(
                    f"experiment-store integrity check failed: {integrity}"
                )
            if version == SCHEMA_VERSION:
                # Schema 4 was not published before the controller queue landed.
                # Bring an earlier local schema-4 queue projection forward
                # transactionally without changing the public schema number.
                con.execute("BEGIN EXCLUSIVE")
                try:
                    self._create_v4_tables(con)
                    self._require_schema(con)
                    con.commit()
                except BaseException:
                    con.rollback()
                    raise
                return
            con.execute("BEGIN EXCLUSIVE")
            try:
                version = int(con.execute("PRAGMA user_version").fetchone()[0])
                if version == SCHEMA_VERSION:
                    self._require_schema(con)
                    con.commit()
                    return
                self._backup(con, version)
                self._assert_no_orphans(con)
                self._migrate(con, version)
                con.commit()
            except BaseException:
                con.rollback()
                raise

    def _backup(self, con: sqlite3.Connection, version: int) -> None:
        # A migration never mutates the only copy of an existing database.
        if not self._existed_at_open:
            return
        backup = self.path.with_name(f"{self.path.name}.v{version}.bak")
        if backup.exists():
            backup = backup.with_name(f"{backup.name}.{uuid.uuid4().hex}")
        if not con.in_transaction:
            raise RuntimeError("migration backup requires exclusive writer ownership")
        temporary = backup.with_name(f".{backup.name}.{uuid.uuid4().hex}.tmp")
        try:
            # Backing up the connection holding BEGIN EXCLUSIVE can wait on itself.
            with (
                closing(
                    sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
                ) as source,
                closing(sqlite3.connect(temporary)) as target,
            ):
                source.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("migration backup failed integrity checking")
            temporary.replace(backup)
        finally:
            temporary.unlink(missing_ok=True)

    def _assert_no_orphans(self, con: sqlite3.Connection) -> None:
        tables = {
            str(row[0])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "runs" not in tables:
            return
        for table in (
            "metrics",
            "events",
            "stage_history",
            "checkpoints",
            "manifests",
            "calibration",
        ):
            if table not in tables:
                continue
            orphan = con.execute(
                f"SELECT {table}.run_id FROM {table} "
                f"LEFT JOIN runs ON runs.run_id={table}.run_id "
                "WHERE runs.run_id IS NULL LIMIT 1"
            ).fetchone()
            if orphan is not None:
                raise ValueError(f"legacy orphan in {table}: run_id={orphan[0]!r}")
        foreign_key_errors = con.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise ValueError(f"foreign-key check failed: {foreign_key_errors[0]}")

    def _migrate(self, con: sqlite3.Connection, version: int) -> None:
        # v1 is the historical runs/metrics/events shape; v2 added stages.  A
        # nominal v2 database is deliberately upgraded to v3 rather than being
        # treated as if its missing replication tables already existed.
        if version == 0:
            self._create_legacy_tables(con)
            con.execute("PRAGMA user_version=1")
            version = 1
        if version == 1:
            con.execute(
                "CREATE TABLE IF NOT EXISTS stage_history("
                "run_id TEXT NOT NULL,sequence INTEGER NOT NULL,stage TEXT NOT NULL,"
                "status TEXT NOT NULL,step INTEGER NOT NULL,tokens_seen INTEGER NOT NULL,"
                "started_at TEXT NOT NULL,finished_at TEXT,payload_json TEXT NOT NULL,"
                "PRIMARY KEY(run_id,sequence),"
                "FOREIGN KEY(run_id) REFERENCES runs(run_id))"
            )
            con.execute("PRAGMA user_version=2")
            version = 2
        if version == 2:
            self._create_v3_tables(con)
            con.execute("PRAGMA user_version=3")
            version = 3
        if version != 3:
            raise ValueError(
                f"cannot migrate experiment-store schema version: {version}"
            )
        self._create_v4_tables(con)
        con.execute("PRAGMA user_version=4")

    def _create_legacy_tables(self, con: sqlite3.Connection) -> None:
        _execute_ddl(
            con,
            "CREATE TABLE IF NOT EXISTS runs("
            "run_id TEXT PRIMARY KEY,name TEXT NOT NULL,status TEXT NOT NULL,"
            "parent_run_id TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,"
            "config_json TEXT NOT NULL,metadata_json TEXT NOT NULL,latest_checkpoint TEXT);"
            "CREATE TABLE IF NOT EXISTS metrics("
            "run_id TEXT NOT NULL,step INTEGER NOT NULL,tokens_seen INTEGER NOT NULL,"
            "wall_time REAL NOT NULL,name TEXT NOT NULL,value REAL NOT NULL,"
            "PRIMARY KEY(run_id,step,name),"
            "FOREIGN KEY(run_id) REFERENCES runs(run_id));"
            "CREATE TABLE IF NOT EXISTS events("
            "id INTEGER PRIMARY KEY,run_id TEXT NOT NULL,step INTEGER NOT NULL,"
            "tokens_seen INTEGER NOT NULL,wall_time REAL NOT NULL,kind TEXT NOT NULL,"
            "payload_json TEXT NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(run_id));",
        )

    def _create_v3_tables(self, con: sqlite3.Connection) -> None:
        _execute_ddl(
            con,
            "CREATE TABLE IF NOT EXISTS manifests("
            "run_id TEXT PRIMARY KEY,digest TEXT NOT NULL,json TEXT NOT NULL,"
            "FOREIGN KEY(run_id) REFERENCES runs(run_id));"
            "CREATE TABLE IF NOT EXISTS checkpoints("
            "run_id TEXT NOT NULL,checkpoint_id TEXT NOT NULL,relative_path TEXT NOT NULL,"
            "digest TEXT NOT NULL,step INTEGER NOT NULL,tokens_seen INTEGER NOT NULL,"
            "created_at TEXT NOT NULL,size_bytes INTEGER NOT NULL,validation_loss REAL,"
            "verified_at TEXT,verification_status TEXT NOT NULL,resume_level TEXT NOT NULL,"
            "backend TEXT NOT NULL,PRIMARY KEY(run_id,checkpoint_id),"
            "FOREIGN KEY(run_id) REFERENCES runs(run_id));"
            "CREATE INDEX IF NOT EXISTS checkpoints_run_step ON checkpoints(run_id,step);"
            "CREATE TABLE IF NOT EXISTS calibration("
            "key_hash TEXT NOT NULL,run_id TEXT NOT NULL,observation_json TEXT NOT NULL,"
            "PRIMARY KEY(key_hash,run_id),FOREIGN KEY(run_id) REFERENCES runs(run_id));"
            "CREATE TABLE IF NOT EXISTS store_metadata("
            "singleton INTEGER PRIMARY KEY CHECK(singleton=1),origin_id TEXT NOT NULL,"
            "next_sequence INTEGER NOT NULL CHECK(next_sequence >= 1));"
            "CREATE TABLE IF NOT EXISTS outbox("
            "sequence INTEGER PRIMARY KEY,run_id TEXT NOT NULL,record_json TEXT NOT NULL,"
            "record_sha256 TEXT NOT NULL UNIQUE,"
            "FOREIGN KEY(run_id) REFERENCES runs(run_id));"
            "CREATE TABLE IF NOT EXISTS ingested_records("
            "origin_id TEXT NOT NULL,sequence INTEGER NOT NULL,sha256 TEXT NOT NULL,"
            "PRIMARY KEY(origin_id,sequence));"
            "CREATE INDEX IF NOT EXISTS ingested_records_origin_sequence "
            "ON ingested_records(origin_id,sequence);",
        )
        metadata = con.execute(
            "SELECT origin_id,next_sequence FROM store_metadata WHERE singleton=1"
        ).fetchone()
        if metadata is None:
            con.execute(
                "INSERT INTO store_metadata(singleton,origin_id,next_sequence) VALUES(1,?,1)",
                (str(uuid.uuid4()),),
            )
        elif int(metadata[1]) < 1:
            raise ValueError("invalid experiment-store sequence metadata")

    def _create_v4_tables(self, con: sqlite3.Connection) -> None:
        """Create the controller-local durable queue projection.

        These tables deliberately contain no foreign keys into a remote worker
        store: only the controller opens this database and projects remote
        records through ``import_records``.  Receipt evidence and its controller
        ingestion are distinct durable facts: a disconnect after a worker has
        finished must not erase the finished receipt.
        """
        _execute_ddl(
            con,
            "CREATE TABLE IF NOT EXISTS workers("
            "worker_id TEXT PRIMARY KEY,record_json TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS experiments("
            "experiment_id TEXT PRIMARY KEY,specification_json TEXT NOT NULL,"
            "status TEXT NOT NULL,submitted_at TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS attempts("
            "attempt_id TEXT PRIMARY KEY,experiment_id TEXT NOT NULL,"
            "run_id TEXT NOT NULL UNIQUE,worker_id TEXT,status TEXT NOT NULL,"
            "receipt_json TEXT,terminal_receipt_json TEXT,queued_reason TEXT,"
            "ingestion_status TEXT NOT NULL DEFAULT 'PENDING',ingestion_error TEXT,"
            "FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id));"
            "CREATE INDEX IF NOT EXISTS attempts_status ON attempts(status);"
            "CREATE INDEX IF NOT EXISTS attempts_worker_status "
            "ON attempts(worker_id,status);",
        )
        columns = {str(row[1]) for row in con.execute("PRAGMA table_info(attempts)")}
        for name, definition in (
            ("terminal_receipt_json", "TEXT"),
            ("ingestion_status", "TEXT NOT NULL DEFAULT 'PENDING'"),
            ("ingestion_error", "TEXT"),
        ):
            if name not in columns:
                con.execute(f"ALTER TABLE attempts ADD COLUMN {name} {definition}")
        con.execute(
            "CREATE INDEX IF NOT EXISTS attempts_ingestion "
            "ON attempts(ingestion_status)"
        )

    def _require_schema(self, con: sqlite3.Connection) -> None:
        required = {
            "runs",
            "metrics",
            "events",
            "stage_history",
            "manifests",
            "checkpoints",
            "calibration",
            "store_metadata",
            "outbox",
            "ingested_records",
            "workers",
            "experiments",
            "attempts",
        }
        present = {
            str(row[0])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                "schema version 4 is missing required tables: " + ", ".join(missing)
            )
        columns = {
            "runs": "run_id name status parent_run_id created_at updated_at config_json metadata_json latest_checkpoint",
            "metrics": "run_id step tokens_seen wall_time name value",
            "events": "id run_id step tokens_seen wall_time kind payload_json",
            "stage_history": "run_id sequence stage status step tokens_seen started_at finished_at payload_json",
            "manifests": "run_id digest json",
            "checkpoints": "run_id checkpoint_id relative_path digest step tokens_seen created_at size_bytes validation_loss verified_at verification_status resume_level backend",
            "calibration": "key_hash run_id observation_json",
            "store_metadata": "singleton origin_id next_sequence",
            "outbox": "sequence run_id record_json record_sha256",
            "ingested_records": "origin_id sequence sha256",
            "workers": "worker_id record_json",
            "experiments": "experiment_id specification_json status submitted_at",
            "attempts": "attempt_id experiment_id run_id worker_id status receipt_json terminal_receipt_json queued_reason ingestion_status ingestion_error",
        }
        for table, names in columns.items():
            actual = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
            if not set(names.split()).issubset(actual):
                raise ValueError(f"schema version 4 has malformed columns in {table}")
        metadata = con.execute(
            "SELECT origin_id,next_sequence FROM store_metadata"
        ).fetchall()
        if len(metadata) != 1:
            raise ValueError("missing or ambiguous store identity")
        origin, sequence = metadata[0]
        if not isinstance(origin, str) or str(uuid.UUID(origin)) != origin:
            raise ValueError("malformed store origin identity")
        last = con.execute("SELECT COALESCE(MAX(sequence),0) FROM outbox").fetchone()[0]
        if type(sequence) is not int or sequence != last + 1:
            raise ValueError("store sequence metadata disagrees with committed outbox")

    def _append_outbox(
        self,
        con: sqlite3.Connection,
        run_id: str,
        kind: str,
        payload: Mapping[str, object],
    ) -> None:
        origin_id, sequence = con.execute(
            "SELECT origin_id,next_sequence FROM store_metadata WHERE singleton=1"
        ).fetchone()
        envelope: dict[str, object] = {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "origin_id": origin_id,
            "sequence": int(sequence),
            "run_id": run_id,
            "kind": kind,
            "payload": dict(payload),
        }
        digest = _envelope_digest(envelope)
        envelope["sha256"] = digest
        encoded = _canonical_json(envelope)
        if len(encoded.encode("utf-8")) > MAX_ENVELOPE_BYTES:
            raise ValueError("envelope exceeds 64 KiB; use a hashed artifact reference")
        con.execute(
            "INSERT INTO outbox(sequence,run_id,record_json,record_sha256) VALUES(?,?,?,?)",
            (sequence, run_id, encoded, digest),
        )
        con.execute(
            "UPDATE store_metadata SET next_sequence=next_sequence+1 WHERE singleton=1"
        )
        con.execute(
            "UPDATE runs SET updated_at=datetime('now') WHERE run_id=?", (run_id,)
        )

    def _write(
        self, run_id: str, kind: str, payload: Mapping[str, object], operation: Any
    ) -> None:
        with self._connect() as con:
            try:
                operation(con)
                self._append_outbox(con, run_id, kind, payload)
                con.commit()
            except BaseException:
                con.rollback()
                raise

    def create_run(
        self,
        run_id: str,
        config: object,
        metadata: object,
        parent_run_id: str | None = None,
    ) -> None:
        config_json = _canonical_json(config)
        metadata_json = _canonical_json(metadata)
        payload = {
            "name": getattr(config, "name", None)
            or (config.get("name", run_id) if isinstance(config, Mapping) else run_id),
            "config": json.loads(config_json),
            "metadata": json.loads(metadata_json),
            "parent_run_id": parent_run_id,
        }
        self._write(
            run_id,
            "run_created",
            payload,
            lambda con: con.execute(
                "INSERT INTO runs VALUES(?,?,'running',?,datetime('now'),datetime('now'),?,?,NULL)",
                (
                    run_id,
                    str(payload["name"]),
                    parent_run_id,
                    config_json,
                    metadata_json,
                ),
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
        if not values or any(not math.isfinite(value) for value in values.values()):
            raise ValueError("nonfinite or empty metric set")
        normalized = {str(key): float(value) for key, value in values.items()}
        payload = {
            "step": step,
            "tokens_seen": tokens_seen,
            "wall_time": wall_time,
            "values": normalized,
        }
        self._write(
            run_id,
            "metrics",
            payload,
            lambda con: con.executemany(
                "INSERT INTO metrics VALUES(?,?,?,?,?,?)",
                [
                    (run_id, step, tokens_seen, wall_time, name, value)
                    for name, value in normalized.items()
                ],
            ),
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
        encoded = _canonical_json(payload)
        record = {
            "step": step,
            "tokens_seen": tokens_seen,
            "wall_time": wall_time,
            "kind": kind,
            "payload": json.loads(encoded),
        }
        self._write(
            run_id,
            "event",
            record,
            lambda con: con.execute(
                "INSERT INTO events(run_id,step,tokens_seen,wall_time,kind,payload_json) VALUES(?,?,?,?,?,?)",
                (run_id, step, tokens_seen, wall_time, kind, encoded),
            ),
        )

    def record_stage(
        self,
        run_id: str,
        sequence: int,
        stage: str,
        status: str,
        step: int,
        tokens_seen: int,
        started_at: str,
        finished_at: str | None = None,
        payload: object | None = None,
    ) -> None:
        encoded = _canonical_json(payload or {})
        record = {
            "sequence": sequence,
            "stage": stage,
            "status": status,
            "step": step,
            "tokens_seen": tokens_seen,
            "started_at": started_at,
            "finished_at": finished_at,
            "payload": json.loads(encoded),
        }
        self._write(
            run_id,
            "stage",
            record,
            lambda con: con.execute(
                "INSERT INTO stage_history VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    sequence,
                    stage,
                    status,
                    step,
                    tokens_seen,
                    started_at,
                    finished_at,
                    encoded,
                ),
            ),
        )

    def register_manifest(self, run_id: str, digest: str, payload: object) -> None:
        encoded = _canonical_json(payload)
        self._write(
            run_id,
            "manifest",
            {"digest": digest, "payload": json.loads(encoded)},
            lambda con: con.execute(
                "INSERT INTO manifests(run_id,digest,json) VALUES(?,?,?)",
                (run_id, digest, encoded),
            ),
        )

    def record_checkpoint(self, run_id: str, record: object) -> None:
        source = record if isinstance(record, Mapping) else vars(record)
        checkpoint_id = str(
            source.get("checkpoint_id")
            or source.get("generation_id")
            or source["relative_path"]
        )
        fields = {
            "checkpoint_id": checkpoint_id,
            "relative_path": str(source["relative_path"]),
            "digest": str(
                source.get("digest")
                or source.get("checkpoint_digest")
                or source.get("manifest_sha256")
            ),
            "step": int(source["step"]),
            "tokens_seen": int(source["tokens_seen"]),
            "created_at": str(source["created_at"]),
            "size_bytes": int(source.get("size_bytes", source.get("bytes", 0))),
            "validation_loss": source.get("validation_loss"),
            "verified_at": source.get("verified_at"),
            "verification_status": str(source.get("verification_status", "verified")),
            "resume_level": str(source.get("resume_level", "full")),
            "backend": str(source.get("backend", "unknown")),
        }
        if fields["validation_loss"] is not None and not math.isfinite(
            float(fields["validation_loss"])
        ):
            raise ValueError("nonfinite validation loss")
        self._write(
            run_id,
            "checkpoint",
            fields,
            lambda con: con.execute(
                "INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, *fields.values()),
            ),
        )

    def record_calibration(
        self, key_hash: str, run_id: str, observation: object
    ) -> None:
        encoded = _canonical_json(observation)
        payload = {"key_hash": key_hash, "observation": json.loads(encoded)}
        self._write(
            run_id,
            "calibration",
            payload,
            lambda con: con.execute(
                "INSERT INTO calibration(key_hash,run_id,observation_json) VALUES(?,?,?)",
                (key_hash, run_id, encoded),
            ),
        )

    @staticmethod
    def get_calibration(root_dir: Path, key_hash: str) -> list[dict[str, object]]:
        """Read existing observations without creating or migrating an experiment store."""
        path = root_dir / "experiments.sqlite3"
        if not path.is_file():
            return []
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as con:
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError("unsupported newer calibration store schema")
            if not con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='calibration'"
            ).fetchone():
                return []
            return [
                {"run_id": row[0], "observation": _strict_json_loads(row[1])}
                for row in con.execute(
                    "SELECT run_id,observation_json FROM calibration WHERE key_hash=? ORDER BY run_id",
                    (key_hash,),
                )
            ]

    def finish_run(
        self, run_id: str, status: str, checkpoint: str | None = None
    ) -> None:
        payload = {"status": status, "checkpoint": checkpoint}
        self._write(
            run_id,
            "run_finished",
            payload,
            lambda con: con.execute(
                "UPDATE runs SET status=?,latest_checkpoint=? WHERE run_id=?",
                (status, checkpoint, run_id),
            ),
        )

    def export_records(
        self, after_sequence: int = 0, limit: int = 1000, max_bytes: int = 4_194_304
    ) -> dict[str, object]:
        if (
            any(type(value) is not int for value in (after_sequence, limit, max_bytes))
            or not 0 <= after_sequence <= 2**63 - 1
            or not 1 <= limit <= 1000
            or not 2 <= max_bytes <= 4_194_304
        ):
            raise ValueError("invalid export cursor, limit, or byte budget")
        with self._connect() as con:
            origin_id = con.execute(
                "SELECT origin_id FROM store_metadata WHERE singleton=1"
            ).fetchone()[0]
            rows = con.execute(
                "SELECT sequence,record_json FROM outbox WHERE sequence>? ORDER BY sequence LIMIT ?",
                (after_sequence, limit),
            ).fetchall()
        records: list[dict[str, object]] = []
        used = 2  # The records.json attachment includes its array delimiters.
        for sequence, raw in rows:
            if sequence != after_sequence + len(records) + 1:
                raise ValueError("outbox sequence gap")
            encoded = raw.encode("utf-8")
            size = len(encoded) + bool(records)
            if used + size > max_bytes:
                if records:
                    break
                raise ValueError("LIMIT_TOO_SMALL: next envelope exceeds max_bytes")
            record = self._validate_envelope(raw)
            if record["sequence"] != sequence or record["origin_id"] != origin_id:
                raise ValueError("outbox envelope identity mismatch")
            records.append(record)
            used += size
        next_sequence = int(records[-1]["sequence"]) if records else after_sequence
        with self._connect() as con:
            has_more = (
                con.execute(
                    "SELECT EXISTS(SELECT 1 FROM outbox WHERE sequence>?)",
                    (next_sequence,),
                ).fetchone()[0]
                == 1
            )
        return {
            "origin_id": origin_id,
            "records": records,
            "next_sequence": next_sequence,
            "has_more": has_more,
        }

    def import_records(
        self, records: Iterable[Mapping[str, object] | str | bytes]
    ) -> None:
        parsed = [self._validate_envelope(record) for record in records]
        with self._connect() as con:
            try:
                expected: dict[str, int] = {}
                for envelope in parsed:
                    origin = str(envelope["origin_id"])
                    sequence = int(envelope["sequence"])
                    existing = con.execute(
                        "SELECT sha256 FROM ingested_records WHERE origin_id=? AND sequence=?",
                        (origin, sequence),
                    ).fetchone()
                    if existing is not None:
                        if existing[0] != envelope["sha256"]:
                            raise ValueError("conflicting imported sequence")
                        continue
                    expected_sequence = expected.setdefault(
                        origin,
                        int(
                            con.execute(
                                "SELECT COALESCE(MAX(sequence),0)+1 FROM ingested_records WHERE origin_id=?",
                                (origin,),
                            ).fetchone()[0]
                        ),
                    )
                    if sequence != expected_sequence:
                        raise ValueError(
                            f"import sequence gap for {origin}: expected {expected_sequence}, got {sequence}"
                        )
                    self._project_import(con, envelope)
                    con.execute(
                        "INSERT INTO ingested_records(origin_id,sequence,sha256) VALUES(?,?,?)",
                        (origin, sequence, envelope["sha256"]),
                    )
                    expected[origin] = sequence + 1
                con.commit()
            except BaseException:
                con.rollback()
                raise

    def _validate_envelope(
        self, record: Mapping[str, object] | str | bytes
    ) -> dict[str, object]:
        if isinstance(record, (str, bytes)):
            size = (
                len(record.encode("utf-8")) if isinstance(record, str) else len(record)
            )
            if size > MAX_ENVELOPE_BYTES:
                raise ValueError("envelope exceeds 64 KiB")
        raw: object = (
            _strict_json_loads(record) if isinstance(record, (str, bytes)) else record
        )
        if not isinstance(raw, Mapping):
            raise TypeError("envelope must be an object")
        required = {
            "schema_version",
            "origin_id",
            "sequence",
            "run_id",
            "kind",
            "payload",
            "sha256",
        }
        if (
            set(raw) != required
            or type(raw["schema_version"]) is not int
            or raw["schema_version"] != ENVELOPE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported or malformed envelope")
        if any(
            not isinstance(raw[name], str) or not raw[name]
            for name in ("origin_id", "run_id", "kind")
        ):
            raise TypeError("malformed envelope identity")
        if str(uuid.UUID(raw["origin_id"])) != raw["origin_id"]:
            raise ValueError("malformed envelope origin identity")
        if (
            type(raw["sequence"]) is not int
            or not 1 <= raw["sequence"] <= 2**63 - 1
            or not isinstance(raw["payload"], Mapping)
        ):
            raise ValueError("malformed envelope sequence or payload")
        digest = _envelope_digest(raw)
        if not isinstance(raw["sha256"], str) or digest != raw["sha256"]:
            raise ValueError("envelope checksum mismatch")
        # Canonical encoding also catches nonfinite values supplied as a mapping.
        if len(_canonical_json(raw).encode("utf-8")) > MAX_ENVELOPE_BYTES:
            raise ValueError("envelope exceeds 64 KiB")
        return dict(raw)

    def _project_import(
        self, con: sqlite3.Connection, envelope: Mapping[str, object]
    ) -> None:
        run_id, kind, payload = (
            str(envelope["run_id"]),
            str(envelope["kind"]),
            envelope["payload"],
        )
        assert isinstance(payload, Mapping)
        if kind == "run_created":
            con.execute(
                "INSERT OR IGNORE INTO runs VALUES(?,?,'running',?,datetime('now'),datetime('now'),?,?,NULL)",
                (
                    run_id,
                    str(payload["name"]),
                    payload.get("parent_run_id"),
                    _canonical_json(payload["config"]),
                    _canonical_json(payload["metadata"]),
                ),
            )
        elif kind == "metrics":
            con.executemany(
                "INSERT OR IGNORE INTO metrics VALUES(?,?,?,?,?,?)",
                [
                    (
                        run_id,
                        payload["step"],
                        payload["tokens_seen"],
                        payload["wall_time"],
                        name,
                        value,
                    )
                    for name, value in dict(payload["values"]).items()
                ],
            )
        elif kind == "event":
            con.execute(
                "INSERT INTO events(run_id,step,tokens_seen,wall_time,kind,payload_json) VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    payload["step"],
                    payload["tokens_seen"],
                    payload["wall_time"],
                    payload["kind"],
                    _canonical_json(payload["payload"]),
                ),
            )
        elif kind == "stage":
            con.execute(
                "INSERT OR IGNORE INTO stage_history VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    payload["sequence"],
                    payload["stage"],
                    payload["status"],
                    payload["step"],
                    payload["tokens_seen"],
                    payload["started_at"],
                    payload["finished_at"],
                    _canonical_json(payload["payload"]),
                ),
            )
        elif kind == "manifest":
            con.execute(
                "INSERT OR IGNORE INTO manifests(run_id,digest,json) VALUES(?,?,?)",
                (run_id, payload["digest"], _canonical_json(payload["payload"])),
            )
        elif kind == "checkpoint":
            con.execute(
                "INSERT OR IGNORE INTO checkpoints VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    *(
                        payload[key]
                        for key in (
                            "checkpoint_id",
                            "relative_path",
                            "digest",
                            "step",
                            "tokens_seen",
                            "created_at",
                            "size_bytes",
                            "validation_loss",
                            "verified_at",
                            "verification_status",
                            "resume_level",
                            "backend",
                        )
                    ),
                ),
            )
        elif kind == "calibration":
            con.execute(
                "INSERT OR IGNORE INTO calibration(key_hash,run_id,observation_json) VALUES(?,?,?)",
                (payload["key_hash"], run_id, _canonical_json(payload["observation"])),
            )
        elif kind == "run_finished":
            con.execute(
                "UPDATE runs SET status=?,latest_checkpoint=?,updated_at=datetime('now') WHERE run_id=?",
                (payload["status"], payload["checkpoint"], run_id),
            )
        else:
            raise ValueError(f"unknown envelope kind: {kind}")
