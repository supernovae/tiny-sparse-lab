from __future__ import annotations

import sqlite3

import pytest

from sparselab.training.metrics import SCHEMA_VERSION, ExperimentStore, _envelope_digest


def _run(store: ExperimentStore, run_id: str = "run") -> None:
    store.create_run(run_id, {"name": run_id}, {"purpose": "test"})


def test_reopen_keeps_origin_and_exposes_committed_records(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    first = store.export_records()
    reopened = ExperimentStore(tmp_path)
    second = reopened.export_records()
    assert first["origin_id"] == second["origin_id"]
    assert second["next_sequence"] == 1
    assert second["records"][0]["kind"] == "run_created"


def test_v2_migrates_with_backup_and_preserves_stage(tmp_path) -> None:
    path = tmp_path / "experiments.sqlite3"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE runs(run_id TEXT PRIMARY KEY,name TEXT,status TEXT,parent_run_id TEXT,created_at TEXT,updated_at TEXT,config_json TEXT,metadata_json TEXT,latest_checkpoint TEXT);"
        "CREATE TABLE metrics(run_id TEXT,step INTEGER,tokens_seen INTEGER,wall_time REAL,name TEXT,value REAL,PRIMARY KEY(run_id,step,name));"
        "CREATE TABLE events(id INTEGER PRIMARY KEY,run_id TEXT,step INTEGER,tokens_seen INTEGER,wall_time REAL,kind TEXT,payload_json TEXT);"
        "CREATE TABLE stage_history(run_id TEXT,sequence INTEGER,stage TEXT,status TEXT,step INTEGER,tokens_seen INTEGER,started_at TEXT,finished_at TEXT,payload_json TEXT,PRIMARY KEY(run_id,sequence));"
        "INSERT INTO runs VALUES('old','old','complete',NULL,'t','t','{}','{}',NULL);"
        "INSERT INTO stage_history VALUES('old',1,'TRAINING','COMPLETE',1,2,'t','t','{}');"
        "PRAGMA user_version=2;"
    )
    con.commit()
    con.close()

    store = ExperimentStore(tmp_path)
    assert (tmp_path / "experiments.sqlite3.v2.bak").is_file()
    assert store.path.exists()
    with sqlite3.connect(store.path) as migrated:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert (
            migrated.execute("SELECT stage FROM stage_history").fetchone()[0]
            == "TRAINING"
        )


def test_unknown_schema_and_orphan_legacy_rows_are_rejected(tmp_path) -> None:
    path = tmp_path / "experiments.sqlite3"
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="unsupported"):
        ExperimentStore(tmp_path)

    path.unlink()
    with sqlite3.connect(path) as con:
        con.executescript(
            "CREATE TABLE runs(run_id TEXT PRIMARY KEY);"
            "CREATE TABLE metrics(run_id TEXT);"
            "INSERT INTO metrics VALUES('missing'); PRAGMA user_version=2;"
        )
    with pytest.raises(ValueError, match="orphan"):
        ExperimentStore(tmp_path)


def test_failed_domain_write_rolls_back_its_outbox_record(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    with pytest.raises(sqlite3.IntegrityError):
        _run(store)
    exported = store.export_records()
    assert [record["kind"] for record in exported["records"]] == ["run_created"]


def test_import_is_checked_contiguous_idempotent_and_never_echoes(tmp_path) -> None:
    source = ExperimentStore(tmp_path / "source")
    _run(source)
    source.log_metrics("run", 1, 8, 0.1, {"train/loss": 1.0})
    batch = source.export_records()
    target = ExperimentStore(tmp_path / "target")
    target.import_records(batch["records"])
    target.import_records(batch["records"])
    assert target.export_records()["records"] == []

    corrupt = dict(batch["records"][0])
    corrupt["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        target.import_records([corrupt])
    conflicting_replay = dict(batch["records"][0])
    conflicting_replay["payload"] = {
        **conflicting_replay["payload"],
        "name": "different",
    }
    conflicting_replay["sha256"] = _envelope_digest(conflicting_replay)
    with pytest.raises(ValueError, match="conflicting"):
        target.import_records([conflicting_replay])

    gap = dict(batch["records"][1])
    gap["origin_id"] = "00000000-0000-4000-8000-000000000001"
    gap["sha256"] = _envelope_digest(gap)
    with pytest.raises(ValueError, match="gap"):
        target.import_records([gap])


def test_stage_checkpoint_and_calibration_are_preserved_in_replication(
    tmp_path,
) -> None:
    source = ExperimentStore(tmp_path / "source")
    _run(source)
    source.record_stage("run", 1, "TRAINING", "RUNNING", 0, 0, "now")
    source.record_checkpoint(
        "run",
        {
            "generation_id": 1,
            "relative_path": "checkpoints/one",
            "manifest_sha256": "digest",
            "step": 1,
            "tokens_seen": 8,
            "created_at": "now",
            "bytes": 10,
            "validation_loss": 0.5,
            "engine": "pytorch",
            "backend": "cpu",
        },
    )
    source.record_calibration("key", "run", {"peak": 10})
    target = ExperimentStore(tmp_path / "target")
    target.import_records(source.export_records()["records"])
    assert ExperimentStore.get_calibration(tmp_path / "target", "key") == [
        {"run_id": "run", "observation": {"peak": 10}}
    ]
    with sqlite3.connect(target.path) as con:
        assert con.execute("SELECT COUNT(*) FROM stage_history").fetchone()[0] == 1
        assert con.execute("SELECT digest FROM checkpoints").fetchone()[0] == "digest"


def test_duplicate_keys_and_nonfinite_values_are_rejected(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    with pytest.raises(ValueError, match="nonfinite"):
        store.log_metrics("run", 1, 1, 1.0, {"train/loss": float("nan")})
    with pytest.raises(ValueError, match="duplicate JSON key"):
        store.import_records(['{"schema_version":1,"schema_version":1}'])


@pytest.mark.parametrize("field", ["schema_version", "sequence"])
def test_boolean_protocol_numbers_are_rejected_before_projection(tmp_path, field):
    source = ExperimentStore(tmp_path / "source")
    _run(source)
    record = source.export_records()["records"][0]
    record[field] = True
    record["sha256"] = _envelope_digest(record)
    target = ExperimentStore(tmp_path / "target")
    with pytest.raises(ValueError):
        target.import_records([record])
    with sqlite3.connect(target.path) as con:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_export_byte_budget_includes_canonical_array_framing(tmp_path):
    from sparselab.training.metrics import _canonical_json

    store = ExperimentStore(tmp_path)
    _run(store)
    store.log_metrics("run", 1, 8, 0.1, {"train/loss": 1.0})
    first = store.export_records(limit=1)["records"]
    size = len(_canonical_json(first).encode("utf-8"))
    with pytest.raises(ValueError):
        store.export_records(max_bytes=size - 1)
    batch = store.export_records(max_bytes=size)
    assert batch["records"] == first
    assert batch["next_sequence"] == 1 and batch["has_more"]
    assert store.export_records(after_sequence=1)["next_sequence"] == 2


def test_oversized_envelope_rolls_back_domain_write(tmp_path):
    store = ExperimentStore(tmp_path)
    with pytest.raises(ValueError):
        store.create_run("large", {"name": "large"}, {"report": "x" * 65_536})
    assert store.export_records()["records"] == []
    with sqlite3.connect(store.path) as con:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    _run(store)
    assert store.export_records()["next_sequence"] == 1


def test_reopening_rejects_missing_columns_and_sequence_disagreement(tmp_path):
    store = ExperimentStore(tmp_path / "columns")
    with sqlite3.connect(store.path) as con:
        con.execute("ALTER TABLE calibration RENAME COLUMN observation_json TO broken")
    with pytest.raises(ValueError):
        ExperimentStore(tmp_path / "columns")
    store = ExperimentStore(tmp_path / "sequence")
    _run(store)
    with sqlite3.connect(store.path) as con:
        con.execute("UPDATE store_metadata SET next_sequence=9")
    with pytest.raises(ValueError):
        ExperimentStore(tmp_path / "sequence")


def test_v3_migrates_queue_tables_with_backup_and_preserves_history(tmp_path) -> None:
    source = ExperimentStore(tmp_path)
    _run(source)
    with sqlite3.connect(source.path) as con:
        con.execute("PRAGMA user_version=3")
        con.execute("DROP TABLE attempts")
        con.execute("DROP TABLE experiments")
        con.execute("DROP TABLE workers")
    migrated = ExperimentStore(tmp_path)
    assert (tmp_path / "experiments.sqlite3.v3.bak").is_file()
    with sqlite3.connect(migrated.path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert con.execute("SELECT run_id FROM runs").fetchone()[0] == "run"
        assert con.execute("SELECT COUNT(*) FROM workers").fetchone()[0] == 0
