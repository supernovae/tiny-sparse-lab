from __future__ import annotations

import sqlite3

import pytest

from sparselab.dashboard.queries import snapshot


def _legacy_store(tmp_path) -> sqlite3.Connection:
    path = tmp_path / "experiments.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE runs("
        "run_id TEXT PRIMARY KEY,name TEXT,status TEXT,parent_run_id TEXT,"
        "created_at TEXT,updated_at TEXT,config_json TEXT,metadata_json TEXT,"
        "latest_checkpoint TEXT);"
        "CREATE TABLE metrics("
        "run_id TEXT,step INTEGER,tokens_seen INTEGER,wall_time REAL,"
        "name TEXT,value REAL);"
        "INSERT INTO runs VALUES("
        "'legacy','legacy','complete',NULL,'t','t','{}','{}',NULL);"
        "PRAGMA user_version=1;"
    )
    connection.commit()
    return connection


def test_dashboard_snapshot_reads_legacy_store_without_migration(tmp_path) -> None:
    connection = _legacy_store(tmp_path)
    connection.close()

    view = snapshot(tmp_path)

    assert [record.run_id for record in view.runs] == ["legacy"]
    assert view.stages == ()
    assert view.checkpoints == ()
    assert view.manifests == {}
    with sqlite3.connect(tmp_path / "experiments.sqlite3") as check:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 1


def test_dashboard_snapshot_fails_read_only_for_missing_projection(tmp_path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        snapshot(tmp_path)
