from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.dashboard.queries import snapshot
from sparselab.training.metrics import ExperimentStore


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


def test_dashboard_reads_latest_runtime_snapshot_without_migration(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    store.create_run("live", {"name": "live"}, {"purpose": "training"})
    payload = {
        "schema_version": 1,
        "phase": "training",
        "event": "heartbeat",
        "state": "RUNNING",
        "derived": {"live": {"eta_status": "available"}},
    }
    store.upsert_runtime_progress_snapshot("live", 3, 96, 12.0, payload)

    view = snapshot(tmp_path)

    assert view.runtime_progress == (
        {
            "run_id": "live",
            "step": 3,
            "tokens_seen": 96,
            "wall_time": 12.0,
            "payload_json": (
                '{"derived":{"live":{"eta_status":"available"}},'
                '"event":"heartbeat","phase":"training","schema_version":1,'
                '"state":"RUNNING"}'
            ),
        },
    )


def test_training_triage_panel_rejects_forged_and_symlinked_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.dashboard import app

    messages: list[str] = []
    monkeypatch.setattr(
        app,
        "st",
        SimpleNamespace(
            subheader=lambda text: None,
            warning=lambda text: messages.append(text),
            info=lambda text: messages.append(text),
            write=lambda text: messages.append(text),
        ),
    )
    run = tmp_path / "run"
    reports = run / "post-train-triage"
    reports.mkdir(parents=True)
    (reports / ("0" * 64 + ".json")).write_text("{}")
    app._triage_panel(tmp_path, ["run"])
    assert "UNKNOWN" in messages[-1] and "invalid" in messages[-1]
    (reports / ("0" * 64 + ".json")).unlink()
    (reports / ("1" * 64 + ".json")).symlink_to(tmp_path / "outside")
    app._triage_panel(tmp_path, ["run"])
    assert "UNKNOWN" in messages[-1] and "invalid" in messages[-1]
