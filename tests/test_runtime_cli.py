from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sparselab.cli.main import build_parser
from sparselab.training.metrics import ExperimentStore


def test_runtime_status_suspends_stale_eta_without_writing_store(
    tmp_path: Path, capsys
) -> None:
    store = ExperimentStore(tmp_path)
    store.create_run("run", {"name": "run"}, {})
    observed_at = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    record = {
        "schema_version": 1,
        "operation_id": "run",
        "run_id": "run",
        "phase": "training",
        "event": "heartbeat",
        "observed_at_utc": observed_at,
        "elapsed_seconds": 600.0,
        "completed_work": 10,
        "total_work": 100,
        "unit": "targets",
        "raw_counters": {"optimizer_step": 2, "completed_targets": 10},
        "derived": {
            "live": {
                "state": "RUNNING",
                "last_meaningful_progress_age_seconds": 0.0,
                "stalled_after_seconds": 300.0,
                "eta_low_seconds": 9.0,
                "eta_high_seconds": 18.0,
                "optimizer_only_eta": {
                    "low_seconds": 9.0,
                    "high_seconds": 18.0,
                    "status": "available",
                    "basis": "recent_and_long",
                },
                "eta_status": "available",
            }
        },
        "state": "RUNNING",
    }
    store.upsert_runtime_progress_snapshot("run", 2, 10, 600.0, record)
    before = store.export_records()

    arguments = build_parser().parse_args(
        ["runtime", "status", "run", "--runs-dir", str(tmp_path), "--json"]
    )
    arguments.handler(arguments)
    status = json.loads(capsys.readouterr().out)

    assert status["availability"] == "available"
    assert status["latest"]["state"] == "NO_PROGRESS"
    assert status["latest"]["derived"]["live"]["eta_low_seconds"] is None
    assert status["latest"]["derived"]["live"]["eta_high_seconds"] is None
    assert (
        status["latest"]["derived"]["live"]["optimizer_only_eta"]["low_seconds"] is None
    )
    assert (
        status["latest"]["derived"]["live"]["optimizer_only_eta"]["high_seconds"]
        is None
    )
    assert (
        status["latest"]["derived"]["live"]["optimizer_only_eta"]["status"]
        == "suspended"
    )
    assert status["latest"]["derived"]["live"]["eta_status"] == "suspended"
    assert store.export_records() == before


def test_runtime_status_preserves_terminal_observation_and_final_timing(
    tmp_path: Path, capsys
) -> None:
    store = ExperimentStore(tmp_path)
    store.create_run("run", {"name": "run"}, {})
    observed_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    progress = {
        "schema_version": 1,
        "run_id": "run",
        "phase": "training",
        "event": "finished",
        "observed_at_utc": observed_at,
        "derived": {
            "live": {
                "state": "COMPLETE",
                "eta_low_seconds": 0.0,
                "eta_high_seconds": 0.0,
                "eta_status": "complete",
                "last_meaningful_progress_age_seconds": 0.0,
                "stalled_after_seconds": 300.0,
            }
        },
        "state": "COMPLETE",
    }
    final = {
        "schema_version": 1,
        "run_status": "completed",
        "phases": {
            "preparation": {"seconds": 1.5, "availability": "observed"},
            "optimizer_update": {"seconds": 2.0, "availability": "observed"},
            "evaluation": {"seconds": None, "availability": "not_observed"},
        },
    }
    store.upsert_runtime_progress_snapshot("run", 3, 100, 20.0, progress)
    store.log_event("run", 3, 100, 20.1, "runtime_final_observation", final)

    arguments = build_parser().parse_args(
        ["runtime", "status", "run", "--runs-dir", str(tmp_path), "--json"]
    )
    arguments.handler(arguments)
    status = json.loads(capsys.readouterr().out)

    assert status["latest"]["state"] == "COMPLETE"
    assert status["latest"]["derived"]["live"]["eta_low_seconds"] == 0.0
    assert status["final_observed"] == final
    assert [item["kind"] for item in status["records"]] == [
        "runtime_progress",
        "runtime_final_observation",
    ]


def test_runtime_status_returns_explicit_unavailability_for_unknown_run(
    tmp_path: Path, capsys
) -> None:
    arguments = build_parser().parse_args(
        ["runtime", "status", "missing", "--runs-dir", str(tmp_path), "--json"]
    )
    arguments.handler(arguments)

    status = json.loads(capsys.readouterr().out)

    assert status["availability"] == "unavailable"
    assert status["latest"] is None
    assert status["final_observed"] is None
