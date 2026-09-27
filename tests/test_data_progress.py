from __future__ import annotations

import json

from sparselab.progress import PROGRESS_SCHEMA_VERSION, ProgressReporter, progress_phase


def test_preprocessing_progress_events_are_versioned_stderr_jsonl(capsys) -> None:
    with progress_phase(
        "blocked-stage",
        operation_id="operation-1",
        completed_work=0,
        total_work=2,
        unit="documents",
        raw_counters={"documents": 0},
        interval_seconds=None,
    ) as progress:
        progress.update(
            completed_work=1,
            total_work=2,
            unit="documents",
            raw_counters={"documents": 1, "source_bytes": 64},
        )

    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert captured.out == ""
    assert [event["event"] for event in events] == [
        "started",
        "progress",
        "finished",
    ]
    for event in events:
        assert event["schema_version"] == PROGRESS_SCHEMA_VERSION
        assert event["operation_id"] == "operation-1"
        assert event["phase"] == "blocked-stage"
        assert "elapsed_seconds" in event
        assert "last_meaningful_progress_at_monotonic" in event
        assert "raw_counters" in event
    assert events[1]["completed_work"] == 1
    assert events[1]["total_work"] == 2
    assert events[1]["unit"] == "documents"
    assert events[1]["raw_counters"] == {"documents": 1, "source_bytes": 64}
    assert events[1]["last_meaningful_progress_at_monotonic"] is not None


def test_heartbeat_reports_stall_and_later_progress_without_stopping(
    capsys, monkeypatch
) -> None:
    clock = [10.0]
    monkeypatch.setattr("sparselab.progress.time.monotonic", lambda: clock[0])
    reporter = ProgressReporter(
        "long-phase",
        interval_seconds=None,
        stalled_after_seconds=5.0,
        completed_work=0,
        total_work=2,
        unit="targets",
        derived={
            "live": {
                "eta_low_seconds": 2.0,
                "eta_high_seconds": 4.0,
                "eta_status": "available",
                "optimizer_only_eta": {
                    "low_seconds": 2.0,
                    "high_seconds": 4.0,
                    "status": "available",
                    "basis": "recent_and_long",
                },
            }
        },
    )
    clock[0] = 16.0
    reporter.heartbeat()
    clock[0] = 17.0
    reporter.update(completed_work=1)
    reporter.close()

    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert events[1]["event"] == "heartbeat"
    assert events[1]["state"] == "NO_PROGRESS"
    assert events[1]["derived"]["live"]["optimizer_only_eta"]["status"] == "suspended"
    assert events[1]["derived"]["live"]["optimizer_only_eta"]["low_seconds"] is None
    assert events[2]["event"] == "progress"
    assert events[2]["state"] == "RUNNING"
    assert events[2]["completed_work"] == 1
    assert events[3]["event"] == "finished"
