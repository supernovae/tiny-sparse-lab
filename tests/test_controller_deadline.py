from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.training.metrics import ExperimentStore
from sparselab.workers.controller import Controller


def _entry(index: int) -> dict[str, object]:
    return {
        "experiment_id": f"experiment-{index}",
        "attempt_id": f"attempt-{index}",
        "run_id": f"run-{index}",
        "spec": {"schema_version": 1, "experiment_id": f"experiment-{index}"},
    }


def test_expired_tick_preserves_queued_and_terminal_attempts(
    tmp_path, monkeypatch
) -> None:
    controller = Controller(tmp_path)
    controller.store.enqueue_many([_entry(1), _entry(2)])
    controller.store.assign("attempt-2", "worker-2")
    receipt = {"state": "COMPLETE", "attempt_id": "attempt-2"}
    controller.store.set_receipt("attempt-2", receipt, "COMPLETE")
    before = controller.store.attempts()

    def unexpected(*args, **kwargs):
        raise AssertionError("expired tick must not contact workers")

    monkeypatch.setattr(controller, "_eligible_worker", unexpected)
    monkeypatch.setattr(controller, "_retry_ingestion", unexpected)
    monkeypatch.setattr(controller, "_poll_active", unexpected)
    monkeypatch.setattr(controller, "_replay_assigned", unexpected)
    assert controller.tick(deadline=-1.0) == {"queued": 1, "assigned": 0}
    assert controller.store.attempts() == before
    assert controller.store.pending_ingestion()[0]["terminal_receipt"] == receipt


def test_record_transfer_shares_single_tick_budget(tmp_path, monkeypatch) -> None:
    controller = Controller(tmp_path / "controller")
    remote = ExperimentStore(tmp_path / "remote")
    remote.create_run("run", {"name": "run"}, {})
    remote.log_metrics("run", 1, 16, 1.0, {"loss": 2.0})
    remote.log_metrics("run", 2, 32, 2.0, {"loss": 1.0})
    origin = remote.export_records()["origin_id"]
    clock = [100.0]
    monkeypatch.setattr("sparselab.workers.controller.time.monotonic", lambda: clock[0])

    def records(worker, op, payload, *, receive_dir: Path, timeout: float):
        required = 7.0 if payload["after_sequence"] == 0 else 4.0
        clock[0] += min(timeout, required)
        if timeout < required:
            raise TimeoutError
        page = remote.export_records(after_sequence=payload["after_sequence"], limit=1)
        attachment = receive_dir / "records.json"
        attachment.write_text(json.dumps(page.pop("records")))
        return SimpleNamespace(result=page, attachments={"records.json": attachment})

    monkeypatch.setattr("sparselab.workers.transport.call_worker", records)
    with pytest.raises(TimeoutError):
        controller._ingest_records(object(), origin, drain=True, deadline=110.0)
    assert controller.store.imported_sequence(origin) == 1
    controller._ingest_records(object(), origin, drain=True)
    assert controller.store.imported_sequence(origin) == 3
