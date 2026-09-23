from __future__ import annotations

import sqlite3

import pytest

from sparselab.workers.store import ControllerStore


def _entry(index: int) -> dict[str, object]:
    return {
        "experiment_id": f"experiment-{index}",
        "attempt_id": f"attempt-{index}",
        "run_id": f"run-{index}",
        "spec": {"schema_version": 1, "experiment_id": f"experiment-{index}"},
    }


def test_enqueue_many_is_atomic_on_late_duplicate(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    first, duplicate = _entry(1), _entry(2)
    duplicate["run_id"] = first["run_id"]
    with pytest.raises(sqlite3.IntegrityError):
        store.enqueue_many([first, duplicate])
    assert store.attempts() == []


def test_queue_cancel_is_durable_and_never_reopens_terminal_attempt(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    entry = _entry(1)
    store.enqueue_many([entry])
    assert store.request_cancel("run-1")["status"] == "CANCELLED"
    assert store.request_cancel("run-1")["status"] == "CANCELLED"
    assert store.attempt_by_run("run-1")["status"] == "CANCELLED"
    assert store.attempt_by_run("run-1")["ingestion_status"] == "NOT_REQUIRED"


def test_assignment_is_compare_and_set_and_keeps_identity(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    entry = _entry(1)
    store.enqueue_many([entry])
    assert store.assign("attempt-1", "worker-1")
    assert not store.assign("attempt-1", "worker-2")
    attempt = store.attempt_by_run("run-1")
    assert attempt is not None
    assert attempt["worker_id"] == "worker-1"
    assert attempt["status"] == "ASSIGNED"


def test_durable_cancel_survives_running_receipt_and_connection_loss(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    store.enqueue_many([_entry(1)])
    store.assign("attempt-1", "worker-1")
    store.request_cancel("run-1")
    store.set_receipt("attempt-1", {"state": "RUNNING"}, "RUNNING")
    store.mark_unknown_if_nonterminal("attempt-1", "connection lost")
    assert (
        ControllerStore(tmp_path).attempt_by_run("run-1")["status"]
        == "CANCEL_REQUESTED"
    )
    store.set_receipt(
        "attempt-1",
        {"state": "INTERRUPTED", "cancellation_acknowledged": True},
        "CANCELLED",
    )
    store.set_receipt("attempt-1", {"state": "RUNNING"}, "RUNNING")
    assert store.attempt_by_run("run-1")["status"] == "CANCELLED"


def test_receipt_for_another_attempt_cannot_change_durable_assignment(tmp_path) -> None:
    from test_training import config as training_config

    from sparselab.workers.controller import Controller
    from sparselab.workers.transport import ProtocolError

    controller = Controller(tmp_path / "controller")
    submission = controller.submit(training_config(tmp_path / "inputs"))
    controller.store.assign(submission.attempt_id, "worker")
    attempt = controller.store.attempt_by_run(submission.run_id)
    spec = controller._model("ExperimentSpec", attempt["spec"])
    receipt = {
        "attempt_id": "another-attempt",
        "run_id": submission.run_id,
        "experiment_id": submission.experiment_id,
        "worker_id": "worker",
        "spec_digest": spec.digest(),
        "bundle_digest": spec.dispatch_bundle_digest,
        "state": "COMPLETE",
    }
    with pytest.raises(ProtocolError):
        controller._reconcile_receipt(attempt, receipt)
    observed = controller.store.attempt_by_run(submission.run_id)
    assert observed["status"] == "ASSIGNED"
    assert observed["terminal_receipt"] is None


def test_failure_before_run_creation_finishes_ingestion(tmp_path) -> None:
    from sparselab.workers.controller import Controller

    controller = Controller(tmp_path)
    controller.store.enqueue_many([_entry(1)])
    controller.store.assign("attempt-1", "worker")
    receipt = {
        "state": "FAILED",
        "training_started_at": "2026-09-23T00:00:00+00:00",
        "artifacts": [
            {"relative_path": "stderr.log", "size_bytes": 0, "sha256": "0" * 64}
        ],
    }
    controller.store.set_receipt("attempt-1", receipt, "FAILED")
    controller._ingest(
        controller.store.attempt_by_run("run-1"), None, receipt, "worker"
    )
    observed = controller.store.attempt_by_run("run-1")
    assert observed["status"] == "FAILED"
    assert observed["ingestion_status"] == "NOT_REQUIRED"
