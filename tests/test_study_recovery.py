"""Model-free publication fault injection; submission never prepares a bundle."""

import json
from types import SimpleNamespace

import pytest

from sparselab.experiments import study as subject
from sparselab.workers.controller import Controller
from sparselab.workers.store import ControllerStore
from test_study import _variants, _write_study


@pytest.fixture
def submission(tmp_path, monkeypatch):
    study = subject.plan_study(_write_study(tmp_path, _variants(), [], seeds=False))
    store = tmp_path / "runs"
    receipt = tmp_path / "receipt.json"
    calls = []

    def fake_submit(controller, requests):
        calls.append(requests)
        entries = []
        for index, request in enumerate(requests):
            config = request["config"].model_dump(mode="json")
            entries.append({
                "experiment_id": f"experiment-{index}",
                "attempt_id": f"attempt-{index}",
                "run_id": f"run-{index}",
                "spec": {
                    "experiment_id": f"experiment-{index}",
                    "config": config,
                    "config_sha256": subject.config_sha256(config),
                    "matrix": request["matrix"],
                    "declaration_provenance": request.get("declaration_provenance"),
                },
            })
        controller.store.enqueue_many(entries)
        return [SimpleNamespace(**{key: entry[key] for key in (
            "experiment_id", "attempt_id", "run_id"
        )}) for entry in entries]

    monkeypatch.setattr(Controller, "submit_many", fake_submit)
    return study, store, receipt, calls


def test_injected_publication_failure_retains_native_jobs(submission, monkeypatch):
    study, store, receipt, calls = submission
    writer = subject._write_json_exclusive

    def fail(path, payload):
        if path == receipt:
            path.write_bytes(b'{"partial":')
            raise OSError("injected publication failure")
        writer(path, payload)

    monkeypatch.setattr(subject, "_write_json_exclusive", fail)
    with pytest.raises(OSError, match="study runs were submitted"):
        subject.submit_study(study, receipt, store=store)
    assert len(calls) == 1
    assert len(ControllerStore(store, read_only=True).attempts()) == 3
    assert receipt.read_bytes() == b'{"partial":'
    with pytest.raises(FileExistsError):
        subject.submit_study(study, receipt, store=store)
    assert len(calls) == 1
