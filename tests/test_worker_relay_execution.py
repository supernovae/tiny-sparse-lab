from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sparselab.config.models import RunConfig
from sparselab.training.trainer import train
from sparselab.workers.controller import Controller
from sparselab.workers.models import WorkerDefinition
from sparselab.workers.relay_collection import collect_relay
from sparselab.workers.relay_models import RelayLocation, RelayProfile


def _tiny(root: Path) -> RunConfig:
    from test_training import config

    value = config(root).model_dump(mode="json")
    value["training"].update(max_steps=4, max_tokens=128)
    value["checkpoint"].update(every_steps=2, keep_periodic=True)
    value["evaluation"].update(every_steps=2, max_batches=1)
    value["staging"].update(smoke_steps=1, warmup_steps=1)
    return RunConfig.model_validate(value)


def test_failed_checkpoint_publication_stops_before_next_optimizer_update(
    tmp_path: Path,
) -> None:
    config = _tiny(tmp_path)
    published = []

    def boundary(run, record):
        published.append(record.step)
        if record.step == 2:
            raise OSError("owned fixture upload interrupted")

    with pytest.raises(OSError, match="upload interrupted"):
        train(config, run_id="failed-transfer", checkpoint_committed=boundary)
    import json

    run = config.logging.root_dir / "failed-transfer"
    progress = json.loads((run / "progress.json").read_bytes())
    assert progress["status"] == "failed"
    assert progress["step"] == 2
    assert published[-1] == 2
    from sparselab.training.checkpoints import CheckpointManager

    verified = CheckpointManager(run).verify(
        run / "checkpoints/latest.json", require_training_state=True
    )
    assert verified.valid


def test_authenticated_offline_checkpoint_recovers_as_new_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.workers import execution, relay_execution, transport

    config = _tiny(tmp_path / "inputs")
    controller = Controller(tmp_path / "controller")
    location = RelayLocation(kind="file", root=str(tmp_path / "durable"))
    profile = RelayProfile(namespace="owned-loss", controller=location, worker=location)
    worker = WorkerDefinition(
        worker_id="owned-worker",
        name="owned-worker",
        transport="local",
        python=Path(sys.executable).absolute(),
        root=tmp_path / "worker",
        engine="pytorch",
        backend="cpu",
        device_index=0,
        relay=profile.binding(),
    )
    controller.register(worker)
    submission = controller.submit(config, worker=worker.name)
    attempt = controller.store.attempt_by_run(submission.run_id)
    assert attempt is not None
    capability, reason = controller._eligible_worker(attempt)
    assert reason is None and capability is not None
    # Deliberately stop delivery after the real durable prepare. Execution below
    # uses the native trainer, not a mocked optimizer or emitted progress fixture.
    native_call = transport.call_worker

    def prepare_only(definition, op, payload, **kwargs):
        return native_call(
            definition, "prepare" if op == "launch" else op, payload, **kwargs
        )

    monkeypatch.setattr(transport, "call_worker", prepare_only)
    controller._launch(attempt, capability)
    attempt = controller.store.attempt_by_run(submission.run_id)
    assert attempt is not None
    definition = controller._worker_for_attempt(attempt)
    assert definition is not None
    original = relay_execution.publish_boundary

    def fail_before_four(definition, attempt_id, *, checkpoint=None, terminal=False):
        if terminal or (checkpoint is not None and checkpoint.step >= 4):
            raise OSError("owned fixture runtime lost before next durable boundary")
        return original(
            definition, attempt_id, checkpoint=checkpoint, terminal=terminal
        )

    monkeypatch.setattr(relay_execution, "publish_boundary", fail_before_four)
    with pytest.raises(OSError, match="runtime lost"):
        execution.execute_attempt(definition, submission.attempt_id)
    monkeypatch.setattr(relay_execution, "publish_boundary", original)
    executor_root = definition.root
    executor_root.rename(tmp_path / "unavailable-worker")
    result = collect_relay(
        controller,
        submission.run_id,
        profile,
        recover=True,
        confirm_worker_lost=True,
        flush=False,
    )
    assert result["status"] == "UNKNOWN"
    assert result["terminal_receipt"] is None
    assert result["receipt"]["state"] == "RUNNING"
    assert result["ingestion_status"] == "COMPLETE"
    assert result["durable_checkpoint"]["step"] == 2
    repeated = collect_relay(
        controller,
        submission.run_id,
        profile,
        recover=True,
        confirm_worker_lost=True,
        flush=False,
    )
    assert repeated["recovery_observation"] == result["recovery_observation"]
    assert controller.cancel(submission.run_id)["status"] == "UNKNOWN"
    assert (
        controller.store.attempt_by_run(submission.run_id)["recovery_observation"]
        == result["recovery_observation"]
    )
    replacement = worker.model_copy(
        update={
            "worker_id": "replacement",
            "name": "replacement",
            "root": tmp_path / "replacement",
        }
    )
    controller.register(replacement)
    child = controller.resume(submission.run_id, worker="replacement")
    assert (
        child.run_id != submission.run_id and child.attempt_id != submission.attempt_id
    )
    child_attempt = controller.store.attempt_by_run(child.run_id)
    assert child_attempt is not None
    new_capability, reason = controller._eligible_worker(child_attempt)
    assert reason is None and new_capability is not None
    unavailable = next(
        row for row in controller.workers(refresh=False) if row.name == worker.name
    )
    assert unavailable.status == "unknown"
    assert unavailable.instance_id == definition.instance_id
    controller._launch(child_attempt, new_capability)
    assigned = controller.store.attempt_by_run(child.run_id)
    assert assigned is not None
    new_definition = controller._worker_for_attempt(assigned)
    assert new_definition is not None
    execution.execute_attempt(new_definition, child.attempt_id)
    completed = collect_relay(controller, child.run_id, profile, flush=False)
    assert completed["status"] == "COMPLETE"
    assert completed["ingestion_status"] == "COMPLETE"
    assert completed["durable_checkpoint"]["step"] == 4
    reference = train(config, run_id="uninterrupted")
    from test_training import equal

    from sparselab.training.checkpoints import CheckpointManager

    left = CheckpointManager(controller.root / child.run_id).load(
        controller.root / child.run_id / "checkpoints/latest.json"
    )
    right_run = config.logging.root_dir / reference
    right = CheckpointManager(right_run).load(right_run / "checkpoints/latest.json")
    for key in (
        "model",
        "optimizer",
        "rng",
        "cursor",
        "step",
        "tokens_seen",
        "scaler",
        "schedule",
    ):
        equal(getattr(left, key), getattr(right, key))
    # A falsely confirmed loss can later produce genuine terminal evidence.
    # Keep the recovery envelope immutable and fence any further child dispatch.
    import shutil

    # Refresh recreated only an empty replacement endpoint, not the original run.
    shutil.rmtree(executor_root)
    (tmp_path / "unavailable-worker").rename(executor_root)
    relay_execution.flush_terminal(definition, submission.attempt_id)
    with pytest.raises(ValueError, match="CONFLICT"):
        collect_relay(
            controller,
            submission.run_id,
            profile,
            recover=True,
            confirm_worker_lost=True,
            flush=False,
        )
    conflicted = controller.store.attempt_by_run(submission.run_id)
    assert conflicted is not None
    assert conflicted["terminal_receipt"] is None
    assert conflicted["recovery_observation"] == result["recovery_observation"]
    assert "CONFLICT" in conflicted["ingestion_error"]
    with pytest.raises(ValueError):
        controller.resume(submission.run_id, worker="replacement")
