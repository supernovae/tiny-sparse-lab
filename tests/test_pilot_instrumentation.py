"""Operational pilot events must not change the configured scientific run."""

from __future__ import annotations

import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from test_staging import _config
from test_training import equal

from sparselab.staging import _read_sealed, pilot_config, stage
from sparselab.training import pilot, pilot_progress
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import canonical_json, read_manifest


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []
        self.phases: list[str] = []

    def emit(self, kind: str, phase: str, **counters: object) -> None:
        self.events.append((kind, phase, counters))
        if kind == "start":
            self.phases.append(phase)
        elif kind == "complete" and self.phases and self.phases[-1] == phase:
            self.phases.pop()

    @property
    def current_phase(self) -> str | None:
        return self.phases[-1] if self.phases else None


def test_pilot_events_observe_exact_configured_updates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "staging": config.staging.model_copy(
                update={"smoke_steps": 2, "warmup_steps": 5}
            )
        }
    )
    root = stage(config, tmp_path / "stage", through="validate")
    original = config.model_dump(mode="json")
    for purpose, steps in (("smoke", 2), ("warmup", 5)):
        recorder = _Recorder()
        token = pilot_progress._ACTIVE.set(recorder)
        try:
            report_path = pilot.run_pilot(root, purpose)
        finally:
            pilot_progress._ACTIVE.reset(token)
        report = _read_sealed(report_path)
        derived = pilot_config(config, purpose, root)
        assert config.model_dump(mode="json") == original
        assert report["step"] == report["timed_updates"] == steps
        assert report["derived_config"] == derived.model_dump(mode="json")
        assert report["tokens_seen"] == sum(
            observation["targets"] for observation in report["update_observations"]
        )
        assert report["checkpoint_reload"] == "verified_full_state_and_finite_forward"
        events = recorder.events
        starts = [
            event for event in events if event[:2] == ("start", "optimizer_update")
        ]
        completes = [
            event for event in events if event[:2] == ("complete", "optimizer_update")
        ]
        assert [event[2]["current_step"] for event in starts] == list(
            range(1, steps + 1)
        )
        assert [event[2]["completed_steps"] for event in starts] == list(range(steps))
        assert [event[2]["current_step"] for event in completes] == list(
            range(1, steps + 1)
        )
        assert [event[2]["completed_targets"] for event in completes] == [
            sum(row["targets"] for row in report["update_observations"][:index])
            for index in range(1, steps + 1)
        ]
        phases = [phase for kind, phase, _ in events if kind == "start"]
        assert phases.index("stage_bundle_verification") < phases.index("data_open")
        assert phases.index("data_open") < phases.index("model_initialization")
        assert phases.index("model_initialization") < phases.index("training")
        assert phases.index("training") < phases.index("checkpoint_verification")
        assert phases.index("checkpoint_verification") < phases.index("model_reload")
        assert phases.index("model_reload") < phases.index("finite_forward")
        assert events[-1][:2] == ("complete", "pilot_complete")
        assert (
            len([event for event in events if event[:2] == ("start", "checkpoint")])
            >= 2
        )
        if purpose == "smoke":
            run = root / "pilots" / purpose / report["run_id"]
            manifest = read_manifest(run / "manifest.json")
            manager = CheckpointManager(
                run,
                manifest_sha256=hashlib.sha256(canonical_json(manifest)).hexdigest(),
            )
            recorded = manager.load(manager.root / "latest.json")
            baseline = _read_sealed(pilot.run_pilot(root, purpose))
            baseline_run = root / "pilots" / purpose / baseline["run_id"]
            baseline_manifest = read_manifest(baseline_run / "manifest.json")
            baseline_manager = CheckpointManager(
                baseline_run,
                manifest_sha256=hashlib.sha256(
                    canonical_json(baseline_manifest)
                ).hexdigest(),
            )
            uninstrumented = baseline_manager.load(
                baseline_manager.root / "latest.json"
            )
            assert (recorded.step, recorded.tokens_seen, recorded.cursor) == (
                uninstrumented.step,
                uninstrumented.tokens_seen,
                uninstrumented.cursor,
            )
            equal(recorded.model, uninstrumented.model)
            equal(recorded.optimizer, uninstrumented.optimizer)
            equal(recorded.rng, uninstrumented.rng)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (InterruptedError("cancelled"), "cancelled"),
        (torch.OutOfMemoryError("allocation failed"), "out_of_memory"),
        (pilot.EngineOutOfMemory("backend allocation failed"), "out_of_memory"),
    ],
)
def test_pilot_failure_distinguishes_cancel_and_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception, expected: str
) -> None:
    root = tmp_path / "staged"
    (root / "pilots" / "smoke").mkdir(parents=True)
    monkeypatch.setattr(
        pilot, "activate_pilot_progress", lambda **_kwargs: nullcontext()
    )
    monkeypatch.setattr(pilot, "ensure_work_dir", lambda: None)

    def fail_pilot(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(pilot, "run_pilot", fail_pilot)
    monkeypatch.setattr(sys, "argv", ["pilot", str(root), "smoke"])
    with pytest.raises(type(failure)):
        pilot.main()
    result = _read_sealed(root / "pilots" / "smoke" / "failure.json")
    assert result["kind"] == expected
    assert result["exception_type"] == type(failure).__name__
