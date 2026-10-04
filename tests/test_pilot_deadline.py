"""Pilot deadline and authenticated operational progress regressions."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
from pathlib import Path

import psutil
import pytest

from sparselab.operational_monitor import MonitorPolicy
from sparselab.training.pilot_deadline import (
    PilotCancelled,
    PilotDeadlinePolicy,
    PilotDeadlineState,
    PilotProcessFailure,
    PilotResourceFailure,
    PilotTimeout,
    PurposeDeadlines,
    load_pilot_deadline_policy,
    supervise_pilot,
)
from sparselab.training.pilot_progress import (
    activate_pilot_progress,
    current_pilot_progress,
    emit_pilot_progress,
    pilot_phase,
    validate_event,
)


def _event(
    sequence: int, phase: str, kind: str, **changes: object
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "phase": phase,
        "kind": kind,
        "current_step": None,
        "completed_steps": None,
        "completed_targets": None,
        "counter": None,
        "value": None,
        "total": None,
        "subject": None,
        **changes,
    }


def test_defaults_and_explicit_overrides(tmp_path: Path) -> None:
    default = PilotDeadlinePolicy()
    assert default.initialization_timeout_seconds > 900
    assert default.no_progress_timeout_seconds > 900
    assert default.absolute_timeout_seconds > default.initialization_timeout_seconds
    path = tmp_path / "pilot.yaml"
    path.write_text(
        "pilot_deadline_version: 1\nsmoke:\n  initialization_timeout_seconds: 10\n  no_progress_timeout_seconds: 2\n  absolute_timeout_seconds: 30\nwarmup:\n  initialization_timeout_seconds: 20\n  no_progress_timeout_seconds: 3\n  absolute_timeout_seconds: 40\n"
    )
    chosen = load_pilot_deadline_policy(path)
    assert chosen.for_purpose("smoke").absolute_timeout_seconds == 30
    assert chosen.for_purpose("warmup").initialization_timeout_seconds == 20
    path.write_text(
        "pilot_deadline_version: 1\nwarmup:\n  absolute_timeout_seconds: 9000\n"
    )
    partial = load_pilot_deadline_policy(path)
    assert partial.for_purpose("warmup").absolute_timeout_seconds == 9000
    assert partial.for_purpose("warmup").initialization_timeout_seconds == 1800
    with pytest.raises(ValueError):
        PilotDeadlinePolicy(pilot_deadline_version=True)
    with pytest.raises(ValueError):
        PilotDeadlinePolicy(absolute_timeout_seconds=float("inf"))
    path.write_text("initialization_timeout_seconds: 10\n")
    with pytest.raises(ValueError, match="explicit integer version"):
        load_pilot_deadline_policy(path)
    path.write_text("pilot_deadline_version: 1\nscientific_target_steps: 2\n")
    with pytest.raises(ValueError):
        load_pilot_deadline_policy(path)
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        load_pilot_deadline_policy(alias)


@pytest.mark.parametrize(
    "duration", [0, -1, True, "1200", float("nan"), float("inf"), 10**400]
)
def test_invalid_duration_is_rejected(duration: object) -> None:
    with pytest.raises(ValueError):
        PilotDeadlinePolicy(no_progress_timeout_seconds=duration)


@pytest.mark.parametrize(
    "settings",
    [
        {"pilot_deadline_version": 2},
        {"warmup": {"micro_batch_size": 1}},
        {"warmup": {"absolute_timeout_seconds": 1}},
        {"initialization_timeout_seconds": 8000},
    ],
)
def test_unknown_or_inconsistent_policy_is_rejected(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PilotDeadlinePolicy.model_validate(settings)


def test_virtual_deadlines_and_long_progress() -> None:
    ticks = [0.0]

    def clock() -> float:
        return ticks[0]

    policy = PurposeDeadlines(
        initialization_timeout_seconds=1000,
        no_progress_timeout_seconds=100,
        absolute_timeout_seconds=3000,
    )
    state = PilotDeadlineState(policy, clock=clock, expected_steps=200)
    ticks[0] = 101
    assert state.deadline() == "no_progress"
    state = PilotDeadlineState(policy, clock=clock)
    ticks[0] = 1102
    assert state.deadline() == "initialization"
    state = PilotDeadlineState(policy, clock=clock, expected_steps=200)
    state.observe(_event(1, "training", "start"))
    for index in range(1, 16):
        ticks[0] += 90
        assert state.deadline() is None
        assert state.observe(
            _event(
                index + 1,
                "training",
                "progress",
                counter="steps",
                value=index,
                total=200,
                completed_steps=index,
                current_step=index,
            )
        )
    assert ticks[0] - state.started > 900
    ticks[0] = state.started + 3000
    assert state.deadline() == "absolute"


def test_repeated_phase_or_heartbeat_does_not_extend_idle() -> None:
    ticks = [0.0]
    state = PilotDeadlineState(
        PurposeDeadlines(
            initialization_timeout_seconds=100,
            no_progress_timeout_seconds=10,
            absolute_timeout_seconds=200,
        ),
        clock=lambda: ticks[0],
    )
    state.observe(_event(1, "stage_bundle_verification", "start"))
    ticks[0] = 5
    assert state.observe(_event(2, "stage_bundle_verification", "complete"))
    assert not state.observe(_event(3, "stage_bundle_verification", "start"))
    ticks[0] = 16
    assert state.deadline() == "no_progress"
    with pytest.raises(ValueError, match="counter outside"):
        state.observe(
            _event(4, "training", "progress", counter="bytes", value=1, total=2)
        )
    with pytest.raises(ValueError, match="regressing"):
        state.observe(
            _event(
                4,
                "stage_bundle_verification",
                "progress",
                counter="bytes",
                value=1,
                total=2,
                completed_steps=-1,
            )
        )


def test_nested_phase_last_complete_and_counter() -> None:
    state = PilotDeadlineState(
        PurposeDeadlines(
            initialization_timeout_seconds=10,
            no_progress_timeout_seconds=10,
            absolute_timeout_seconds=100,
        )
    )
    state.observe(_event(1, "training", "start"))
    state.observe(_event(2, "optimizer_update", "start", current_step=1))
    state.observe(_event(3, "optimizer_update", "complete", completed_steps=1))
    assert state.last_complete_phase == "optimizer_update"
    assert state.current_phase == "training"
    with pytest.raises(ValueError, match="matching start"):
        state.observe(_event(4, "checkpoint", "complete"))
    state.observe(_event(4, "training", "complete"))
    state.observe(_event(5, "pilot_complete", "start"))
    state.observe(_event(6, "pilot_complete", "complete"))
    with pytest.raises(ValueError, match="after pilot completion"):
        state.observe(_event(7, "checkpoint", "start"))


def test_authenticated_wire_rejects_forged_owner_sequence_and_timer() -> None:
    secret = b"s" * 32
    payload: dict[str, object] = {
        "version": 1,
        "purpose": "smoke",
        "pid": 4242,
        "create_time": 123.5,
        "sequence": 1,
        "kind": "progress",
        "phase": "training",
        "timestamp": 1.0,
        "current_step": None,
        "completed_steps": None,
        "completed_targets": None,
        "elapsed_phase_seconds": 0.0,
        "counter": "steps",
        "value": 1,
        "total": 2,
        "subject": None,
    }

    def wire() -> bytes:
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        mac = hmac.new(secret, raw, hashlib.sha256).hexdigest()
        return json.dumps({"payload": payload, "mac": mac}).encode() + b"\n"

    assert (
        validate_event(
            wire(),
            secret,
            purpose="smoke",
            pid=4242,
            create_time=123.5,
            last_sequence=0,
        )["value"]
        == 1
    )
    with pytest.raises(ValueError):
        validate_event(
            wire(),
            secret,
            purpose="smoke",
            pid=4243,
            create_time=123.5,
            last_sequence=0,
        )
    with pytest.raises(ValueError):
        validate_event(
            wire(),
            secret,
            purpose="smoke",
            pid=4242,
            create_time=123.6,
            last_sequence=0,
        )
    with pytest.raises(ValueError):
        validate_event(
            wire(),
            secret,
            purpose="smoke",
            pid=4242,
            create_time=123.5,
            last_sequence=1,
        )
    payload.update(counter=None, value=None, total=None)
    with pytest.raises(ValueError):
        validate_event(
            wire(),
            secret,
            purpose="smoke",
            pid=4242,
            create_time=123.5,
            last_sequence=0,
        )
    payload["timestamp"] = float("nan")
    with pytest.raises(ValueError):
        validate_event(
            wire(),
            secret,
            purpose="smoke",
            pid=4242,
            create_time=123.5,
            last_sequence=0,
        )


def test_activation_is_explicit_and_journal_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert current_pilot_progress() is None
    with pilot_phase("training"):
        emit_pilot_progress("progress", "training", counter="steps", value=1, total=2)
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv("SPARSELAB_PILOT_PROGRESS_FD", str(write_fd))
    monkeypatch.setenv("SPARSELAB_PILOT_PROGRESS_SECRET", "ab" * 32)
    with (
        activate_pilot_progress(purpose="smoke", directory=tmp_path),
        pilot_phase("training"),
    ):
        emit_pilot_progress("progress", "training", counter="steps", value=1, total=2)
    raw = os.read(read_fd, 4096)
    os.close(read_fd)
    assert len(raw.splitlines()) == 3
    assert len((tmp_path / "progress.jsonl").read_bytes().splitlines()) == 3
    assert b"secret" not in raw


def _policy(*, idle: float = 2, absolute: float = 5) -> PilotDeadlinePolicy:
    return PilotDeadlinePolicy(
        initialization_timeout_seconds=3,
        no_progress_timeout_seconds=idle,
        absolute_timeout_seconds=absolute,
        termination_grace_seconds=0.1,
    )


def test_success_and_real_metrics_without_fabrication(tmp_path: Path) -> None:
    directory = tmp_path / "pilot"
    result = supervise_pilot(
        [sys.executable, "-c", "import time; time.sleep(.25)"],
        purpose="smoke",
        directory=directory,
        policy=_policy(),
    )
    assert result.returncode == 0
    evidence = json.loads((directory / "supervisor-completion.json").read_text())
    assert evidence["root"]["pid"] > 0 and evidence["root"]["create_time"] > 0
    assert evidence["sample_count"] > 0
    assert (
        evidence["peak_tree_rss_bytes"] is None or evidence["peak_tree_rss_bytes"] > 0
    )
    assert (
        evidence["sha256"]
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in evidence.items() if key != "sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )


def test_supervised_authenticated_phase_and_counter(tmp_path: Path) -> None:
    directory = tmp_path / "signed"
    script = "import sys\nfrom pathlib import Path\nfrom sparselab.training.pilot_progress import activate_pilot_progress, pilot_phase, emit_pilot_progress\nwith activate_pilot_progress(purpose='warmup', directory=Path(sys.argv[1])):\n    with pilot_phase('training'):\n        emit_pilot_progress('progress', 'training', counter='steps', value=1, total=2, current_step=1, completed_steps=1)\n        emit_pilot_progress('progress', 'training', counter='steps', value=2, total=2, current_step=2, completed_steps=2)\n    with pilot_phase('pilot_complete', completed_steps=2):\n        pass"
    supervise_pilot(
        [sys.executable, "-c", script, str(directory)],
        purpose="warmup",
        directory=directory,
        policy=_policy(),
        expected_steps=2,
    )
    evidence = json.loads((directory / "supervisor-completion.json").read_text())
    assert evidence["sequence"] == 6
    assert evidence["completed_steps"] == 2
    assert evidence["last_complete_phase"] == "pilot_complete"
    assert evidence["last_progress_timestamp"] is not None


def test_hung_owned_child_terminated_and_phase_preserved(tmp_path: Path) -> None:
    directory = tmp_path / "hung"
    script = (
        "import os,subprocess,sys,time; "
        "from pathlib import Path; "
        "from sparselab.training.pilot_progress import activate_pilot_progress,pilot_phase; "
        "d=Path(sys.argv[1]); "
        "ctx=activate_pilot_progress(purpose='smoke',directory=d); "
        "ctx.__enter__(); "
        "p=pilot_phase('stage_bundle_verification');p.__enter__(); "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        "(d/'child.pid').write_text(str(child.pid));time.sleep(30)"
    )
    with pytest.raises(PilotTimeout) as caught:
        supervise_pilot(
            [sys.executable, "-c", script, str(directory)],
            purpose="smoke",
            directory=directory,
            policy=_policy(idle=1.5, absolute=5),
        )
    evidence = caught.value.evidence
    assert evidence["reason"] == "no_progress"
    assert evidence["current_phase"] == "stage_bundle_verification"
    child_pid = int((directory / "child.pid").read_text())
    assert (
        not psutil.pid_exists(child_pid)
        or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    )


def test_cancelled_pilot(tmp_path: Path) -> None:
    directory = tmp_path / "cancel"
    cancel_path = tmp_path / "cancel.flag"
    cancel_path.touch()
    with pytest.raises(PilotCancelled):
        supervise_pilot(
            [sys.executable, "-c", "pass"],
            purpose="smoke",
            directory=directory,
            cancel_path=cancel_path,
        )
    assert not directory.exists()


def test_runtime_cancellation_is_distinct_from_timeout(tmp_path: Path) -> None:
    directory = tmp_path / "running-cancel"
    cancel_path = tmp_path / "cancel.flag"
    timer = threading.Timer(0.5, cancel_path.touch)
    timer.start()
    try:
        with pytest.raises(PilotCancelled) as caught:
            supervise_pilot(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                purpose="warmup",
                directory=directory,
                policy=_policy(idle=3, absolute=5),
                cancel_path=cancel_path,
            )
    finally:
        timer.join()
    assert caught.value.evidence["reason"] == "cancelled"
    assert (directory / "supervisor-failure.json").is_file()


def test_free_text_is_bounded_and_not_trusted_progress(tmp_path: Path) -> None:
    directory = tmp_path / "noisy"
    script = (
        "import sys;sys.stdout.write('x'*1100000);sys.stdout.flush();"
        "sys.stderr.write('fatal marker\\n');sys.stderr.flush()"
    )
    supervise_pilot(
        [sys.executable, "-c", script],
        purpose="smoke",
        directory=directory,
        policy=PilotDeadlinePolicy(
            initialization_timeout_seconds=30,
            no_progress_timeout_seconds=30,
            absolute_timeout_seconds=60,
        ),
    )
    evidence = json.loads((directory / "supervisor-completion.json").read_text())
    assert evidence["sequence"] == 0
    assert evidence["execution_log_omitted_bytes"] > 0
    log = (directory / "execution.log").read_bytes()
    assert len(log) < 1_048_576
    assert b"fatal marker" in log


def test_nonzero_exit_keeps_traceback(tmp_path: Path) -> None:
    directory = tmp_path / "failed"
    script = "import time;time.sleep(.15);raise RuntimeError('diagnostic marker')"
    with pytest.raises(PilotProcessFailure) as caught:
        supervise_pilot(
            [sys.executable, "-c", script],
            purpose="smoke",
            directory=directory,
            policy=_policy(),
        )
    assert caught.value.evidence["returncode"] != 0
    assert b"diagnostic marker" in (directory / "execution.log").read_bytes()


def test_resource_guard_distinct_from_timeout(tmp_path: Path) -> None:
    directory = tmp_path / "resource"
    with pytest.raises(PilotResourceFailure) as caught:
        supervise_pilot(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            purpose="smoke",
            directory=directory,
            policy=_policy(),
            monitor_policy=MonitorPolicy(
                monitor_policy_version=1,
                max_tree_rss_bytes=1,
            ),
        )
    assert caught.value.evidence["reason"] == "resource"
    assert caught.value.evidence["peak_tree_rss_bytes"] is not None


def test_nonadvancing_and_invalid_counters_do_not_refresh_or_mutate_state() -> None:
    ticks = [0.0]
    state = PilotDeadlineState(
        PurposeDeadlines(
            initialization_timeout_seconds=100,
            no_progress_timeout_seconds=10,
            absolute_timeout_seconds=200,
        ),
        clock=lambda: ticks[0],
        expected_steps=2,
    )
    state.observe(_event(1, "training", "start"))
    ticks[0] = 1
    state.observe(
        _event(
            2,
            "training",
            "progress",
            counter="steps",
            value=1,
            total=2,
            current_step=1,
            completed_steps=1,
        )
    )
    ticks[0] = 2
    assert not state.observe(
        _event(
            3,
            "training",
            "progress",
            counter="steps",
            value=1,
            total=2,
            current_step=1,
            completed_steps=1,
        )
    )
    with pytest.raises(ValueError, match="regressing or changed"):
        state.observe(
            _event(
                4,
                "training",
                "progress",
                counter="steps",
                value=0,
                total=2,
                current_step=2,
                completed_steps=2,
            )
        )
    with pytest.raises(ValueError, match="exceed expected"):
        state.observe(_event(4, "training", "complete", completed_steps=3))
    assert state.current_phase == "training"
    assert state.current_step == state.completed_steps == 1
    assert state.sequence == 3
    ticks[0] = 11
    assert state.deadline() == "no_progress"


def test_successful_exit_with_missing_updates_fails_closed(tmp_path: Path) -> None:
    directory = tmp_path / "incomplete"
    script = "import sys\nfrom pathlib import Path\nfrom sparselab.training.pilot_progress import activate_pilot_progress, emit_pilot_progress, pilot_phase\nwith activate_pilot_progress(purpose='smoke', directory=Path(sys.argv[1])):\n    with pilot_phase('training'):\n        emit_pilot_progress('progress', 'training', counter='steps', value=1, total=2, current_step=1, completed_steps=1)\n    with pilot_phase('pilot_complete', completed_steps=1):\n        pass"
    with pytest.raises(PilotProcessFailure) as caught:
        supervise_pilot(
            [sys.executable, "-c", script, str(directory)],
            purpose="smoke",
            directory=directory,
            policy=_policy(),
            expected_steps=2,
        )
    assert caught.value.evidence["reason"] == "incomplete_pilot_progress"
    assert caught.value.evidence["completed_steps"] == 1
    assert not (directory / "supervisor-completion.json").exists()
