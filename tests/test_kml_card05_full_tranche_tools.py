"""Offline contracts for the one-shot Card 05 full-tranche launcher."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools/kernel-memory-lab"
LAUNCHER = TOOLS / "run-full-tranche-phase.sh"


def _launch(tmp_path: Path, scenario: str, phase: str = "stage") -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    if sys.platform != "linux":
        pytest.skip("launcher requires Linux")
    task = tmp_path / "task"
    root = task / "attempt"
    root.mkdir(parents=True)
    (root / "profile-baseline-bytes.txt").write_text("0\n")
    (root / "profile-baseline-inodes.txt").write_text("0\n")
    (root / "profile-baseline-sha256.txt").write_text(hashlib.sha256(b"0\n0\n").hexdigest() + "\n")
    if scenario != "missing_sampler_marker":
        (root / "profile-baseline-sampler.txt").write_text("sample-task-root-v1\n")
    (root / "profile-monitor-policy.yaml").write_text("monitor_policy_version: 1\n")
    (root / "profile-resource-envelope.yaml").write_text("resource_envelope_version: 1\n")
    (root / "attempt-budget.sqlite").write_text("mock only\n")
    (tmp_path / "fake-rocm").mkdir()
    (task / "card04-synthetic/runs/kml-card05-full-tranche-v1/evaluations").mkdir(parents=True)
    if scenario == "disk_cap":
        with (root / "oversize.bin").open("wb") as stream:
            stream.truncate(68_719_476_737)
    if scenario == "inode_cap":
        for index in range(2001):
            (root / f"inode-{index}").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = root / "calls.txt"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in *bounded-measurement.py*) shift 4; exec "$MOCK_PYTHON" "$@" ;; esac\n'
        'case "$*" in *sample-task-root.py*) shift 4; exec "$MOCK_PYTHON" "$@" ;; esac\n'
        'case "$*" in *run-owned-phase-command.py*) shift 4; exec "$MOCK_PYTHON" "$@" ;; esac\n'
        'printf "%s\\n" "$*" >> "$MOCK_CALLS"\n'
        'case "$*" in\n'
        '  *full-tranche-phase-deadline.py*) printf "%s\\n" "$(($(date +%s)+60))000000000" ;;\n'
        '  *validate-full-tranche-phase.py*) exit 0 ;;\n'
        '  *read-vram-bytes.py*)\n'
        '    count=$(cat "$MOCK_COUNT" 2>/dev/null || printf 0)\n'
        '    count=$((count+1)); printf "%s\\n" "$count" > "$MOCK_COUNT"\n'
        '    if [[ "$MOCK_SCENARIO" == sensor_loss || "$MOCK_SCENARIO" == sensor_loss_worker ]] && [[ "$count" -gt 1 ]]; then exit 9; fi\n'
        '    if [[ "$MOCK_SCENARIO" == over_vram ]]; then printf "21474836481\\n"; else printf "1073741824\\n"; fi ;;\n'
        '  *"sparselab monitor"*)\n'
        '    if [[ "$MOCK_SCENARIO" == orphan_worker ]]; then exec "$MOCK_PYTHON" "$MOCK_ORPHAN_SCRIPT"; fi\n'
        '    if [[ "$MOCK_SCENARIO" == sensor_loss_worker || "$MOCK_SCENARIO" == owner_dies ]]; then exec "$MOCK_PYTHON" "$MOCK_LIVE_SCRIPT"; fi\n'
        '    if [[ "$MOCK_SCENARIO" == sensor_loss ]]; then /bin/sleep 20; fi ;;\n'
        '  *) exit 8 ;;\n'
        'esac\n'
    )
    fake_uv.chmod(0o700)
    orphan_script = tmp_path / "orphan-parent.py"
    orphan_script.write_text(
        "import os, subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'], "
        "start_new_session=True)\n"
        "open(os.environ['MOCK_WORKER_PID'], 'w').write(str(p.pid))\n"
    )
    live_script = tmp_path / "live-parent.py"
    live_script.write_text(orphan_script.read_text() + "import time; time.sleep(30)\n")
    env = os.environ.copy()
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        KML_PROFILE_ROOT=str(root),
        KML_TASK_ROOT=str(task),
        KML_EXPECTED_GPU_UUID="expected-uuid",
        KML_CHECKOUT=str(Path(__file__).resolve().parents[1]),
        UV_PROJECT_ENVIRONMENT=str(tmp_path / "fake-rocm"),
        SPARSELAB_ATTEMPT_BUDGET_LEDGER=str(root / "attempt-budget.sqlite"),
        MOCK_PYTHON=sys.executable,
        MOCK_SCENARIO=scenario,
        MOCK_CALLS=str(calls),
        MOCK_COUNT=str(root / "count.txt"),
        MOCK_ORPHAN_SCRIPT=str(orphan_script),
        MOCK_LIVE_SCRIPT=str(live_script),
        MOCK_WORKER_PID=str(root / "worker-pid.txt"),
    )
    process = subprocess.Popen(
        ["bash", str(LAUNCHER), phase],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        if scenario == "owner_dies":
            until = time.monotonic() + 6
            while not (root / "worker-pid.txt").exists() and time.monotonic() < until:
                time.sleep(0.01)
            assert (root / "worker-pid.txt").exists(), "worker never started"
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    leaked_group: list[int] = []
    for candidate in psutil.process_iter():
        try:
            if candidate.pid != process.pid and candidate.status() != psutil.STATUS_ZOMBIE and os.getpgid(candidate.pid) == process.pid:
                leaked_group.append(candidate.pid)
        except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if leaked_group:
        os.killpg(process.pid, signal.SIGKILL)
        pytest.fail(f"launcher returned with live same-group workers: {leaked_group}")
    result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    return result, root, calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize("phase", ["stage", "train", "evaluate"])
def test_declared_phase_commands(tmp_path: Path, phase: str) -> None:
    result, root, calls = _launch(tmp_path, "success", phase)
    assert result.returncode == 0, result.stderr
    assert sum("full-tranche-phase-deadline.py" in call for call in calls) == 1
    assert any("validate-full-tranche-phase.py" in call for call in calls)
    monitors = [call for call in calls if "sparselab monitor" in call]
    assert monitors
    assert all("--reserve-bytes 68719476736 --reserve-inodes 2000" in call for call in monitors)
    command = monitors[0]
    if phase == "stage":
        assert f"--through validate --output {root}/stage-validate" in command
        assert "--prepared-inputs" not in command
    elif phase == "train":
        assert "--run-id kml-card05-full-tranche-v1" in command
        assert "--stop-after-step" not in command and "--resume" not in command
        assert len(monitors) == 2
    else:
        assert "run-full-tranche-evaluation.py" in command


@pytest.mark.parametrize("scenario", ["over_vram", "sensor_loss", "disk_cap", "inode_cap"])
def test_sensor_failure_stops_before_or_during_work(tmp_path: Path, scenario: str) -> None:
    result, root, calls = _launch(tmp_path, scenario)
    assert result.returncode != 0
    assert (root / "stage-cap-event.txt").is_file()
    assert (root / "stage-exit-code.txt").read_text() != "0\n"
    assert sum("sparselab monitor" in call for call in calls) == (scenario == "sensor_loss")


def test_baseline_sampler_identity_required_before_measurement(tmp_path: Path) -> None:
    result, _, calls = _launch(tmp_path, "missing_sampler_marker")
    assert result.returncode != 0
    assert calls == []


def test_claimed_phase_cannot_launch(tmp_path: Path) -> None:
    # The claim directory is created exclusively by the launcher. A replay
    # fails before a deadline helper, device read, or native command.
    result, root, _ = _launch(tmp_path, "success")
    assert result.returncode == 0
    assert (root / "stage-launch-claim").is_dir()
    calls_before = (root / "calls.txt").read_text()
    env = os.environ.copy()
    env.update(
        KML_PROFILE_ROOT=str(root), KML_TASK_ROOT=str(root.parent),
        KML_EXPECTED_GPU_UUID="expected-uuid",
        KML_CHECKOUT=str(Path(__file__).resolve().parents[1]),
        UV_PROJECT_ENVIRONMENT=str(tmp_path / "fake-rocm"),
        SPARSELAB_ATTEMPT_BUDGET_LEDGER=str(root / "attempt-budget.sqlite"),
    )
    replay = subprocess.run(["bash", str(LAUNCHER), "stage"], env=env, capture_output=True, text=True, check=False)
    assert replay.returncode != 0
    assert (root / "calls.txt").read_text() == calls_before


def test_invalid_reservation_rejected_before_input_access(tmp_path: Path) -> None:
    import runpy

    from sparselab.training.attempt_budget import AttemptBudget

    task = tmp_path / "task"
    root = task / "attempt"
    root.mkdir(parents=True)
    ledger = AttemptBudget.create(root / "budget.sqlite", max_updates=4883, max_wall_seconds=10800)
    module = runpy.run_path(str(TOOLS / "validate-full-tranche-phase.py"))
    validate = module["validate"]
    with pytest.raises(ValueError, match="reservations"):
        validate("stage", root, task, Path(__file__).resolve().parents[1], ledger.path)
    ledger.reserve("command: bash /checkout/run-full-tranche-phase.sh stage", 0)
    ledger.reserve("command: bash /checkout/run-full-tranche-phase.sh train", 4882)
    with pytest.raises(ValueError, match="reservations"):
        validate("train", root, task, Path(__file__).resolve().parents[1], ledger.path)


def test_partial_generation_preserved_without_success_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import runpy

    root = tmp_path / "attempt"
    root.mkdir()
    (root / "selected-checkpoint.json").write_text(json.dumps({
        "run_id": "kml-card05-full-tranche-v1", "selected_step": 0,
        "checkpoint": "step_00000000_gen_000001", "checkpoint_sha256": "sealed",
    }))
    evaluate = runpy.run_path(str(TOOLS / "run-full-tranche-evaluation.py"))["evaluate"]
    g = evaluate.__globals__
    for name, value in {
        "evaluation_config": lambda *_: object(),
        "profile_for_id": lambda *_: object(),
        "authorize_profile": lambda *_: object(),
        "run_suite": lambda *_args, **_kwargs: tmp_path / "index.json",
        "verify_evaluation_index": lambda *_: {"run_id": "kml-card05-full-tranche-v1", "checkpoint_sha256": "sealed"},
        "run_panel": lambda *_args, **_kwargs: tmp_path / "panel.json",
        "verify_panel_result": lambda *_: {"rows": [{"status": "COMPLETED"}] * 199 + [{"status": "FAILED"}]},
    }.items():
        monkeypatch.setitem(g, name, value)
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(root, Path(__file__).resolve().parents[1])
    assert not (root / "evaluation-generation-complete.json").exists()


def test_launcher_reaps_early_parent_separate_group_worker_without_touching_sentinel(tmp_path: Path) -> None:
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    worker_pid: int | None = None
    leaked = False
    try:
        result, root, calls = _launch(tmp_path, "orphan_worker")
        worker_pid = int((root / "worker-pid.txt").read_text())
        leaked = psutil.pid_exists(worker_pid)
        assert result.returncode != 0
        assert any("sparselab monitor" in call for call in calls)
        assert (root / "stage-cap-event.txt").is_file()
        assert not leaked, f"owned worker {worker_pid} survived launcher return"
        assert sentinel.poll() is None, "unrelated sentinel was affected"
        receipt = root / "stage-launch-claim/owned-completion.json"
        assert '"living_descendants": 0' in receipt.read_text()
    finally:
        # Cleanup after the assertions is only hygiene; a leaked worker has
        # already failed the test and cannot be hidden by this teardown.
        if worker_pid is not None and psutil.pid_exists(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)
        sentinel.kill()
        sentinel.wait(timeout=5)


def test_watchdog_failure_kills_separate_group_term_ignoring_worker(tmp_path: Path) -> None:
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    worker_pid: int | None = None
    leaked = False
    try:
        result, root, _ = _launch(tmp_path, "sensor_loss_worker")
        worker_pid = int((root / "worker-pid.txt").read_text())
        leaked = psutil.pid_exists(worker_pid)
        assert result.returncode != 0
        assert (root / "stage-cap-event.txt").read_text().strip() == "Device-memory measurement unavailable"
        assert not leaked, f"owned worker {worker_pid} survived launcher return"
        assert sentinel.poll() is None, "unrelated sentinel was affected"
        receipt = root / "stage-launch-claim/owned-completion.json"
        assert '"living_descendants": 0' in receipt.read_text()
    finally:
        if worker_pid is not None and psutil.pid_exists(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)
        sentinel.kill()
        sentinel.wait(timeout=5)


def test_supervisor_reaps_worker_when_launcher_group_exits_early(tmp_path: Path) -> None:
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    worker_pid: int | None = None
    leaked = False
    try:
        result, root, _ = _launch(tmp_path, "owner_dies")
        worker_pid = int((root / "worker-pid.txt").read_text())
        until = time.monotonic() + 5
        while psutil.pid_exists(worker_pid) and time.monotonic() < until:
            time.sleep(0.02)
        leaked = psutil.pid_exists(worker_pid)
        assert result.returncode != 0
        assert not leaked, f"worker {worker_pid} survived owner death"
        assert sentinel.poll() is None, "unrelated sentinel was affected"
        receipt = root / "stage-launch-claim/owned-completion.json"
        assert '"living_descendants": 0' in receipt.read_text()
        assert "launcher owner exited" in receipt.read_text()
    finally:
        if worker_pid is not None and psutil.pid_exists(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)
        sentinel.kill()
        sentinel.wait(timeout=5)
