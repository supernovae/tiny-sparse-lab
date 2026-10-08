"""CPU and mocked contracts for the bounded Card 05 real-data launcher."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.config.loading import load_config
from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError
from sparselab.training.manifest import config_sha256

TOOLS = Path(__file__).resolve().parents[1] / "tools/kernel-memory-lab"
LAUNCHER = TOOLS / "run-real-data-measurement.sh"
VALIDATOR = TOOLS / "validate-real-data-phase.py"
CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments/research/kernel-memory-lab/card05-real-data-measurement.yaml"
)


def _mock_launch(
    tmp_path: Path, scenario: str, *, phase: str = "stage"
) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    if sys.platform != "linux":
        pytest.skip("launcher requires Linux/WSL")
    task_root = tmp_path / "task"
    root = task_root / "attempt"
    root.mkdir(parents=True)
    (
        task_root
        / "corpora/kernel-memory-lab-card03-scale-retry1/prep/prepared-bundle-v1"
    ).mkdir(parents=True)
    (
        task_root
        / "corpora/kernel-memory-lab-card03-scale-retry1/prep/prepared-bundle-v1/inputs.json"
    ).write_text("mock bundle")
    (root / "profile-baseline-bytes.txt").write_text("0\n")
    (root / "profile-baseline-inodes.txt").write_text("0\n")
    (root / "profile-baseline-sha256.txt").write_text(
        ("bad" if scenario == "bad_baseline" else hashlib.sha256(b"0\n0\n").hexdigest())
        + "\n"
    )
    (root / "profile-monitor-policy.yaml").write_text("monitor_policy_version: 1\n")
    (root / "profile-resource-envelope.yaml").write_text(
        "resource_envelope_version: 1\n"
    )
    (root / "attempt-budget.sqlite").write_text("mock only\n")
    (tmp_path / "fake-rocm").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = root / "calls.txt"
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in *bounded-measurement.py*) shift 4; exec "$MOCK_REAL_PYTHON" "$@" ;; esac\n'
        'printf "%s\\n" "$*" >> "$MOCK_CALLS"\n'
        'case "$*" in\n'
        "  *validate-real-data-phase.py*) exit 0 ;;\n"
        "  *read-vram-bytes.py*)\n"
        '    count=$(cat "$MOCK_COUNT" 2>/dev/null || printf 0)\n'
        '    count=$((count+1)); printf "%s\\n" "$count" > "$MOCK_COUNT"\n'
        '    if [[ "$MOCK_SCENARIO" == bad_vram ]]; then exit 9; fi\n'
        '    if [[ "$MOCK_SCENARIO" == midrun_loss && "$count" -gt 1 ]]; then exit 9; fi\n'
        '    if [[ "$MOCK_SCENARIO" == over_vram ]]; then printf "21474836481\\n"; exit 0; fi\n'
        '    printf "1073741824\\n" ;;\n'
        '  *"sparselab monitor"*)\n'
        '    if [[ "$MOCK_SCENARIO" == midrun_loss || "$MOCK_SCENARIO" == disk_cap ]]; then /bin/sleep 20; fi ;;\n'
        "  *) exit 8 ;;\n"
        "esac\n"
    )
    uv.chmod(0o700)
    if scenario == "disk_cap":
        fake_du = bin_dir / "du"
        fake_du.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ $(cat "$MOCK_COUNT") == 1 ]]; then exec /usr/bin/du "$@"; fi\n'
            "printf '21474836481\\troot\\n'\n"
        )
        fake_du.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "KML_PROFILE_ROOT": str(root),
            "KML_TASK_ROOT": str(task_root),
            "KML_EXPECTED_GPU_UUID": "expected-uuid",
            "KML_CHECKOUT": str(Path(__file__).resolve().parents[1]),
            "UV_PROJECT_ENVIRONMENT": str(tmp_path / "fake-rocm"),
            "SPARSELAB_ATTEMPT_BUDGET_LEDGER": str(root / "attempt-budget.sqlite"),
            "MOCK_REAL_PYTHON": sys.executable,
            "MOCK_SCENARIO": scenario,
            "MOCK_CALLS": str(calls),
            "MOCK_COUNT": str(root / "count.txt"),
        }
    )
    process = subprocess.Popen(
        ["bash", str(LAUNCHER), phase],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=12)
        result = subprocess.CompletedProcess(
            process.args, process.returncode, stdout, stderr
        )
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    return result, root, calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize("phase", ["stage", "train"])
def test_launcher_uses_exact_zero_pilot_and_32_update_commands(
    tmp_path: Path, phase: str
) -> None:
    result, root, calls = _mock_launch(tmp_path, "success", phase=phase)
    assert result.returncode == 0, result.stderr
    assert any("validate-real-data-phase.py" in call for call in calls)
    monitor = next(call for call in calls if "sparselab monitor" in call)
    assert "--reserve-bytes 21474836480 --reserve-inodes 1000" in monitor
    assert "card05-real-data-measurement.yaml" in monitor
    if phase == "stage":
        assert "--through validate --prepared-inputs" in monitor
        assert f"--output {root}/stage-validate" in monitor
        assert "--through warmup" not in monitor
    else:
        assert f"--stage-bundle {root}/stage-validate" in monitor
        assert "--stop-after-step 32" in monitor
        assert "--run-id kml-card05-realdata-measurement-p1" in monitor
        assert "--resume" not in monitor


@pytest.mark.parametrize(
    "scenario", ["bad_vram", "over_vram", "midrun_loss", "disk_cap"]
)
def test_launcher_fails_closed_on_resource_fault(tmp_path: Path, scenario: str) -> None:
    result, root, calls = _mock_launch(tmp_path, scenario)
    assert result.returncode != 0
    assert (root / "stage-cap-event.txt").is_file()
    assert (root / "stage-exit-code.txt").read_text() != "0\n"
    assert sum("sparselab monitor" in call for call in calls) == (
        scenario in {"midrun_loss", "disk_cap"}
    )


def test_launcher_rejects_changed_baseline_before_gpu_read(tmp_path: Path) -> None:
    result, _, calls = _mock_launch(tmp_path, "bad_baseline")
    assert result.returncode != 0
    assert "storage baseline changed" in result.stderr
    assert calls == []


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "unreserved",
        "wrong_updates",
        "wall_time",
        "loose_rss",
        "loose_headroom",
        "bad_baseline",
        "changed_inputs",
        "existing_run",
        "train_without_stage",
        "valid_train",
    ],
)
def test_phase_validator_enforces_bound_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    prepared = (
        tmp_path
        / "corpora/kernel-memory-lab-card03-scale-retry1/prep/prepared-bundle-v1"
    )
    prepared.mkdir(parents=True)
    config = load_config(CONFIG)
    dataset = config.dataset.model_copy(update={"cache_dir": tmp_path / "cache"})
    logging = config.logging.model_copy(update={"root_dir": tmp_path / "runs"})
    config = config.model_copy(update={"dataset": dataset, "logging": logging})
    (prepared / "inputs.json").write_text(
        json.dumps(
            {
                "requested_config": config.model_dump(mode="json"),
                "source_identity_sha256": "source",
            }
        )
    )
    policy = {
        "monitor_policy_version": 1,
        "max_tree_rss_bytes": 25_769_803_776,
        "min_projected_disk_free_bytes": 2_147_483_648,
        "min_projected_disk_free_inodes": 1000,
        "interval_seconds": 1,
    }
    if problem == "loose_rss":
        policy["max_tree_rss_bytes"] += 1
    if problem == "loose_headroom":
        policy["min_projected_disk_free_bytes"] -= 1
    (root / "profile-monitor-policy.yaml").write_text(yaml.safe_dump(policy))
    (root / "profile-resource-envelope.yaml").write_text(
        yaml.safe_dump(
            {
                "resource_envelope_version": 1,
                "max_rss_bytes": 25_769_803_776,
                "min_disk_bytes": 23_622_320_128,
                "min_inodes": 2000,
            }
        )
    )
    (root / "profile-baseline-bytes.txt").write_text("100\n")
    (root / "profile-baseline-inodes.txt").write_text("10\n")
    (root / "profile-baseline-sha256.txt").write_text(
        "bad\n"
        if problem == "bad_baseline"
        else hashlib.sha256(b"100\n10\n").hexdigest()
    )
    ledger = AttemptBudget.create(
        root / "budget.sqlite",
        max_updates=32,
        max_wall_seconds=1801 if problem == "wall_time" else 1800,
    )
    if problem != "unreserved":
        ledger.reserve(
            "command: bash /checkout/run-real-data-measurement.sh stage",
            1 if problem == "wrong_updates" else 0,
        )
    if problem == "changed_inputs":
        (prepared / "inputs.json").write_text("changed")
    if problem == "existing_run":
        (tmp_path / "runs/kml-card05-realdata-measurement-p1").mkdir(parents=True)
    module = SimpleNamespace(**runpy.run_path(str(VALIDATOR)))
    monkeypatch.setitem(module.validate.__globals__, "load_config", lambda _: config)
    monkeypatch.setitem(
        module.validate.__globals__,
        "_CONFIG_SHA256",
        config_sha256(config.model_dump(mode="json")),
    )
    monkeypatch.setitem(
        module.validate.__globals__,
        "sha256_file",
        lambda _: "wrong" if problem == "changed_inputs" else "expected",
    )
    monkeypatch.setitem(module.validate.__globals__, "_INPUTS_SHA256", "expected")
    monkeypatch.setitem(
        module.validate.__globals__, "source_identity", lambda: {"sha256": "source"}
    )
    phase = "train" if problem in {"train_without_stage", "valid_train"} else "stage"
    if phase == "train":
        ledger.reserve("command: bash /checkout/run-real-data-measurement.sh train", 32)
    if problem == "valid_train":
        (root / "stage-exit-code.txt").write_text("0\n")
        (root / "monitor-stage").mkdir()
        (root / "monitor-stage/completion.json").write_text("mock")
        monkeypatch.setitem(
            module.validate.__globals__,
            "MonitorCompletion",
            SimpleNamespace(
                model_validate_json=lambda _: SimpleNamespace(
                    status="COMPLETE", returncode=0
                )
            ),
        )
    if problem is None or problem == "valid_train":
        module.validate(phase, root, tmp_path, tmp_path, ledger.path)
    else:
        with pytest.raises(
            (
                ValueError,
                AttemptBudgetError,
                sqlite3.DatabaseError,
                FileNotFoundError,
                FileExistsError,
            )
        ):
            module.validate(phase, root, tmp_path, tmp_path, ledger.path)
