"""CPU-only contracts for the Card 04 external GPU monitor wrapper."""

from __future__ import annotations

import os
import runpy
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
import yaml

from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError

TOOLS = Path(__file__).resolve().parents[1] / "tools/kernel-memory-lab"
READER = TOOLS / "read-vram-bytes.py"
LAUNCHER = TOOLS / "run-profile-phase.sh"


def _reader_module():
    return SimpleNamespace(**runpy.run_path(str(READER), run_name="kml_card04_reader"))


def _fake_amdsmi(
    monkeypatch: pytest.MonkeyPatch, *, used: int = 100
) -> SimpleNamespace:
    device = object()
    fake = SimpleNamespace(
        AmdSmiMemoryType=SimpleNamespace(VRAM=object()),
        amdsmi_init=Mock(),
        amdsmi_shut_down=Mock(),
        amdsmi_get_processor_handles=Mock(return_value=[device]),
        amdsmi_get_gpu_device_uuid=Mock(return_value="expected-uuid"),
        amdsmi_get_gpu_memory_usage=Mock(return_value=used),
        amdsmi_get_gpu_memory_total=Mock(return_value=1000),
    )
    monkeypatch.setitem(sys.modules, "amdsmi", fake)
    return fake


def test_reader_returns_whole_device_bytes_and_checks_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _fake_amdsmi(monkeypatch)
    _reader_module().main(["--expected-uuid", "expected-uuid"])
    assert capsys.readouterr().out == "100\n"
    fake.amdsmi_get_gpu_memory_usage.assert_called_once_with(
        fake.amdsmi_get_processor_handles.return_value[0],
        fake.AmdSmiMemoryType.VRAM,
    )
    fake.amdsmi_shut_down.assert_called_once()


@pytest.mark.parametrize(
    "failure",
    [
        "absent",
        "ambiguous",
        "uuid",
        "negative",
        "over_total",
        "bool",
        "float",
        "zero_total",
        "api_error",
    ],
)
def test_reader_fails_closed_on_untrusted_reading(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    fake = _fake_amdsmi(monkeypatch)
    if failure == "absent":
        fake.amdsmi_get_processor_handles.return_value = []
    elif failure == "ambiguous":
        fake.amdsmi_get_processor_handles.return_value = [object(), object()]
    elif failure == "uuid":
        fake.amdsmi_get_gpu_device_uuid.return_value = "other-uuid"
    elif failure == "negative":
        fake.amdsmi_get_gpu_memory_usage.return_value = -1
    elif failure == "over_total":
        fake.amdsmi_get_gpu_memory_usage.return_value = 1001
    elif failure == "bool":
        fake.amdsmi_get_gpu_memory_usage.return_value = True
    elif failure == "float":
        fake.amdsmi_get_gpu_memory_usage.return_value = float("nan")
    elif failure == "zero_total":
        fake.amdsmi_get_gpu_memory_total.return_value = 0
    else:
        fake.amdsmi_get_gpu_memory_usage.side_effect = RuntimeError("sensor lost")
    with pytest.raises(RuntimeError):
        _reader_module().read_vram_bytes("expected-uuid")
    fake.amdsmi_shut_down.assert_called_once()


def _mock_launch(
    tmp_path: Path,
    scenario: str,
    *,
    phase: str = "stage",
    budget: bool = True,
    outside_root: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    if sys.platform != "linux":
        pytest.skip("Card 04 launcher requires Linux/WSL GNU tools and Bash 5.1+")
    task_root = tmp_path / "task"
    root = (tmp_path if outside_root else task_root) / "attempt"
    root.mkdir(parents=True)
    task_root.mkdir(exist_ok=True)
    (root / "profile-baseline-bytes.txt").write_text("0\n")
    (root / "profile-baseline-inodes.txt").write_text("0\n")
    (root / "profile-monitor-policy.yaml").write_text("monitor_policy_version: 1\n")
    (root / "profile-resource-envelope.yaml").write_text(
        "resource_envelope_version: 1\n"
    )
    ledger = root / "attempt-budget.sqlite"
    if budget:
        ledger.write_text("mock only\n")
    (tmp_path / "fake-rocm").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = root / "mock-calls.txt"
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$MOCK_UV_CALLS"\n'
        'case "$*" in\n'
        "  *validate-profile-phase.py*) exit 0 ;;\n"
        "  *read-vram-bytes.py*)\n"
        '    count=$(cat "$MOCK_UV_COUNT" 2>/dev/null || printf 0)\n'
        "    count=$((count+1))\n"
        '    printf \'%s\\n\' "$count" > "$MOCK_UV_COUNT"\n'
        '    case "$MOCK_SCENARIO" in\n'
        "      preflight_failure) exit 9 ;;\n"
        "      preflight_hang) /bin/sleep 20 & wait ;;\n"
        "      midrun_hang) if (( count > 1 )); then /bin/sleep 20 & wait; fi ;;\n"
        "      midrun_failure|stubborn_child) if (( count > 1 )); then exit 9; fi ;;\n"
        "      over_cap) printf '21474836481\\n'; exit 0 ;;\n"
        "      invalid) printf 'unknown\\n'; exit 0 ;;\n"
        "      malformed_octal) printf '08\\n'; exit 0 ;;\n"
        "      oversized) printf '999999999999999999999\\n'; exit 0 ;;\n"
        "    esac\n"
        "    printf '1073741824\\n' ;;\n"
        "  *'sparselab monitor'*)\n"
        '    if [[ "$MOCK_SCENARIO" == stubborn_child ]]; then\n'
        '      bash -c \'trap "" TERM; echo $$ > "$KML_PROFILE_ROOT/stubborn.pid"; while :; do /bin/sleep 1; done\' &\n'
        "    fi\n"
        '    if [[ "$MOCK_SCENARIO" != success ]]; then /bin/sleep 20; fi ;;\n'
        "  *) exit 8 ;;\n"
        "esac\n"
    )
    uv.chmod(0o700)
    if scenario in {"disk_cap", "inode_cap", "watchdog_death"}:
        if scenario == "disk_cap":
            name, content = "du", "printf '21474836481\\troot\\n'\n"
        elif scenario == "inode_cap":
            name, content = "find", "for ((i=0; i<1001; i++)); do printf '\\0'; done\n"
        else:
            name, content = "sleep", 'kill -KILL "$PPID"\n'
        fake = bin_dir / name
        # Storage passes preflight, then crosses its cap while native work lives.
        fake.write_text(
            "#!/usr/bin/env bash\n"
            + (
                f'if [[ $(cat "$MOCK_UV_COUNT") == 1 ]]; then exec /usr/bin/{name} "$@"; fi\n'
                if name != "sleep"
                else ""
            )
            + content
        )
        fake.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "KML_PROFILE_ROOT": str(root),
            "KML_TASK_ROOT": str(task_root),
            "KML_EXPECTED_GPU_UUID": "expected-uuid",
            "KML_CHECKOUT": str(Path(__file__).resolve().parents[1]),
            "UV_PROJECT_ENVIRONMENT": str(tmp_path / "fake-rocm"),
            "SPARSELAB_ATTEMPT_BUDGET_LEDGER": str(ledger),
            "MOCK_SCENARIO": scenario,
            "MOCK_UV_CALLS": str(calls),
            "MOCK_UV_COUNT": str(root / "mock-count.txt"),
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
        if scenario == "stubborn_child":
            pid = int((root / "stubborn.pid").read_text())
            assert (
                not psutil.pid_exists(pid)
                or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    return result, root, calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize(
    "scenario",
    ["preflight_failure", "over_cap", "invalid", "malformed_octal", "oversized"],
)
def test_launcher_stops_before_native_monitor_on_bad_preflight(
    tmp_path: Path, scenario: str
) -> None:
    result, root, calls = _mock_launch(tmp_path, scenario)
    assert result.returncode != 0
    assert (root / "stage-cap-event.txt").is_file()
    assert len(calls) == 2
    assert "read-vram-bytes.py" in calls[1]
    assert not (root / "stage-monitor-result.json").exists()


def test_launcher_stops_owned_group_on_midrun_reader_failure(tmp_path: Path) -> None:
    result, root, calls = _mock_launch(tmp_path, "midrun_failure")
    assert result.returncode != 0
    assert (root / "stage-cap-event.txt").read_text() == (
        "Device-memory measurement unavailable\n"
    )
    assert sum("read-vram-bytes.py" in call for call in calls) == 2
    assert sum("sparselab monitor" in call for call in calls) == 1


@pytest.mark.parametrize("scenario", ["disk_cap", "inode_cap"])
def test_launcher_stops_owned_group_on_added_storage_cap(
    tmp_path: Path, scenario: str
) -> None:
    result, root, calls = _mock_launch(tmp_path, scenario)
    assert result.returncode != 0
    assert "Added disk/inodes cap" in (root / "stage-cap-event.txt").read_text()
    assert sum("sparselab monitor" in call for call in calls) == 1


@pytest.mark.parametrize("phase", ["stage", "train"])
def test_launcher_preserves_native_phase_command(tmp_path: Path, phase: str) -> None:
    result, root, calls = _mock_launch(tmp_path, "success", phase=phase)
    assert result.returncode == 0
    monitor = next(call for call in calls if "sparselab monitor" in call)
    assert "--reserve-bytes 21474836480" in monitor
    assert "--reserve-inodes 1000" in monitor
    if phase == "stage":
        assert "sparselab stage" in monitor
        assert "--through warmup" in monitor
        assert f"--output {root}/stage-v3" in monitor
    else:
        assert "sparselab train" in monitor
        assert f"--stage-bundle {root}/stage-v3" in monitor
        assert "--run-id kml-card04-synthetic-profile-v3" in monitor


def test_launcher_requires_budget_ledger(tmp_path: Path) -> None:
    result, root, calls = _mock_launch(tmp_path, "success", budget=False)
    assert result.returncode != 0
    assert calls == []
    assert not (root / "stage-monitor-result.json").exists()


def test_launcher_rejects_unmonitored_attempt_root(tmp_path: Path) -> None:
    result, root, calls = _mock_launch(tmp_path, "success", outside_root=True)
    assert result.returncode != 0
    assert "inside the monitored task root" in result.stderr
    assert calls == []
    assert not (root / "stage-monitor-result.json").exists()


@pytest.mark.parametrize(
    "scenario", ["preflight_hang", "midrun_hang", "watchdog_death", "stubborn_child"]
)
def test_launcher_stops_on_hung_sensor_or_dead_watchdog(
    tmp_path: Path, scenario: str
) -> None:
    result, root, calls = _mock_launch(tmp_path, scenario)
    assert result.returncode != 0
    assert (root / "stage-cap-event.txt").is_file()
    assert (root / "stage-exit-code.txt").read_text() != "0\n"
    assert sum("sparselab monitor" in call for call in calls) == (
        scenario != "preflight_hang"
    )


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "rss_missing",
        "rss_loose",
        "disk_margin",
        "inode_margin",
        "unreserved",
        "wall_time",
        "wrong_phase",
        "outside_outputs",
        "corrupt_ledger",
        "train_without_stage",
    ],
)
def test_native_launch_input_validation(tmp_path: Path, problem: str | None) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    policy = {
        "monitor_policy_version": 1,
        "max_tree_rss_bytes": 25769803776,
        "min_projected_disk_free_bytes": 2147483648,
        "min_projected_disk_free_inodes": 1000,
    }
    if problem == "rss_missing":
        del policy["max_tree_rss_bytes"]
    elif problem == "rss_loose":
        policy["max_tree_rss_bytes"] += 1
    elif problem == "disk_margin":
        policy["min_projected_disk_free_bytes"] -= 1
    elif problem == "inode_margin":
        policy["min_projected_disk_free_inodes"] -= 1
    (root / "profile-monitor-policy.yaml").write_text(yaml.safe_dump(policy))
    (root / "profile-resource-envelope.yaml").write_text(
        "resource_envelope_version: 1\n"
    )
    config = (
        tmp_path
        / "experiments/research/kernel-memory-lab/card04-synthetic-profile.yaml"
    )
    config.parent.mkdir(parents=True)
    output = tmp_path.parent if problem == "outside_outputs" else tmp_path
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"cache_dir": str(output / "cache")},
                "logging": {"root_dir": str(output / "runs")},
            }
        )
    )
    ledger = AttemptBudget.create(
        root / "budget.sqlite",
        max_updates=120,
        max_wall_seconds=1801 if problem == "wall_time" else 1800,
    )
    phase = "train" if problem == "train_without_stage" else "stage"
    if problem != "unreserved":
        label = "train" if problem == "wrong_phase" else "stage"
        ledger.reserve(f"command: bash /checkout/run-profile-phase.sh {label}", 7)
    if phase == "train":
        ledger.reserve("command: bash /checkout/run-profile-phase.sh train", 113)
    if problem == "corrupt_ledger":
        ledger.path.write_text("not a ledger")
    module = SimpleNamespace(**runpy.run_path(str(TOOLS / "validate-profile-phase.py")))
    if problem is None:
        module.validate(phase, root, tmp_path, tmp_path, ledger.path)
    else:
        with pytest.raises(
            (ValueError, AttemptBudgetError, sqlite3.DatabaseError, FileNotFoundError)
        ):
            module.validate(phase, root, tmp_path, tmp_path, ledger.path)
