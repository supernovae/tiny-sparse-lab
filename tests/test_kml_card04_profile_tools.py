"""CPU-only contracts for the Card 04 external GPU monitor wrapper."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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
    "failure", ["absent", "ambiguous", "uuid", "negative", "over_total"]
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
    else:
        fake.amdsmi_get_gpu_memory_usage.return_value = 1001
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
        "  *read-vram-bytes.py*)\n"
        '    count=$(cat "$MOCK_UV_COUNT" 2>/dev/null || printf 0)\n'
        "    count=$((count+1))\n"
        '    printf \'%s\\n\' "$count" > "$MOCK_UV_COUNT"\n'
        '    case "$MOCK_SCENARIO" in\n'
        "      preflight_failure) exit 9 ;;\n"
        "      midrun_failure) if (( count > 1 )); then exit 9; fi ;;\n"
        "      over_cap) printf '21474836481\\n'; exit 0 ;;\n"
        "      invalid) printf 'unknown\\n'; exit 0 ;;\n"
        "      malformed_octal) printf '08\\n'; exit 0 ;;\n"
        "      oversized) printf '999999999999999999999\\n'; exit 0 ;;\n"
        "    esac\n"
        "    printf '1073741824\\n' ;;\n"
        "  *'sparselab monitor'*)\n"
        '    if [[ "$MOCK_SCENARIO" == midrun_failure || "$MOCK_SCENARIO" == disk_cap || "$MOCK_SCENARIO" == inode_cap ]]; then sleep 20; fi ;;\n'
        "  *) exit 8 ;;\n"
        "esac\n"
    )
    uv.chmod(0o700)
    if scenario == "disk_cap":
        fake_du = bin_dir / "du"
        fake_du.write_text("#!/usr/bin/env bash\nprintf '21474836481\\t%s\\n' \"$2\"\n")
        fake_du.chmod(0o700)
    if scenario == "inode_cap":
        fake_find = bin_dir / "find"
        fake_find.write_text(
            "#!/usr/bin/env bash\nfor ((i=0; i<1001; i++)); do printf 'inode\\n'; done\n"
        )
        fake_find.chmod(0o700)
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
    result = subprocess.run(
        ["bash", str(LAUNCHER), phase],
        env=environment,
        capture_output=True,
        text=True,
        timeout=8,
        start_new_session=True,
        check=False,
    )
    return result, root, calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize(
    "scenario",
    ["preflight_failure", "over_cap", "invalid", "malformed_octal", "oversized"],
)
def test_launcher_stops_before_native_monitor_on_bad_preflight(
    tmp_path: Path, scenario: str
) -> None:
    result, root, calls = _mock_launch(tmp_path, scenario)
    assert result.returncode == 2
    assert (root / "stage-cap-event.txt").is_file()
    assert len(calls) == 1
    assert "read-vram-bytes.py" in calls[0]
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
    assert result.returncode == 2
    assert calls == []
    assert not (root / "stage-monitor-result.json").exists()


def test_launcher_rejects_unmonitored_attempt_root(tmp_path: Path) -> None:
    result, root, calls = _mock_launch(tmp_path, "success", outside_root=True)
    assert result.returncode == 2
    assert "inside the monitored task root" in result.stderr
    assert calls == []
    assert not (root / "stage-monitor-result.json").exists()
