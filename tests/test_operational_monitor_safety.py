"""CPU-only, zero-update safety checks for native operational monitoring."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from sparselab import operational_monitor as monitor


def _policy(path: Path, **fields: object) -> monitor.MonitorPolicy:
    policy = monitor.MonitorPolicy(monitor_policy_version=1, **fields)
    path.write_text(
        "monitor_policy_version: 1\n"
        + "".join(f"{key}: {value}\n" for key, value in fields.items())
    )
    return policy


def test_absent_optional_fields_preserve_legacy_policy_and_receipt_serialization() -> (
    None
):
    policy = monitor.MonitorPolicy(monitor_policy_version=1)
    data = policy.model_dump()
    assert set(data) == {
        "monitor_policy_version",
        "max_tree_rss_bytes",
        "max_tree_swap_bytes",
        "min_host_available_ram_bytes",
        "min_host_free_swap_bytes",
        "min_disk_free_bytes",
        "min_disk_free_inodes",
        "min_projected_disk_free_bytes",
        "min_projected_disk_free_inodes",
        "interval_seconds",
        "termination_grace_seconds",
    }
    assert policy.model_dump_json() == json.dumps(
        {
            "monitor_policy_version": 1,
            "max_tree_rss_bytes": None,
            "max_tree_swap_bytes": None,
            "min_host_available_ram_bytes": None,
            "min_host_free_swap_bytes": None,
            "min_disk_free_bytes": None,
            "min_disk_free_inodes": None,
            "min_projected_disk_free_bytes": None,
            "min_projected_disk_free_inodes": None,
            "interval_seconds": 1.0,
            "termination_grace_seconds": 5.0,
        },
        separators=(",", ":"),
    )
    assert (
        "device_memory_bytes"
        not in monitor.MonitorSample(
            elapsed_seconds=0,
            processes=[],
            tree_rss_bytes=None,
            tree_swap_bytes=None,
            host_available_ram_bytes=None,
            host_free_swap_bytes=None,
            host_used_swap_bytes=None,
            disk_free_bytes=None,
            disk_free_inodes=None,
            projected_disk_free_bytes=None,
            projected_disk_free_inodes=None,
            violations=[],
        ).model_dump()
    )


def test_legacy_monitor_receipts_have_no_new_optional_fields(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy = _policy(policy_path, interval_seconds=0.02)
    result = monitor.monitor_command(
        [sys.executable, "-c", "import time; time.sleep(0.1)"],
        policy,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        policy_path=policy_path,
    )
    assert result.status == "COMPLETE"
    assert "baseline_sha256" not in result.launch.model_dump()
    assert "owned_source_sha256" not in result.launch.model_dump()
    assert "peak_device_memory_bytes" not in result.model_dump()
    assert "peak_added_workspace_bytes" not in result.model_dump()
    assert "peak_added_workspace_inodes" not in result.model_dump()
    receipt = json.loads((tmp_path / "logs" / "completion.json").read_text())
    assert receipt["launch"]["policy"] == policy.model_dump()
    assert "baseline_sha256" not in receipt["launch"]


def test_baseline_identity_is_immutable_and_live_files_counted(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "db.sqlite").write_bytes(b"a")
    baseline_path = tmp_path / "baseline.json"
    baseline = monitor.capture_workspace_baseline(root, baseline_path)
    assert monitor.load_workspace_baseline(baseline_path, root) == baseline
    with pytest.raises(FileExistsError):
        monitor.capture_workspace_baseline(root, baseline_path)
    (root / "db.sqlite-wal").write_bytes(b"wal")
    (root / "db.sqlite-shm").write_bytes(b"shm")
    now_bytes, now_inodes = monitor.sample_workspace_tree(root)
    assert now_bytes - baseline.apparent_bytes == 6
    assert now_inodes - baseline.inodes == 2
    altered = json.loads(baseline_path.read_text())
    altered["apparent_bytes"] = 0
    baseline_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="digest mismatch"):
        monitor.load_workspace_baseline(baseline_path, root)


def test_replaced_same_filesystem_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    baseline_path = tmp_path / "baseline.json"
    baseline = monitor.capture_workspace_baseline(root, baseline_path)
    root.rename(tmp_path / "old-root")
    root.mkdir()
    with pytest.raises(ValueError, match="identity|filesystem"):
        monitor.load_workspace_baseline(baseline_path, root)
    policy = monitor.MonitorPolicy(
        monitor_policy_version=1, max_added_workspace_inodes=100
    )
    # Sample uses the same root identity check after launch; no fresh baseline.
    result = monitor.sample(
        monitor.ProcessIdentity(
            pid=os.getpid(), create_time=psutil.Process().create_time()
        ),
        {
            os.getpid(): monitor.ProcessIdentity(
                pid=os.getpid(), create_time=psutil.Process().create_time()
            )
        },
        policy,
        root,
        baseline=baseline,
    )
    assert "workspace_sampling" in result.violations
    assert result.model_dump()["added_workspace_inodes"] is None


def test_transient_path_restarts_entire_sample_but_real_io_error_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "db.sqlite-wal").write_bytes(b"live")
    original = monitor.os.stat
    calls = 0

    def transient(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal calls
        if str(path).endswith("db.sqlite-wal") and calls == 0:
            calls += 1
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(monitor.os, "stat", transient)
    assert monitor.sample_workspace_tree(root, seconds=0.5)[1] == 2
    assert calls == 1

    def denied(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if str(path).endswith("db.sqlite-wal"):
            raise PermissionError(errno.EACCES, "denied", str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(monitor.os, "stat", denied)
    with pytest.raises(PermissionError):
        monitor.sample_workspace_tree(root, seconds=0.5)


def test_persistent_disappearance_exhausts_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        monitor,
        "_sample_tree_once",
        lambda _root, _deadline: (_ for _ in ()).throw(FileNotFoundError("gone")),
    )
    with pytest.raises(TimeoutError, match="sampling deadline exhausted"):
        monitor.sample_workspace_tree(tmp_path, seconds=0.02)


def test_added_cap_and_sensor_cap_are_sampled_after_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    baseline = monitor.capture_workspace_baseline(root, tmp_path / "baseline.json")
    (root / "new").write_bytes(b"over")
    monkeypatch.setattr(monitor, "read_device_memory_bytes", lambda _uuid: 101)
    identity = monitor.ProcessIdentity(
        pid=os.getpid(), create_time=psutil.Process().create_time()
    )
    result = monitor.sample(
        identity,
        {identity.pid: identity},
        monitor.MonitorPolicy(
            monitor_policy_version=1,
            max_device_memory_bytes=100,
            expected_device_uuid="uuid",
            max_added_workspace_bytes=3,
            max_added_workspace_inodes=0,
        ),
        root,
        baseline=baseline,
    )
    assert set(result.violations) == {
        "max_device_memory_bytes",
        "max_added_workspace_bytes",
        "max_added_workspace_inodes",
    }
    assert result.device_memory_bytes == 101
    assert result.added_workspace_bytes == 4
    assert result.added_workspace_inodes == 1


def test_insufficient_margin_rejected_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy = _policy(policy_path, min_projected_disk_free_bytes=50)
    monkeypatch.setattr(monitor, "_storage", lambda _root: (100, 100))
    with pytest.raises(ValueError, match="insufficient free workspace margin"):
        monitor.monitor_command(
            [sys.executable, "-c", "pass"],
            policy,
            workspace=tmp_path,
            log_dir=tmp_path / "never-created",
            policy_path=policy_path,
            reserved_bytes=51,
        )
    assert not (tmp_path / "never-created").exists()


def test_nonfinite_wall_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        monitor.MonitorPolicy(monitor_policy_version=1, max_wall_seconds=float("inf"))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_baseline_mutation_during_command_fails_completion(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    baseline_path = tmp_path / "baseline.json"
    monitor.capture_workspace_baseline(root, baseline_path)
    policy_path = tmp_path / "policy.yaml"
    policy = _policy(
        policy_path,
        max_added_workspace_bytes=100000,
        max_added_workspace_inodes=100,
        max_wall_seconds=3,
        interval_seconds=0.02,
    )
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(baseline_path)!r}).write_text('tampered')",
    ]
    result = monitor.monitor_command(
        command,
        policy,
        workspace=root,
        log_dir=root / "logs",
        policy_path=policy_path,
        baseline_path=baseline_path,
    )
    assert result.status == "MONITOR_ERROR"
    assert any("workspace baseline unverified" in value for value in result.violations)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
@pytest.mark.parametrize("child_seconds", [0.1, 0.3])
def test_final_receipts_cannot_cross_cap_and_still_claim_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child_seconds: float
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    baseline_path = tmp_path / "baseline.json"
    baseline = monitor.capture_workspace_baseline(root, baseline_path)
    policy_path = tmp_path / "policy.yaml"
    cap = 100000
    policy = _policy(
        policy_path,
        max_added_workspace_bytes=cap,
        max_added_workspace_inodes=100,
        max_wall_seconds=3,
        interval_seconds=0.02,
    )
    native_projection = monitor._final_storage_projection
    prefinal_bytes = []

    def leave_one_byte_before_final_receipts(result, *args):
        # Sampling cadence and telemetry length differ between command runs.
        # Place this run one byte below its fixed cap at the real finalization
        # boundary; the native projection must account for both final receipts.
        assert result.status == "COMPLETE"
        assert result.violations == []
        current = monitor.sample_workspace_tree(root)[0] - baseline.apparent_bytes
        assert current < cap - 1
        (root / "finalization-fixture-padding").write_bytes(b"x" * (cap - 1 - current))
        prefinal_bytes.append(
            monitor.sample_workspace_tree(root)[0] - baseline.apparent_bytes
        )
        assert prefinal_bytes == [cap - 1]
        return native_projection(result, *args)

    monkeypatch.setattr(
        monitor, "_final_storage_projection", leave_one_byte_before_final_receipts
    )
    stopped = monitor.monitor_command(
        [sys.executable, "-c", f"import time; time.sleep({child_seconds})"],
        policy,
        workspace=root,
        log_dir=root / "logs",
        policy_path=policy_path,
        baseline_path=baseline_path,
    )
    events = [
        json.loads(line)
        for line in (root / "logs" / "events.jsonl").read_text().splitlines()
    ]
    samples = [row for row in events if row["kind"] == "sample"]
    assert samples
    assert all(row["added_workspace_bytes"] <= cap for row in samples)
    assert prefinal_bytes == [cap - 1]
    actual = monitor.sample_workspace_tree(root)[0] - baseline.apparent_bytes
    assert stopped.status == "VIOLATED"
    assert "max_added_workspace_bytes" in stopped.violations
    assert stopped.final_added_workspace_bytes == actual
    assert actual > cap
    assert (
        json.loads((root / "logs" / "completion.json").read_text())["status"]
        == "VIOLATED"
    )
    owned = json.loads((root / "logs" / "owned-completion.json").read_text())
    assert owned["living_descendants"] == 0
    assert owned["returncode"] == stopped.returncode == 0


def test_missing_or_wrong_sensor_fails_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy = _policy(
        policy_path,
        max_device_memory_bytes=100,
        expected_device_uuid="expected",
    )
    monkeypatch.setattr(
        monitor,
        "read_device_memory_bytes",
        lambda _uuid: (_ for _ in ()).throw(RuntimeError("wrong UUID")),
    )
    with pytest.raises(RuntimeError, match="wrong UUID"):
        monitor.monitor_command(
            [sys.executable, "-c", "pass"],
            policy,
            workspace=tmp_path,
            log_dir=tmp_path / "never-created",
            policy_path=policy_path,
        )
    assert not (tmp_path / "never-created").exists()


def test_requested_missing_sensor_is_explicit_null_and_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        monitor,
        "read_device_memory_bytes",
        lambda _uuid: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    root = monitor.ProcessIdentity(
        pid=os.getpid(), create_time=psutil.Process().create_time()
    )
    result = monitor.sample(
        root,
        {root.pid: root},
        monitor.MonitorPolicy(
            monitor_policy_version=1,
            max_device_memory_bytes=100,
            expected_device_uuid="uuid",
        ),
        tmp_path,
    )
    assert result.model_dump()["device_memory_bytes"] is None
    assert "max_device_memory_bytes" in result.violations
    assert (
        monitor.MonitorSample.model_validate_json(
            result.model_dump_json()
        ).model_dump_json()
        == result.model_dump_json()
    )


@pytest.mark.parametrize(
    "error", [ImportError, OSError, RuntimeError, ValueError, AttributeError]
)
def test_unavailable_requested_device_reading_survives_receipt_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    calls: list[str] = []

    def unavailable(uuid: str) -> int:
        calls.append(uuid)
        raise error("deterministically unavailable")

    monkeypatch.setattr(monitor, "read_device_memory_bytes", unavailable)
    root = monitor.ProcessIdentity(
        pid=os.getpid(), create_time=psutil.Process().create_time()
    )
    result = monitor.sample(
        root,
        {root.pid: root},
        monitor.MonitorPolicy(
            monitor_policy_version=1,
            max_device_memory_bytes=100,
            expected_device_uuid="requested-uuid",
        ),
        tmp_path,
    )
    assert calls == ["requested-uuid"]
    assert result.violations == ["max_device_memory_bytes"]
    wire = json.loads(result.model_dump_json())
    assert "device_memory_bytes" in wire
    assert wire["device_memory_bytes"] is None
    assert "requested_device_memory" not in wire
    assert "added_workspace_bytes" not in wire
    restored = monitor.MonitorSample.model_validate(wire)
    assert restored.requested_device_memory
    assert restored.model_dump() == result.model_dump()
    assert restored.model_dump_json() == result.model_dump_json()


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_late_sensor_failure_stops_owned_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readings = 0

    def sensor(_uuid: str) -> int:
        nonlocal readings
        readings += 1
        if readings == 1:
            return 10  # Preflight succeeds; the live sample must fail closed.
        raise RuntimeError("sensor lost")

    monkeypatch.setattr(monitor, "read_device_memory_bytes", sensor)
    policy_path = tmp_path / "policy.yaml"
    policy = _policy(
        policy_path,
        max_device_memory_bytes=100,
        expected_device_uuid="uuid",
        max_wall_seconds=3,
        interval_seconds=0.02,
        termination_grace_seconds=0.05,
    )
    result = monitor.monitor_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        policy,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        policy_path=policy_path,
    )
    assert readings >= 2
    assert result.status == "VIOLATED"
    assert "max_device_memory_bytes" in result.violations
    samples = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        row.get("kind") == "sample"
        and row.get("device_memory_bytes", "missing") is None
        and "max_device_memory_bytes" in row["violations"]
        for row in samples
    )
    assert (
        json.loads((tmp_path / "logs" / "owned-completion.json").read_text())[
            "living_descendants"
        ]
        == 0
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_early_parent_and_detached_term_ignoring_worker_are_gone_before_return(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    baseline_path = tmp_path / "baseline.json"
    monitor.capture_workspace_baseline(root, baseline_path)
    policy_path = tmp_path / "policy.yaml"
    policy = _policy(
        policy_path,
        max_added_workspace_bytes=100000,
        max_added_workspace_inodes=100,
        max_wall_seconds=2,
        interval_seconds=0.02,
        termination_grace_seconds=0.05,
    )
    worker_pid_path = root / "worker.pid"
    worker = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(worker_pid_path)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    parent = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable, '-c', {worker!r}], start_new_session=True); "
        "import time; time.sleep(0.1)"
    )
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = monitor.monitor_command(
            [sys.executable, "-c", parent],
            policy,
            workspace=root,
            log_dir=root / "logs",
            policy_path=policy_path,
            baseline_path=baseline_path,
        )
        assert result.status != "COMPLETE"
        assert sentinel.poll() is None
        assert worker_pid_path.exists()
        assert not _alive(int(worker_pid_path.read_text()))
        receipt = json.loads((root / "logs" / "owned-completion.json").read_text())
        assert receipt["living_descendants"] == 0
    finally:
        sentinel.kill()
        sentinel.wait(timeout=2)


@pytest.mark.parametrize("mask", [0o000, 0o002, 0o022])
@pytest.mark.parametrize("spawn_error", [False, True])
def test_private_monitor_outputs_and_baseline(tmp_path, monkeypatch, mask, spawn_error):
    policy_path = tmp_path / "policy.yaml"
    policy = _policy(policy_path, interval_seconds=0.02)
    logs = tmp_path / "new-parent" / "nested" / "logs"
    baseline = tmp_path / "baseline.json"
    parent_mode = tmp_path.stat().st_mode
    if spawn_error:

        def fail(*args, **kwargs):
            raise OSError("injected spawn error")

        monkeypatch.setattr(monitor.subprocess, "Popen", fail)
    previous = os.umask(mask)
    try:
        monitor.capture_workspace_baseline(tmp_path, baseline)
        options = {"workspace": tmp_path, "log_dir": logs, "policy_path": policy_path}
        if spawn_error:
            with pytest.raises(OSError, match="injected spawn error"):
                monitor.monitor_command(
                    [sys.executable, "-c", "pass"], policy, **options
                )
        else:
            result = monitor.monitor_command(
                [sys.executable, "-c", "print('tiny')"], policy, **options
            )
            assert result.status == "COMPLETE"
        retained = baseline.read_bytes()
        with pytest.raises(FileExistsError):
            monitor.capture_workspace_baseline(tmp_path, baseline)
        assert baseline.read_bytes() == retained
        with pytest.raises(FileExistsError):
            monitor.monitor_command([sys.executable, "-c", "pass"], policy, **options)
    finally:
        observed = os.umask(previous)
    assert observed == mask
    assert tmp_path.stat().st_mode == parent_mode
    for directory in (logs, logs.parent, logs.parent.parent):
        assert directory.stat().st_mode & 0o777 == 0o700
    assert baseline.stat().st_mode & 0o777 == 0o600
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in logs.iterdir())
    assert (logs / "completion.json").exists() is not spawn_error


@pytest.mark.skipif(sys.platform != "linux", reason="Linux owned helper/subreaper")
def test_pending_readiness_is_private_on_publication_error(tmp_path):
    # Run the real helper in isolation: prctl and signal state belong to the child.
    code = """
import os
from pathlib import Path
from unittest.mock import patch
from sparselab.owned_process import supervise
root = Path(__import__('sys').argv[1])
os.umask(0)
with patch('os.replace', side_effect=OSError('publication failed')):
    try:
        supervise(['must-not-launch'], completion=root / 'completion.json', ready=root / 'ready.json')
    except OSError as error:
        assert str(error) == 'publication failed'
    else:
        raise AssertionError('expected publication failure')
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=10)
    assert (tmp_path / "ready.json.pending").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "completion.json").stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "ready.json").exists()
