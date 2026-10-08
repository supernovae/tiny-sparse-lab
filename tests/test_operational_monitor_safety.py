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
def test_final_receipts_cannot_cross_cap_and_still_claim_complete(
    tmp_path: Path,
) -> None:
    def run(
        name: str, cap: int
    ) -> tuple[monitor.MonitorCompletion, list[dict[str, object]], int]:
        base = tmp_path / name
        base.mkdir()
        root = base / "root"
        root.mkdir()
        baseline_path = base / "baseline.json"
        baseline = monitor.capture_workspace_baseline(root, baseline_path)
        policy_path = base / "policy.yaml"
        policy = _policy(
            policy_path,
            max_added_workspace_bytes=cap,
            max_added_workspace_inodes=100,
            max_wall_seconds=3,
            interval_seconds=0.02,
        )
        result = monitor.monitor_command(
            [sys.executable, "-c", "import time; time.sleep(0.1)"],
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
        actual = monitor.sample_workspace_tree(root)[0] - baseline.apparent_bytes
        return result, events, actual

    probe, probe_events, probe_actual = run("probe", 100000)
    live_peak = max(
        row["added_workspace_bytes"]
        for row in probe_events
        if row["kind"] == "sample" and row["added_workspace_bytes"] is not None
    )
    assert probe.status == "COMPLETE"
    assert probe.final_added_workspace_bytes == probe_actual
    assert probe_actual > live_peak + 100
    cap = live_peak + (probe_actual - live_peak) // 2
    stopped, events, actual = run("bounded", cap)
    assert all(
        row["added_workspace_bytes"] <= cap
        for row in events
        if row["kind"] == "sample" and row["added_workspace_bytes"] is not None
    )
    assert stopped.status == "VIOLATED"
    assert "max_added_workspace_bytes" in stopped.violations
    assert stopped.final_added_workspace_bytes == actual
    assert actual > cap
    assert (
        json.loads(
            (tmp_path / "bounded" / "root" / "logs" / "completion.json").read_text()
        )["status"]
        == "VIOLATED"
    )


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
