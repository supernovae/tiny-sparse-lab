from __future__ import annotations

import os
from pathlib import Path

from sparselab.workers.leases import acquire_lease, lease_key, process_matches


def test_accelerator_aliases_contend_across_worker_roots(tmp_path: Path) -> None:
    state = tmp_path / "host-state"
    first = acquire_lease(
        worker_id="mps", backend="mps", physical_device_id=None, state_dir=state
    )
    assert first is not None
    try:
        assert (
            acquire_lease(
                worker_id="mlx",
                backend="metal",
                physical_device_id=None,
                state_dir=state,
            )
            is None
        )
    finally:
        first.close()


def test_unknown_accelerator_identity_contends_with_known_device(
    tmp_path: Path,
) -> None:
    known = acquire_lease(
        worker_id="cuda",
        backend="cuda",
        physical_device_id="pci-0000",
        state_dir=tmp_path,
    )
    assert known is not None
    try:
        assert (
            acquire_lease(
                worker_id="unknown",
                backend="rocm",
                physical_device_id=None,
                state_dir=tmp_path,
            )
            is None
        )
    finally:
        known.close()


def test_distinct_known_accelerators_share_gate_but_keep_device_slots(
    tmp_path: Path,
) -> None:
    first = acquire_lease(
        worker_id="one",
        backend="cuda",
        physical_device_id="pci-one",
        state_dir=tmp_path,
    )
    second = acquire_lease(
        worker_id="two",
        backend="cuda",
        physical_device_id="pci-two",
        state_dir=tmp_path,
    )
    assert first is not None and second is not None
    try:
        assert (
            acquire_lease(
                worker_id="one",
                backend="cuda",
                physical_device_id="pci-three",
                state_dir=tmp_path,
            )
            is None
        )
    finally:
        first.close()
        second.close()


def test_cpu_workers_use_distinct_logical_slots(tmp_path: Path) -> None:
    one = acquire_lease(
        worker_id="cpu-one", backend="cpu", physical_device_id=None, state_dir=tmp_path
    )
    two = acquire_lease(
        worker_id="cpu-two", backend="cpu", physical_device_id=None, state_dir=tmp_path
    )
    assert one is not None and two is not None
    try:
        assert one.key != two.key
    finally:
        one.close()
        two.close()


def test_duplicate_executor_claim_is_nonblocking(tmp_path: Path) -> None:
    from sparselab.workers.execution import _claim_file

    first = _claim_file(tmp_path)
    assert first is not None
    try:
        assert _claim_file(tmp_path) is None
    finally:
        first.close()


def test_dead_running_receipt_becomes_unknown_without_relaunch(tmp_path: Path) -> None:
    from sparselab.workers.execution import (
        _receipt_payload,
        _write_receipt,
        status_worker,
    )
    from sparselab.workers.models import WorkerDefinition

    worker = WorkerDefinition(
        worker_id=f"worker-{tmp_path.name}",
        name=f"worker-{tmp_path.name}",
        transport="local",
        host=None,
        python=Path(os.__file__).resolve(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    payload = _receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "run",
            "experiment_id": "experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    payload.update(state="RUNNING", pid=999999, process_start="0", boot_id="wrong")
    _write_receipt(worker, payload)
    holder = acquire_lease(
        worker_id=worker.worker_id, backend="cpu", physical_device_id=None
    )
    assert holder is not None
    try:
        assert status_worker(worker, "attempt")["receipt"]["state"] == "RUNNING"
    finally:
        holder.close()
    assert status_worker(worker, "attempt")["receipt"]["state"] == "UNKNOWN"


def test_cancel_marks_attempt_without_signalling_pid(tmp_path: Path) -> None:
    from sparselab.workers.execution import (
        _receipt_payload,
        _write_receipt,
        cancel_attempt,
    )
    from sparselab.workers.models import WorkerDefinition

    worker = WorkerDefinition(
        worker_id="worker",
        name="worker",
        transport="local",
        host=None,
        python=Path(os.__file__).resolve(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    payload = _receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "run",
            "experiment_id": "experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    _write_receipt(worker, payload)
    receipt = cancel_attempt(worker, "attempt")
    assert receipt["cancellation_requested"]
    assert receipt["cancellation_acknowledged"]
    assert receipt["state"] == "INTERRUPTED"
    assert (tmp_path / "attempts" / "attempt" / "cancel.json").is_file()
    from sparselab.workers.execution import execute_attempt

    assert cancel_attempt(worker, "future") is None
    future = _receipt_payload(
        worker,
        {
            "attempt_id": "future",
            "run_id": "future-run",
            "experiment_id": "future-experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    _write_receipt(worker, future)
    acknowledged = execute_attempt(worker, "future")
    assert acknowledged["state"] == "INTERRUPTED"
    assert acknowledged["cancellation_acknowledged"]
    assert not (worker.root / "runs/future-run").exists()


def test_late_cancel_cannot_regress_complete_receipt(tmp_path: Path) -> None:
    from sparselab.workers.execution import (
        _receipt_payload,
        _write_receipt,
        cancel_attempt,
    )
    from sparselab.workers.models import WorkerDefinition

    worker = WorkerDefinition(
        worker_id="worker",
        name="worker",
        transport="local",
        host=None,
        python=Path(os.__file__).resolve(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    receipt = _receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "run",
            "experiment_id": "experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    receipt.update(state="COMPLETE", phase="complete")
    _write_receipt(worker, receipt)
    assert cancel_attempt(worker, "attempt")["state"] == "COMPLETE"
    assert not (tmp_path / "attempts" / "attempt" / "cancel.json").exists()


def test_stale_heartbeat_mutation_cannot_replace_terminal_receipt(
    tmp_path: Path,
) -> None:
    from sparselab.workers.execution import (
        _mutate_receipt,
        _receipt_payload,
        _write_receipt,
    )
    from sparselab.workers.models import WorkerDefinition

    worker = WorkerDefinition(
        worker_id="worker",
        name="worker",
        transport="local",
        host=None,
        python=Path(os.__file__).resolve(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    receipt = _receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "run",
            "experiment_id": "experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    receipt.update(state="COMPLETE", phase="complete")
    _write_receipt(worker, receipt)

    result = _mutate_receipt(
        worker,
        "attempt",
        lambda stale: stale | {"state": "RUNNING", "heartbeat_at": "later"},
    )
    assert result["state"] == "COMPLETE"


def test_recycled_or_dead_pid_never_counts_as_live_claim() -> None:
    assert not process_matches(os.getpid(), "0", "wrong-boot")


def test_unknown_accelerator_identity_uses_host_wide_key() -> None:
    assert lease_key(
        worker_id="a", backend="mps", physical_device_id=None
    ) == lease_key(worker_id="b", backend="metal", physical_device_id=None)


def test_inherited_pilot_keeps_lease_after_wrapper_closes(tmp_path: Path) -> None:
    import subprocess
    import sys

    lease = acquire_lease(
        worker_id="owner", backend="mps", physical_device_id=None, state_dir=tmp_path
    )
    assert lease is not None
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; print('ready', flush=True); sys.stdin.buffer.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        pass_fds=lease.inherited_fds,
    ) as pilot:
        try:
            assert pilot.stdout.readline() == b"ready\n"
            lease.close()
            assert (
                acquire_lease(
                    worker_id="contender",
                    backend="metal",
                    physical_device_id=None,
                    state_dir=tmp_path,
                )
                is None
            )
        finally:
            pilot.communicate(timeout=10)
            lease.close()
    next_owner = acquire_lease(
        worker_id="contender",
        backend="metal",
        physical_device_id=None,
        state_dir=tmp_path,
    )
    assert next_owner is not None
    next_owner.close()
