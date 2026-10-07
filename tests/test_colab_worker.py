"""Behavior contracts for the bounded foreground Colab adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from sparselab.training.manifest import canonical_json
from sparselab.workers.colab import (
    ColabTransportError,
    download_hosted_status,
    download_kernel_status,
    record_prepared_delivery,
    upload_cancel_intent,
)
from sparselab.workers.models import ColabEndpoint, WorkerDefinition
from sparselab.workers.relay_models import RelayBinding, RelayLocation


def _worker(tmp_path: Path) -> WorkerDefinition:
    return WorkerDefinition(
        worker_id="colab-test",
        name="colab-test",
        transport="colab",
        host=None,
        colab=ColabEndpoint(session="test-session", instance_id="boot-1"),
        relay=RelayBinding(
            namespace="colab-test",
            worker=RelayLocation(kind="file", root=str(tmp_path / "relay-worker")),
            controller=RelayLocation(
                kind="file", root=str(tmp_path / "relay-controller")
            ),
        ),
        python=Path("/content/runtime/bin/python"),
        root=tmp_path / "worker",
        engine="pytorch",
        backend="cuda",
        device_index=0,
    )


class _Transfer:
    def __init__(
        self,
        status: dict[str, object] | None = None,
        kernel: dict[str, object] | None = None,
    ) -> None:
        self.status = status
        self.kernel = kernel
        self.uploaded: tuple[Path, str] | None = None

    def download(self, source: str, destination: Path) -> None:
        if source.endswith("hosted-status.json"):
            assert self.status is not None
            destination.write_bytes(canonical_json(self.status))
        elif source.endswith("kernel-status.json"):
            assert self.kernel is not None
            destination.write_bytes(canonical_json(self.kernel))
        elif source.endswith("/stat"):
            destination.write_text(
                "1 (python) R " + " ".join(["0"] * 18 + ["42"]) + " 0\n"
            )
        elif source == "/proc/sys/kernel/random/boot_id":
            destination.write_text("boot-id\n")
        else:
            pytest.fail(f"unexpected download {source}")

    def upload_exact(self, source: Path, destination: str) -> None:
        self.uploaded = (source, destination)


def _status(*, observed_at: float, heartbeat_at: str) -> dict[str, object]:
    return {
        "hosted_status_version": 1,
        "instance_id": "boot-1",
        "observed_at": observed_at,
        "capability": {},
        "receipt": {"heartbeat_at": heartbeat_at},
        "origin_id": "origin",
        "newest_commit": None,
    }


def _kernel(
    *, observed_at: float, state: str = "IDLE", instance_id: str = "boot-1"
) -> dict[str, object]:
    return {
        "kernel_status_version": 1,
        "instance_id": instance_id,
        "boot_id": "boot-id",
        "pid": 1,
        "process_start": 42,
        "state": state,
        "observed_at": observed_at,
    }


def test_active_status_uses_no_kernel_exec_and_rejects_nonce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    transfer = _Transfer(
        _status(observed_at=now.timestamp(), heartbeat_at=now.isoformat())
    )
    monkeypatch.setattr(
        "sparselab.workers.colab._file_transport", lambda *_a, **_k: transfer
    )
    monkeypatch.setattr(
        "sparselab.workers.colab.subprocess.run",
        lambda *_a, **_k: pytest.fail("active status must not invoke colab exec"),
    )
    download_hosted_status(_worker(tmp_path), tmp_path / "status.json")
    transfer.status = {
        **_status(observed_at=now.timestamp(), heartbeat_at=now.isoformat()),
        "instance_id": "reused-session",
    }
    with pytest.raises(ColabTransportError, match="nonce changed"):
        download_hosted_status(_worker(tmp_path), tmp_path / "nonce.json")


def test_status_requires_fresh_receipt_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    transfer = _Transfer(
        _status(observed_at=now.timestamp(), heartbeat_at=now.isoformat())
    )
    monkeypatch.setattr(
        "sparselab.workers.colab._file_transport", lambda *_a, **_k: transfer
    )
    assert (
        download_hosted_status(_worker(tmp_path), tmp_path / "status.json")["origin_id"]
        == "origin"
    )
    transfer.status = _status(
        observed_at=now.timestamp(), heartbeat_at="2000-01-01T00:00:00+00:00"
    )
    with pytest.raises(ColabTransportError, match="heartbeat is stale"):
        download_hosted_status(_worker(tmp_path), tmp_path / "stale.json")


def test_cancel_upload_is_exact_and_authenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transfer = _Transfer()
    monkeypatch.setattr(
        "sparselab.workers.colab._file_transport", lambda *_a, **_k: transfer
    )
    monkeypatch.setattr(
        "sparselab.workers.colab.subprocess.run",
        lambda *_a, **_k: pytest.fail("cancellation must not invoke colab exec"),
    )
    digest = "a" * 64
    upload_cancel_intent(
        _worker(tmp_path),
        attempt_id="attempt-1",
        spec_digest=digest,
        attempt_key=b"k" * 32,
        local_root=tmp_path,
    )
    assert transfer.uploaded is not None
    payload = __import__("json").loads(transfer.uploaded[0].read_text())
    assert set(payload) == {"instance_id", "attempt_id", "spec_digest", "hmac_sha256"}
    assert transfer.uploaded[1].endswith("worker/attempts/attempt-1/colab-cancel.json")


def test_kernel_observer_refuses_busy_stale_or_reused_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    transfer = _Transfer(kernel=_kernel(observed_at=now.timestamp()))
    monkeypatch.setattr(
        "sparselab.workers.colab._file_transport", lambda *_a, **_k: transfer
    )
    assert (
        download_kernel_status(_worker(tmp_path), tmp_path / "kernel.json")["state"]
        == "IDLE"
    )
    transfer.kernel = _kernel(observed_at=now.timestamp(), state="BUSY")
    with pytest.raises(ColabTransportError, match="occupancy"):
        from sparselab.workers.colab import require_colab_idle

        require_colab_idle(_worker(tmp_path), tmp_path / "busy.json")
    transfer.kernel = _kernel(observed_at=0, instance_id="reused-session")
    with pytest.raises(ColabTransportError, match="nonce changed"):
        download_kernel_status(_worker(tmp_path), tmp_path / "reused.json")
    transfer.kernel = _kernel(observed_at=0, state="BUSY")
    with pytest.raises(ColabTransportError, match="stale"):
        download_kernel_status(_worker(tmp_path), tmp_path / "stale-kernel.json")


def test_old_idle_observer_is_valid_when_process_identity_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transfer = _Transfer(kernel=_kernel(observed_at=0))
    monkeypatch.setattr(
        "sparselab.workers.colab._file_transport", lambda *_a, **_k: transfer
    )
    assert (
        download_kernel_status(_worker(tmp_path), tmp_path / "old-idle.json")["state"]
        == "IDLE"
    )


def test_unrelated_prepared_delivery_blocks_second_foreground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    transfer = _Transfer(kernel=_kernel(observed_at=now.timestamp()))
    monkeypatch.setattr(
        "sparselab.workers.colab._file_transport", lambda *_a, **_k: transfer
    )
    worker = _worker(tmp_path)
    record_prepared_delivery(
        worker, attempt_id="attempt-1", spec_digest="a" * 64, controller_root=tmp_path
    )
    with pytest.raises(ColabTransportError, match="another Colab"):
        record_prepared_delivery(
            worker,
            attempt_id="attempt-2",
            spec_digest="b" * 64,
            controller_root=tmp_path,
        )


def test_terminal_receipt_does_not_expire_during_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old = datetime(2025, 12, 31, 23, 50, tzinfo=UTC).isoformat()
    status = _status(observed_at=now.timestamp(), heartbeat_at=old)
    status["receipt"].update(state="COMPLETE", updated_at=old)
    transfer = _Transfer(status=status)
    monkeypatch.setattr(
        "sparselab.workers.colab._file_transport", lambda *_a, **_k: transfer
    )
    monkeypatch.setattr("sparselab.workers.colab.time.time", lambda: now.timestamp())
    worker = _worker(tmp_path)
    assert (
        download_hosted_status(worker, tmp_path / "status.json")["receipt"]["state"]
        == "COMPLETE"
    )

    # A live attempt still requires its own fresh heartbeat.
    status["receipt"]["state"] = "RUNNING"
    with pytest.raises(ColabTransportError):
        download_hosted_status(worker, tmp_path / "status.json")

    # Immutable terminal truth does not excuse stale transport or future clocks.
    status["receipt"]["state"] = "COMPLETE"
    status["observed_at"] = now.timestamp() - 31
    with pytest.raises(ColabTransportError):
        download_hosted_status(worker, tmp_path / "status.json")
    status["observed_at"] = now.timestamp()
    status["receipt"]["updated_at"] = datetime(2026, 1, 1, 0, 1, tzinfo=UTC).isoformat()
    with pytest.raises(ColabTransportError):
        download_hosted_status(worker, tmp_path / "status.json")
