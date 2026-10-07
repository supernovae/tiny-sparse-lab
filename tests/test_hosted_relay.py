"""Security boundaries for the hosted immutable relay."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sparselab.training.manifest import ArtifactIdentity
from sparselab.workers.relay import (
    RelayStore,
    RelayVerificationError,
    create_attempt_key,
    read_attempt_key,
)
from sparselab.workers.relay_models import RelayLocation, RelayProfile


def _item(path: Path, name: str = "artifact.bin") -> ArtifactIdentity:
    return ArtifactIdentity(
        name, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
    )


def _store(tmp_path: Path) -> RelayStore:
    profile = RelayProfile(
        namespace="hosted-test",
        controller=RelayLocation(kind="file", root=str(tmp_path / "relay")),
        worker=RelayLocation(kind="file", root=str(tmp_path / "relay")),
    )
    return RelayStore(profile)


def _commit(
    store: RelayStore,
    key: bytes,
    item: ArtifactIdentity,
    *,
    sequence: int = 0,
    previous: str | None = None,
):
    return store.build_commit(
        key=key,
        sequence=sequence,
        previous_commit_sha256=previous,
        worker_id="worker-1",
        attempt_id="attempt-1",
        run_id="run-1",
        experiment_id="experiment-1",
        spec_digest="1" * 64,
        bundle_digest="2" * 64,
        source_digest="3" * 64,
        instance_id="instance-1",
        receipt={
            "attempt_id": "attempt-1",
            "run_id": "run-1",
            "experiment_id": "experiment-1",
            "spec_digest": "1" * 64,
            "bundle_digest": "2" * 64,
            "worker_id": "worker-1",
            "state": "PREPARED",
            "phase": "prepare",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        checkpoint=None,
        inventory=(item,),
        outbox_pages=(),
    )


def test_file_relay_rejects_absent_and_altered_objects(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = tmp_path / "source"
    source.write_bytes(b"immutable")
    item = _item(source)
    with pytest.raises(RelayVerificationError):
        store.get_verified(item, tmp_path / "missing")
    store.put_verified(source, item)
    object_path = (
        tmp_path
        / "relay"
        / "hosted-test"
        / "objects"
        / "sha256"
        / item.sha256[:2]
        / item.sha256
    )
    object_path.write_bytes(b"altered")
    with pytest.raises(RelayVerificationError):
        store.put_verified(source, item)


def test_commit_requires_authenticated_complete_closure_and_chain(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source = tmp_path / "source"
    source.write_bytes(b"immutable")
    item = _item(source)
    key = read_attempt_key(create_attempt_key(tmp_path, "attempt-1"))
    descriptor = _commit(store, key, item)
    with pytest.raises(RelayVerificationError):
        store.publish_commit(descriptor, key=key)
    store.put_verified(source, item)
    first = store.publish_commit(descriptor, key=key)
    second = _commit(store, key, item, sequence=2, previous=first.digest)
    store.publish_commit(second, key=key)
    with pytest.raises(RelayVerificationError, match="gap or fork"):
        store.verified_chain("attempt-1", key=key)


def test_commit_hmac_and_unsafe_inventory_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = tmp_path / "source"
    source.write_bytes(b"immutable")
    item = _item(source)
    key = read_attempt_key(create_attempt_key(tmp_path, "attempt-1"))
    store.put_verified(source, item)
    descriptor = _commit(store, key, item)
    forged = descriptor.model_copy(update={"hmac_sha256": "0" * 64})
    with pytest.raises(RelayVerificationError, match="HMAC"):
        store.publish_commit(forged, key=key)
    with pytest.raises(ValueError):
        _commit(store, key, ArtifactIdentity("../escape", item.sha256, item.size_bytes))


def test_relay_refuses_symlink_source_and_preserves_attempt_key(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = tmp_path / "source"
    source.write_bytes(b"immutable")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(RelayVerificationError):
        store.put_verified(link, _item(source))
    key_path = create_attempt_key(tmp_path, "attempt-1")
    assert key_path.stat().st_mode & 0o777 == 0o600
    assert (
        create_attempt_key(tmp_path, "attempt-1").read_bytes() == key_path.read_bytes()
    )
