"""Frozen-stage copying and in-operation proof reuse."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab import owned_copy, staging
from sparselab.data import verification
from sparselab.training.manifest import sha256_file
from sparselab.verification_proofs import ProofStore


def test_copy_hashes_output_once_and_reuses_sealed_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "array.npy").write_bytes(b"\x93NUMPY" + b"payload" * 1000)
    proof = verification.verify_file(source / "array.npy")
    destination = tmp_path / "assets"
    copied = staging._copy_tree(source, destination, verified={"array.npy": proof})

    def unexpected_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
        raise AssertionError(f"already hashed copied output: {path}")

    monkeypatch.setattr(staging, "sha256_file", unexpected_hash)
    inventory = staging._inventory(destination, copied=copied)
    assert inventory == [
        {
            "relative_path": "array.npy",
            "size_bytes": proof.size_bytes,
            "sha256": proof.sha256,
        }
    ]
    (destination / "array.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="copied stage asset changed"):
        staging._inventory(destination, copied=copied)


def test_copied_bytes_progress_has_bounded_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.training import pilot_progress

    # This test checks chunk thresholds; native clones report completion once.
    # Native mechanisms remain covered by test_owned_copy's private-copy test.
    def unavailable(*_args: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "native clones disabled for progress test")

    monkeypatch.setattr(owned_copy.fcntl, "ioctl", unavailable)
    monkeypatch.setattr(owned_copy, "_clonefile", unavailable)
    monkeypatch.delattr(owned_copy.os, "copy_file_range", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    payload = b"x" * (1024 * 1024)
    with (source / "large.npy").open("wb") as handle:
        for _ in range(65):
            handle.write(payload)
    events: list[dict[str, object]] = []
    monkeypatch.setattr(pilot_progress, "current_pilot_progress", lambda: object())
    monkeypatch.setattr(
        pilot_progress,
        "emit_pilot_progress",
        lambda kind, phase, **fields: events.append(
            {"kind": kind, "phase": phase, **fields}
        ),
    )
    staging._copy_tree(source, tmp_path / "copied")
    assert [event["value"] for event in events] == [64 << 20, 65 << 20]
    assert all(
        event["counter"] == "bytes"
        and event["total"] == 65 << 20
        and event["phase"] == "run_input_materialization"
        for event in events
    )
    assert len({event["subject"] for event in events}) == 1

    events.clear()
    digest = sha256_file(
        source / "large.npy", progress_phase="stage_bundle_verification"
    )
    expected = hashlib.sha256()
    for _ in range(65):
        expected.update(payload)
    assert digest == expected.hexdigest()
    assert [event["value"] for event in events] == [64 << 20, 65 << 20]
    assert all(
        event["phase"] == "stage_bundle_verification"
        and event["counter"] == "bytes"
        and event["total"] == 65 << 20
        for event in events
    )


def test_same_inode_rewrite_with_restored_mtime_rejects_cached_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.npy"
    path.write_bytes(b"original")
    proof = verification.verify_file(path)
    before = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_ino == before.st_ino
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="digest mismatch"):
        verification.verify_file(
            path,
            expected_sha256=proof.sha256,
            memo={(proof.path, proof.fingerprint): proof},
        )


@pytest.mark.parametrize("reuse", ["memo", "signed"])
def test_ctime_change_alone_forces_actual_byte_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reuse: str
) -> None:
    # Control only the reported ctime, independently of filesystem clock
    # resolution. The inode, write, restored mtime and SHA reads remain real.
    path = tmp_path / "data.npy"
    path.write_bytes(b"original")
    ctime = 10
    lstat = Path.lstat

    def observed_lstat(candidate: Path, *args, **kwargs):
        info = lstat(candidate, *args, **kwargs)
        if candidate != path:
            return info
        fields = {
            name: getattr(info, name) for name in dir(info) if name.startswith("st_")
        }
        return SimpleNamespace(**{**fields, "st_ctime_ns": ctime})

    monkeypatch.setattr(Path, "lstat", observed_lstat)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    store = ProofStore(tmp_path)
    options = (
        {"memo": {}}
        if reuse == "memo"
        else {"proof_store": store, "verification_mode": "verified_reuse"}
    )
    digest = hashlib.sha256(b"original").hexdigest()
    proof = verification.verify_file(path, expected_sha256=digest, **options)
    hash_file = verification.manifest_module.sha256_file
    reads: list[Path] = []

    def counted(candidate: Path) -> str:
        reads.append(candidate)
        return hash_file(candidate)

    monkeypatch.setattr(verification.manifest_module, "sha256_file", counted)
    assert (
        verification.verify_file(path, expected_sha256=digest, **options).sha256
        == digest
    )
    assert reads == []
    ctime += 1
    refreshed = verification.verify_file(path, expected_sha256=digest, **options)
    assert refreshed.sha256 == digest
    assert refreshed.cold_verified
    assert refreshed.fingerprint[:-1] == proof.fingerprint[:-1]
    assert refreshed.fingerprint[-1] != proof.fingerprint[-1]
    assert reads == [path]
    if reuse == "signed":
        assert store.hits == 1
        assert store.recorded == 2
        assert store.diagnostics()["events"][-1]["reason"] == "changed_fingerprint"
    assert (
        verification.verify_file(path, expected_sha256=digest, **options).sha256
        == digest
    )
    assert reads == [path]
    proof = refreshed
    reads.clear()
    before = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    ctime += 1
    after = verification._fingerprint(path)
    assert after[:-1] == proof.fingerprint[:-1]
    assert after[-1] != proof.fingerprint[-1]
    with pytest.raises(ValueError, match="digest mismatch"):
        verification.verify_file(path, expected_sha256=digest, **options)
    assert reads == [path]
    if reuse == "signed":
        assert store.hits == 2
        assert store.recorded == 2
        assert store.diagnostics()["events"][-1]["reason"] == "changed_fingerprint"


def test_reconstructed_copy_proof_cannot_skip_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data.npy"
    path.write_bytes(b"verified actual bytes")
    proof = verification.verify_file(path)
    cloned = copy.copy(proof)
    original = verification.manifest_module.sha256_file
    observed: list[Path] = []

    def counted(candidate: Path) -> str:
        observed.append(candidate)
        return original(candidate)

    monkeypatch.setattr(verification.manifest_module, "sha256_file", counted)
    verification.verify_file(path, memo={(cloned.path, cloned.fingerprint): cloned})
    assert observed == [path]
    with pytest.raises(ValueError, match="copied stage asset changed"):
        staging._inventory(tmp_path, copied={path.name: cloned})


def test_later_worker_rehashes_transferred_in_process_proof(tmp_path: Path) -> None:
    import pickle
    import subprocess
    import sys

    path = tmp_path / "data.npy"
    path.write_bytes(b"independently verified worker input")
    proof = verification.verify_file(path)
    (tmp_path / "witness.pkl").write_bytes(pickle.dumps(proof))
    script = """import json, pickle, sys
from pathlib import Path
from sparselab.data import verification
root = Path(sys.argv[1])
path = root / 'data.npy'
proof = pickle.loads((root / 'witness.pkl').read_bytes())
original = verification.manifest_module.sha256_file
observed = []
def counted(candidate):
    observed.append(candidate)
    return original(candidate)
verification.manifest_module.sha256_file = counted
cold = verification.verify_file(path, memo={(proof.path, proof.fingerprint): proof})
print(json.dumps({'sha256': cold.sha256, 'hashes': len(observed)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "sha256": hashlib.sha256(b"independently verified worker input").hexdigest(),
        "hashes": 1,
    }
