"""Worker CAS receipts never replace transfer authentication or private copies."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sparselab.training.manifest import sha256_file
from sparselab.verification_proofs import ProofStore
from sparselab.workers.bundles import (
    install_dispatch_bundle,
    materialize_dispatch_bundle,
    prepare_dispatch_bundle,
    verify_dispatch_bundle,
)


def test_signed_cache_reuse_and_private_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_training import config as training_config

    from sparselab.data import verification

    key_root = tmp_path / "keys"
    key_root.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(key_root))
    config = training_config(tmp_path / "source")
    bundle_root = tmp_path / "dispatch"
    manifest = prepare_dispatch_bundle(config, bundle_root)
    worker = tmp_path / "worker"
    attachments = {
        f"assets/{item.sha256}": bundle_root / item.relative_path
        for item in manifest.files
    }
    store = ProofStore(worker)
    observations: list[dict[str, object]] = []
    install_dispatch_bundle(
        worker,
        manifest,
        attachments,
        check_only=False,
        proof_store=store,
        verification_mode="verified_reuse",
        copy_observations=observations,
    )
    assert store.recorded == len({item.sha256 for item in manifest.files})
    assert verify_dispatch_bundle(bundle_root).digest() == manifest.digest()
    cached = worker / ".dispatch-cache" / "assets"
    cache_hashes: list[Path] = []
    original_hash = verification.manifest_module.sha256_file

    def counted_hash(path: Path) -> str:
        if Path(path).parent == cached:
            cache_hashes.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(verification.manifest_module, "sha256_file", counted_hash)
    install_dispatch_bundle(
        worker,
        manifest,
        {},
        check_only=False,
        proof_store=store,
        verification_mode="verified_reuse",
    )
    destination = tmp_path / "materialized"
    materialize_dispatch_bundle(
        worker,
        manifest.digest(),
        destination,
        proof_store=store,
        verification_mode="verified_reuse",
        copy_observations=observations,
    )
    assert not cache_hashes
    assert any(row.get("cache_misses") for row in observations)
    assert any(row.get("cache_hit") is True for row in observations)
    assert store.hits >= 2 * len(manifest.files)
    code = """
import json, sys
from pathlib import Path
from sparselab.training import manifest as hashes
from sparselab.verification_proofs import ProofStore
from sparselab.workers.bundles import materialize_dispatch_bundle
root = Path(sys.argv[1])
cache = root / '.dispatch-cache/assets'
reads=[]
original=hashes.sha256_file
def counted(path, *args, **kwargs):
    if path.parent == cache: reads.append(str(path))
    return original(path,*args,**kwargs)
hashes.sha256_file=counted
store=ProofStore(root)
bundle=materialize_dispatch_bundle(root,sys.argv[2],Path(sys.argv[3]),proof_store=store,verification_mode='verified_reuse')
print(json.dumps({'reads':reads,'digest':bundle.digest(),'hits':store.hits}))
"""
    cross_process = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                code,
                str(worker),
                manifest.digest(),
                str(tmp_path / "other-process"),
            ],
            text=True,
        )
    )
    assert cross_process["reads"] == []
    assert cross_process["digest"] == manifest.digest()
    assert cross_process["hits"] >= len(manifest.files)
    for item in manifest.files:
        source = bundle_root / item.relative_path
        private = destination / item.relative_path
        cas = cached / item.sha256
        assert sha256_file(private) == item.sha256
        assert (source.stat().st_dev, source.stat().st_ino) != (
            cas.stat().st_dev,
            cas.stat().st_ino,
        )
        assert (cas.stat().st_dev, cas.stat().st_ino) != (
            private.stat().st_dev,
            private.stat().st_ino,
        )
    item = next(
        item for item in manifest.files if item.relative_path.endswith("train.npy")
    )
    victim = cached / item.sha256
    original = victim.read_bytes()
    stamp = victim.stat().st_mtime_ns
    victim.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    os.utime(victim, ns=(stamp, stamp))
    with pytest.raises(ValueError, match="corrupt"):
        materialize_dispatch_bundle(
            worker,
            manifest.digest(),
            tmp_path / "tampered",
            proof_store=store,
            verification_mode="verified_reuse",
        )
    assert not (tmp_path / "tampered").exists()


def test_cold_override_rehashes_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_training import config as training_config

    from sparselab.data import verification

    key_root = tmp_path / "keys"
    key_root.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(key_root))
    config = training_config(tmp_path / "source")
    bundle_root = tmp_path / "dispatch"
    manifest = prepare_dispatch_bundle(config, bundle_root)
    worker = tmp_path / "worker"
    store = ProofStore(worker)
    install_dispatch_bundle(
        worker,
        manifest,
        {
            f"assets/{item.sha256}": bundle_root / item.relative_path
            for item in manifest.files
        },
        check_only=False,
        proof_store=store,
        verification_mode="verified_reuse",
    )
    calls: list[Path] = []
    original_hash = verification.manifest_module.sha256_file

    def counted_hash(path: Path) -> str:
        calls.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(verification.manifest_module, "sha256_file", counted_hash)
    materialize_dispatch_bundle(worker, manifest.digest(), tmp_path / "cold")
    cached = worker / ".dispatch-cache" / "assets"
    assert {path for path in calls if path.parent == cached} == {
        cached / item.sha256 for item in manifest.files
    }
