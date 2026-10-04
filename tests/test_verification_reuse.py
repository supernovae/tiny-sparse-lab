from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
from test_training import config as training_config

from sparselab.data.packing import load_prepared_data, prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.data.verification import verify_file
from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.plan import Artifact
from sparselab.training import manifest as manifests
from sparselab.verification_proofs import ProofStore, file_binding


@pytest.fixture
def trusted_prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    config = training_config(root)
    prepared = prepare_data(config, load_tokenizer(config.tokenizer.path))
    return root, config, prepared


def _load(prepared, store, mode="verified_reuse"):
    return load_prepared_data(
        prepared.root, byte_enabled=False, proof_store=store, verification_mode=mode
    )


def test_separate_process_reuses_arrays_without_payload_sha(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    assert store.recorded == 2
    code = """
import json, sys
from pathlib import Path
from sparselab.training import manifest
from sparselab.data.packing import load_prepared_data
from sparselab.verification_proofs import ProofStore
reads=[]
original=manifest.sha256_file
def counted(path, *a, **kw):
    if path.suffix == '.npy': reads.append(path.name)
    return original(path,*a,**kw)
manifest.sha256_file=counted
store=ProofStore(Path(sys.argv[1]))
data=load_prepared_data(Path(sys.argv[2]),byte_enabled=False,proof_store=store,verification_mode=sys.argv[3])
print(json.dumps({'reads':reads,'hits':store.hits,'sha':data.manifest['manifest_sha256']}))
"""
    result = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                code,
                str(root),
                str(prepared.root),
                "verified_reuse",
            ],
            text=True,
        )
    )
    assert result == {
        "reads": [],
        "hits": 2,
        "sha": prepared.manifest["manifest_sha256"],
    }
    cold = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(root), str(prepared.root), "cold"],
            text=True,
        )
    )
    assert sorted(cold["reads"]) == ["train.npy", "validation.npy"]
    assert cold["hits"] == 0


@pytest.mark.parametrize("mutation", ["bytes", "inode", "truncate", "manifest"])
def test_changed_array_or_manifest_fails(trusted_prepared, mutation):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    path = prepared.root / "train.npy"
    before = path.stat()
    raw = path.read_bytes()
    if mutation == "manifest":
        manifest_path = prepared.root / "manifest.json"
        document = json.loads(manifest_path.read_text())
        document["train"]["tokens"] += 1
        manifest_path.write_text(json.dumps(document))
    elif mutation == "inode":
        new = path.with_name("replacement")
        new.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        new.replace(path)
    elif mutation == "truncate":
        path.write_bytes(raw[:-1])
    else:
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        _load(prepared, store)
    with pytest.raises(ValueError):
        _load(prepared, store, "cold")


@pytest.mark.parametrize(
    "invalid",
    ["missing", "forged", "stale", "missing_key", "foreign_key", "permissions"],
)
def test_invalid_receipt_falls_back_to_cold(trusted_prepared, invalid, monkeypatch):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    receipts = list(store.directory.glob("*.json"))
    if invalid == "missing":
        for path in receipts:
            path.unlink()
    elif invalid in {"forged", "stale"}:
        for path in receipts:
            payload = json.loads(path.read_text())
            if invalid == "forged":
                payload.pop("hmac")
            else:
                payload["proof"]["verification_schema"] += 1
            path.write_text(json.dumps(payload))
    elif invalid == "missing_key":
        store.key_path.unlink()
    elif invalid == "foreign_key":
        store.key_path.write_bytes(b"x" * 32)
    else:
        os.chmod(root, 0o777)
    reads = []
    original = manifests.sha256_file

    def counted(path, *args, **kwargs):
        if path.suffix == ".npy":
            reads.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(manifests, "sha256_file", counted)
    _load(prepared, store)
    assert sorted(reads) == ["train.npy", "validation.npy"]


def test_relocation_and_links_are_not_reuse(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    moved = root / "relocated"
    shutil.copytree(prepared.root, moved)
    data = load_prepared_data(
        moved, byte_enabled=False, proof_store=store, verification_mode="verified_reuse"
    )
    assert all(p.cold_verified for p in data.receipt.proofs.values())
    linked = root / "linked"
    linked.symlink_to(prepared.root, target_is_directory=True)
    assert not store.trusted(linked)
    hard = root / "hard.npy"
    os.link(prepared.root / "train.npy", hard)
    assert not store.trusted(hard)


def test_only_exact_cold_process_owned_evidence_can_record(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    path = tmp_path / "asset"
    path.write_bytes(b"original")
    store = ProofStore(tmp_path)
    digest = manifests.sha256_file(path)
    binding = file_binding(path, digest)
    with pytest.raises(TypeError, match="process-owned"):
        store.record(binding, {"sha256": digest})
    proof = verify_file(path, expected_sha256=digest)
    store.record(binding, proof)
    assert store.lookup(binding)
    warm = verify_file(
        path,
        expected_sha256=digest,
        proof_store=store,
        verification_mode="verified_reuse",
    )
    with pytest.raises(TypeError, match="process-owned"):
        store.record(binding, warm)
    assert oct(store.directory.stat().st_mode & 0o777) == "0o700"
    assert oct(store.key_path.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(ValueError, match="binding differs"):
        verify_file(
            path,
            expected_sha256="a" * 64,
            proof_store=store,
            verification_mode="verified_reuse",
            binding=binding,
        )


def test_typed_dependency_changes_miss(trusted_prepared):
    root, config, _ = trusted_prepared
    path = config.tokenizer.path
    artifact = Artifact(
        kind="tokenizer",
        version=1,
        producer="fixture",
        identifier=path.parent.name,
        sha256=manifests.sha256_file(path),
        path=str(path),
    )
    store = ProofStore(root)
    before = verify_artifact(
        artifact,
        root / "plan.yaml",
        proof_store=store,
        verification_mode="verified_reuse",
    )
    assert (
        verify_artifact(
            artifact,
            root / "plan.yaml",
            proof_store=store,
            verification_mode="verified_reuse",
        )
        == before
    )
    manifest = path.with_name("tokenizer_manifest.json")
    payload = json.loads(manifest.read_text())
    payload["vocab_size"] += 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        verify_artifact(
            artifact,
            root / "plan.yaml",
            proof_store=store,
            verification_mode="verified_reuse",
        )


def test_prepared_worker_choices_preserve_exact_manifest_and_sha(
    trusted_prepared, monkeypatch
):
    from dataclasses import replace

    from sparselab.data import packing
    from sparselab.host_capacity import sha_work_plan

    _, _, prepared = trusted_prepared
    manifest_bytes = (prepared.root / "manifest.json").read_bytes()
    expected = {name: proof.sha256 for name, proof in prepared.receipt.proofs.items()}
    plan = sha_work_plan(operator_cap=4)
    for workers in (1, 2, plan.workers):
        monkeypatch.setattr(
            packing,
            "sha_work_plan",
            lambda workers=workers: replace(plan, workers=workers),
        )
        loaded = load_prepared_data(prepared.root, byte_enabled=False)
        assert {
            name: proof.sha256 for name, proof in loaded.receipt.proofs.items()
        } == expected
        assert (prepared.root / "manifest.json").read_bytes() == manifest_bytes


def test_unchanged_array_node_remains_reusable(trusted_prepared, monkeypatch):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    # Identical bytes at a new inode still require cold verification for that node.
    path = prepared.root / "train.npy"
    replacement = path.with_name("replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    original = manifests.sha256_file
    reads = []

    def counted(path, *args, **kwargs):
        if path.suffix == ".npy":
            reads.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(manifests, "sha256_file", counted)
    _load(prepared, store)
    assert reads == ["train.npy"]


def test_unsafe_config_ancestry_never_creates_key_outside_selected_path(
    trusted_prepared, tmp_path, monkeypatch
):
    root, _, prepared = trusted_prepared
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "config-alias"
    alias.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(alias / "new-config"))
    store = ProofStore(root)
    verified = _load(prepared, store)
    assert all(proof.cold_verified for proof in verified.receipt.proofs.values())
    assert store.recorded == 0
    assert not (outside / "new-config").exists()
