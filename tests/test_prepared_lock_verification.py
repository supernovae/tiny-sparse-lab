from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_training import config as training_config

from sparselab.data import packing
from sparselab.data.tokenizer import load_tokenizer
from sparselab.experiments import artifacts as artifact_module
from sparselab.experiments import lock as lock_module
from sparselab.experiments.artifacts import _VerifiedArtifact
from sparselab.experiments.lock import (
    ResolvedExperimentPlan,
    open_lock,
    publish_lock,
    resolve_plan,
)
from sparselab.experiments.plan import Artifact, ExperimentPlan
from sparselab.training import manifest as manifest_module


def test_repeated_prepared_lock_inputs_scan_once_and_reject_tampering(
    tmp_path, monkeypatch
):
    config = training_config(tmp_path)
    prepared = packing.prepare_data(config, load_tokenizer(config.tokenizer.path))
    packed = {
        "kind": "prepared_data",
        "version": 1,
        "producer": "sparselab",
        "identifier": prepared.manifest["settings_sha256"],
        "sha256": prepared.manifest["manifest_sha256"],
        "path": str(prepared.root),
    }
    tokenizer = {
        "kind": "tokenizer",
        "version": 1,
        "producer": "sparselab",
        "identifier": config.tokenizer.path.parent.name,
        "sha256": manifest_module.sha256_file(config.tokenizer.path),
        "path": str(config.tokenizer.path),
    }
    plan = ExperimentPlan.model_validate(
        {
            "plan_version": 1,
            "id": "deep-once",
            "base_run": config,
            "artifacts": {
                "tokenizer": tokenizer,
                "packed": packed,
                "packed_again": packed,
            },
            "inputs": {
                "tokenizer": "tokenizer",
                "training": "packed",
                "audit": "packed_again",
            },
        }
    )
    (tmp_path / "plan.yaml").write_text(plan.model_dump_json())
    original = manifest_module.sha256_file
    reads: list[Path] = []

    def counted(path, *args, **kwargs):
        if path.suffix == ".npy":
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(manifest_module, "sha256_file", counted)
    locked = resolve_plan(plan, tmp_path / "plan.yaml")
    assert locked.cells[0].config == config
    assert sorted(path.name for path in reads) == ["train.npy", "validation.npy"]
    reads.clear()
    assert (
        resolve_plan(plan, tmp_path / "plan.yaml").scientific_sha256
        == locked.scientific_sha256
    )
    assert sorted(path.name for path in reads) == ["train.npy", "validation.npy"]
    path = prepared.root / "train.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        resolve_plan(plan, tmp_path / "plan.yaml")


@pytest.fixture
def resolved_prepared_lock(tmp_path):
    config = training_config(tmp_path)
    prepared = packing.prepare_data(config, load_tokenizer(config.tokenizer.path))
    plan = ExperimentPlan.model_validate(
        {
            "plan_version": 1,
            "id": "sealed-publication",
            "base_run": config,
            "artifacts": {
                "tokenizer": {
                    "kind": "tokenizer",
                    "version": 1,
                    "producer": "sparselab",
                    "identifier": config.tokenizer.path.parent.name,
                    "sha256": manifest_module.sha256_file(config.tokenizer.path),
                    "path": str(config.tokenizer.path),
                },
                "packed": {
                    "kind": "prepared_data",
                    "version": 1,
                    "producer": "sparselab",
                    "identifier": prepared.manifest["settings_sha256"],
                    "sha256": prepared.manifest["manifest_sha256"],
                    "path": str(prepared.root),
                },
            },
            "inputs": {"tokenizer": "tokenizer", "training": "packed"},
        }
    )
    source = tmp_path / "plan.yaml"
    source.write_text(plan.model_dump_json())
    return resolve_plan(plan, source), prepared.root, tmp_path


def test_publish_reuses_only_sealed_resolver_verifications(
    resolved_prepared_lock, monkeypatch
):
    locked, _, root = resolved_prepared_lock
    original = artifact_module._verify_domain
    calls: list[str] = []

    def counted(artifact, path):
        calls.append(artifact.kind)
        return original(artifact, path)

    monkeypatch.setattr(artifact_module, "_verify_domain", counted)
    path = publish_lock(locked, root)
    assert calls == []
    assert path.read_bytes().endswith(b"\n")
    assert b"_artifact_proof" not in path.read_bytes()
    assert open_lock(path).plan_sha256 == locked.plan_sha256
    assert sorted(calls) == ["prepared_data", "tokenizer"]


def test_reconstructed_and_borrowed_proof_cannot_skip_cold_verification(
    resolved_prepared_lock, monkeypatch
):
    locked, _, root = resolved_prepared_lock
    reconstructed = ResolvedExperimentPlan.model_validate(
        locked.model_dump(mode="json")
    )
    reconstructed._artifact_proof = locked._artifact_proof
    copied = locked.model_copy()
    original = artifact_module._verify_domain
    calls: list[str] = []

    def counted(artifact, path):
        calls.append(artifact.kind)
        return original(artifact, path)

    monkeypatch.setattr(artifact_module, "_verify_domain", counted)
    publish_lock(reconstructed, root)
    assert sorted(calls) == [
        "prepared_data",
        "prepared_data",
        "tokenizer",
        "tokenizer",
    ]
    calls.clear()
    publish_lock(copied, root)
    assert sorted(calls) == [
        "prepared_data",
        "prepared_data",
        "tokenizer",
        "tokenizer",
    ]
    calls.clear()
    publish_lock(locked, root)
    assert calls == []


def test_changed_prepared_array_blocks_publication_before_lock_write(
    resolved_prepared_lock,
):
    locked, prepared, root = resolved_prepared_lock
    array = prepared / "train.npy"
    previous = array.stat()
    data = bytearray(array.read_bytes())
    data[-1] ^= 1
    array.write_bytes(data)
    import os

    os.utime(array, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError, match="changed since resolution"):
        publish_lock(locked, root)
    assert not list((root / "locks").glob("*.json"))


def test_artifact_identity_and_canonical_availability_bound_on_readback(
    resolved_prepared_lock,
):
    locked, _, root = resolved_prepared_lock
    path = publish_lock(locked, root)
    sidecar = path.with_name(path.stem + ".availability.json")
    binding = sidecar.read_bytes()
    sidecar.write_bytes(binding + b" ")
    with pytest.raises(ValueError, match="canonical"):
        open_lock(path)
    sidecar.write_bytes(binding)
    assert open_lock(path).plan_sha256 == locked.plan_sha256
    altered = json.loads(binding)
    altered["availability"]["artifacts"]["packed"] = str(
        Path(locked.availability["artifacts"]["tokenizer"])
    )
    altered["sha256"] = lock_module._hash(
        "sparselab-experiment-availability-v1",
        {
            "plan_sha256": locked.plan_sha256,
            "availability": altered["availability"],
        },
    )
    sidecar.write_bytes(manifest_module.canonical_json(altered) + b"\n")
    with pytest.raises(ValueError):
        open_lock(path)
    sidecar.write_bytes(binding)
    tokenizer = Path(locked.availability["artifacts"]["tokenizer"])
    tokenizer.write_bytes(tokenizer.read_bytes() + b" ")
    with pytest.raises(ValueError):
        open_lock(path)


def test_unsigned_artifact_receipt_is_not_reusable(resolved_prepared_lock):
    locked, _, root = resolved_prepared_lock
    proof = locked._artifact_proof
    assert proof is not None
    key, genuine = next(iter(proof.memo.items()))
    counterfeit = _VerifiedArtifact(
        dict(genuine.identity), key, _seal=artifact_module._MEMO_SEAL
    )
    spec = Artifact.model_validate({**dict(genuine.identity), "producer": "sparselab"})
    with pytest.raises(ValueError, match="changed since resolution"):
        artifact_module._reuse_verified_artifact(
            spec, root / "plan.yaml", {key: counterfeit}
        )


def test_changed_tokenizer_blocks_publication_before_lock_write(
    resolved_prepared_lock,
):
    locked, _, root = resolved_prepared_lock
    tokenizer = Path(locked.availability["artifacts"]["tokenizer"])
    tokenizer.write_bytes(tokenizer.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed since resolution"):
        publish_lock(locked, root)
    assert not list((root / "locks").glob("*.json"))


def test_changed_tokenizer_provenance_blocks_publication(
    resolved_prepared_lock,
):
    locked, _, root = resolved_prepared_lock
    tokenizer = Path(locked.availability["artifacts"]["tokenizer"])
    manifest = tokenizer.with_name("tokenizer_manifest.json")
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed since resolution"):
        publish_lock(locked, root)
    assert not list((root / "locks").glob("*.json"))


def test_forked_process_cannot_reuse_parent_verification(tmp_path):
    import os
    import subprocess
    import sys

    if not hasattr(os, "fork"):
        pytest.skip("requires a genuinely inherited process context")
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "format": "dense_lm_decoding_prompts_v1",
                "split": "development",
                "prompts": [
                    {
                        "id": "owned",
                        "text": "def parse_config(path):",
                        "category": "code",
                    }
                ],
            }
        )
    )
    # Keep this fixture single-threaded; prepared-data loading starts native threads.
    script = """
import multiprocessing, sys
from pathlib import Path
import psutil
from sparselab.experiments import artifacts
from sparselab.experiments.plan import Artifact
from sparselab.training.manifest import sha256_file
panel = Path(sys.argv[1])
spec = Artifact.model_validate({
    "kind": "prompt_set", "version": 1, "producer": "sparselab",
    "identifier": panel.stem, "path": str(panel), "sha256": sha256_file(panel),
})
memo = {}
artifacts.verify_artifact(spec, panel, memo=memo)
original = artifacts._verify_domain
calls = []
def counted(artifact, path):
    calls.append(artifact.kind)
    return original(artifact, path)
artifacts._verify_domain = counted
context = multiprocessing.get_context("fork")
parent, child = context.Pipe()
def cold_child():
    try:
        artifacts.verify_artifact(spec, panel, memo=memo)
        child.send(calls)
    finally:
        child.close()
assert psutil.Process().num_threads() == 1, "fork fixture must be single threaded"
process = context.Process(target=cold_child)
process.start()
child.close()
process.join()
assert process.exitcode == 0
observed = parent.recv()
parent.close()
assert observed == ["prompt_set"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(panel)],
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
