"""Story tokenizer proofs retain source closure across reuse and relocation."""

from __future__ import annotations

import json
import shutil

import pytest
from test_experiment_story_inputs import story_plan

from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
from sparselab.training.manifest import sha256_file
from sparselab.verification_proofs import ProofStore


@pytest.mark.parametrize("reuse", ["memo", "persistent"])
@pytest.mark.parametrize("member", ["excluded.jsonl", "manifest.json", "train.jsonl"])
def test_story_artifact_reuse_rechecks_changed_source(
    tmp_path, monkeypatch, reuse, member
):
    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "local_stories")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    store = ProofStore(tmp_path)
    memo = {}
    kwargs = (
        {"memo": memo}
        if reuse == "memo"
        else {"proof_store": store, "verification_mode": "verified_reuse"}
    )
    artifact = plan.artifacts["tokenizer"]
    dataset = plan.base_run.dataset
    first = verify_artifact(artifact, declaration, dataset=dataset, **kwargs)
    assert verify_artifact(artifact, declaration, dataset=dataset, **kwargs) == first
    if reuse == "persistent":
        assert store.recorded == 1
        assert store.hits == 1
        # A new index instance must reject the previously signed source closure.
        kwargs["proof_store"] = ProofStore(tmp_path)
    target = tmp_path / "snapshot" / member
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        verify_artifact(artifact, declaration, dataset=dataset, **kwargs)


@pytest.mark.parametrize("reuse", ["memo", "persistent"])
def test_story_artifact_relocation_requires_matching_snapshot(
    tmp_path, monkeypatch, reuse
):
    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "local_stories")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    store = ProofStore(tmp_path)
    memo = {}
    kwargs = (
        {"memo": memo}
        if reuse == "memo"
        else {"proof_store": store, "verification_mode": "verified_reuse"}
    )
    artifact = plan.artifacts["tokenizer"]
    dataset = plan.base_run.dataset
    first = verify_artifact(artifact, declaration, dataset=dataset, **kwargs)
    relocated = tmp_path / "relocated"
    shutil.copytree(tmp_path / "snapshot", relocated)
    moved = dataset.model_copy(
        update={
            "source_manifest_path": relocated / "manifest.json",
            "train_path": relocated / "train.jsonl",
            "validation_path": relocated / "validation.jsonl",
        }
    )
    assert verify_artifact(artifact, declaration, dataset=moved, **kwargs) == first
    if reuse == "persistent":
        assert store.recorded == 2
        assert store.hits == 0
    else:
        assert len(memo) == 2
    # Extra manifest metadata leaves a structurally valid snapshot but changes
    # its pinned identity; the old tokenizer must not authenticate it.
    manifest_path = relocated / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["annotation"] = "another snapshot identity"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="snapshot identity mismatch"):
        verify_artifact(artifact, declaration, dataset=moved, **kwargs)
    assert verify_artifact(artifact, declaration, dataset=dataset, **kwargs) == first


def test_reopen_binds_prepared_snapshot_despite_updated_tokenizer_provenance(
    tmp_path, monkeypatch
):
    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "local_stories")
    path = publish_lock(resolve_plan(plan, declaration), tmp_path / "experiment")
    manifest_path = tmp_path / "snapshot/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["annotation"] = "a different valid snapshot identity"
    manifest_path.write_text(json.dumps(manifest))
    tokenizer_manifest = tmp_path / "tokenizer/tokenizer_manifest.json"
    provenance = json.loads(tokenizer_manifest.read_text())
    provenance["source_manifest_sha256"] = sha256_file(manifest_path)
    provenance["training_contract"]["source_manifest_sha256"] = sha256_file(
        manifest_path
    )
    tokenizer_manifest.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="pinned story snapshot"):
        open_lock(path)
