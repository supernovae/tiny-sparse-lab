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


def test_unused_story_tokenizer_keeps_verification_context(tmp_path, monkeypatch):
    from sparselab.experiments.plan import ExperimentPlan

    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "local_stories")
    shutil.copytree(tmp_path / "tokenizer", tmp_path / "spare")
    raw = plan.model_dump(mode="json")
    raw["artifacts"]["spare"] = {
        **raw["artifacts"]["tokenizer"],
        "identifier": "spare",
        "path": str(tmp_path / "spare/tokenizer.json"),
    }
    locked = resolve_plan(ExperimentPlan.model_validate(raw), declaration)
    path = publish_lock(locked, tmp_path / "experiment")
    reopened = open_lock(path)
    assert set(reopened.availability["artifact_datasets"]) == {"tokenizer", "spare"}
    assert reopened.scientific_sha256 == locked.scientific_sha256


def test_effective_story_paths_preserve_verification_context(tmp_path, monkeypatch):
    from sparselab.experiments.plan import ExperimentPlan

    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "local_stories")
    shutil.copytree(tmp_path / "snapshot", tmp_path / "relocated")
    raw = plan.model_dump(mode="json")
    raw["phases"] = [
        {
            "id": "main",
            "set": {
                f"dataset.{field}": str(tmp_path / "relocated" / member)
                for field, member in (
                    ("train_path", "train.jsonl"),
                    ("validation_path", "validation.jsonl"),
                    ("source_manifest_path", "manifest.json"),
                )
            },
        }
    ]
    locked = resolve_plan(ExperimentPlan.model_validate(raw), declaration)
    path = publish_lock(locked, tmp_path / "experiment")
    reopened = open_lock(path)
    assert reopened.availability["artifact_datasets"]["tokenizer"]["train_path"] == str(
        tmp_path / "snapshot/train.jsonl"
    )
    assert (
        reopened.cells[0].config.dataset.train_path
        == tmp_path / "relocated/train.jsonl"
    )
    target = tmp_path / "relocated/train.jsonl"
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        open_lock(path)


@pytest.mark.parametrize("invalid", ["missing", "null", "unknown", "non-tokenizer"])
def test_malformed_story_verification_context_fails_closed(
    tmp_path, monkeypatch, invalid
):
    from sparselab.experiments.lock import ResolvedExperimentPlan

    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "local_stories")
    raw = resolve_plan(plan, declaration).model_dump(mode="json")
    contexts = raw["availability"]["artifact_datasets"]
    if invalid == "missing":
        contexts.clear()
    elif invalid == "null":
        contexts["tokenizer"] = None
    else:
        contexts["unknown" if invalid == "unknown" else "prepared"] = contexts[
            "tokenizer"
        ]
    with pytest.raises(ValueError):
        publish_lock(
            ResolvedExperimentPlan.model_validate(raw), tmp_path / "experiment"
        )
