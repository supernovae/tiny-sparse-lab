"""Generic snapshot inputs keep their native source identity through plan binding."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from dataset_fixtures import offline_hub_identity

from sparselab.config.loading import load_config
from sparselab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
from sparselab.data import sources
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.direct_inputs import bind_direct_inputs
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
from sparselab.training.manifest import canonical_json
from sparselab.verification_proofs import ProofStore


@pytest.fixture
def snapshot_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    revision = "a" * 40
    monkeypatch.setattr(
        sources,
        "_hub_identity",
        offline_hub_identity,
    )
    monkeypatch.setattr(
        sources,
        "_stream",
        lambda source, split, cache, **kw: iter(
            {"body": f"{split} document {i}: A small fox visited the quiet forest."}
            for i in range(4)
        ),
    )
    declaration = tmp_path / "source.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "repo_id": "fixture/second-dataset",
                "revision": revision,
                "config": "default",
                "splits": {"train": "training", "validation": "heldout"},
                "text_field": "body",
                "attribution": "Offline fixture",
                "license": "MIT",
                "selection": {
                    "mode": "bounded",
                    "documents": {"train": 4, "validation": 2},
                },
                "resources": {
                    "max_source_records": 20,
                    "max_text_bytes": 100000,
                    "max_record_bytes": 10000,
                    "max_work_bytes": 10000000,
                    "min_free_bytes": 0,
                },
            }
        )
    )
    lock = sources.lock_source(
        declaration, tmp_path / "source.lock.json", tmp_path / "hub"
    )
    manifest = sources.snapshot_source(lock, tmp_path / "snapshot", tmp_path / "hub")
    dataset = DatasetConfig(
        source="snapshot",
        revision=revision,
        license="MIT",
        cache_dir=tmp_path / "cache",
        train_path=manifest.parent / "train.jsonl",
        validation_path=manifest.parent / "validation.jsonl",
        source_manifest_path=manifest,
        train_max_documents=4,
        validation_max_documents=2,
        train_max_tokens=512,
        validation_max_tokens=256,
    )
    tokenizer = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            min_frequency=1,
            max_documents=4,
            output_dir=tmp_path / "tokenizer",
            dataset=dataset,
        )
    )
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    config = RunConfig.model_validate(
        {
            **base.model_dump(mode="python"),
            "dataset": dataset,
            "tokenizer": {"path": tokenizer},
            "model": {**base.model.model_dump(), "vocab_size": 260},
            "logging": {**base.logging.model_dump(), "root_dir": tmp_path / "runs"},
            "training": {
                **base.training.model_dump(),
                "max_steps": 2,
                "max_tokens": 64,
            },
            "optimizer": {**base.optimizer.model_dump(), "warmup_steps": 0},
        }
    )
    prepared = prepare_data(config, load_tokenizer(tokenizer))
    run = tmp_path / "run.yaml"
    run.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "plan_version": 1,
                "id": "generic-snapshot",
                "base_run": "run.yaml",
                "execution": {"backend": "cpu"},
            }
        )
    )
    output = tmp_path / "plan.yaml"
    plan = bind_direct_inputs(run, template, prepared.root, output)
    return plan, output, prepared


def test_snapshot_bind_lock_publish_reopen(snapshot_inputs, tmp_path):
    plan, declaration, prepared = snapshot_inputs
    locked = resolve_plan(plan, declaration)
    path = publish_lock(locked, tmp_path / "experiment")
    reopened = open_lock(path)
    assert reopened.plan_sha256 == locked.plan_sha256
    assert (
        reopened.availability["artifact_datasets"]["tokenizer"]["source"] == "snapshot"
    )
    assert {value["kind"] for value in reopened.cells[0].artifacts.values()} == {
        "tokenizer",
        "prepared_data",
    }
    identity = prepared.manifest["cache_identity"]
    assert "local_stories_source_sha256" not in identity
    assert (
        identity["snapshot_source_sha256"]
        == hashlib.sha256(
            canonical_json(sources.verify_snapshot(plan.base_run.dataset))
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "member",
    [
        "train.jsonl",
        "validation.jsonl",
        "manifest.json",
        "excluded.jsonl",
        "events.jsonl",
    ],
)
def test_snapshot_publication_rechecks_raw_inputs(snapshot_inputs, tmp_path, member):
    plan, declaration, _ = snapshot_inputs
    locked = resolve_plan(plan, declaration)
    target = tmp_path / "snapshot" / member
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        publish_lock(locked, tmp_path / "experiment")


@pytest.mark.parametrize("reuse", ["memo", "persistent"])
def test_snapshot_reuse_relocation_and_missing_raw_source(
    snapshot_inputs, tmp_path, reuse
):
    plan, declaration, _ = snapshot_inputs
    artifact = plan.artifacts["tokenizer"]
    dataset = plan.base_run.dataset
    store = ProofStore(tmp_path)
    options = (
        {"memo": {}}
        if reuse == "memo"
        else {"proof_store": store, "verification_mode": "verified_reuse"}
    )
    first = verify_artifact(artifact, declaration, dataset=dataset, **options)
    assert verify_artifact(artifact, declaration, dataset=dataset, **options) == first
    if reuse == "persistent":
        assert store.recorded == 1, store.diagnostics()
        assert store.hits == 1
    relocated = tmp_path / "relocated"
    shutil.copytree(tmp_path / "snapshot", relocated)
    moved = dataset.model_copy(
        update={
            field: relocated / getattr(dataset, field).name
            for field in ("train_path", "validation_path", "source_manifest_path")
        }
    )
    assert verify_artifact(artifact, declaration, dataset=moved, **options) == first
    (relocated / "events.jsonl").unlink()
    with pytest.raises(ValueError):
        verify_artifact(artifact, declaration, dataset=moved, **options)


def test_snapshot_lock_sidecar_requires_raw_source_after_reopen(
    snapshot_inputs, tmp_path
):
    plan, declaration, _ = snapshot_inputs
    path = publish_lock(resolve_plan(plan, declaration), tmp_path / "experiment")
    (tmp_path / "snapshot/events.jsonl").unlink()
    with pytest.raises(ValueError):
        open_lock(path)


@pytest.mark.parametrize("change", ["missing", "legacy", "wrong"])
def test_snapshot_lock_requires_its_own_canonical_source_identity(
    snapshot_inputs, change
):
    from sparselab.experiments.lock import _check_cell_inputs

    plan, _, prepared = snapshot_inputs
    manifest_path = prepared.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = manifest["cache_identity"]
    digest = identity.pop("snapshot_source_sha256")
    if change == "legacy":
        identity["local_stories_source_sha256"] = digest
    elif change == "wrong":
        identity["snapshot_source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    identities = {
        name: plan.artifacts[reference].model_dump(mode="json")
        for name, reference in plan.inputs.items()
    }
    paths = {name: value["path"] for name, value in identities.items()}
    with pytest.raises(ValueError, match="pinned source snapshot"):
        _check_cell_inputs(
            plan.base_run, identities, paths, identity["source_identity_sha256"]
        )


@pytest.mark.parametrize("change", ["missing", "wrong-source"])
def test_snapshot_lock_cannot_drop_or_relabel_dataset_context(
    snapshot_inputs, tmp_path, change
):
    from sparselab.experiments.lock import ResolvedExperimentPlan

    plan, declaration, _ = snapshot_inputs
    raw = resolve_plan(plan, declaration).model_dump(mode="json")
    contexts = raw["availability"]["artifact_datasets"]
    if change == "missing":
        contexts.clear()
    else:
        contexts["tokenizer"]["source"] = "synthetic"
    with pytest.raises(ValueError):
        publish_lock(ResolvedExperimentPlan.model_validate(raw), tmp_path / "rejected")
