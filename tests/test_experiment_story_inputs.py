"""Direct story locks use real tokenizer/packing verifiers and bounded offline streams."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.config.loading import load_config
from sparselab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
from sparselab.data import datasets, local_stories
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
from sparselab.experiments.plan import ExperimentPlan
from sparselab.training.manifest import sha256_file


def story_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str):
    """Prepare tiny actual artifacts without contacting Hugging Face."""
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    tmp_path.mkdir(parents=True, exist_ok=True)

    def stream(_name, *, split, revision, **_kwargs):
        assert revision == local_stories.REVISION
        return iter(
            {"text": f"{split} story {i}: A small fox visited the quiet forest."}
            for i in range(4)
        )

    monkeypatch.setattr(datasets, "load_dataset", stream)
    monkeypatch.setattr("datasets.load_dataset", stream)
    data = {
        "source": source,
        "revision": local_stories.REVISION,
        "cache_dir": tmp_path / "cache",
        "train_max_documents": 4,
        "validation_max_documents": 2,
        "train_max_tokens": 512,
        "validation_max_tokens": 256,
    }
    if source == "local_stories":
        snapshot = tmp_path / "snapshot"
        local_stories.snapshot(snapshot, train_count=4, validation_count=2)
        data.update(
            license=local_stories.LICENSE,
            train_path=snapshot / "train.jsonl",
            validation_path=snapshot / "validation.jsonl",
            source_manifest_path=snapshot / "manifest.json",
        )
    dataset = DatasetConfig.model_validate(data)
    tokenizer = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            min_frequency=1,
            max_documents=4,
            output_dir=tmp_path / "tokenizer",
            dataset=dataset.model_copy(update={"train_max_tokens": 4096}),
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
    artifacts = {
        "tokenizer": {
            "kind": "tokenizer",
            "version": 1,
            "producer": "sparselab",
            "identifier": tokenizer.parent.name,
            "sha256": sha256_file(tokenizer),
            "path": str(tokenizer),
        },
        "prepared": {
            "kind": "prepared_data",
            "version": 1,
            "producer": "sparselab",
            "identifier": prepared.manifest["settings_sha256"],
            "sha256": prepared.manifest["manifest_sha256"],
            "path": str(prepared.root),
        },
    }
    plan = ExperimentPlan(
        plan_version=1,
        id="direct-stories",
        base_run=config,
        artifacts=artifacts,
        inputs={"tokenizer": "tokenizer", "training": "prepared"},
    )
    declaration = tmp_path / "plan.json"
    declaration.write_text(plan.model_dump_json())
    return plan, declaration, prepared


@pytest.mark.parametrize("source", ["tinystories", "local_stories"])
def test_direct_story_prepare_lock_publish_reopen(tmp_path, monkeypatch, source):
    plan, declaration, prepared = story_plan(tmp_path, monkeypatch, source)

    def offline(*_args, **_kwargs):
        raise AssertionError("locking must not acquire remote source data")

    monkeypatch.setattr(datasets, "load_dataset", offline)
    monkeypatch.setattr("datasets.load_dataset", offline)
    locked = resolve_plan(plan, declaration)
    path = publish_lock(locked, tmp_path / "experiment")
    reopened = open_lock(path)
    assert reopened.plan_sha256 == locked.plan_sha256
    assert {value["kind"] for value in reopened.cells[0].artifacts.values()} == {
        "tokenizer",
        "prepared_data",
    }
    assert reopened.cells[0].config.dataset.source == source
    assert reopened.cells[0].config.dataset.corpus_release_path is None
    assert (
        prepared.manifest["cache_identity"]["dataset"]["revision"]
        == local_stories.REVISION
    )


@pytest.mark.parametrize("source", ["tinystories", "local_stories"])
@pytest.mark.parametrize(
    "member", ["train.npy", "validation.npy", "tokenizer_manifest.json"]
)
def test_direct_story_reopen_rejects_changed_artifact(
    tmp_path, monkeypatch, source, member
):
    plan, declaration, prepared = story_plan(tmp_path, monkeypatch, source)
    path = publish_lock(resolve_plan(plan, declaration), tmp_path / "experiment")
    target = (
        prepared.root / member
        if member.endswith(".npy")
        else tmp_path / "tokenizer" / member
    )
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        open_lock(path)


@pytest.mark.parametrize(
    "member", ["train.jsonl", "validation.jsonl", "excluded.jsonl", "manifest.json"]
)
def test_local_story_source_change_blocks_publication(tmp_path, monkeypatch, member):
    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "local_stories")
    locked = resolve_plan(plan, declaration)
    target = tmp_path / "snapshot" / member
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        publish_lock(locked, tmp_path / "experiment")


def test_tinystories_lock_rejects_moving_revision(tmp_path, monkeypatch):
    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "tinystories")
    raw = plan.model_dump(mode="json")
    raw["base_run"]["dataset"]["revision"] = "main"
    with pytest.raises(ValueError):
        resolve_plan(ExperimentPlan.model_validate(raw), declaration)


@pytest.mark.parametrize("source", ["tinystories", "local_stories"])
def test_story_lock_rejects_different_preparation_settings(
    tmp_path, monkeypatch, source
):
    plan, declaration, _ = story_plan(tmp_path, monkeypatch, source)
    raw = plan.model_dump(mode="json")
    raw["base_run"]["dataset"]["train_max_tokens"] += 1
    with pytest.raises(ValueError, match="prepared cache"):
        resolve_plan(ExperimentPlan.model_validate(raw), declaration)


def test_story_lock_rejects_tokenizer_from_other_source(tmp_path, monkeypatch):
    plan, declaration, _ = story_plan(tmp_path, monkeypatch, "tinystories")
    manifest_path = tmp_path / "tokenizer/tokenizer_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = "synthetic"
    manifest["revision"] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance|source"):
        resolve_plan(plan, declaration)
