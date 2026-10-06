"""Existing inputs are authenticated before authoring; locking remains a separate gate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from test_experiment_variant_tokenizer import authored
from test_training import config as synthetic_config

from sparselab.config.loading import load_config
from sparselab.config.models import RunConfig, TokenizerTrainConfig
from sparselab.data import datasets, local_stories
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.experiments.direct_inputs import bind_direct_inputs
from sparselab.experiments.export_config import export_effective_config
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
from sparselab.experiments.plan import load_plan

# Register the shared Forge fixture under this module's descriptive alias.
forge_authored = authored


def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str = "synthetic"):
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    if source == "synthetic":
        config = synthetic_config(tmp_path)
    else:

        def stream(_name, *, split, revision, **_kwargs):
            assert revision == local_stories.REVISION
            return iter(
                {"text": f"{split} story {i}: A small fox visited the forest."}
                for i in range(4)
            )

        monkeypatch.setattr(datasets, "load_dataset", stream)
        monkeypatch.setattr("datasets.load_dataset", stream)
        dataset = {
            "source": "tinystories",
            "revision": local_stories.REVISION,
            "cache_dir": tmp_path / "cache",
            "train_max_documents": 4,
            "validation_max_documents": 2,
            "train_max_tokens": 512,
            "validation_max_tokens": 256,
        }
        tokenizer = train_tokenizer(
            TokenizerTrainConfig.model_validate(
                {
                    "schema_version": 1,
                    "vocab_size": 260,
                    "min_frequency": 1,
                    "max_documents": 4,
                    "output_dir": tmp_path / "tokenizer",
                    "dataset": {**dataset, "train_max_tokens": 4096},
                }
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
    prepared = prepare_data(config, load_tokenizer(config.tokenizer.path))
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "plan_version": 1,
                "id": "direct-bind-test",
                "base_run": "run.yaml",
                "phases": [{"id": "pretrain", "transition": "fresh"}],
                "execution": {"backend": "cpu"},
            }
        )
    )
    return config, prepared.root, run_path, template


@pytest.mark.parametrize("source", ["synthetic", "tinystories"])
def test_bind_publish_reopen_export_and_relocation(tmp_path, monkeypatch, source):
    config, root, run, template = inputs(tmp_path, monkeypatch, source)
    if source == "tinystories":
        monkeypatch.setattr(
            datasets, "load_dataset", lambda *_a, **_k: pytest.fail("download")
        )
        monkeypatch.setattr(
            "datasets.load_dataset", lambda *_a, **_k: pytest.fail("download")
        )
    output = tmp_path / "plan.yaml"
    plan = bind_direct_inputs(run, template, root, output)
    assert plan.base_run == config
    assert (
        plan.artifacts["packed"].sha256
        == json.loads((root / "manifest.json").read_text())["manifest_sha256"]
    )
    with pytest.raises(FileExistsError):
        bind_direct_inputs(run, template, root, output)
    moved = tmp_path / "relocated"
    moved.mkdir()
    relocated = moved / "plan.yaml"
    shutil.copyfile(output, relocated)
    assert load_plan(relocated) == plan
    lock_path = publish_lock(
        resolve_plan(load_plan(relocated), relocated), tmp_path / "experiment"
    )
    locked = open_lock(lock_path)
    assert locked.cells[0].id == "pretrain:single"
    effective = tmp_path / "effective.yaml"
    digest = export_effective_config(lock_path, "pretrain:single", effective)
    assert digest == locked.cells[0].config_sha256
    assert load_config(effective) == locked.cells[0].config
    with pytest.raises(FileExistsError):
        export_effective_config(lock_path, "pretrain:single", effective)
    with pytest.raises(ValueError, match="unknown cell"):
        export_effective_config(lock_path, "wrong", tmp_path / "wrong.yaml")
    assert not (tmp_path / "wrong.yaml").exists()


def test_export_reopens_and_rejects_changed_prepared_array(tmp_path, monkeypatch):
    _, root, run, template = inputs(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.yaml"
    plan = bind_direct_inputs(run, template, root, plan_path)
    lock = publish_lock(resolve_plan(plan, plan_path), tmp_path / "experiment")
    array = root / "train.npy"
    array.write_bytes(array.read_bytes() + b"tampered")
    output = tmp_path / "effective.yaml"
    with pytest.raises(ValueError):
        export_effective_config(lock, "pretrain:single", output)
    assert not output.exists()


@pytest.mark.parametrize(
    "change", ["revision", "tokenizer", "array", "cache", "source"]
)
def test_bind_rejects_mismatched_or_changed_existing_inputs(
    tmp_path, monkeypatch, change
):
    config, root, run, template = inputs(tmp_path, monkeypatch, "tinystories")
    if change == "revision":
        raw = config.model_dump(mode="json")
        raw["dataset"]["revision"] = "a" * 40
        run.write_text(yaml.safe_dump(raw))
    elif change == "tokenizer":
        manifest = config.tokenizer.path.with_name("tokenizer_manifest.json")
        raw = json.loads(manifest.read_text())
        raw["revision"] = "b" * 40
        manifest.write_text(json.dumps(raw))
    elif change == "array":
        path = root / "train.npy"
        path.write_bytes(path.read_bytes() + b"changed")
    elif change == "cache":
        raw = config.model_dump(mode="json")
        raw["dataset"]["train_max_tokens"] += 1
        run.write_text(yaml.safe_dump(raw))
    else:
        manifest_path = root / "manifest.json"
        raw = json.loads(manifest_path.read_text())
        raw["cache_identity"]["source_identity_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(raw))
    output = tmp_path / "plan.yaml"
    with pytest.raises((ValueError, KeyError)):
        bind_direct_inputs(run, template, root, output)
    assert not output.exists()


def test_valid_but_wrong_tokenizer_rejected(tmp_path, monkeypatch):
    config, root, run, template = inputs(tmp_path, monkeypatch, "tinystories")
    other = tmp_path / "other"
    revision = "a" * 40
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_a, **_k: iter(
            {"text": f"Another story {i} about a curious fox."} for i in range(4)
        ),
    )
    monkeypatch.setattr("datasets.load_dataset", datasets.load_dataset)
    alternate = train_tokenizer(
        TokenizerTrainConfig.model_validate(
            {
                "schema_version": 1,
                "vocab_size": 260,
                "min_frequency": 1,
                "max_documents": 4,
                "output_dir": other,
                "dataset": {
                    **config.dataset.model_dump(mode="python"),
                    "revision": revision,
                    "train_max_tokens": 4096,
                },
            }
        )
    )
    raw = yaml.safe_load(run.read_text())
    raw["tokenizer"]["path"] = str(alternate)
    run.write_text(yaml.safe_dump(raw))
    output = tmp_path / "plan.yaml"
    with pytest.raises(ValueError, match="tokenizer|prepared"):
        bind_direct_inputs(run, template, root, output)
    assert not output.exists()


def test_forge_bindings_include_release_and_export(forge_authored, tmp_path):
    _, _, authored_plan = forge_authored
    # The shared Forge fixture has already frozen the release and export.
    tokenizer_path = Path(authored_plan["artifacts"]["shared_tokenizer"]["path"])
    export = tokenizer_path.parent.parent
    config = load_config(export / "run.yaml")
    prepared = prepare_data(config, load_tokenizer(tokenizer_path))
    run = tmp_path / "forge-run.yaml"
    run.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    template = tmp_path / "forge-template.yaml"
    template.write_text("plan_version: 1\nid: direct-forge\nbase_run: forge-run.yaml\n")
    output = tmp_path / "forge-plan.yaml"
    plan = bind_direct_inputs(run, template, prepared.root, output)
    assert {artifact.kind for artifact in plan.artifacts.values()} == {
        "corpus_release",
        "corpus_export",
        "tokenizer",
        "prepared_data",
    }
    assert set(plan.inputs) == {
        "corpus_release",
        "corpus_export",
        "tokenizer",
        "prepared_data",
    }


def test_bind_external_full_state_parent_survives_declaration_relocation(
    tmp_path, monkeypatch
):
    from sparselab.training.checkpoints import CheckpointManager
    from sparselab.training.trainer import train

    config, root, run, template = inputs(tmp_path, monkeypatch)
    raw = config.model_dump(mode="json")
    raw["training"].update(max_steps=4, max_tokens=128)
    raw["optimizer"]["decay_steps"] = 4
    config = RunConfig.model_validate(raw)
    run.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    run_id = train(config, run_id="parent")
    manager = CheckpointManager(config.logging.root_dir / run_id)
    generation = manager._resolve(manager.root / "latest.json")
    snapshot = manager.load(generation)
    document = yaml.safe_load(template.read_text())
    document["artifacts"] = {
        "parent": {
            "kind": "checkpoint",
            "version": 2,
            "producer": "sparselab",
            "identifier": generation.name,
            "sha256": snapshot.checkpoint_sha256,
            "path": generation.relative_to(tmp_path).as_posix(),
            "state": "full",
        }
    }
    document["phases"] = [
        {
            "id": "continue",
            "transition": "extend_budget",
            "checkpoint": "parent",
            "set": {
                "training.max_steps": 8,
                "training.max_tokens": 256,
                "optimizer.decay_steps": 4,
            },
        }
    ]
    template.write_text(yaml.safe_dump(document))
    moved = tmp_path / "declarations"
    moved.mkdir()
    output = moved / "plan.yaml"
    plan = bind_direct_inputs(run, template, root, output)
    locked = resolve_plan(plan, output)
    assert (
        locked.cells[0].artifacts["parent_checkpoint"]["sha256"]
        == snapshot.checkpoint_sha256
    )
    assert locked.cells[0].config.training.max_steps == 8
    assert Path(plan.artifacts["parent"].path) == generation
    state = generation / "training_state.pt"
    state.write_bytes(state.read_bytes() + b"corrupt")
    rejected = moved / "rejected.yaml"
    with pytest.raises(ValueError):
        bind_direct_inputs(run, template, root, rejected)
    assert not rejected.exists()
