"""Snapshot artifact proofs bind raw availability as well as tokenizer bytes."""

from __future__ import annotations

import json
import shutil

import pytest

from sparselab.config.models import DatasetConfig
from sparselab.experiments.artifacts import _artifact_key
from sparselab.experiments.plan import Artifact
from sparselab.training.manifest import sha256_file


@pytest.fixture
def snapshot_binding(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    for member in (
        "train.jsonl",
        "validation.jsonl",
        "excluded.jsonl",
        "events.jsonl",
        "test.jsonl",
    ):
        (root / member).write_text("{}\n")
    (root / "manifest.json").write_text(
        json.dumps({"excluded": {"path": "excluded.jsonl"}})
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    tokenizer.with_name("tokenizer_manifest.json").write_text(
        json.dumps({"source": "snapshot"})
    )
    artifact = Artifact(
        kind="tokenizer",
        version=1,
        producer="test",
        identifier=tmp_path.name,
        path=str(tokenizer),
        sha256=sha256_file(tokenizer),
    )
    dataset = DatasetConfig(
        source="snapshot",
        cache_dir=tmp_path / "cache",
        train_max_documents=4,
        validation_max_documents=2,
        train_max_tokens=512,
        validation_max_tokens=256,
        revision="a" * 40,
        license="MIT",
        train_path=root / "train.jsonl",
        validation_path=root / "validation.jsonl",
        source_manifest_path=root / "manifest.json",
    )
    return artifact, tokenizer, dataset


@pytest.mark.parametrize(
    "member",
    [
        "train.jsonl",
        "validation.jsonl",
        "excluded.jsonl",
        "manifest.json",
        "events.jsonl",
        "test.jsonl",
    ],
)
def test_snapshot_raw_drift_invalidates_proof_key(snapshot_binding, member):
    artifact, tokenizer, dataset = snapshot_binding
    original = _artifact_key(artifact, tokenizer, dataset)
    target = dataset.source_manifest_path.parent / member
    target.write_bytes(target.read_bytes() + b" ")
    assert _artifact_key(artifact, tokenizer, dataset) != original


@pytest.mark.parametrize(
    "member", ["train.jsonl", "validation.jsonl", "excluded.jsonl", "manifest.json"]
)
def test_snapshot_raw_availability_required(snapshot_binding, member):
    artifact, tokenizer, dataset = snapshot_binding
    (dataset.source_manifest_path.parent / member).unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        _artifact_key(artifact, tokenizer, dataset)


def test_snapshot_relocation_requires_new_proof_binding(snapshot_binding, tmp_path):
    artifact, tokenizer, dataset = snapshot_binding
    original = _artifact_key(artifact, tokenizer, dataset)
    relocated = tmp_path / "relocated"
    shutil.copytree(dataset.source_manifest_path.parent, relocated)
    moved = dataset.model_copy(
        update={
            field: relocated / getattr(dataset, field).name
            for field in ("train_path", "validation_path", "source_manifest_path")
        }
    )
    rebound = _artifact_key(artifact, tokenizer, moved)
    assert rebound != original
    assert (
        json.loads(rebound[8])["dataset_context"]
        == json.loads(original[8])["dataset_context"]
    )
    relocated.joinpath("train.jsonl").unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        _artifact_key(artifact, tokenizer, moved)


@pytest.mark.parametrize("context", [None, "synthetic", "local_stories"])
def test_snapshot_proof_rejects_absent_or_other_source_context(
    snapshot_binding, context
):
    artifact, tokenizer, dataset = snapshot_binding
    wrong = None if context is None else dataset.model_copy(update={"source": context})
    with pytest.raises(ValueError, match="snapshot tokenizer requires"):
        _artifact_key(artifact, tokenizer, wrong)


@pytest.mark.parametrize("unsafe", ["symlink", "traversal"])
def test_snapshot_excluded_dependency_cannot_escape_safe_paths(
    snapshot_binding, tmp_path, unsafe
):
    artifact, tokenizer, dataset = snapshot_binding
    manifest = dataset.source_manifest_path
    if unsafe == "symlink":
        excluded = manifest.parent / "excluded.jsonl"
        excluded.unlink()
        excluded.symlink_to(dataset.train_path)
    else:
        manifest.write_text(
            json.dumps({"excluded": {"path": "../snapshot/excluded.jsonl"}})
        )
    with pytest.raises(ValueError, match="symlink|traversal"):
        _artifact_key(artifact, tokenizer, dataset)
