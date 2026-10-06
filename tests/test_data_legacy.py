from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.config.models import DatasetConfig
from sparselab.data import legacy, local_stories
from sparselab.data.sources import load_source, verify_snapshot

RESOURCES = {
    "max_source_records": 100,
    "max_text_bytes": 100000,
    "max_record_bytes": 10000,
    "max_work_bytes": 10000000,
    "min_free_bytes": 0,
}


def _dataset() -> dict:
    return {
        "source": "tinystories",
        "revision": local_stories.REVISION,
        "cache_dir": "cache",
        "train_max_documents": 3,
        "validation_max_documents": 2,
        "train_max_tokens": 500,
        "validation_max_tokens": 100,
    }


def _tokenizer(path: Path, dataset: dict | None = None) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "vocab_size": 512,
                "min_frequency": 1,
                "max_documents": 2,
                "output_dir": "tokenizer",
                "dataset": _dataset() if dataset is None else dataset,
            }
        )
    )
    return path


def _old_snapshot(monkeypatch: pytest.MonkeyPatch, root: Path) -> DatasetConfig:
    rows = {"train": ["a", "a", "b", "c"], "validation": ["a", "d", "e"]}
    monkeypatch.setattr(
        local_stories,
        "_hub_stream",
        lambda split, _cache: ({"text": text} for text in rows[split]),
    )
    local_stories.snapshot(root, train_count=3, validation_count=2)
    return DatasetConfig.model_validate(
        {
            **_dataset(),
            "source": "local_stories",
            "license": local_stories.LICENSE,
            "train_path": root / "train.jsonl",
            "validation_path": root / "validation.jsonl",
            "source_manifest_path": root / "manifest.json",
        }
    )


def test_new_execution_guard_leaves_historical_model_valid() -> None:
    old = DatasetConfig.model_validate(_dataset())
    with pytest.raises(ValueError, match="sparselab data migrate"):
        legacy.require_current_dataset(old)
    current = DatasetConfig.model_validate(
        {**_dataset(), "source": "synthetic", "revision": None}
    )
    assert legacy.require_current_dataset(current) is None


def test_direct_requires_policy_ack_and_explicit_resources(tmp_path: Path) -> None:
    path = _tokenizer(tmp_path / "old.yaml")
    with pytest.raises(ValueError, match="Explicit policy migration required"):
        legacy.migrate(path, tmp_path / "new", resources=RESOURCES)
    with pytest.raises(ValueError, match="resource bounds"):
        legacy.migrate(path, tmp_path / "new", accept_policy_change=True)
    assert not (tmp_path / "new").exists()


def test_tokenizer_migration_preserves_budgets_paths_and_source_bytes(
    tmp_path: Path,
) -> None:
    path = _tokenizer(tmp_path / "old.yaml")
    original = path.read_bytes()
    destination = tmp_path / "new"
    receipt_path = legacy.migrate(
        path, destination, resources=RESOURCES, accept_policy_change=True
    )
    receipt = json.loads(receipt_path.read_text())
    old = load_tokenizer_config(path)
    new = load_tokenizer_config(destination / "config.yaml")
    source = load_source(destination / "source.yaml")
    assert path.read_bytes() == original
    assert new.output_dir == old.output_dir
    assert new.max_documents == old.max_documents
    assert new.dataset.cache_dir == old.dataset.cache_dir
    for key in (
        "train_max_documents",
        "validation_max_documents",
        "train_max_tokens",
        "validation_max_tokens",
    ):
        assert getattr(new.dataset, key) == getattr(old.dataset, key)
    assert new.dataset.source == "snapshot"
    assert new.dataset.train_path == destination / "snapshot/train.jsonl"
    assert source.dedup.within_split == source.dedup.overlap == "keep"
    assert source.selection.documents == {"train": 3, "validation": 2}
    assert receipt["snapshot_status"] == "pending_acquisition"
    assert receipt["identity_preserved"] is False
    assert receipt["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert "empty/null" in receipt["warnings"][0]
    assert not (destination / "snapshot").exists()


def test_run_migration_preserves_scientific_fields(tmp_path: Path) -> None:
    from sparselab.research.scaffold import scaffold_lesson

    scaffold = scaffold_lesson(
        "dense", tmp_path / "lesson", scale="smoke", data="offline"
    )
    path = scaffold / "model.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["dataset"] = _dataset()
    path.write_text(yaml.safe_dump(raw))
    old = load_config(path)
    legacy.migrate(
        path, tmp_path / "new", resources=RESOURCES, accept_policy_change=True
    )
    new = load_config(tmp_path / "new/config.yaml")
    before, after = old.model_dump(mode="json"), new.model_dump(mode="json")
    before.pop("dataset")
    after.pop("dataset")
    assert before == after


@pytest.mark.parametrize("destination_kind", ["directory", "symlink"])
def test_migration_never_replaces_existing_output(
    tmp_path: Path, destination_kind: str
) -> None:
    path = _tokenizer(tmp_path / "old.yaml")
    destination = tmp_path / "new"
    if destination_kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(FileExistsError):
        legacy.migrate(
            path, destination, resources=RESOURCES, accept_policy_change=True
        )


def test_direct_rejects_moving_revision_without_publishing(tmp_path: Path) -> None:
    path = _tokenizer(tmp_path / "old.yaml", {**_dataset(), "revision": "main"})
    with pytest.raises(ValueError, match="immutable"):
        legacy.migrate(
            path, tmp_path / "new", resources=RESOURCES, accept_policy_change=True
        )
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("bare_manifest", [False, True])
def test_import_legacy_snapshot_retains_old_identity_and_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bare_manifest: bool
) -> None:
    dataset = _old_snapshot(monkeypatch, tmp_path / "old")
    originals = {p: p.read_bytes() for p in (tmp_path / "old").iterdir()}
    path = (
        dataset.source_manifest_path
        if bare_manifest
        else _tokenizer(tmp_path / "old.yaml", dataset.model_dump(mode="json"))
    )
    receipt_path = legacy.migrate(path, tmp_path / "new")
    assert all(p.read_bytes() == content for p, content in originals.items())
    old = local_stories.verify_snapshot(dataset)
    new = verify_snapshot(tmp_path / "new/snapshot/manifest.json")
    assert (
        new["manifest_sha256"]
        != hashlib.sha256(originals[dataset.source_manifest_path]).hexdigest()
    )
    assert new["splits"]["train"]["count"] == old["splits"]["train"]["count"]
    assert json.loads(receipt_path.read_text())["snapshot_status"] == "imported"
    if not bare_manifest:
        config = load_tokenizer_config(tmp_path / "new/config.yaml")
        verify_snapshot(config.dataset)
        assert config.dataset.train_max_tokens == dataset.train_max_tokens
    else:
        binding = yaml.safe_load((tmp_path / "new/dataset-binding.yaml").read_text())[
            "dataset"
        ]
        assert "train_max_tokens" not in binding


def test_import_rejects_tampered_old_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _old_snapshot(monkeypatch, tmp_path / "old")
    dataset.train_path.write_text('{"ordinal":0,"text":"tampered"}\n')
    with pytest.raises(ValueError):
        legacy.migrate(dataset.source_manifest_path, tmp_path / "new")
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".data-migrate-*"))


def test_local_guard_preserves_historical_verifier_and_rejects_nested_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _old_snapshot(monkeypatch, tmp_path / "old")
    with pytest.raises(ValueError, match="sparselab data migrate"):
        legacy.require_current_dataset(dataset)
    assert local_stories.verify_snapshot(dataset)["splits"]["train"]["count"] == 3
    with pytest.raises(ValueError, match="outside the historical snapshot"):
        legacy.migrate(dataset.source_manifest_path, tmp_path / "old/new")
    assert not (tmp_path / "old/new").exists()


def test_parser_dispatch_emits_json(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = _tokenizer(tmp_path / "old.yaml")
    bounds = tmp_path / "resources.yaml"
    bounds.write_text(yaml.safe_dump(RESOURCES))
    parser = argparse.ArgumentParser()
    legacy.register_parser(parser.add_subparsers())
    args = parser.parse_args(
        [
            "migrate",
            str(path),
            "--output",
            str(tmp_path / "new"),
            "--resources",
            str(bounds),
            "--accept-policy-change",
            "--json",
        ]
    )
    args.handler(args)
    assert json.loads(capsys.readouterr().out)["policy_change_accepted"] is True
