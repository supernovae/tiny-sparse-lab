from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.data.sources import load_source
from sparselab.research import catalog, scaffold


@pytest.mark.parametrize("kind", ["research", "learn"])
def test_snapshot_scaffold_binds_resource_and_paths(tmp_path: Path, kind: str) -> None:
    if kind == "research":
        output = scaffold.scaffold_research(
            "engram-ffn-substitution-v1",
            tmp_path / "new",
            scale="smoke",
            data="tinystories",
        )
        metadata = json.loads((output / "research.json").read_text())
        configs = [output / row["path"] for row in metadata["coordinates"]]
    else:
        output = scaffold.scaffold_lesson(
            "dense", tmp_path / "new", scale="smoke", data="tinystories"
        )
        metadata = json.loads((output / "lesson.json").read_text())
        configs = [output / "model.yaml"]
    source = load_source(output / "source.yaml")
    tokenizer = load_tokenizer_config(output / "tokenizer.yaml")
    assert tokenizer.dataset.source == "snapshot"
    assert tokenizer.dataset.license == source.license
    assert (
        tokenizer.dataset.source_manifest_path
        == output / "artifacts/snapshot/manifest.json"
    )
    for path in configs:
        config = load_config(path)
        assert config.dataset.source == "snapshot"
        assert config.dataset.train_path == output / "artifacts/snapshot/train.jsonl"
        assert (
            config.dataset.source_manifest_path
            == tokenizer.dataset.source_manifest_path
        )
    inputs = {row["path"]: row["sha256"] for row in metadata["inputs"]}
    assert (
        inputs["source.yaml"]
        == hashlib.sha256((output / "source.yaml").read_bytes()).hexdigest()
    )
    assert "research_sources/dataset_sources/tinystories.json" in inputs
    readme = (output / "README.md").read_text()
    assert (
        readme.index("sparselab data lock")
        < readme.index("sparselab data snapshot")
        < readme.index("sparselab tokenizer train")
    )
    assert "source.yaml" in readme
    assert not (output / "artifacts").exists()


def test_source_mapping_is_generic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profiles = catalog.load_datasets()
    profile = profiles.datasets["tinystories"]
    source = catalog.load_dataset_source(profile)
    source.update(repo_id="another/corpus", license="CC0-1.0", revision="b" * 40)
    resource = tmp_path / "other.json"
    resource.write_text(json.dumps(source))
    other = profile.model_copy(
        update={"source_declaration": "other.json", "revision": "b" * 40}
    )
    profiles = profiles.model_copy(update={"datasets": {"another": other}})
    monkeypatch.setattr(catalog, "_RESOURCE_ROOT", tmp_path)
    monkeypatch.setattr(scaffold, "_RESOURCE_ROOT", tmp_path)
    monkeypatch.setattr(scaffold, "load_datasets", lambda: profiles)
    mapping = scaffold._dataset_mapping("another")
    assert mapping["license"] == "CC0-1.0"
    assert mapping["revision"] == "b" * 40
    assert mapping["source"] == "snapshot"
    output = tmp_path / "out"
    scaffold._write_dataset_source(output, "another")
    assert (
        yaml.safe_load((output / "source.yaml").read_text())["repo_id"]
        == "another/corpus"
    )


def test_historical_dataset_profiles_remain_loadable() -> None:
    values = catalog.load_datasets().datasets["tinystories"].model_dump(mode="json")
    values.pop("source_declaration")
    values["source"] = "tinystories"
    profile = catalog.DatasetProfile.model_validate(values)
    assert profile.model_dump(mode="json") == values


@pytest.mark.parametrize("reference", ["../escape.json", "/absolute.json", "a\\b.json"])
def test_profile_rejects_unsafe_source_resource(reference: str) -> None:
    values = catalog.load_datasets().datasets["tinystories"].model_dump(mode="json")
    with pytest.raises(ValueError, match="safe relative"):
        catalog.DatasetProfile.model_validate(
            {**values, "source_declaration": reference}
        )
