"""Removal-sensitive storage wiring and refusal tests on tiny native fixtures."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_corpus_admission_draft import _setup
from test_corpus_admission_inspection import _fixture
from test_corpus_record_admission import _fixture as admission_fixture

from sparselab.corpus.admission_draft import draft_admission_manifest
from sparselab.corpus.admission_inspection import (
    inspect_admission,
    verify_admission_inspection,
)
from sparselab.corpus.bounded_metadata import (
    DiskRows,
    MetadataStore,
    metadata_operation,
)
from sparselab.corpus.project import load_project
from sparselab.corpus.rights import verify_record_admission
from sparselab.corpus.split_inventory import write_split_inventory
from sparselab.training.manifest import canonical_json


def inspect(root, recipe, policy, draft, selection, output):
    return inspect_admission(
        load_project(recipe / "acquire.yaml"),
        recipe / "acquire.yaml",
        root,
        draft,
        policy,
        selection,
        output,
        max_input_bytes=1048576,
        max_excerpt_bytes=80,
        max_output_bytes=262144,
    )


@pytest.mark.parametrize("entrypoint", ["draft", "inspect", "verify", "split"])
def test_entrypoint_activates_metadata_storage(tmp_path, monkeypatch, entrypoint):
    root = tmp_path / "work"
    root.mkdir()
    recipe, policy, draft, selection, receipt = _fixture(root)
    loaded = {}
    native_load = MetadataStore.load

    def tracked(self, path, **limits):
        result = native_load(self, path, **limits)
        loaded[Path(path)] = result
        if path.name == "manifest.json":
            assert isinstance(result["files"], DiskRows)
            for shard in result.get("retrieval", {}).get("shards", []):
                if "selected_rows" in shard:
                    assert isinstance(shard["selected_rows"], DiskRows)
        if path == draft:
            for source in result["sources"]:
                assert isinstance(source.get("records", source.get("files")), DiskRows)
        return result

    monkeypatch.setattr(MetadataStore, "load", tracked)
    project = load_project(recipe / "acquire.yaml")
    if entrypoint == "draft":
        draft_admission_manifest(
            project,
            root,
            recipe / "application.json",
            policy,
            recipe / "new-draft.json",
        )
        assert recipe / "application.json" in loaded
    elif entrypoint == "inspect":
        inspect(root, recipe, policy, draft, selection, recipe / "new-inspection.json")
        assert draft in loaded
    elif entrypoint == "verify":
        verify_admission_inspection(receipt, root)
        assert draft in loaded
    else:
        write_split_inventory(project, root, recipe / "inventory.jsonl")
    snapshots = set(
        (root / "corpora" / project.config.id / "snapshots").glob("*/*/manifest.json")
    )
    assert snapshots and snapshots <= loaded.keys()
    assert not list(root.rglob(".metadata-*"))


def test_git_file_arrays_do_not_consume_header_allowance(tmp_path, monkeypatch):
    native_load = MetadataStore.load
    exercised = []

    def capped(self, path, **limits):
        payload = json.loads(path.read_text())
        # Scale the 4 MiB boundary down: retain all headers, remove only file
        # rows when deriving the allowance. The actual input keeps every row.
        reduced = deepcopy(payload)
        removed = 0

        def strip(value):
            nonlocal removed
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "files" and isinstance(child, list) and child:
                        removed += len(canonical_json(child))
                        value[key] = []
                    else:
                        strip(child)
            elif isinstance(value, list):
                for child in value:
                    strip(child)

        strip(reduced)
        if removed:
            limits["max_header_bytes"] = path.stat().st_size - removed + 16
            exercised.append(path)
        return native_load(self, path, **limits)

    monkeypatch.setattr(MetadataStore, "load", capped)
    root = tmp_path / "work"
    root.mkdir()
    _recipe, _, draft, _, receipt = _fixture(root)
    verify_admission_inspection(receipt, root)
    assert draft in exercised
    assert any(path.name == "manifest.json" for path in exercised)


def test_inspection_enforces_metadata_input_cap_wiring(tmp_path, monkeypatch):
    from sparselab.corpus import bounded_metadata

    root = tmp_path / "work"
    root.mkdir()
    recipe, policy, draft, selection, _ = _fixture(root)
    output = recipe / "refused.json"
    monkeypatch.setattr(bounded_metadata, "_MAX_INPUT_BYTES", draft.stat().st_size - 1)
    with (
        metadata_operation(root / "explicit-capped-scratch"),
        pytest.raises(ValueError, match="metadata input-byte cap exceeded"),
    ):
        inspect(root, recipe, policy, draft, selection, output)
    assert not output.exists()


def test_symlinked_scratch_refuses_before_draft_output(tmp_path, monkeypatch):
    project, root, template, policy = _setup(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    scratch = root / "corpora" / project.config.id / "metadata-scratch"
    scratch.symlink_to(outside, target_is_directory=True)
    output = root / "draft.json"
    with pytest.raises(
        ValueError, match="metadata scratch directory must not be a symlink"
    ):
        draft_admission_manifest(project, root, template, policy, output)
    assert not output.exists() and not list(outside.iterdir())


def test_non_hf_split_file_cap_exact_boundary(tmp_path, monkeypatch):
    from sparselab.corpus import split_inventory

    root = tmp_path / "work"
    root.mkdir()
    recipe, _, _, _, _ = _fixture(root)
    project = load_project(recipe / "acquire.yaml")
    lock = json.loads(
        (root / "corpora" / project.config.id / "acquisition.json").read_text()
    )
    sizes = [
        file.stat().st_size
        for source in project.sources
        if source.kind != "huggingface_dataset"
        for file in (Path(lock["sources"][source.id]["snapshot_path"]) / "files").rglob(
            "*"
        )
        if file.is_file()
    ]
    bound = max(sizes)
    monkeypatch.setattr(split_inventory, "_MAX_FILE_BYTES", bound)
    write_split_inventory(project, root, recipe / "passing.jsonl")
    monkeypatch.setattr(split_inventory, "_MAX_FILE_BYTES", bound - 1)
    output = recipe / "refused.jsonl"
    with pytest.raises(ValueError, match="split inventory file-byte cap exceeded"):
        write_split_inventory(project, root, output)
    assert not output.exists() and not list(recipe.glob(".split-inventory-*"))


def test_extra_unselected_decision_refuses(tmp_path):
    manifest, source, snapshot, _, _ = admission_fixture(tmp_path)
    extra = deepcopy(manifest["sources"][0]["records"][0])
    extra["source_row_index"] = 999
    manifest["sources"][0]["records"].append(extra)
    with (
        metadata_operation(tmp_path / "scratch"),
        pytest.raises(
            ValueError, match="record admission decisions do not cover selected rows"
        ),
    ):
        verify_record_admission(
            manifest, {source["id"]: source}, {source["id"]: snapshot}, tmp_path
        )


@pytest.mark.parametrize("entrypoint", ["verify", "split"])
def test_local_source_declaration_stays_plain_at_native_entrypoints(
    tmp_path, monkeypatch, entrypoint
):
    from corpus_fixtures import neutral_corpus
    from test_corpus_structure_cleaning import _local_source

    from sparselab.corpus.acquisition import acquire, verify_acquisition
    from sparselab.corpus.project import Project

    recipe = tmp_path / "recipe"
    recipe.mkdir()
    (recipe / "doc.md").write_text(
        "# Fixture\nLocal prose preserves source evidence.\n"
    )
    payload = neutral_corpus(recipe).model_dump(mode="python")
    payload["sources"] = [_local_source().model_dump(mode="python")]
    payload["config"]["sources"] = ["sources/docs.yaml"]
    payload["transforms"][0]["inputs"] = ["docs"]
    payload["splits"]["assignments"] = {"fixture-family": "train"}
    payload["release"] = {
        "schema_version": 1,
        "mixture": {"technical_docs": 1.0},
        "lm": {"selected": True, "training_splits": ["train"]},
        "chat": {"selected": False, "training_splits": []},
    }
    project = Project.model_validate(payload)
    root = tmp_path / "work"
    acquire(project, root)
    expected = verify_acquisition(project, root)
    loaded = []
    native_load = MetadataStore.load

    def tracked(self, path, **limits):
        result = native_load(self, path, **limits)
        if path.name == "manifest.json":
            assert isinstance(result["files"], DiskRows)
            files = result["declaration"]["acquisition"]["files"]
            assert type(files) is list and len(files) == 1
            loaded.append(path)
        return result

    monkeypatch.setattr(MetadataStore, "load", tracked)
    if entrypoint == "verify":
        with metadata_operation(root / "scratch"):
            assert verify_acquisition(project, root) == expected
    else:
        output = root / "inventory.jsonl"
        result = write_split_inventory(project, root, output)
        assert result["documents"] == 1
        assert json.loads(output.read_text())["source_id"] == "docs"
    assert loaded
    assert not list(root.rglob(".metadata-*"))


def test_build_keeps_metadata_operation_active_through_native_snapshot_consumers(
    tmp_path, monkeypatch
):
    from test_corpus_pipeline import _project

    from sparselab.corpus.acquisition import acquire
    from sparselab.corpus.pipeline import build
    from sparselab.corpus.release import verify_build

    project = _project()
    acquire(project, tmp_path)
    loaded = []
    native_load = MetadataStore.load

    def tracked(self, path, **limits):
        result = native_load(self, path, **limits)
        if path.name == "manifest.json":
            assert isinstance(result["files"], DiskRows)
            loaded.append(path)
        return result

    monkeypatch.setattr(MetadataStore, "load", tracked)
    output = build(project, tmp_path, offline=True)
    assert loaded
    assert not list(tmp_path.rglob(".metadata-*"))
    verify_build(output)


def test_native_metadata_uses_file_bound_instead_of_fixed_input_quota(
    tmp_path, monkeypatch
):
    from sparselab.corpus import bounded_metadata

    root = tmp_path / "work"
    root.mkdir()
    recipe, policy, draft, selection, _ = _fixture(root)
    monkeypatch.setattr(bounded_metadata, "_MAX_INPUT_BYTES", 1)
    limits = []
    original = MetadataStore.load

    def tracked(self, path, **kwargs):
        limits.append(
            (self.max_database_bytes, kwargs["max_input_bytes"], path.stat().st_size)
        )
        return original(self, path, **kwargs)

    monkeypatch.setattr(MetadataStore, "load", tracked)
    inspect(root, recipe, policy, draft, selection, recipe / "streaming.json")
    assert limits and all(
        quota is None and bound == size for quota, bound, size in limits
    )
    assert not list(root.rglob(".metadata-*"))
