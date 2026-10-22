"""Split-family policy changes preserve immutable acquisition evidence."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_corpus_row_qualification import code_policy, declared_fixture
from test_corpus_split_freeze import _fixture as freeze_fixture

from sparselab.corpus import split_freeze, split_inventory
from sparselab.corpus.acquisition import _project_sha, declaration_sha256
from sparselab.corpus.project import Project, SourceDeclaration, SplitDeclaration
from sparselab.training.manifest import canonical_json, sha256_file


def _project(tmp_path, source, fields):
    return Project.model_validate(
        {
            "root": tmp_path,
            "config": {
                "schema_version": 1,
                "id": "prompt-families",
                "sources": ["sources/fixture.yaml"],
                "transforms": [],
                "splits": "splits.yaml",
                "release": "release.yaml",
            },
            "sources": [source],
            "transforms": [],
            "splits": {
                "schema_version": 1,
                "unit": "source_repository",
                "assignments": {source.id: "train"},
                "family_fields_by_source": fields,
            },
            "release": {
                "schema_version": 2,
                "publication_mode": "metadata_reconstruction_only",
                "mixture": {"synthetic": 1.0},
                "lm": {"selected": True, "training_splits": ["train"]},
                "chat": {"selected": False, "training_splits": []},
            },
        }
    )


@pytest.mark.parametrize("same_prompt", [False, True])
def test_native_inventory_uses_exact_prompt_and_preserves_acquisition(
    tmp_path, monkeypatch, same_prompt
):
    prompt = "A snow story.\n\u00e9\u2028"
    rows = [
        {
            "text": "First generated response with enough explanatory words. " * 5,
            "seed_data": "fineweb",
            "prompt": prompt,
        },
        {
            "text": "Different generated response with enough explanatory words. " * 5,
            "seed_data": "ultrachat" if same_prompt else "fineweb",
            "prompt": prompt if same_prompt else prompt + " ",
        },
    ]
    _, source_payload, snapshot, raw = declared_fixture(
        tmp_path,
        policy={
            "profile": "synthetic_text",
            "family_fields": ["seed_data"],
            "synthetic_provenance_fields": ["seed_data"],
        },
        rows=rows,
    )
    source = SourceDeclaration.model_validate(source_payload)
    original = canonical_json(source_payload)
    digest = declaration_sha256(source)
    snapshot_dir = tmp_path / source.id / snapshot["snapshot_sha256"]
    manifest = snapshot_dir / "manifest.json"
    manifest.write_bytes(canonical_json(snapshot))
    manifest_sha = sha256_file(manifest)
    project = _project(tmp_path, source, {source.id: ["prompt"]})
    without_override = _project(tmp_path, source, {})
    assert _project_sha(project) == _project_sha(without_override)
    monkeypatch.setattr(
        split_inventory,
        "verify_acquisition",
        lambda candidate, root: {
            "sources": {
                source.id: {
                    "snapshot_path": str(snapshot_dir),
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                }
            }
        },
    )
    first = tmp_path / "inventory-first.jsonl"
    second = tmp_path / "inventory-second.jsonl"
    split_inventory.write_split_inventory(project, tmp_path, first)
    split_inventory.write_split_inventory(project, tmp_path, second)
    observed = [json.loads(line) for line in first.read_text().splitlines()]
    assert first.read_bytes() == second.read_bytes()
    assert len(observed) == 2
    expected = [
        "synthetic_text-family:"
        + hashlib.sha256(
            json.dumps(
                [row["prompt"]],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for row in rows
    ]
    assert [row["family_hint"] for row in observed] == expected
    assert (expected[0] == expected[1]) is same_prompt
    assert canonical_json(source_payload) == original
    assert declaration_sha256(source) == digest
    assert sha256_file(manifest) == manifest_sha
    file = snapshot_dir / "files" / snapshot["files"][0]["path"]
    assert file.read_bytes() == raw


def test_project_refuses_unknown_source_and_repository_partition(tmp_path):
    _, payload, _, _ = declared_fixture(tmp_path)
    source = SourceDeclaration.model_validate(payload)
    with pytest.raises(ValueError, match="unknown source"):
        _project(tmp_path, source, {"another_source": ["prompt"]})
    without_qualification = {**payload, "acquisition": dict(payload["acquisition"])}
    without_qualification["acquisition"].pop("qualification")
    unqualified = SourceDeclaration.model_validate(without_qualification)
    with pytest.raises(ValueError, match="declared HF qualification"):
        _project(tmp_path, unqualified, {unqualified.id: ["prompt"]})
    payload["acquisition"]["qualification"] = code_policy().model_dump(mode="json")
    code_source = SourceDeclaration.model_validate(payload)
    with pytest.raises(ValueError, match="repository family"):
        _project(tmp_path, code_source, {code_source.id: ["path"]})


@pytest.mark.parametrize(
    "fields",
    [
        [],
        ["prompt", "prompt"],
        ["a..b"],
        ["_sparselab_source.source_row_index"],
        ["prompt"] * 9,
    ],
)
def test_split_family_field_paths_refuse(fields):
    with pytest.raises(ValueError):
        SplitDeclaration.model_validate(
            {
                "schema_version": 1,
                "unit": "source_repository",
                "assignments": {"synthetic": "train"},
                "family_fields_by_source": {"synthetic": fields},
            }
        )


def test_freeze_persists_policy_and_receipt_authenticates_it(tmp_path, monkeypatch):
    project, inventory, clusters, _, _ = freeze_fixture(tmp_path, monkeypatch)
    project.splits = SimpleNamespace(family_fields_by_source={"pages": ("prompt",)})
    output = tmp_path / "splits.yaml"
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    frozen = SplitDeclaration.model_validate(yaml.safe_load(output.read_text()))
    assert frozen.family_fields_by_source == {"pages": ("prompt",)}
    split_freeze.verify_split_bundle(output)
    altered = yaml.safe_load(output.read_text())
    altered["family_fields_by_source"]["pages"] = ["seed_data"]
    output.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError, match="digest|SHA|mismatch"):
        split_freeze.verify_split_bundle(output)


def test_freeze_and_cold_inventory_replay_use_the_saved_prompt_rule(
    tmp_path, monkeypatch
):
    project_id = "prompt-families"
    snapshots_root = tmp_path / "corpora" / project_id / "snapshots"
    rows = [
        {
            "text": f"Original generated fixture response {i} with enough prose. " * 4,
            "seed_data": "fineweb",
            "prompt": prompt,
        }
        for i, prompt in enumerate(
            ["First prompt", "Second prompt", "Third prompt", "First prompt"]
        )
    ]
    manifest, payload, snapshot, _ = declared_fixture(
        snapshots_root,
        policy={
            "profile": "synthetic_text",
            "family_fields": ["seed_data"],
            "synthetic_provenance_fields": ["seed_data"],
        },
        rows=rows,
    )
    source = SourceDeclaration.model_validate(payload)
    snapshot_dir = snapshots_root / source.id / snapshot["snapshot_sha256"]
    (snapshot_dir / "manifest.json").write_bytes(canonical_json(snapshot))
    for row in manifest["sources"][0]["records"]:
        row["decision"] = "qualify"
        row["reason"] = "Known generated test fixture, no external source rights claim"
    manifest["schema_version"] = 2
    admission = tmp_path / "admission.json"
    admission.write_bytes(canonical_json(manifest))
    project = _project(tmp_path, source, {source.id: ["prompt"]})
    project_payload = project.model_dump(mode="python")
    project_payload["release"]["record_admission"] = {
        "path": "admission.json",
        "sha256": sha256_file(admission),
    }
    project = Project.model_validate(project_payload)
    lock = {
        "sources": {
            source.id: {
                "snapshot_path": str(snapshot_dir),
                "snapshot_sha256": snapshot["snapshot_sha256"],
            }
        }
    }
    monkeypatch.setattr(split_inventory, "verify_acquisition", lambda *_: lock)
    monkeypatch.setattr(split_freeze, "verify_acquisition", lambda *_: lock)
    inventory = tmp_path / "inventory.jsonl"
    split_inventory.write_split_inventory(project, tmp_path, inventory)
    clusters = tmp_path / "clusters.json"
    clusters.write_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "inventory_sha256": sha256_file(inventory),
                "seed": "fixed-test-seed",
                "source_strata": {source.id: "synthetic"},
                "split_weights": {"train": 1, "validation": 1, "test": 1},
                "merges": [],
                "reviewer": "Fixture",
                "reviewed_on": "2026-10-22",
            }
        )
    )
    output = tmp_path / "splits.yaml"
    report = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    assert report["family_counts"]["synthetic"]["families"] == 3
    frozen = SplitDeclaration.model_validate(yaml.safe_load(output.read_text()))
    cold_project = project.model_copy(update={"splits": frozen})
    replay = tmp_path / "cold-inventory.jsonl"
    split_inventory.write_split_inventory(cold_project, tmp_path, replay)
    assert replay.read_bytes() == inventory.read_bytes()
    candidates = [
        json.loads(line)
        for line in Path(report["family_candidates"]).read_text().splitlines()
    ]
    by_id = {row["document_id"]: row for row in candidates}
    original = [json.loads(line) for line in inventory.read_text().splitlines()]
    first, fourth = [by_id[original[index]["document_id"]] for index in (0, 3)]
    assert first["family_id"] == fourth["family_id"]
    assert first["split"] == fourth["split"]
    assert {row["split"] for row in candidates} == {"train", "validation", "test"}
    second = tmp_path / "same-splits.yaml"
    split_freeze.freeze_splits(cold_project, tmp_path, inventory, clusters, second)
    assert second.read_bytes() == output.read_bytes()


def test_missing_prompt_never_falls_back_to_dataset_label(tmp_path, monkeypatch):
    rows = [
        {
            "text": "A complete first generated fixture with sufficient text. " * 4,
            "seed_data": "fineweb",
            "prompt": "The retained prompt",
        },
        {
            "text": "A complete second generated fixture with sufficient text. " * 4,
            "seed_data": "fineweb",
        },
    ]
    _, payload, snapshot, _ = declared_fixture(
        tmp_path,
        policy={
            "profile": "synthetic_text",
            "family_fields": ["seed_data"],
            "synthetic_provenance_fields": ["seed_data"],
        },
        rows=rows,
    )
    source = SourceDeclaration.model_validate(payload)
    directory = tmp_path / source.id / snapshot["snapshot_sha256"]
    (directory / "manifest.json").write_bytes(canonical_json(snapshot))
    project = _project(tmp_path, source, {source.id: ["prompt"]})
    monkeypatch.setattr(
        split_inventory,
        "verify_acquisition",
        lambda *_: {
            "sources": {
                source.id: {
                    "snapshot_path": str(directory),
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                }
            }
        },
    )
    output = tmp_path / "inventory.jsonl"
    split_inventory.write_split_inventory(project, tmp_path, output)
    observed = [json.loads(line) for line in output.read_text().splitlines()]
    assert not observed[0]["family_hint"].startswith("unresolved-family:")
    assert (
        observed[1]["family_hint"] == "unresolved-family:" + observed[1]["document_id"]
    )
