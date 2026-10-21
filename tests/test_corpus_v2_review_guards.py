"""Public entrypoint refusals and replay/dispatch boundaries from PR203 review."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_kml_50m_preparation_sequence as fixture
import yaml
from test_corpus_acquisition import _bounded_hf_fixture, _mock_hf_stream

from sparselab.corpus import large_build, pipeline
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.project import Project, load_project
from sparselab.corpus.release import freeze, verify_build, verify_release
from sparselab.training.manifest import canonical_json, sha256_file


def project_fixture(root, *, normalizer=None):
    root.mkdir()
    recipe, strata = fixture._sources(root, "review-fixture")
    template, policy = fixture._policy(recipe, strata)
    admission = recipe / "admission.json"
    fixture.draft_admission_manifest(
        load_project(recipe / "acquire.yaml"), root, template, policy, admission
    )
    fixture._reviewed_release(recipe, admission)
    release = yaml.safe_load((recipe / "release-reviewed.yaml").read_text())
    release["normalizer"] = normalizer
    fixture._write_yaml(recipe / "release-reviewed.yaml", release)
    project = load_project(recipe / "pre-freeze.yaml")
    inventory = recipe / "inventory.jsonl"
    fixture.write_split_inventory(project, root, inventory)
    clusters = recipe / "clusters.json"
    fixture._clusters(clusters, inventory, strata)
    frozen = recipe / "splits-reviewed.yaml"
    fixture.freeze_splits(project, root, inventory, clusters, frozen)
    fixture._write_yaml(
        recipe / "build.yaml",
        {
            **yaml.safe_load((recipe / "pre-freeze.yaml").read_text()),
            "transforms": ["transforms/lm.yaml"],
            "splits": frozen.name,
        },
    )
    return load_project(recipe / "build.yaml")


@pytest.mark.parametrize("schema", [1, 2])
def test_structure_only_large_row_build_and_cold_verify_agree(
    tmp_path, monkeypatch, schema
):
    raw = (
        canonical_json({"text": "q" * (8 * 1024**2 + 1024)})
        + b"\n"
        + canonical_json({"text": "Small fixture row."})
        + b"\n"
    )
    recipe = _bounded_hf_fixture(tmp_path, raw, "default/train.jsonl", rows=2)
    source_path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["acquisition"].update(max_bytes=2 * len(raw))
    source["acquisition"]["bounded_shards"][0].update(
        hash_modulus=1,
        hash_remainders=[0],
    )
    fixture._write_yaml(source_path, source)
    spec = load_project(recipe).model_dump(mode="json")
    spec["splits"] = {
        "schema_version": 1,
        "unit": "source_repository",
        "assignments": {"docs": "train"},
    }
    spec["transforms"] = [
        {
            "id": "lm",
            "version": "1",
            "kind": "lm_text",
            "inputs": ["one"],
            "parameters": {},
        }
    ]
    spec["release"].update(
        normalizer="normalizer-structure-v3",
        lm={"selected": True, "training_splits": ["train"]},
    )
    if schema == 2:
        spec["sources"][0].pop("redistribution")
        spec["sources"][0].update(
            schema_version=2,
            license_url="https://example.invalid/license",
            rights={
                "training_eligibility": "eligible",
                "redistribution_mode": "metadata_reconstruction_only",
                "spdx_expression": "MIT",
                "license_references": ["https://example.invalid/license"],
            },
        )
        spec["release"].update(
            schema_version=2, publication_mode="metadata_reconstruction_only"
        )
    project = Project.model_validate(spec)
    _mock_hf_stream(monkeypatch, raw)
    root = tmp_path / "work"
    acquire(project, root)
    built = pipeline.build(project, root, offline=True)
    assert verify_build(built)["build_id"] == built.name
    release = freeze(built, root)
    assert verify_release(release)["release_id"] == release.name


@pytest.mark.parametrize("fault", ["no_admission", "fraction", "local"])
def test_v2_dispatch_refuses_unsupported_combinations(fault):
    project = SimpleNamespace(
        release=SimpleNamespace(
            schema_version=2,
            lm=SimpleNamespace(selected=True),
            chat=SimpleNamespace(selected=False),
            record_admission=object(),
            fraction=None,
        ),
        sources=[SimpleNamespace(schema_version=2, kind="git")],
    )
    transforms = [SimpleNamespace(kind="lm_text")]
    assert pipeline._uses_disk_lm_builder(project, transforms)
    if fault == "no_admission":
        project.release.record_admission = None
    elif fault == "fraction":
        project.release.fraction = object()
    else:
        project.sources[0].kind = "local"
    assert not pipeline._uses_disk_lm_builder(project, transforms)


def test_public_build_refuses_missing_qualified_document(tmp_path, monkeypatch):
    root = tmp_path / "work"
    project = project_fixture(root)
    original = large_build._prepare_file

    def omit_qualified(**kwargs):
        path = original(**kwargs)
        if kwargs["source"].kind == "huggingface_dataset":
            receipt_path = path / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            for name in ("docs.jsonl", "spans.jsonl"):
                (path / name).write_bytes(b"")
                receipt["files"][name] = {"sha256": sha256_file(path / name), "size": 0}
            receipt["counts"]["documents"] = 0
            receipt_path.write_bytes(canonical_json(receipt) + b"\n")
        return path

    monkeypatch.setattr(large_build, "_prepare_file", omit_qualified)
    with pytest.raises(
        ValueError, match="admitted source row inventory differs from documents"
    ):
        pipeline.build(project, root, offline=True)
    assert not list(
        (root / "corpora" / project.config.id / "builds").glob("*/build.json")
    )


def reseal_file(build, name, rows):
    path = build / name
    path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    receipt_path = build / "build.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files"][name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    receipt_path.write_bytes(canonical_json(receipt) + b"\n")


@pytest.mark.parametrize(
    "fault,match",
    [
        ("missing", "admitted source row inventory differs"),
        ("duplicate", "duplicate admitted source row"),
        ("text", "admitted document differs from selected source row"),
    ],
)
def test_public_cold_build_admission_refusals(tmp_path, fault, match):
    root = tmp_path / "work"
    project = project_fixture(root)
    built = pipeline.build(project, root, offline=True)
    docs = [
        json.loads(line)
        for line in (built / "documents.jsonl").read_text().splitlines()
    ]
    spans = [
        json.loads(line) for line in (built / "spans.jsonl").read_text().splitlines()
    ]
    doc = next(row for row in docs if row["source_id"] == "books_rows")
    span = next(row for row in spans if row["record_id"] == doc["document_id"])
    if fault == "missing":
        docs.remove(doc)
        spans.remove(span)
    elif fault == "duplicate":
        docs.append({**doc, "document_id": "f" * 64})
        spans.append({**span, "record_id": "f" * 64})
    else:
        doc["text"] = "Wrong but internally rehashed fixture text."
        doc["content_sha256"] = hashlib.sha256(doc["text"].encode()).hexdigest()
    reseal_file(built, "documents.jsonl", docs)
    reseal_file(built, "spans.jsonl", spans)
    with pytest.raises(ValueError, match=match):
        verify_build(built)


def test_public_cold_replay_streams_rows_in_source_order(tmp_path, monkeypatch):
    native_json = fixture.canonical_json

    def two_qualified(value):
        if isinstance(value, dict) and "text" in value and "metadata" in value:
            value = {
                **value,
                "text": value["text"] + " Ordered fixture B.",
                "metadata": {**value["metadata"], "license": "Public Domain"},
            }
        return native_json(value)

    monkeypatch.setattr(fixture, "canonical_json", two_qualified)
    root = tmp_path / "work"
    project = project_fixture(root)
    built = pipeline.build(project, root, offline=True)
    docs = [
        json.loads(line)
        for line in (built / "documents.jsonl").read_text().splitlines()
    ]
    admitted = [doc for doc in docs if doc["source_id"] == "books_rows"]
    assert sorted(doc["metadata"]["source_row_index"] for doc in admitted) == [0, 1]
    released = freeze(built, root)
    # Document IDs include the adapter identity. Force adversarial reader order
    # explicitly rather than depending on one code digest sorting rows backwards.
    from sparselab.corpus import release as release_module

    native_rows = release_module._rows

    def reversed_documents(path):
        rows = native_rows(path)
        if path.name == "documents.jsonl":
            return sorted(
                rows,
                key=lambda doc: doc.get("metadata", {}).get("source_row_index", -1),
                reverse=True,
            )
        return rows

    monkeypatch.setattr(release_module, "_rows", reversed_documents)
    original_read = Path.read_bytes

    def no_shard_materialization(path):
        if path.name.endswith(".sample.jsonl"):
            pytest.fail("cold verifier read a whole source shard")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", no_shard_materialization)
    assert verify_release(released)["release_id"] == released.name


def test_dropped_only_family_split_keeps_native_audit_warning(tmp_path, monkeypatch):
    native_json = fixture.canonical_json

    def metadata_only(value):
        if isinstance(value, dict) and "text" in value and "metadata" in value:
            value = {
                **value,
                "metadata": {**value["metadata"], "license": "Public Domain"},
            }
            if value["id"] == "101":
                value["text"] = "---\ntitle: Metadata only\nweight: 1\n---\n"
        return native_json(value)

    monkeypatch.setattr(fixture, "canonical_json", metadata_only)
    root = tmp_path / "work"
    project = project_fixture(root, normalizer="normalizer-structure-v3")
    inventory = [
        json.loads(line)
        for line in (project.root / "inventory.jsonl").read_text().splitlines()
    ]
    spec = project.model_dump(mode="json")
    from sparselab.corpus.bounded_metadata import metadata_operation
    from sparselab.corpus.split_assignments import split_policy

    with metadata_operation(root):
        spec["splits"]["assignments"] = dict(split_policy(project)["assignments"])
    spec["splits"].pop("assignment_file", None)
    for row in inventory:
        if row["source_id"] == "books_rows":
            spec["splits"]["assignments"][row["document_id"]] = (
                "test" if row["source_row_index"] == 1 else "train"
            )
    project = Project.model_validate(spec)
    disk = pipeline.build(project, root, offline=True)
    monkeypatch.setattr(pipeline, "_uses_disk_lm_builder", lambda *_: False)
    memory = pipeline.build(project, root, offline=True)
    docs = [
        json.loads(line) for line in (disk / "documents.jsonl").read_text().splitlines()
    ]
    assert any(
        doc["source_id"] == "books_rows"
        and doc["split"] == "test"
        and doc["drop_reason"] == "metadata_only_front_matter"
        for doc in docs
    )
    for name in ("audit.json", "report.json"):
        assert (disk / name).read_bytes() == (memory / name).read_bytes()
    assert "books_rows" in (disk / "audit.json").read_text()


def test_split_freeze_uses_disk_metadata_and_decisions(tmp_path, monkeypatch):
    from sparselab.corpus import split_freeze
    from sparselab.corpus.bounded_metadata import DiskDecisions, DiskRows

    native = split_freeze.verify_record_admission
    observed = []

    def tracked(manifest, *args, **kwargs):
        assert any(
            isinstance(source.get("records"), DiskRows)
            for source in manifest["sources"]
        )
        result = native(manifest, *args, **kwargs)
        assert any(
            isinstance(row.get("decisions"), DiskDecisions) for row in result.values()
        )
        observed.append(True)
        return result

    monkeypatch.setattr(split_freeze, "verify_record_admission", tracked)
    project_fixture(tmp_path / "work")
    assert observed == [True]


@pytest.mark.parametrize("disk", [True, False])
def test_public_build_refuses_changed_admission_copy(tmp_path, monkeypatch, disk):
    import shutil

    root = tmp_path / "work"
    project = project_fixture(root)
    native_copy = shutil.copyfile

    def corrupt_copy(source, target, *args, **kwargs):
        result = native_copy(source, target, *args, **kwargs)
        if Path(target).name == "record-admission.json":
            with Path(target).open("ab") as stream:
                stream.write(b" ")
        return result

    monkeypatch.setattr(shutil, "copyfile", corrupt_copy)
    monkeypatch.setattr(pipeline, "_uses_disk_lm_builder", lambda *_: disk)
    with pytest.raises(ValueError, match="record admission changed during build"):
        pipeline.build(project, root, offline=True)
