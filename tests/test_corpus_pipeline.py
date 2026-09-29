"""Corpus normalization, ancestry, deduplication and immutable release checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from sparselab.corpus.acquisition import acquire
from sparselab.corpus.pipeline import _records_for_file, _split, build
from sparselab.corpus.project import SourceDeclaration, load_project
from sparselab.corpus.release import freeze, lineage, review, sample, verify_release

PROJECT = Path(__file__).resolve().parents[1] / "corpora/devmind-sample-v0/corpus.yaml"


def _project():
    return load_project(PROJECT)


def test_markdown_ancestry_fences_and_raw_evidence() -> None:
    raw = b"Introduction.\n# Alpha\n- preserve list\n```md\n# not a heading\n```\n## Beta\nkey = value\n"
    source = SourceDeclaration.model_validate(
        {
            "schema_version": 1,
            "id": "docs",
            "kind": "local",
            "canonical_uri": "fixture:docs",
            "revision": "v1",
            "license": "MIT",
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["markdown"],
            "source_family": "docs",
            "acquisition": {
                "files": [{"path": "any.md", "name": "any.md"}],
                "max_bytes": 1000,
            },
        }
    )
    docs = _records_for_file(raw, "any.md", source, "a" * 64)
    assert [row[0]["section_path"] for row in docs] == [
        [],
        ["Alpha"],
        ["Alpha", "Beta"],
    ]
    assert (
        docs[1][0]["text"] == "# Alpha\n- preserve list\n```md\n# not a heading\n```\n"
    )
    assert docs[1][1]["line_start"] == 2
    assert docs[1][1]["line_end"] == 6
    assert (
        raw[docs[1][1]["byte_start"] : docs[1][1]["byte_end"]]
        == docs[1][0]["text"].encode()
    )
    assert docs[2][0]["content_sha256"] != docs[1][0]["content_sha256"]


def test_normalized_duplicates_preserve_distinct_raw_origins() -> None:
    source = SourceDeclaration.model_validate(
        {
            "schema_version": 1,
            "id": "docs",
            "kind": "local",
            "canonical_uri": "fixture:docs",
            "revision": "v1",
            "license": "MIT",
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["plain_text"],
            "source_family": "docs",
            "acquisition": {
                "files": [{"path": "any.txt", "name": "any.txt"}],
                "max_bytes": 1000,
            },
        }
    )
    composed, _ = _records_for_file("café\r\n".encode(), "any.txt", source, "a" * 64)[0]
    decomposed, _ = _records_for_file(
        "cafe\u0301\n".encode(), "any.txt", source, "b" * 64
    )[0]
    assert composed["text"] == decomposed["text"] == "café\n"
    assert composed["content_sha256"] == decomposed["content_sha256"]
    assert composed["raw_content_sha256"] != decomposed["raw_content_sha256"]
    assert composed["document_id"] != decomposed["document_id"]


def test_transform_parameter_changes_produce_distinct_builds(tmp_path: Path) -> None:
    original = _project()
    acquire(original, tmp_path)
    old = build(original, tmp_path, offline=True)
    changed = tuple(
        stage.model_copy(update={"parameters": {"recognizer_revision": "other"}})
        if stage.kind == "lexical_candidates"
        else stage
        for stage in original.transforms
    )
    revised = original.model_copy(update={"transforms": changed})
    acquire(revised, tmp_path)
    new = build(revised, tmp_path, offline=True)
    assert new != old
    assert (old / "build.json").exists()
    assert build(revised, tmp_path, offline=True) == new


def test_split_requires_complete_consistent_family_assignments() -> None:
    policy = {
        "unit": "source_document_family",
        "assignments": {"docs": "train", "code": "test", "world": "validation"},
    }
    assert _split({"source_family_ids": ["docs"]}, policy) == "train"
    assert _split({"scenario_family_id": "world"}, policy) == "validation"
    with pytest.raises(ValueError, match="conflicting"):
        _split({"source_family_ids": ["docs", "code"]}, policy)
    with pytest.raises(ValueError, match="missing"):
        _split({"source_family_ids": ["unassigned"]}, policy)


def test_build_freeze_replay_spans_semantics_and_tamper(tmp_path: Path) -> None:
    project = _project()
    acquire(project, tmp_path)
    first = build(project, tmp_path, offline=True)
    assert build(project, tmp_path, offline=True) == first
    published = freeze(first, tmp_path)
    assert freeze(first, tmp_path) == published
    manifest = verify_release(published)
    docs = [
        json.loads(line)
        for line in (published / "documents.jsonl").read_text().splitlines()
    ]
    assert any(row["section_path"] for row in docs)
    item = next(row for row in docs if "technical_docs" in row["domains"])
    trace = lineage(published, item["document_id"])
    assert trace["parents"][0]["source"]["license"] == "MIT"
    assert trace["parents"][0]["span"]["raw_sha256"]
    chat = next(
        row
        for row in review(published, kind="chat", limit=100)["records"]
        if row["verification"]["status"] == "oracle_verified"
    )
    assert lineage(published, chat["record_id"])["scenario"]["oracle_answer"] == ".py"
    assert sample(published, domain="systems_scenarios", limit=1)["records"][0][
        "domains"
    ] == ["systems_scenarios"]
    semantic = [
        json.loads(line)
        for line in (published / "semantic/candidates.jsonl").read_text().splitlines()
    ]
    assert semantic
    assert all(
        next(d for d in docs if d["document_id"] == row["evidence_document_id"])[
            "text"
        ][slice(*row["evidence_span"])]
        == row["evidence_passage"]
        for row in semantic
    )
    lexical = [
        json.loads(line)
        for line in (published / "lexical/candidates.jsonl").read_text().splitlines()
    ]
    assert any(
        row["occurrences"] >= 1 and row["document_ids"] and row["domain_counts"]
        for row in lexical
    )
    candidate = lexical[0]
    assert {
        parent["document"]["document_id"]
        for parent in lineage(published, candidate["record_id"])["parents"]
    } == set(candidate["document_ids"])
    assert (
        lineage(published, semantic[0]["record_id"])["parents"][0]["document"][
            "document_id"
        ]
        == semantic[0]["evidence_document_id"]
    )
    artifact = published / "semantic/candidates.jsonl"
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="tampered"):
        verify_release(published)
    assert manifest["release_id"] == published.name


def test_cross_split_identical_text_is_rejected(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe"
    shutil.copytree(PROJECT.parent, recipe)
    duplicate = yaml.safe_load((recipe / "sources/docs.yaml").read_text())
    duplicate["id"] = "copied_docs"
    duplicate["canonical_uri"] = "fixture:copied_docs"
    duplicate["source_family"] = "sample_code_validation"
    (recipe / "sources/copied.yaml").write_text(yaml.safe_dump(duplicate))
    corpus_path = recipe / "corpus.yaml"
    corpus = yaml.safe_load(corpus_path.read_text())
    corpus["sources"] = sorted([*corpus["sources"], "sources/copied.yaml"])
    corpus_path.write_text(yaml.safe_dump(corpus))
    project = load_project(corpus_path)
    acquire(project, tmp_path / "work")
    with pytest.raises(ValueError, match="cross-split exact text overlap"):
        build(project, tmp_path / "work", offline=True)
    diagnostics = list(
        (tmp_path / "work/corpora/devmind-sample-v0/builds/diagnostics").glob("*.json")
    )
    assert len(diagnostics) == 1
    audit = json.loads(diagnostics[0].read_text())
    assert any(
        {"train", "validation"} == set(group["splits"]) and len(group["origins"]) > 1
        for group in audit["duplicates"]
    )


def test_empty_selected_training_view_cannot_replace_previous_release(
    tmp_path: Path,
) -> None:
    project = _project()
    acquire(project, tmp_path)
    published = freeze(build(project, tmp_path, offline=True), tmp_path)
    original_digest = (published / "manifest.json").read_bytes()
    revised = project.model_copy(
        update={
            "transforms": tuple(
                stage
                for stage in project.transforms
                if stage.kind not in {"chat_sft", "tool_episode"}
            )
        }
    )
    acquire(revised, tmp_path)
    candidate = build(revised, tmp_path, offline=True)
    with pytest.raises(ValueError, match="selected chat/train training view is empty"):
        freeze(candidate, tmp_path)
    assert verify_release(published)["release_id"] == published.name
    assert (published / "manifest.json").read_bytes() == original_digest
