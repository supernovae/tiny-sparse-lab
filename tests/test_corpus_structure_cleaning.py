"""Selected offline checks for opt-in corpus structure cleaning and reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.config.loading import load_config
from sparselab.config.models import RunConfig
from sparselab.corpus.large_build import _deduplicate, _index_prepared, _prepare_file
from sparselab.corpus.pipeline import (
    _records_for_file,
    _require_consistent_cleaning_decisions,
)
from sparselab.corpus.progress import BuildProgress
from sparselab.corpus.project import (
    ReleaseDeclaration,
    SourceDeclaration,
    SplitDeclaration,
    release_declaration_payload,
)
from sparselab.corpus.release import _verify_structure_replay
from sparselab.corpus.split_inventory import write_split_inventory
from sparselab.corpus.structure_cleaning import (
    NORMALIZER_V2,
    NORMALIZER_V3,
    clean_structure,
)


def _local_source() -> SourceDeclaration:
    return SourceDeclaration.model_validate(
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
            "source_family": "fixture-family",
            "acquisition": {
                "files": [{"path": "doc.md", "name": "doc.md"}],
                "max_bytes": 4096,
            },
        }
    )


def test_front_matter_vs_real_configuration_and_metadata_only() -> None:
    mixed = clean_structure(
        "---\ntitle: Playbook\nweight: 4\ncategories:\n- k8s\n---\n# Diagnose\n- inspect logs\n",
        source_id="scoutflo",
    )
    assert mixed["text"] == "# Diagnose\n- inspect logs\n"
    assert (
        mixed["structure"]["removed_spans"][0]["reason"]
        == "recognized_yaml_front_matter"
    )
    assert mixed["structure"]["decision"] == "retain"

    metadata_only = clean_structure(
        "---\ntitle: Empty\nweight: 1\n---\n", source_id="scoutflo"
    )
    assert metadata_only["drop_reason"] == "metadata_only_front_matter"
    assert metadata_only["structure"]["decision"] == "exclude_lm_metadata_only"
    assert metadata_only["text"].startswith("---\n")  # source row stays reviewable

    configuration = "---\napiVersion: v1\nkind: Pod\n---\ncontainers:\n- name: app\n"
    kept = clean_structure(configuration, source_id="scoutflo")
    assert kept["text"] == configuration
    assert kept["structure"]["flags"] == ["configuration_or_ambiguous_yaml"]


def test_wrappers_preserve_body_and_ambiguous_openings() -> None:
    source = "Produced by A. Editor\n\n[Transcriber's note]\n\nHamlet speaks.\n"
    cleaned = clean_structure(source, source_id="project_gutenberg")
    assert cleaned["text"] == "[Transcriber's note]\n\nHamlet speaks.\n"
    assert (
        cleaned["structure"]["removed_spans"][0]["reason"]
        == "gutenberg_production_credit"
    )
    assert clean_structure(source, source_id="ordinary")["text"] == source
    marked = clean_structure(
        "Preface\n*** START OF THE PROJECT GUTENBERG EBOOK TITLE ***\n"
        "Body prose.\n*** END OF THE PROJECT GUTENBERG EBOOK TITLE ***\nFooter\n",
        source_id="project_gutenberg",
    )
    assert marked["text"] == "Body prose.\n"
    assert [row["reason"] for row in marked["structure"]["removed_spans"]] == [
        "gutenberg_header_through_start_marker",
        "gutenberg_footer_from_end_marker",
    ]
    ambiguous = "Project Gutenberg title\nThis preface matters.\n"
    kept = clean_structure(ambiguous, source_id="project_gutenberg")
    assert kept["text"] == ambiguous
    assert "ambiguous_gutenberg_header" in kept["structure"]["flags"]


def test_new_source_rules_are_separate_from_frozen_v2() -> None:
    credit = "Produced by A. Editor\n\nThe body remains.\n"
    assert (
        clean_structure(credit, source_id="kml_scale_project_gutenberg")["text"]
        == credit
    )
    assert (
        clean_structure(
            credit, source_id="kml_scale_project_gutenberg", normalizer=NORMALIZER_V3
        )["text"]
        == "The body remains.\n"
    )
    pagerduty = (
        "---\ncover: image.png\ndescription: A real incident guide\n---\n# Response\n"
    )
    assert (
        clean_structure(pagerduty, source_id="kml_scale_pagerduty")["text"] == pagerduty
    )
    assert (
        clean_structure(
            pagerduty, source_id="kml_scale_pagerduty", normalizer=NORMALIZER_V3
        )["text"]
        == "# Response\n"
    )
    config = "---\napiVersion: v1\nkind: Pod\n---\n# Keep this\n"
    assert (
        clean_structure(
            config, source_id="kml_scale_pagerduty", normalizer=NORMALIZER_V3
        )["text"]
        == config
    )


def test_fences_headings_lists_links_code_and_equations_survive() -> None:
    text = (
        "# Real heading\n[read this](https://example.org/doc)\n"
        "```python\n# code comment, not heading\n  value = 2\n```\n"
        "- ordered instructions\n    indented code\n$$ E=mc^2 $$\n"
    )
    result = clean_structure(text, source_id="docs")
    assert result["text"] == text
    assert [item["kind"] for item in result["structure"]["blocks"]] == [
        "heading",
        "fence_open",
        "code",
        "code",
        "fence_close",
        "list",
        "indented",
        "equation",
    ]


def test_document_identity_and_source_span_replay_are_versioned() -> None:
    source = _local_source()
    raw = b"---\ntitle: Playbook\n---\n# Meaningful heading\n```md\n# code\n```\n"
    old = _records_for_file(raw, "doc.md", source, "a" * 64)
    new = _records_for_file(
        raw, "doc.md", source, "a" * 64, normalizer_version=NORMALIZER_V2
    )
    assert len(old) == len(new) == 2
    for (prior, _), (document, span) in zip(old, new, strict=True):
        assert prior["document_id"] != document["document_id"]
        assert document["source_family"] == prior["source_family"]
        assert document["metadata"]["normalizer"] == span["normalizer"] == NORMALIZER_V2
        source_slice = raw[span["byte_start"] : span["byte_end"]]
        _verify_structure_replay(document, source_slice.decode())
        with pytest.raises(ValueError, match="replay mismatch"):
            _verify_structure_replay(
                {**document, "text": document["text"] + "altered"},
                source_slice.decode(),
            )
    assert new[0][0]["drop_reason"] == "metadata_only_front_matter"
    assert new[1][0]["text"].startswith("# Meaningful heading")
    assert (
        _records_for_file(
            raw, "doc.md", source, "a" * 64, normalizer_version=NORMALIZER_V2
        )
        == new
    )


def test_v3_source_rule_changes_identity_and_replays_exact_content() -> None:
    source = _local_source().model_copy(update={"id": "kml_scale_project_gutenberg"})
    raw = b"Produced by A. Editor\n\nBody prose with useful details.\n"
    v2 = _records_for_file(
        raw, "doc.md", source, "c" * 64, normalizer_version=NORMALIZER_V2
    )[0]
    v3 = _records_for_file(
        raw, "doc.md", source, "c" * 64, normalizer_version=NORMALIZER_V3
    )[0]
    assert v2[0]["text"] != v3[0]["text"]
    assert v2[0]["document_id"] != v3[0]["document_id"]
    for document, span in (v2, v3):
        _verify_structure_replay(
            document,
            raw[span["byte_start"] : span["byte_end"]].decode(),
            expected_normalizer=span["normalizer"],
        )
    with pytest.raises(ValueError, match="normalizer mismatch"):
        _verify_structure_replay(v3[0], raw.decode(), expected_normalizer=NORMALIZER_V2)


def test_large_dedup_preserves_metadata_only_document_but_excludes_lm(
    tmp_path: Path,
) -> None:
    source = _local_source()
    snapshot = "d" * 64
    pairs = _records_for_file(
        b"---\ntitle: Metadata only\n---\n# Useful heading\nBody prose.\n",
        "doc.md",
        source,
        snapshot,
        normalizer_version=NORMALIZER_V3,
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name, index in (("docs.jsonl", 0), ("spans.jsonl", 1)):
        with (prepared / name).open("wb") as output:
            for pair in pairs:
                output.write(json.dumps(pair[index], sort_keys=True).encode() + b"\n")
    splits = SplitDeclaration.model_validate(
        {
            "schema_version": 1,
            "unit": "document",
            "family_key": None,
            "assignments": {doc["document_id"]: "train" for doc, _ in pairs},
        }
    )
    project = SimpleNamespace(sources=[source], splits=splits)
    progress = BuildProgress(tmp_path / "progress.jsonl", "structure-dedup")
    connection = _index_prepared(
        tmp_path / "index.sqlite", [prepared], project, progress
    )
    try:
        _deduplicate(
            connection,
            tmp_path / "groups.jsonl",
            diagnostics=tmp_path / "diagnostics.json",
            progress=progress,
        )
        observed = dict(
            connection.execute("SELECT document_id,drop_reason FROM documents")
        )
        assert observed[pairs[0][0]["document_id"]] == "metadata_only_front_matter"
        assert observed[pairs[1][0]["document_id"]] is None
    finally:
        connection.close()


def test_large_shard_prepare_uses_same_opt_in_parser_without_acquisition(
    tmp_path: Path,
) -> None:
    from sparselab.corpus.project import load_project
    from sparselab.corpus.rights import resolve_file_rights

    source = next(
        item
        for item in load_project("corpora/devmind-v2/corpus.yaml").sources
        if item.id == "fineweb_edu_sample_10bt"
    )
    path = tmp_path / "sample.jsonl"
    content = (
        "---\ntitle: Article\n---\n# Body\n" + ("Relevant paragraph. " * 12) + "\n"
    )
    raw = (
        json.dumps({"text": content, "url": "https://example.org/article"}).encode()
        + b"\n"
    )
    path.write_bytes(raw)
    file = {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }
    decision = resolve_file_rights(
        source.rights, path.name, b"", prospective_private_research=True
    )
    progress = BuildProgress(tmp_path / "progress.jsonl", "structure-fixture")
    prepared = _prepare_file(
        build_id="structure-fixture",
        prepared_root=tmp_path / "prepared",
        source=source,
        file=file,
        snapshot_sha="b" * 64,
        source_path=path,
        decision=decision,
        progress=progress,
        normalizer_version=NORMALIZER_V2,
    )
    streamed = [
        json.loads(line) for line in (prepared / "docs.jsonl").read_text().splitlines()
    ]
    direct = _records_for_file(
        raw,
        path.name,
        source,
        "b" * 64,
        full_file_sha256=file["sha256"],
        file_rights=decision,
        normalizer_version=NORMALIZER_V2,
    )
    assert streamed == [row[0] for row in direct]
    assert streamed[0]["text"] == "# Body\n" + ("Relevant paragraph. " * 12) + "\n"


def test_optional_absence_keeps_legacy_release_payload() -> None:
    from sparselab.corpus.project import load_project

    project = load_project("corpora/devmind-sample-v0/corpus.yaml")
    legacy = release_declaration_payload(project.release)
    assert "normalizer" not in legacy
    opt_in = ReleaseDeclaration.model_validate({**legacy, "normalizer": NORMALIZER_V2})
    assert release_declaration_payload(opt_in)["normalizer"] == NORMALIZER_V2
    assert (
        release_declaration_payload(ReleaseDeclaration.model_validate(legacy)) == legacy
    )


def test_two_configs_share_exact_dataset_export_and_cache_without_copy(
    tmp_path: Path,
) -> None:
    base = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    shared = tmp_path / "persistent" / "corpora" / "fixture" / "releases" / ("a" * 64)
    exported = (
        tmp_path
        / "persistent"
        / "corpora"
        / "fixture"
        / "exports"
        / ("a" * 64)
        / "lm"
        / ("b" * 64)
    )
    dataset = base.dataset.model_copy(
        update={
            "source": "local_text",
            "revision": "a" * 64,
            "license": "MIT",
            "train_path": shared / "lm/train.jsonl",
            "validation_path": shared / "lm/validation.jsonl",
            "corpus_release_path": shared,
            "corpus_export_path": exported,
            "cache_dir": exported / "prepared",
        }
    )
    for index in (1, 2):
        run = base.model_copy(
            update={
                "name": f"fixture-experiment-{index}",
                "seed": index,
                "dataset": dataset,
            }
        )
        path = tmp_path / f"experiment-{index}.yaml"
        path.write_text(yaml.safe_dump(run.model_dump(mode="json")), encoding="utf-8")
        loaded = load_config(path)
        assert (
            RunConfig.model_validate(loaded.model_dump(mode="json")).dataset == dataset
        )
        assert loaded.dataset.cache_dir == exported / "prepared"
    assert not shared.exists() and not exported.exists()  # declarations never copy data


def test_split_inventory_uses_declared_normalizer_and_preserves_legacy_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _local_source()
    raw = b"---\ntitle: Playbook\n---\nUseful body prose.\n"
    snapshot = tmp_path / "snapshot"
    (snapshot / "files").mkdir(parents=True)
    (snapshot / "files" / "doc.md").write_bytes(raw)
    (snapshot / "manifest.json").write_text(json.dumps({"files": [{"path": "doc.md"}]}))
    monkeypatch.setattr(
        "sparselab.corpus.split_inventory.verify_acquisition",
        lambda *_: {
            "sources": {
                source.id: {
                    "snapshot_path": str(snapshot),
                    "snapshot_sha256": "a" * 64,
                }
            }
        },
    )
    for normalizer in (None, NORMALIZER_V3):
        project = SimpleNamespace(
            config=SimpleNamespace(id="normalizer-inventory-fixture"),
            sources=[source],
            release=SimpleNamespace(normalizer=normalizer),
        )
        destination = tmp_path / f"inventory-{normalizer or 'legacy'}.jsonl"
        write_split_inventory(project, tmp_path, destination)
        inventory = [json.loads(line) for line in destination.read_text().splitlines()]
        expected = _records_for_file(
            raw,
            "doc.md",
            source,
            "a" * 64,
            normalizer_version=normalizer or "normalizer-nfc-markdown-v1",
        )
        assert [row["document_id"] for row in inventory] == [
            document["document_id"] for document, _ in expected
        ]
        assert [row["content_sha256"] for row in inventory] == [
            document["content_sha256"] for document, _ in expected
        ]
    assert inventory[0]["content_sha256"] != hashlib.sha256(raw).hexdigest()


def test_large_dedup_rejects_discordant_metadata_and_config_decisions(
    tmp_path: Path,
) -> None:
    raw = b"---\ncover: image.png\n---\n"
    sources = [
        _local_source().model_copy(update={"id": source_id})
        for source_id in ("kml_scale_pagerduty", "other_config")
    ]
    prepared_roots = []
    pairs = []
    for index, source in enumerate(sources):
        pair = _records_for_file(
            raw,
            "doc.md",
            source,
            str(index + 1) * 64,
            normalizer_version=NORMALIZER_V3,
        )[0]
        pairs.append(pair)
        prepared = tmp_path / source.id
        prepared.mkdir()
        for name, position in (("docs.jsonl", 0), ("spans.jsonl", 1)):
            (prepared / name).write_text(json.dumps(pair[position]) + "\n")
        prepared_roots.append(prepared)
    assert pairs[0][0]["drop_reason"] == "metadata_only_front_matter"
    assert pairs[1][0].get("drop_reason") is None
    with pytest.raises(ValueError, match="discordant metadata-only"):
        _require_consistent_cleaning_decisions([document for document, _ in pairs])
    splits = SplitDeclaration.model_validate(
        {
            "schema_version": 1,
            "unit": "document",
            "family_key": None,
            "assignments": {doc["document_id"]: "train" for doc, _ in pairs},
        }
    )
    project = SimpleNamespace(sources=sources, splits=splits)
    progress = BuildProgress(tmp_path / "progress.jsonl", "discordant-cleaning")
    connection = _index_prepared(
        tmp_path / "index.sqlite", prepared_roots, project, progress
    )
    try:
        with pytest.raises(ValueError, match="discordant metadata-only"):
            _deduplicate(
                connection,
                tmp_path / "groups.jsonl",
                diagnostics=tmp_path / "diagnostics.json",
                progress=progress,
            )
        assert "discordant metadata-only" in (tmp_path / "diagnostics.json").read_text()
    finally:
        connection.close()
