"""The ledger's source, verification, and training form are independent dimensions."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sparselab.corpus.acquisition import acquire
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.provenance import (
    DERIVED_ORIGIN,
    DETERMINISTIC_ORIGIN,
    HUMAN_ORIGIN,
    rendered_digest,
    shape,
    validate_lineage,
    verification,
)

DOC_ID = "d" * 64
DOC = {"source_id": "settings", "modality": "text", "text": "worker_count = 4\n"}


def test_non_text_modality_cannot_be_mislabeled_as_text():
    row = record()
    row["modalities"] = ["image"]
    with pytest.raises(ValueError, match="modality"):
        validate_lineage(row, documents={DOC_ID: DOC})
    row["modalities"] = ["text"]
    with pytest.raises(ValueError, match="modality"):
        validate_lineage(row, documents={DOC_ID: {**DOC, "modality": "audio"}})


def record(*, origin=DERIVED_ORIGIN, shape_id="direct_qa", status="source_entailed"):
    evidence = {
        "document_id": DOC_ID,
        "passage": "worker_count = 4",
        "span": [0, 16],
        "answer": "4",
    }
    if status == "rejected":
        evidence = {"reason": "invalid recorded JSON"}
    return {
        "record_id": "r" * 64,
        "record_kind": "chat_sft",
        "parent_document_ids": [DOC_ID],
        "origin_schema_version": 1,
        "origin": origin,
        "modalities": ["text"],
        "shape": shape(
            shape_id,
            domains=["configs"],
            interaction="single_turn",
            grounding="source",
            answer_style="concise",
            supervision="assistant_only",
            reasoning_depth="direct",
            task_family=shape_id,
        ),
        "verification": verification(status, "verbatim_source_span", evidence),
        "rendered_sha256": rendered_digest(
            {
                "format_version": 2,
                "loss_mode": "assistant_only",
                "messages": [{"role": "assistant", "content": "4"}],
            }
        ),
        "generator_identity": "a" * 64,
    }


def test_same_source_can_supply_distinct_shapes_without_changing_origin():
    first = record(shape_id="direct_qa")
    second = record(shape_id="decision_record")
    assert validate_lineage(first, documents={DOC_ID: DOC})["origin"] == DERIVED_ORIGIN
    assert validate_lineage(second, documents={DOC_ID: DOC})["origin"] == DERIVED_ORIGIN
    assert first["shape"]["id"] != second["shape"]["id"]
    assert first["verification"] == second["verification"]


def test_verifier_and_citation_cannot_be_self_attested():
    baseline = record()
    for corruption in (
        {"verifier_id": "model_self_review"},
        {"verifier_version": "unknown"},
        {"schema_version": 2},
        {"evidence": {"document_id": DOC_ID, "passage": "fabricated", "span": [0, 10]}},
        {
            "evidence": {
                "document_id": "other",
                "passage": "worker_count = 4",
                "span": [0, 16],
            }
        },
        {"status": "oracle_verified"},
        {"status": "human_reviewed"},
        {"status": "cross_source_verified"},
    ):
        forged = deepcopy(baseline)
        forged["verification"].update(corruption)
        with pytest.raises(ValueError):
            validate_lineage(forged, documents={DOC_ID: DOC})
    without_origin_version = deepcopy(baseline)
    without_origin_version.pop("origin_schema_version")
    with pytest.raises(ValueError, match="origin v1"):
        validate_lineage(without_origin_version, documents={DOC_ID: DOC})


def test_answer_cannot_exceed_verbatim_cited_passage():
    unsupported = record()
    unsupported["verification"]["evidence"]["answer"] = "8"
    with pytest.raises(ValueError, match="not copied"):
        validate_lineage(unsupported, documents={DOC_ID: DOC})


def test_rejection_needs_reason_and_synthetic_generation_needs_generator():
    rejected = record(origin=DETERMINISTIC_ORIGIN, status="rejected")
    assert validate_lineage(rejected, documents={DOC_ID: DOC}) is rejected
    without_reason = deepcopy(rejected)
    without_reason["verification"]["evidence"] = {}
    with pytest.raises(ValueError, match="evidence"):
        validate_lineage(without_reason)
    without_generator = deepcopy(rejected)
    without_generator.pop("generator_identity")
    with pytest.raises(ValueError, match="generator identity"):
        validate_lineage(without_generator)


def test_raw_document_origin_and_rendered_text_are_independent():
    source = record(origin=HUMAN_ORIGIN, shape_id="raw_document", status="rejected")
    source.update(
        record_kind="document",
        record_id=DOC_ID,
        rendered_sha256=rendered_digest(DOC["text"], text=True),
    )
    source["verification"] = verification(
        "schema_validated", "normalizer_v1", {"schema_id": "normalized_document_v1"}
    )
    assert validate_lineage(source, documents={DOC_ID: DOC})["origin"] == HUMAN_ORIGIN
    source["rendered_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rendered hash"):
        validate_lineage(source, documents={DOC_ID: DOC})


def test_cross_source_needs_independent_cited_documents():
    other_id = "e" * 64
    other = {
        "source_id": "other_settings",
        "modality": "text",
        "text": "worker_count = 4\n",
    }
    cited = {
        "document_id": DOC_ID,
        "source_id": "settings",
        "passage": "worker_count = 4",
        "span": [0, 16],
    }
    second = {**cited, "document_id": other_id, "source_id": "other_settings"}
    row = record()
    row["parent_document_ids"] = [DOC_ID, other_id]
    row["verification"] = verification(
        "cross_source_verified",
        "independent_source_spans",
        {"citations": [cited, second]},
    )
    assert validate_lineage(row, documents={DOC_ID: DOC, other_id: other}) is row
    same_source = deepcopy(row)
    same_source["verification"]["evidence"]["citations"][1]["document_id"] = DOC_ID
    with pytest.raises(ValueError, match="distinct sources"):
        validate_lineage(same_source, documents={DOC_ID: DOC, other_id: other})


def test_oracle_receipt_must_match_lineage_identity():
    row = record(origin=DETERMINISTIC_ORIGIN, shape_id="tool_trace")
    row["oracle_identity"] = "a" * 64
    world = {"path": "/repo/file.txt", "operation": "suffix"}
    row["verification"] = verification(
        "oracle_verified",
        "pathlib_pureposix_oracle_v1",
        {
            "oracle_identity": rendered_digest([world, ".txt"]),
            "generator_world_id": "world",
            "oracle_answer": ".txt",
            "world_state": world,
            "oracle_implementation": "c" * 64,
            "oracle_version": "1",
            "interpreter": "Python",
            "actual_result": ".txt",
            "comparison_status": "match",
        },
    )
    with pytest.raises(ValueError, match="oracle verification identity"):
        validate_lineage(row)


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_source_fact_has_three_distinct_rendered_shapes_and_filtered_views(
    tmp_path: Path,
):
    project = load_project(Path("corpora/devmind-sample-v0/corpus.yaml"))
    acquire(project, tmp_path)
    built = build(project, tmp_path, offline=True)
    lineage = {row["record_id"]: row for row in _rows(built / "lineage.jsonl")}
    semantic_chats = [
        row for row in _rows(built / "chat/records.jsonl") if "semantic_id" in row
    ]
    groups = {}
    for row in semantic_chats:
        groups.setdefault(row["semantic_id"], []).append(row)
    assert any(
        {row["semantic_shape"] for row in group}
        == {"direct_qa", "troubleshooting_scenario", "decision_record"}
        for group in groups.values()
    )
    group = next(
        group
        for group in groups.values()
        if len({row["semantic_shape"] for row in group}) == 3
    )
    records = [lineage[row["record_id"]] for row in group]
    assert len({tuple(item["parent_document_ids"]) for item in records}) == 1
    assert len({item["rendered_sha256"] for item in records}) == 3
    assert {item["origin"] for item in records} == {DERIVED_ORIGIN}
    assert {item["verification"]["status"] for item in records} == {"source_entailed"}
    raw = lineage[records[0]["parent_document_ids"][0]]
    assert raw["shape"]["id"] == "raw_document"
    assert raw["origin"] == HUMAN_ORIGIN
    assert raw["modalities"] == ["text"]
    chosen = project.model_copy(
        update={
            "release": project.release.model_copy(
                update={
                    "include_shapes": ("raw_document", "direct_qa"),
                    "include_origins": (HUMAN_ORIGIN, DERIVED_ORIGIN),
                }
            ),
        }
    )
    filtered = build(chosen, tmp_path, offline=True)
    assert len(_rows(filtered / "lineage.jsonl")) == len(lineage)
    selected = [
        link["record_id"]
        for split in ("train", "validation", "test")
        for link in _rows(filtered / "chat" / f"{split}.lineage.jsonl")
    ]
    assert selected
    assert {lineage[identifier]["shape"]["id"] for identifier in selected} == {
        "direct_qa"
    }
    assert all(
        len(_rows(filtered / "chat" / f"{split}.jsonl"))
        == len(_rows(filtered / "chat" / f"{split}.lineage.jsonl"))
        for split in ("train", "validation", "test")
    )
