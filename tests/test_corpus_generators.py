"""Oracle and recorded-response derivations are checked against source evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from sparselab.corpus.acquisition import acquire
from sparselab.corpus.pipeline import _path_scenario, build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze
from sparselab.data.conversations import iter_rendered_conversations

PROJECT = Path(__file__).resolve().parents[1] / "corpora/devmind-sample-v0"


def _replay_project(tmp_path: Path, *, bad_prompt: bool = False):
    root = tmp_path / "recipe"
    shutil.copytree(PROJECT, root)
    baseline = load_project(root / "corpus.yaml")
    acquire(baseline, tmp_path / "baseline")
    built = build(baseline, tmp_path / "baseline", offline=True)
    docs = [
        json.loads(line)
        for line in (built / "documents.jsonl").read_text().splitlines()
    ]
    parent = next(doc for doc in docs if doc["source_id"] == "sample_docs")
    template = "Answer using only the cited passage:\n{passages}"
    prompt = template.format(
        passages=f"[{parent['document_id']}] {parent['text'][:2000]}"
    )
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    provider = {
        "provider": "offline_fixture",
        "model": "recorded",
        "model_revision": "pinned-v1",
    }
    responses = []
    for number, answer in enumerate(
        (parent["text"].splitlines()[0], "invented unsupported answer", None)
    ):
        raw = (
            "not-json"
            if answer is None
            else json.dumps({"answer": answer, "citation_id": parent["document_id"]})
        )
        responses.append(
            {
                "request_id": f"request-{number}",
                "source_document_ids": [parent["document_id"]],
                "prompt_sha256": "0" * 64 if bad_prompt else prompt_hash,
                "raw_output": raw,
                **provider,
                "seed": number,
            }
        )
    (root / "sources/fixtures/prompt.txt").write_text(template)
    (root / "sources/fixtures/responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in responses)
    )
    (root / "sources/generator-inputs.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "recordings",
                "kind": "local",
                "canonical_uri": "fixture:responses",
                "revision": "v1",
                "license": "MIT",
                "redistribution": "redistributable",
                "domains": ["technical_docs"],
                "document_kinds": ["data"],
                "source_family": "sample_docs_train",
                "acquisition": {
                    "files": [
                        {"path": "sources/fixtures/prompt.txt", "name": "prompt.txt"},
                        {
                            "path": "sources/fixtures/responses.jsonl",
                            "name": "responses.jsonl",
                        },
                    ],
                    "max_bytes": 100000,
                },
            }
        )
    )
    (root / "transforms/inference.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "sample_inference",
                "version": "1",
                "kind": "inference_qa",
                "inputs": ["sample_docs", "recordings"],
                "parameters": {
                    "generator_source_id": "recordings",
                    "backend": "recorded_responses",
                    **provider,
                    "prompt_template_path": "prompt.txt",
                    "responses_path": "responses.jsonl",
                    "max_input_chars": 2000,
                    "generation_config": {"temperature": 0},
                },
            }
        )
    )
    corpus = yaml.safe_load((root / "corpus.yaml").read_text())
    corpus["sources"] = sorted([*corpus["sources"], "sources/generator-inputs.yaml"])
    corpus["transforms"] = sorted([*corpus["transforms"], "transforms/inference.yaml"])
    (root / "corpus.yaml").write_text(yaml.safe_dump(corpus))
    return load_project(root / "corpus.yaml"), parent


def test_scenario_oracle_is_deterministic_without_filesystem_access() -> None:
    first = _path_scenario(23, "world-23", "stage")
    assert first == _path_scenario(23, "world-23", "stage")
    assert first["oracle_answer"] == Path(first["world_state"]["path"]).suffix
    assert first["world_state"]["operation"] == "suffix"


def test_grounded_unverified_and_rejected_replies_replay_offline(
    tmp_path: Path,
) -> None:
    project, parent = _replay_project(tmp_path)
    work = tmp_path / "work"
    acquire(project, work)
    built = build(project, work, offline=True)
    assert build(project, work, offline=True) == built
    records = [
        json.loads(line)
        for line in (built / "generations.jsonl").read_text().splitlines()
    ]
    assert [row["validation_status"] for row in records] == [
        "source_entailed",
        "unverified",
        "rejected",
    ]
    assert all(
        row["generator_identity"]
        and row["seed"] is not None
        and row["source_document_ids"] == [parent["document_id"]]
        for row in records
    )
    evidence = records[0]["evidence"]
    assert (
        parent["text"][slice(*evidence["span"])]
        == evidence["passage"]
        == records[0]["parsed_output"]["answer"]
    )
    assert records[1]["evidence"] is None
    assert (
        records[-1]["raw_output"] == "not-json" and records[-1]["parsed_output"] is None
    )
    lineage_rows = [
        json.loads(line) for line in (built / "lineage.jsonl").read_text().splitlines()
    ]
    generated_lineage = [
        row for row in lineage_rows if row["record_kind"] == "generation"
    ]
    assert len(generated_lineage) == len(records)
    assert {row["origin"] for row in generated_lineage} == {
        "source_transformed_synthetic"
    }
    assert {row["verification"]["status"] for row in generated_lineage} == {
        "source_entailed",
        "unverified",
        "rejected",
    }
    records_by_id = {row["record_id"]: row for row in records}
    assert all(
        row["generator"] == records_by_id[row["record_id"]]["generator"]
        for row in generated_lineage
    )
    released = freeze(built, work)
    assert any(
        json.loads(line)["record_id"] == records[-1]["record_id"]
        for line in (released / "lineage.jsonl").read_text().splitlines()
    )
    for split in ("train", "validation", "test"):
        assert list(iter_rendered_conversations(released / "chat" / f"{split}.jsonl"))


def test_changed_prompt_digest_rejects_recorded_request(tmp_path: Path) -> None:
    project, _ = _replay_project(tmp_path, bad_prompt=True)
    work = tmp_path / "work"
    acquire(project, work)
    with pytest.raises(ValueError, match="prompt mismatch"):
        build(project, work, offline=True)
    release_root = work / "corpora" / project.config.id / "releases"
    assert not release_root.exists() or not any(release_root.iterdir())
    diagnostics = list(
        (work / "corpora" / project.config.id / "builds/diagnostics").glob("*.json")
    )
    assert len(diagnostics) == 1
    assert json.loads(diagnostics[0].read_text())["request_id"] == "request-0"
