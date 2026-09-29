"""Verified measurements distinguish exported records from exact rendered tokens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from sparselab.corpus.acquisition import acquire
from sparselab.corpus.measurement import (
    capability_matrix,
    measure_views,
    summarize_release,
)
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import _validate_rows, describe, freeze

PROJECT = Path(__file__).resolve().parents[1] / "corpora/devmind-sample-v0/corpus.yaml"


def _release(tmp_path: Path) -> Path:
    project = load_project(PROJECT)
    acquire(project, tmp_path)
    return freeze(build(project, tmp_path, offline=True), tmp_path)


def _tokenizer(path: Path) -> Path:
    model = Tokenizer(
        models.WordLevel({"[UNK]": 0, "The": 1, "path": 2}, unk_token="[UNK]")
    )
    model.pre_tokenizer = pre_tokenizers.Whitespace()
    model.save(str(path))
    return path


def test_streamed_measurement_matches_small_overlap_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "release"
    for view in ("lm", "chat"):
        (root / view).mkdir(parents=True)
        for split in ("train", "validation", "test"):
            (root / view / f"{split}.jsonl").write_text("")
            (root / view / f"{split}.lineage.jsonl").write_text("")
    records = [
        {
            "record_id": f"r{index}",
            "origin": "primary_source",
            "shape": {"id": "source"},
            "verification": {"status": "source_entailed"},
            "parent_document_ids": [f"doc{index // 2}"],
            "source_family_ids": ["family"],
        }
        for index in range(3)
    ]
    texts = ["alpha beta gamma " * 5, "alpha beta gamma " * 5, "other words"]
    (root / "lineage.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    for name in ("generations.jsonl", "scenarios.jsonl"):
        (root / name).write_text("")
    (root / "lexical").mkdir()
    (root / "lexical/candidates.jsonl").write_text('{"term":"alpha"}\n')
    (root / "lm/train.jsonl").write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in texts)
    )
    (root / "lm/train.lineage.jsonl").write_text(
        "".join(json.dumps({"record_id": row["record_id"], "split": "train"}) + "\n"
                for row in records)
    )
    spec = {
        "lm": {"selected": True, "training_splits": ["train"]},
        "chat": {"selected": False, "training_splits": []},
    }
    original_read_text = Path.read_text

    def no_whole_jsonl(self: Path, *args: object, **kwargs: object) -> str:
        if self.suffix == ".jsonl":
            raise AssertionError(f"materialized JSONL: {self}")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_whole_jsonl)
    summary = summarize_release(root, release_spec=spec)
    train = summary["views"]["lm"]["train"]
    concentration = train["concentration"]
    assert train["records"] == summary["training_mixture"]["records"] == 3
    assert train["dimensions"]["origin"]["primary_source"]["records"] == 3
    assert concentration["duplicate_text_rows"] == 1
    assert concentration["normalized_duplicate_rows"] == 1
    assert concentration["prefix_32_rows"] == 1
    assert concentration["ngram_3_total"] == 26
    assert concentration["ngram_3_distinct"] == 3
    assert concentration["parent_document_counts"] == {"doc0": 2, "doc1": 1}
    assert concentration["source_family_counts"] == {"family": 3}
    assert summary["inventory"]["lexical_distinct_terms"] == 1
    assert summary["inventory"]["generated_training_records"] == 0


def test_large_view_omits_ngrams_without_retaining_texts(tmp_path: Path) -> None:
    root = tmp_path / "release"
    for view in ("lm", "chat"):
        (root / view).mkdir(parents=True)
        for split in ("train", "validation", "test"):
            (root / view / f"{split}.jsonl").write_text("")
            (root / view / f"{split}.lineage.jsonl").write_text("")
    row = {
        "record_id": "source", "origin": "primary_source",
        "parent_document_ids": ["parent"], "source_family_ids": [],
    }
    (root / "lineage.jsonl").write_text(json.dumps(row) + "\n")
    for name in ("generations.jsonl", "scenarios.jsonl"):
        (root / name).write_text("")
    text = "a" * 20_000_001
    (root / "lm/train.jsonl").write_text(json.dumps({"text": text}) + "\n")
    (root / "lm/train.lineage.jsonl").write_text(
        '{"record_id":"source","split":"train"}\n'
    )
    concentration = measure_views(root)["views"]["lm"]["train"]["concentration"]
    assert concentration["ngram_3_total"] is None
    assert concentration["ngram_3_distinct"] is None
    assert concentration["ngram_count_reason"] == "omitted_above_20m_character_bound"
    assert concentration["parent_document_counts"] == {"parent": 1}

def test_report_keeps_record_and_exact_token_denominators_separate(
    tmp_path: Path,
) -> None:
    release = _release(tmp_path)
    release_spec = json.loads((release / "manifest.json").read_text())[
        "build_identity"
    ]["release"]
    without_tokenizer = summarize_release(release, release_spec=release_spec)
    assert without_tokenizer["inventory"]["view_record_distribution"]["origin"]
    assert all(
        item["actual_tokens"] is None
        and item["token_count_reason"] == "tokenizer_not_declared"
        for view in without_tokenizer["views"].values()
        for item in view.values()
    )
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    measured = measure_views(release, tokenizer)
    observed = describe(release, tokenizer=tokenizer)
    assert observed["token_totals"] == {
        f"{view}/{split}": item["actual_tokens"]
        for view, splits in measured["views"].items()
        for split, item in splits.items()
    }
    for view, splits in measured["views"].items():
        for split, item in splits.items():
            assert item["records"] == len(
                (release / view / f"{split}.jsonl").read_text().splitlines()
            )
            if item["records"]:
                assert (
                    sum(
                        group["actual_tokens"]
                        for group in item["dimensions"]["origin"].values()
                    )
                    == item["actual_tokens"]
                )
            else:
                assert item["actual_tokens"] == 0
            assert (
                sum(group["records"] for group in item["dimensions"]["shape"].values())
                == item["records"]
            )
    assert capability_matrix(release, tmp_path / "no-model-evidence") is None
    ledger = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (release / "lineage.jsonl").read_text().splitlines()
        )
    }
    heldout_templates = {
        ledger[json.loads(line)["record_id"]].get("template_family_id")
        for view in ("lm", "chat")
        for line in (release / view / "test.lineage.jsonl").read_text().splitlines()
    }
    overlaps = without_tokenizer["inventory"]["generated_heldout_lineage_by_shape"]
    for shape_id, groups in overlaps.items():
        train_templates = {
            row.get("template_family_id")
            for view in ("lm", "chat")
            for line in (release / view / "train.lineage.jsonl")
            .read_text()
            .splitlines()
            if (row := ledger[json.loads(line)["record_id"]])["shape"]["id"] == shape_id
            and row["origin"] not in {"primary_source", "human_authored"}
        }
        assert groups["template_family_id"] == sorted(
            (train_templates & heldout_templates) - {None}
        )


def test_generated_holdout_template_overlap_is_scoped_to_shape(tmp_path: Path) -> None:
    release = _release(tmp_path)
    rows = [
        json.loads(line)
        for line in (release / "lineage.jsonl").read_text().splitlines()
    ]
    train_ids = {
        json.loads(line)["record_id"]
        for view in ("lm", "chat")
        for line in (release / view / "train.lineage.jsonl").read_text().splitlines()
    }
    test_ids = {
        json.loads(line)["record_id"]
        for view in ("lm", "chat")
        for line in (release / view / "test.lineage.jsonl").read_text().splitlines()
    }
    generated = next(
        row
        for row in rows
        if row["record_id"] in train_ids
        and row["origin"] not in {"primary_source", "human_authored"}
    )
    heldout = next(row for row in rows if row["record_id"] in test_ids)
    generated["template_family_id"] = heldout["template_family_id"] = "shared-template"
    (release / "lineage.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    release_spec = json.loads((release / "manifest.json").read_text())[
        "build_identity"
    ]["release"]
    overlaps = summarize_release(release, release_spec=release_spec)["inventory"][
        "generated_heldout_lineage_by_shape"
    ]
    assert "shared-template" in overlaps[generated["shape"]["id"]]["template_family_id"]
    assert all(
        "shared-template" not in group["template_family_id"]
        for shape_id, group in overlaps.items()
        if shape_id != generated["shape"]["id"]
    )


def test_lineage_claim_cannot_forge_verbatim_evidence(tmp_path: Path) -> None:
    release = _release(tmp_path)
    records = [
        json.loads(line)
        for line in (release / "lineage.jsonl").read_text().splitlines()
    ]
    row = next(
        record
        for record in records
        if record["verification"]["status"] == "source_entailed"
    )
    row["verification"]["evidence"]["passage"] = "fabricated unrelated answer"
    (release / "lineage.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    with pytest.raises(ValueError):
        _validate_rows(release)
