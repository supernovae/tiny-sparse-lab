"""Corpus Forge tokenizer fitting and held-out family selection invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers

from sparselab.corpus.tokenizer_bakeoff import (
    GROUPS,
    Declaration,
    _selection,
    _train_token_counts,
    choose_candidate,
    load_declaration,
)


def _spec() -> Declaration:
    return Declaration(
        schema_version=1,
        release_path="release",
        vocab_sizes=(16384, 24576, 32768),
        max_fit_bytes=268435456,
        eval_split="validation",
        eval_max_docs_per_group=200,
        groups=GROUPS,
        near_best_ratio=0.98,
    )


def _fixture(tmp_path: Path, *, overlap: bool = False) -> Path:
    release = tmp_path / "release"
    (release / "lm").mkdir(parents=True)
    docs = []
    for split in ("train", "validation"):
        ids = []
        for group in GROUPS:
            text = f"{split}/{group}: one source document."
            ids.append(f"{split}-{group}")
            docs.append(
                {
                    "document_id": ids[-1],
                    "split": split,
                    "text": text,
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "document_kind": group,
                    "source_location": f"{group}.txt",
                    "source_family": ("train" if overlap else split) + "_family",
                }
            )
        (release / "lm" / f"{split}.lineage.jsonl").write_text(
            "".join(
                json.dumps({"record_id": key, "split": split}) + "\n" for key in ids
            )
        )
    (release / "documents.jsonl").write_text(
        "".join(json.dumps(doc) + "\n" for doc in docs)
    )
    return release


def test_loaded_pilot_declaration_retains_frozen_group_selection(
    tmp_path: Path,
) -> None:
    expected = _spec().model_copy(update={"schema_version": 2, "groups": GROUPS[:6]})
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(expected.model_dump(mode="json")))
    assert load_declaration(path) == expected
    path.write_text(
        yaml.safe_dump(
            expected.model_copy(update={"schema_version": 1}).model_dump(mode="json")
        )
    )
    with pytest.raises(ValueError, match="approved candidates"):
        load_declaration(path)


def test_selection_stable_per_kind_and_family_disjoint(tmp_path: Path) -> None:
    release = _fixture(tmp_path)
    first = _selection(release, _spec())
    second = _selection(release, _spec())
    assert first == second
    fit, heldout, train = first
    for group in GROUPS:
        assert len(fit[group]) == len(heldout[group]) == len(train[group]) == 1
        assert fit[group][0]["split"] == "train"
        assert heldout[group][0]["split"] == "validation"
        assert fit[group][0]["source_family"] != heldout[group][0]["source_family"]


def test_pilot_bakeoff_keeps_observed_groups_without_fitting_probes(
    tmp_path: Path,
) -> None:
    release = _fixture(tmp_path)
    pilot = _spec().model_copy(update={"schema_version": 2, "groups": GROUPS[:-1]})
    with pytest.raises(ValueError, match="measurable source groups omitted"):
        _selection(release, pilot)
    docs_path = release / "documents.jsonl"
    docs = [json.loads(line) for line in docs_path.read_text().splitlines()]
    docs_path.write_text(
        "".join(
            json.dumps(doc) + "\n" for doc in docs if doc["document_kind"] != "logs"
        )
    )
    for split in ("train", "validation"):
        path = release / "lm" / f"{split}.lineage.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in rows
                if not row["record_id"].endswith("-logs")
            )
        )
    fit, heldout, _ = _selection(release, pilot)
    assert tuple(fit) == tuple(heldout) == GROUPS[:-1]
    with pytest.raises(ValueError, match="missing required"):
        _selection(release, _spec())


@pytest.mark.parametrize(
    ("schema_version", "distinct_tokens"),
    [(1, 2), (2, 5)],
)
def test_train_token_accounting_keeps_distinct_source_and_view_scopes(
    tmp_path: Path, schema_version: int, distinct_tokens: int
) -> None:
    release = tmp_path / "release"
    (release / "lm").mkdir(parents=True)
    rows = []
    for identifier, text, split, domains in (
        ("prose", "one two", "train", ["developer_systems"]),
        ("code", "three four five", "train", ["developer_systems"]),
        ("duplicate", "one two", "train", ["general_education"]),
        ("unselected", "six seven eight nine", "train", ["general_education"]),
        ("heldout", "held out", "validation", ["general_education"]),
    ):
        rows.append(
            {
                "document_id": identifier,
                "split": split,
                "text": text,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "domains": domains,
            }
        )
    (release / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (release / "lm/train.lineage.jsonl").write_text(
        "".join(
            json.dumps({"record_id": row["document_id"]}) + "\n" for row in rows[:3]
        )
    )
    (release / "lm/train.jsonl").write_text(
        "".join(json.dumps({"text": row["text"]}) + "\n" for row in rows[:3])
    )
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    assert _train_token_counts(
        release, {"prose": [rows[0]]}, schema_version, tokenizer
    ) == {
        "distinct_normalized_train_tokens": distinct_tokens,
        "total_selected_train_view_tokens": 7,
        "general_education_distinct_train_tokens": 0,
        "general_education_train_view_tokens": 2,
    }


def test_selection_rejects_shared_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="families overlap"):
        _selection(_fixture(tmp_path, overlap=True), _spec())


def test_selection_rejects_shared_content_across_families(tmp_path: Path) -> None:
    release = _fixture(tmp_path)
    path = release / "documents.jsonl"
    docs = [json.loads(line) for line in path.read_text().splitlines()]
    docs[9]["text"] = docs[0]["text"]
    docs[9]["content_sha256"] = docs[0]["content_sha256"]
    path.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    with pytest.raises(ValueError, match="share normalized content"):
        _selection(release, _spec())


def test_selection_rule_near_best_boundary() -> None:
    candidates = [
        {"vocab_size": 16384, "weighted_bytes_per_token": 1.959},
        {"vocab_size": 24576, "weighted_bytes_per_token": 1.96},
        {"vocab_size": 32768, "weighted_bytes_per_token": 2.0},
    ]
    assert choose_candidate(candidates) == 24576
    candidates[0]["weighted_bytes_per_token"] = 1.96
    assert choose_candidate(candidates) == 16384
