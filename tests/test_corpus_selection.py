"""Exact-budget selection and independent release recipes share source snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers

from sparselab.corpus.acquisition import acquire, verify_acquisition
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze, verify_release
from sparselab.corpus.selection import select_fraction


def test_exact_generated_token_fraction_preserves_source_families() -> None:
    candidates = [
        {
            "record_id": "raw-a",
            "tokens": 6,
            "origin": "primary_source",
            "source_family_ids": ["a"],
        },
        {
            "record_id": "raw-b",
            "tokens": 4,
            "origin": "human_authored",
            "source_family_ids": ["b"],
        },
        {
            "record_id": "raw-extra",
            "tokens": 4,
            "origin": "primary_source",
            "source_family_ids": ["a"],
        },
        {
            "record_id": "synth",
            "tokens": 10,
            "origin": "deterministic_synthetic",
            "source_family_ids": [],
        },
    ]
    assert select_fraction(candidates, 0.5, 20) == {"raw-a", "raw-b", "synth"}
    with pytest.raises(ValueError, match="infeasible"):
        select_fraction(candidates, 0.25, 20)


def test_fraction_does_not_silently_round_generated_tokens() -> None:
    with pytest.raises(ValueError, match="integer number of tokens"):
        select_fraction([], 0.1, 3)


def test_shape_ablation_reuses_acquisition_and_changes_release(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "sample"
    shutil.copytree(Path("corpora/devmind-sample-v0"), recipe)
    source_project = recipe / "corpus.yaml"
    original = load_project(source_project)
    work = tmp_path / "work"
    initial_lock = acquire(original, work)
    original_release = freeze(build(original, work, offline=True), work)
    variant_release = yaml.safe_load((recipe / "release.yaml").read_text())
    variant_release["include_shapes"] = [
        "raw_document",
        "direct_qa",
        "troubleshooting_scenario",
        "tool_trace",
    ]
    (recipe / "raw-and-qa.yaml").write_text(yaml.safe_dump(variant_release))
    variant_config = yaml.safe_load(source_project.read_text())
    variant_config["release"] = "raw-and-qa.yaml"
    variant_path = recipe / "corpus-raw-and-qa.yaml"
    variant_path.write_text(yaml.safe_dump(variant_config))
    variant = load_project(variant_path)
    assert verify_acquisition(variant, work) == initial_lock
    selected_release = freeze(build(variant, work, offline=True), work)
    assert selected_release != original_release
    assert (
        verify_release(selected_release)["snapshots"]
        == verify_release(original_release)["snapshots"]
    )
    assert (selected_release / "lm/train.jsonl").read_bytes() == (
        original_release / "lm/train.jsonl"
    ).read_bytes()


def test_zero_generated_fraction_counts_selected_training_tokens(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "sample"
    shutil.copytree(Path("corpora/devmind-sample-v0"), recipe)
    project_path = recipe / "corpus.yaml"
    project = load_project(project_path)
    work = tmp_path / "work"
    acquire(project, work)
    baseline = freeze(build(project, work, offline=True), work)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    (recipe / "tokenizers").mkdir()
    tokenizer_path = recipe / "tokenizers/fixed.json"
    tokenizer.save(str(tokenizer_path))
    train_rows = [
        json.loads(row)
        for row in (baseline / "lm/train.jsonl").read_text().splitlines()
    ]
    budget = sum(len(tokenizer.encode(row["text"]).ids) for row in train_rows)
    declaration = yaml.safe_load((recipe / "release.yaml").read_text())
    declaration["chat"] = {"selected": False, "training_splits": []}
    declaration["fraction"] = {
        "generated_share": 0.0,
        "train_tokens": budget,
        "tokenizer_path": "tokenizers/fixed.json",
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }
    (recipe / "raw-only.yaml").write_text(yaml.safe_dump(declaration))
    config = yaml.safe_load(project_path.read_text())
    config["release"] = "raw-only.yaml"
    variant_path = recipe / "corpus-raw-only.yaml"
    variant_path.write_text(yaml.safe_dump(config))
    variant = load_project(variant_path)
    assert (
        verify_acquisition(variant, work)["sources"]
        == verify_acquisition(project, work)["sources"]
    )
    selected = freeze(build(variant, work, offline=True), work)
    report = json.loads((selected / "report.json").read_text())
    mixture = report["measurement"]["training_mixture"]
    assert mixture["actual_tokens"] == budget
    assert mixture["records"] == len(train_rows)
    assert set(mixture["dimensions"]["origin"]) == {"human_authored"}
