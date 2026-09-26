from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer

from sparselab.config.loading import load_tokenizer_config
from sparselab.data.tokenizer import train_tokenizer
from sparselab.evaluation.capabilities import load_capability_card
from sparselab.experiments.study import (
    ArchitectureStudy,
    StudyComparison,
    StudyPair,
    _comparison_report,
    plan_study,
    study_plan_payload,
)
from sparselab.model.inspection import parameter_inventory

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "runtime_smoke_cpu.yaml"


def _write_study(
    tmp_path: Path,
    variants: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    *,
    seeds: bool = True,
    axis_name: str = "architecture",
) -> Path:
    axes: dict[str, object] = {}
    if seeds:
        axes["seed"] = [
            {"label": "seed-7", "set": {"seed": 7}},
            {"label": "seed-41", "set": {"seed": 41}},
        ]
    axes[axis_name] = variants
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "matrix_version": 1,
                "base_config": str(BASE),
                "axes": axes,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        yaml.safe_dump(
            {
                "study_version": 1,
                "name": "tiny-architecture-study",
                "matrix": matrix_path.name,
                "cards": ["chat-alias-recall-v1"],
                "comparisons": comparisons,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return study_path


def _variants() -> list[dict[str, object]]:
    return [
        {"label": "dense", "set": {}},
        {
            "label": "ngram",
            "set": {
                "model.memory": "ngram",
                "model.memory_table_size": 31,
                "model.memory_ngram_size": 2,
                "model.memory_dim": 8,
            },
        },
        {
            "label": "moe",
            "set": {"model.ffn": "moe", "model.num_experts": 2},
        },
    ]


def test_plan_pairs_architectures_across_seeds_and_reports_size(tmp_path: Path) -> None:
    path = _write_study(
        tmp_path,
        _variants(),
        [
            {
                "id": "ngram-effect",
                "vary": "memory",
                "baseline": {"architecture": "dense"},
                "variant": {"architecture": "ngram"},
            },
            {
                "id": "moe-effect",
                "vary": "ffn",
                "baseline": {"architecture": "dense"},
                "variant": {"architecture": "moe"},
            },
        ],
    )

    plan = plan_study(path)
    payload = study_plan_payload(plan)

    assert len(plan.expanded) == 6
    assert len(plan.pairs) == 4
    comparisons = payload["comparisons"]
    assert isinstance(comparisons, list)
    assert [item["id"] for item in comparisons] == [
        "ngram-effect",
        "moe-effect",
    ]
    run_rows = payload["runs"]
    assert isinstance(run_rows, list)
    dense, ngram, moe = run_rows[:3]
    assert isinstance(dense, dict) and isinstance(ngram, dict) and isinstance(moe, dict)
    assert ngram["parameter_inventory"]["total"] > dense["parameter_inventory"]["total"]
    assert moe["parameter_inventory"]["routed_expert"] > 0
    assert all(len(item["pairs"]) == 2 for item in comparisons)
    interpretation = payload["interpretation"]
    assert isinstance(interpretation, str) and interpretation.startswith("Plan only")


def test_plan_supports_explicit_custom_architecture_fields(tmp_path: Path) -> None:
    path = _write_study(
        tmp_path,
        [
            {"label": "base", "set": {}},
            {"label": "wider-ffn", "set": {"model.ffn_dim": 48}},
        ],
        [
            {
                "id": "ffn-width",
                "vary": "custom",
                "vary_fields": ["model.ffn_dim"],
                "baseline": {"size": "base"},
                "variant": {"size": "wider-ffn"},
            }
        ],
        seeds=False,
        axis_name="size",
    )

    plan = plan_study(path)

    assert len(plan.pairs) == 1
    assert set(plan.pairs[0].differences) == {"model.ffn_dim"}


def test_plan_rejects_unmatched_or_misdeclared_architecture_comparisons(
    tmp_path: Path,
) -> None:
    path = _write_study(
        tmp_path,
        _variants(),
        [
            {
                "id": "wrong-axis",
                "vary": "ffn",
                "baseline": {"architecture": "dense"},
                "variant": {"architecture": "ngram"},
            }
        ],
    )

    with pytest.raises(ValueError, match="not controlled"):
        plan_study(path)


def test_duplicate_study_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text("study_version: 1\nstudy_version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate study YAML key"):
        plan_study(path)


def test_validation_loss_requires_matched_run_identity(tmp_path: Path) -> None:
    study = ArchitectureStudy(
        path=tmp_path / "study.yaml",
        name="identity-mismatch",
        matrix_path=tmp_path / "matrix.yaml",
        matrix_sha256="matrix",
        cards=(),
        expanded=(),
        comparisons=(),
        pairs=(),
        study_sha256="study",
    )
    comparison = StudyComparison(
        "ffn-width",
        "custom",
        {"architecture": "baseline"},
        {"architecture": "variant"},
        ("model.ffn_dim",),
    )
    pair = StudyPair(
        comparison,
        0,
        1,
        {},
        {"model.ffn_dim": {"base": 32, "variant": 48}},
    )
    identity = {
        "step": 1,
        "tokens_seen": 32,
        "tokenizer_sha256": "tokenizer",
        "data_sha256": {"train": "train", "validation": "validation"},
        "source_identity_sha256": "source",
        "runtime": {"engine": "pytorch"},
        "training_runtime": {"engine": "pytorch"},
    }
    baseline = {
        "coordinate": {"architecture": "baseline"},
        "run_id": "baseline-run",
        "identity": identity,
        "validation": {"loss": 4.0},
        "capabilities": {},
    }
    variant = {
        "coordinate": {"architecture": "variant"},
        "run_id": "variant-run",
        "identity": {**identity, "tokens_seen": 31},
        "validation": {"loss": 3.5},
        "capabilities": {},
    }

    report = _comparison_report(study, pair, [baseline, variant])
    validation = report["validation_loss"]

    assert isinstance(validation, dict)
    assert validation["status"] == "inconclusive"
    assert validation["reason"] == "run identity differs at tokens_seen"
    assert "delta_variant_minus_baseline" not in validation


def test_memory_injection_campaign_plan_and_composition_card(tmp_path: Path) -> None:
    study = plan_study(ROOT / "configs/memory_injection_v1.study.yaml")

    assert len(study.expanded) == 9
    assert len(study.pairs) == 9
    assert [item.coordinate["seed"] for item in study.expanded] == [
        seed for seed in ("s17", "s41", "s73") for _ in ("none", "final", "embedding")
    ]
    placement_pairs = [
        pair
        for pair in study.pairs
        if pair.comparison.identifier == "embedding-minus-final"
    ]
    assert len(placement_pairs) == 3
    for pair in placement_pairs:
        assert set(pair.differences) == {"model.memory_injection"}
        baseline = study.expanded[pair.baseline_index].config
        variant = study.expanded[pair.variant_index].config
        assert parameter_inventory(baseline) == parameter_inventory(variant)

    card = load_capability_card(
        ROOT / "data/memory_injection_v1/context_two_hop_v1.card.json"
    )
    assert [case.identifier for case in card.cases] == [
        f"two-hop-{pair}{side}" for pair in range(1, 5) for side in ("a", "b")
    ]
    assert Counter(case.expected for case in card.cases) == Counter(
        {
            "amber": 1,
            "ivory": 1,
            "onyx": 1,
            "pearl": 1,
            "cobalt": 1,
            "sienna": 1,
            "topaz": 1,
            "violet": 1,
        }
    )
    for pair_index in range(0, len(card.cases), 2):
        left, right = card.cases[pair_index : pair_index + 2]
        left_edges = re.findall(r"(\w+) maps to (\w+)", left.prompt)
        right_edges = re.findall(r"(\w+) maps to (\w+)", right.prompt)
        assert left_edges == right_edges
        left_start = re.search(r"Follow two links from (\w+)", left.prompt)
        right_start = re.search(r"Follow two links from (\w+)", right.prompt)
        assert left_start is not None and right_start is not None
        assert left_start.group(1) != right_start.group(1)
        assert left.expected != right.expected
        graph = dict(left_edges)
        assert graph[left_start.group(1)] != left.expected
        assert graph[graph[left_start.group(1)]] == left.expected
        assert graph[right_start.group(1)] != right.expected
        assert graph[graph[right_start.group(1)]] == right.expected
    tokenizer_config = load_tokenizer_config(
        ROOT / "configs/context_study_tokenizer.yaml"
    )
    tokenizer_config = tokenizer_config.model_copy(
        update={
            "output_dir": tmp_path / "context-study-tokenizer",
            "dataset": tokenizer_config.dataset.model_copy(
                update={"cache_dir": tmp_path / "context-study-tokenizer-cache"}
            ),
        }
    )
    tokenizer_path = train_tokenizer(tokenizer_config)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    lengths = [
        len(tokenizer.encode(case.prompt, add_special_tokens=False).ids)
        for case in card.cases
    ]
    assert lengths == [71, 71, 81, 82, 101, 103, 89, 86]
    assert max(lengths) + card.generation["max_new_tokens"] <= 128
