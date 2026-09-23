from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparselab.experiments.study import (
    ArchitectureStudy,
    StudyComparison,
    StudyPair,
    _comparison_report,
    plan_study,
    study_plan_payload,
)

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
