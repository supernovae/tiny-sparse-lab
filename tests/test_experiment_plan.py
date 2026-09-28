from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.experiments.compiler import compare_configs
from sparselab.experiments.plan import load_plan, schema


@pytest.mark.parametrize(
    ("suffix", "document", "error"),
    [
        ("yaml", "plan_version: 1\nplan_version: 1\n", "duplicate YAML key"),
        ("json", '{"plan_version":1,"plan_version":1}', "duplicate JSON key"),
        (
            "yaml",
            "plan_version: !!python/object/apply:os.system ['true']",
            "invalid experiment document",
        ),
        ("json", '{"plan_version":NaN}', "non-finite JSON number"),
    ],
)
def test_authored_plan_rejects_ambiguous_or_executable_data(
    tmp_path: Path, suffix: str, document: str, error: str
) -> None:
    path = tmp_path / f"plan.{suffix}"
    path.write_text(document)
    with pytest.raises(ValueError, match=error):
        load_plan(path)


def test_authored_plan_rejects_unknown_fields_and_duplicate_axis_labels(
    tmp_path: Path,
) -> None:
    base = {
        "plan_version": 1,
        "id": "strict",
        "base_run": "run.yaml",
        "axes": [
            {
                "name": "seed",
                "choices": [
                    {"label": "same", "set": {"seed": 7}},
                    {"label": "same", "set": {"seed": 17}},
                ],
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="duplicate choice labels"):
        load_plan(path)
    base.pop("axes")
    base["training_code"] = "import os"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="extra_forbidden"):
        load_plan(path)


def test_comparison_rejects_hidden_artifact_change_and_unchanged_intervention() -> None:
    baseline = {"training": {"max_steps": 20}, "model": {"hidden_dim": 16}}
    variant = {"training": {"max_steps": 25}, "model": {"hidden_dim": 16}}
    with pytest.raises(ValueError, match="undeclared changed fields"):
        compare_configs(baseline, variant, expected=("model.hidden_dim",))
    with pytest.raises(ValueError, match="declared interventions unchanged"):
        compare_configs(baseline, baseline, expected=("model.hidden_dim",))
    with pytest.raises(ValueError, match="undeclared changed fields"):
        compare_configs(
            baseline,
            baseline,
            expected=(),
            artifacts={
                "base": {"tokenizer": {"sha256": "a" * 64}},
                "variant": {"tokenizer": {"sha256": "b" * 64}},
            },
        )


def test_published_plan_schema_is_current() -> None:
    published = (
        Path(__file__).resolve().parents[1] / "schemas/experiment-plan-v1.schema.json"
    )
    assert json.loads(published.read_text()) == schema()
