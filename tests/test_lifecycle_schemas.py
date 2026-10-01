"""Published strict authoring schemas match the runtime validators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.evaluation.readiness import ModelReadinessPolicy
from sparselab.evaluation.suite import EvaluationSuite
from sparselab.family.manifest import FamilyManifest
from sparselab.recovery.manifest import RecoveryManifest


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("recovery-manifest-v1.schema.json", RecoveryManifest),
        ("evaluation-suite-v1.schema.json", EvaluationSuite),
        ("model-readiness-v1.schema.json", ModelReadinessPolicy),
        ("model-family-v1.schema.json", FamilyManifest),
    ],
)
def test_published_lifecycle_schema(filename, model) -> None:
    path = Path(__file__).resolve().parents[1] / "schemas" / filename
    assert json.loads(path.read_text()) == model.model_json_schema()
