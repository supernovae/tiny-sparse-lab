"""Neutral metadata golden vectors; no personal study inputs or execution."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from report_fixtures import write_report_inputs
from study_identity_vectors import CONFIGS, MATRIX, STUDY

from sparselab.experiments.reporting import (
    _historical_config_sha256,
    _historical_study_identity,
    _identity_for_receipt,
)
from sparselab.experiments.study import plan_study


def test_neutral_historical_identity_matches_fixed_vectors_after_relocation(
    tmp_path: Path,
) -> None:
    root = write_report_inputs(tmp_path / "original", historical=True)
    receipt = {"matrix_sha256": MATRIX, "study_sha256": STUDY}
    for location in (root, tmp_path / "relocated"):
        if location != root:
            shutil.move(root, location)
        study = plan_study(location / "study.yaml")
        assert _historical_study_identity(study) == (MATRIX, STUDY, CONFIGS)
        assert (study.matrix_sha256, study.study_sha256) != (MATRIX, STUDY)
        assert _identity_for_receipt(study, receipt) == (
            STUDY,
            MATRIX,
            CONFIGS,
            "legacy_default_fields",
        )
        current = _identity_for_receipt(
            study,
            {
                "matrix_sha256": study.matrix_sha256,
                "study_sha256": study.study_sha256,
            },
        )
        assert current[3] == "current"


@pytest.mark.parametrize("field", ["matrix_sha256", "study_sha256"])
def test_neutral_historical_receipt_rejects_mismatched_identity(
    tmp_path: Path, field: str
) -> None:
    root = write_report_inputs(tmp_path / "inputs", historical=True)
    receipt = {"matrix_sha256": MATRIX, "study_sha256": STUDY, field: "0" * 64}
    with pytest.raises(ValueError, match="study specification differs from receipt"):
        _identity_for_receipt(plan_study(root / "study.yaml"), receipt)


def test_neutral_historical_receipt_cannot_erase_nondefault_training_semantics(
    tmp_path: Path,
) -> None:
    root = write_report_inputs(tmp_path / "inputs", historical=True)
    base = yaml.safe_load((root / "base.yaml").read_text())
    base["training"]["neural_loss_weight"] = 0.5
    (root / "base.yaml").write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="requires default fields"):
        _identity_for_receipt(
            plan_study(root / "study.yaml"),
            {"matrix_sha256": MATRIX, "study_sha256": STUDY},
        )


def test_retirement_preserves_identity_closure_mapping() -> None:
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "docs/memory-retirement.json").read_text())
    original_bytes = (root / record["prior_closure"]["path"]).read_bytes()
    assert (
        hashlib.sha256(original_bytes).hexdigest() == record["prior_closure"]["sha256"]
    )
    original = {row["source"]: row for row in json.loads(original_bytes)["files"]}
    rows = record["files"]
    assert len(rows) == len({row["source"] for row in rows}) == 12
    assert {row["source"] for row in rows} == set(original)
    assert sum(row["needed_by_historical_identity_reader"] for row in rows) == 7
    for row in rows:
        assert row["disposition"] == "removed_active_copy"
        assert {**row, "disposition": "retained_unchanged"} == original[row["source"]]
        assert not (root / row["source"]).exists()
    assert record["destination_commit"] == "acf5f61fe4b80f31ba4b2e1db4f37a0f25b0f426"


def test_historical_identity_cannot_omit_allocation_manifest(tmp_path: Path) -> None:
    root = write_report_inputs(tmp_path / "inputs", historical=True)
    config = plan_study(root / "study.yaml").expanded[0].config.model_dump(mode="json")
    assert config["dataset"].get("allocation_manifest_path") is None
    assert _historical_config_sha256(config) == CONFIGS[0]
    config["dataset"]["allocation_manifest_path"] = "declared-allocation.json"
    with pytest.raises(
        ValueError, match="historical identity cannot omit an allocation manifest"
    ):
        _historical_config_sha256(config)
    assert not (root / "declared-allocation.json").exists()
