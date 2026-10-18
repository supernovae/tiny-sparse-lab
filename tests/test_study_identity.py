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


def test_memory_identity_closure_matches_retained_local_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "docs/memory-identity-closure.json").read_text())
    rows = record["files"]
    assert len(rows) == len({row["source"] for row in rows}) == 12
    assert sum(row["needed_by_historical_identity_reader"] for row in rows) == 7
    for row in rows:
        data = (root / row["source"]).read_bytes()
        assert len(data) == row["size_bytes"], row["source"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["source"]
        assert (
            hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == row["git_blob"]
        ), row["source"]


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
