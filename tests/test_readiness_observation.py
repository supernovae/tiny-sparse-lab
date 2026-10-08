"""Zero-update coverage checks; no fixture loads or trains a model."""

from pathlib import Path

import pytest

from sparselab.evaluation import panel as panel_module
from sparselab.evaluation import readiness


def _index() -> dict:
    return {
        "index_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "evaluations": [
            {"id": "loss", "status": "COMPLETED"},
            {"id": "capability", "status": "FAILED"},
        ],
    }


def test_completed_execution_does_not_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    index = _index()
    index["evaluations"][1]["status"] = "COMPLETED"
    monkeypatch.setattr(readiness, "verify_evaluation_index", lambda _: index)
    monkeypatch.setattr(
        panel_module,
        "verify_panel_result",
        lambda _: {
            "evaluation_index_sha256": index["index_sha256"],
            "checkpoint_sha256": index["checkpoint_sha256"],
            "rows": [{"status": "COMPLETED"}, {"status": "COMPLETED"}],
        },
    )
    result = readiness.inspect_observation_coverage(Path("index"), Path("panel"))
    assert (result["suite_completed"], result["suite_total"]) == (2, 2)
    assert (result["panel_completed"], result["panel_total"]) == (2, 2)
    assert result["reviewed_scores"] == "MISSING"
    assert result["reader_eligibility"] == "UNESTABLISHED"


def test_incomplete_or_mismatched_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    index = _index()
    monkeypatch.setattr(readiness, "verify_evaluation_index", lambda _: index)
    no_panel = readiness.inspect_observation_coverage(Path("index"))
    assert (no_panel["suite_completed"], no_panel["suite_total"]) == (1, 2)
    assert no_panel["panel_completed"] is None
    monkeypatch.setattr(
        panel_module,
        "verify_panel_result",
        lambda _: {
            "evaluation_index_sha256": index["index_sha256"],
            "checkpoint_sha256": "c" * 64,
            "rows": [],
        },
    )
    with pytest.raises(ValueError, match="different checkpoints"):
        readiness.inspect_observation_coverage(Path("index"), Path("panel"))
