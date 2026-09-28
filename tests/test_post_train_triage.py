from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.evaluation.post_train_triage import _advice, loss_trajectory, read_triage


def _points(losses: tuple[float, ...]) -> list[dict[str, object]]:
    return [
        {"step": index, "targets": target, "loss": loss, "evidence": "verified"}
        for index, (target, loss) in enumerate(
            zip((0, 1_000_000, 2_000_000, 4_000_000), losses, strict=True)
        )
    ]


def test_loss_trajectory_uses_exposures_and_endpoint_rates() -> None:
    curve = loss_trajectory(_points((2.0, 1.8, 1.6, 1.595)))
    assert curve["initial_loss"] == 2.0
    assert curve["terminal_loss"] == 1.595
    assert curve["delta_loss"] == pytest.approx(-0.405)
    assert curve["trend"] == "plateau_possible"
    intervals = curve["trend_intervals"]
    assert intervals["early"]["rate"] == pytest.approx(0.2)
    assert intervals["recent"]["rate"] == pytest.approx(0.0025)
    assert intervals["recent"]["from"]["targets"] == 2_000_000
    assert intervals["recent"]["to"]["targets"] == 4_000_000


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ((2.0, 1.8, 1.6, 1.61), "regressing"),
        ((2.0, 1.8, 1.6, 1.59), "still_improving"),
        ((2.0, 1.8, 1.6, 1.591), "plateau_possible"),
        ((2.0, 1.8, 1.6, 1.595), "plateau_possible"),
        ((2.0, 1.8, 1.6, 1.55), "still_improving"),
    ],
)
def test_endpoint_trend_boundaries(losses: tuple[float, ...], expected: str) -> None:
    assert loss_trajectory(_points(losses))["trend"] == expected


def test_short_or_unverified_curve_cannot_claim_plateau() -> None:
    assert loss_trajectory(_points((2.0, 1.8, 1.6, 1.595))[:3])["trend"] == (
        "insufficient_history"
    )


def test_decoder_sensitivity_distinguishes_missing_from_stable_and_reduced_repetition() -> (
    None
):
    core = {
        "integrity": {"status": "PASS"},
        "loss": {"trend": "insufficient_history", "points": []},
        "runtime": {},
    }
    tier1 = {"sampled": [], "parameter_inventory": {"total": 1000}}
    parent = {"status": "UNKNOWN"}
    panel = {"status": "UNKNOWN"}
    seed = {"fired": None, "comparisons": [], "reason": "no declared siblings"}

    def outcome() -> tuple[dict, dict, list]:
        return _advice(core, tier1, {}, parent, panel, seed)

    unknown, diagnostics, recommendations = outcome()
    assert unknown["DECODER_SENSITIVITY_DETECTED"]["fired"] is None
    assert diagnostics["DECODER_SENSITIVITY_DETECTED"]["status"] == "UNKNOWN"
    assert recommendations == []
    tier1["sampled"] = [
        {
            "id": "simple-continuation",
            "status": "OBSERVED",
            "trigram_excess_difference": 0,
            "settings": {"seed": seed_id},
        }
        for seed_id in range(8)
    ]
    stable, diagnostics, _ = outcome()
    assert stable["DECODER_SENSITIVITY_DETECTED"]["fired"] is False
    assert diagnostics["DECODER_SENSITIVITY_DETECTED"]["status"] == "NOT_NEEDED"
    tier1["sampled"][0]["trigram_excess_difference"] = -3
    sensitive, diagnostics, recommendations = outcome()
    assert sensitive["DECODER_SENSITIVITY_DETECTED"]["fired"] is True
    assert diagnostics["DECODER_SENSITIVITY_DETECTED"]["status"] == "RECOMMENDED"
    assert recommendations[0]["estimated_cost_class"] == "medium"


def test_viewer_rejects_unsafe_run_and_conflicting_artifacts(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    assert read_triage("run", tmp_path) is None
    with pytest.raises(ValueError):
        read_triage("../other", tmp_path)
    directory = tmp_path / "run" / "post-train-triage"
    directory.mkdir(parents=True)
    (directory / "forged.json").write_text(
        json.dumps({"format": "sparselab_post_train_triage_v1"})
    )
    with pytest.raises(ValueError):
        read_triage("run", tmp_path)
    (directory / "forged.json").unlink()
    (directory / "outside.json").symlink_to(tmp_path / "unowned.json")
    with pytest.raises(ValueError):
        read_triage("run", tmp_path)
