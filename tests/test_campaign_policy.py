"""Campaign readiness over real, independently frozen local corpus releases."""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from sparselab.campaign.plan import load_campaign
from sparselab.campaign.policy import CorpusReadinessPolicy, measure_readiness
from sparselab.campaign.state import CampaignStore
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze, verify_release


@pytest.fixture
def local_recipe(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "examples/tiny-campaign"
    destination = tmp_path / "tiny-campaign"
    shutil.copytree(source, destination)
    return destination


def _release(recipe: Path, work: Path) -> Path:
    project = load_project(recipe / "corpus.yaml")
    acquire(project, work, offline=False)
    released = freeze(build(project, work, offline=True), work)
    verify_release(released)
    return released


def _declaration(
    recipe: Path, weights: tuple[str, str], total: int, suffix: str
) -> Path:
    payload = {
        "campaign_version": 1,
        "id": "exact-source-passes",
        "stages": [
            {
                "id": "corpus",
                "kind": "corpus_release",
                "scope": "corpus",
                "project": "tiny-campaign/corpus.yaml",
            },
            {
                "id": "ready",
                "kind": "corpus_readiness",
                "scope": "corpus",
                "requires": ["corpus"],
                "corpus": "corpus",
                "policy": {
                    "passes": {
                        "basis": "bytes",
                        "requested_total": total,
                        "mixture": {"developer": 0.5, "technical_docs": 0.5},
                        "max_required": 1,
                    }
                },
            },
        ],
    }
    path = recipe.parent / f"campaign.{suffix}"
    if suffix == "json":
        text = json.dumps(payload, separators=(",", ":"))
        text = text.replace(
            '"developer":0.5,"technical_docs":0.5',
            f'"developer":{weights[0]},"technical_docs":{weights[1]}',
        )
    else:
        text = yaml.safe_dump(payload)
        text = text.replace("developer: 0.5", f"developer: {weights[0]}")
        text = text.replace("technical_docs: 0.5", f"technical_docs: {weights[1]}")
    path.write_text(text)
    return path


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_exact_authored_weights_change_verified_release_decision(
    local_recipe: Path, tmp_path: Path, suffix: str
) -> None:
    release = _release(local_recipe, tmp_path / "work")
    amounts = measure_readiness(release, CorpusReadinessPolicy(min_heldout_families=1))[
        "measurements"
    ]["unique_train_bytes_by_domain"]
    # Both 0.5 weights fit a single pass at this threshold. The extra 1e-17
    # of developer weight needs a second pass exactly at the integer boundary.
    assert amounts["developer"] > 0 and amounts["technical_docs"] > 0
    total = 2 * amounts["developer"]
    if amounts["developer"] > amounts["technical_docs"]:
        # Make the smaller domain the weighted boundary regardless of recipe size.
        total = 2 * amounts["technical_docs"]
        upper_domain = "technical_docs"
        weights = ("0.49999999999999999", "0.50000000000000001")
    else:
        upper_domain = "developer"
        weights = ("0.50000000000000001", "0.49999999999999999")
    baseline = load_campaign(_declaration(local_recipe, ("0.5", "0.5"), total, suffix))
    baseline_policy = baseline.stages[1].policy
    assert measure_readiness(release, baseline_policy)["state"] == "COMPLETE"
    baseline_sha = CampaignStore(baseline, tmp_path / "work").declaration_sha

    plan = load_campaign(_declaration(local_recipe, weights, total, suffix))
    mixture = plan.stages[1].policy.passes.mixture
    assert mixture[upper_domain] == Decimal("0.50000000000000001")
    assert CampaignStore(plan, tmp_path / "work").declaration_sha != baseline_sha
    decision = measure_readiness(release, plan.stages[1].policy)
    assert decision["state"] == "BLOCKED"
    assert decision["outcome"] == "EXPAND_MORE"
    assert (
        decision["measurements"]["projected_source_passes_by_domain"][upper_domain] == 2
    )
    assert {item["domain"] for item in decision["deficits"]} == {upper_domain}


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_reject_authored_string_and_nonfinite_weights(
    tmp_path: Path, suffix: str
) -> None:
    for bad in ('"0.5"', '"NaN"', "1e9999"):
        path = _declaration(tmp_path, (bad, "0.5"), 2, suffix)
        with pytest.raises((TypeError, ValueError)):
            load_campaign(path)


def test_validation_only_selected_view_does_not_satisfy_train_shape(
    local_recipe: Path, tmp_path: Path
) -> None:
    selection_path = local_recipe / "release.yaml"
    selection = yaml.safe_load(selection_path.read_text())
    selection["lm"]["training_splits"] = ["validation"]
    selection_path.write_text(yaml.safe_dump(selection))
    split_path = local_recipe / "splits.yaml"
    splits = yaml.safe_load(split_path.read_text())
    splits["assignments"]["tiny_test_family"] = "validation"
    split_path.write_text(yaml.safe_dump(splits))
    release = _release(local_recipe, tmp_path / "work")
    assert verify_release(release)["build_identity"]["release"]["lm"]["selected"]
    assert (release / "lm/train.lineage.jsonl").read_text().strip()
    decision = measure_readiness(
        release, CorpusReadinessPolicy(required_nonzero_shapes=("raw_document",))
    )
    assert decision["state"] == "BLOCKED"
    assert decision["outcome"] == "EXPAND_MORE"
    assert decision["measurements"]["selected_train_shapes"] == {}
    assert decision["deficits"] == [
        {
            "dimension": "train_shape",
            "observed": 0,
            "required": 1,
            "domain": "raw_document",
        }
    ]
