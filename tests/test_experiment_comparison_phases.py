"""Phase-scoped comparisons preserve legacy payloads and compile shipped plans."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest
from dataset_fixtures import offline_hub_identity

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.data import sources
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.experiments.direct_inputs import bind_direct_inputs
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
from sparselab.experiments.plan import Comparison, ExperimentPlan


def comparison(**updates):
    return {
        "id": "width",
        "baseline": {"architecture": "baseline"},
        "variant": {"architecture": "wider"},
        "interventions": ["model.ffn_dim"],
        **updates,
    }


def test_empty_phase_selection_preserves_historical_serialization():
    expected = {
        **comparison(),
        "invariants": [],
        "mode": "controlled",
        "confounders": [],
    }
    for raw in (comparison(), comparison(phases=[])):
        value = Comparison.model_validate(raw)
        assert value.model_dump(mode="json") == expected
        assert json.loads(value.model_dump_json()) == expected
        plan = ExperimentPlan.model_validate(
            {
                "plan_version": 1,
                "id": "old",
                "base_run": "run.yaml",
                "comparisons": [raw],
            }
        )
        assert plan.model_dump(mode="json")["comparisons"] == [expected]
        assert "phases" not in plan.model_dump()["comparisons"][0]


@pytest.mark.parametrize("selected", [["missing"], ["main", "main"], [""]])
def test_phase_selection_rejects_unknown_duplicate_or_unsafe_names(selected):
    with pytest.raises(ValueError, match="phases"):
        ExperimentPlan.model_validate(
            {
                "plan_version": 1,
                "id": "invalid",
                "base_run": "run.yaml",
                "comparisons": [comparison(phases=selected)],
            }
        )


def test_phase_selection_validates_declared_and_implicit_main():
    raw = {
        "plan_version": 1,
        "id": "selection",
        "base_run": "run.yaml",
        "comparisons": [comparison(phases=["main"])],
    }
    plan = ExperimentPlan.model_validate(raw)
    assert plan.comparisons[0].model_dump(mode="json")["phases"] == ["main"]
    with pytest.raises(ValueError, match="unknown phases"):
        ExperimentPlan.model_validate({**raw, "phases": [{"id": "pretrain"}]})


def test_shipped_sample_resolves_with_offline_snapshot_and_original_budgets(
    tmp_path, monkeypatch
):
    sample = (
        Path(__file__).resolve().parents[1] / "experiments/samples/tinystories-microlab"
    )
    declarations = tmp_path / "inputs"
    declarations.mkdir()
    for name in (
        "source.yaml",
        "run.yaml",
        "tokenizer.yaml",
        "plan.yaml",
        "suite.yaml",
    ):
        shutil.copyfile(sample / name, declarations / name)
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setattr(sources, "_hub_identity", offline_hub_identity)

    def rows(source, split, cache, **kwargs):
        rng = random.Random(split)
        for index in range(source.selection.documents[split]):
            words = [
                "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8))
                for _ in range(32)
            ]
            yield {"text": f"{split} story {index}: " + " ".join(words)}

    monkeypatch.setattr(sources, "_stream", rows)
    source_lock = sources.lock_source(
        declarations / "source.yaml", tmp_path / "source.lock.json", tmp_path / "hub"
    )
    sources.snapshot_source(source_lock, tmp_path / "snapshot", tmp_path / "hub")
    tokenizer = train_tokenizer(load_tokenizer_config(declarations / "tokenizer.yaml"))
    config = load_config(declarations / "run.yaml")
    prepared = prepare_data(config, load_tokenizer(tokenizer))
    bound_path = declarations / "bound.yaml"
    plan = bind_direct_inputs(
        declarations / "run.yaml", declarations / "plan.yaml", prepared.root, bound_path
    )
    resolved = resolve_plan(plan, bound_path)
    assert len(resolved.cells) == 4
    assert len(resolved.comparisons) == 1
    pair = resolved.comparisons[0]
    assert pair.baseline == "pretrain:architecture=baseline"
    assert pair.variant == "pretrain:architecture=wider"
    assert "artifacts.prepared_data.sha256" in pair.invariants
    assert "artifacts.tokenizer.sha256" in pair.invariants
    for cell in resolved.cells:
        assert cell.config.dataset == config.dataset
        assert cell.config.model.vocab_size == 2048
        assert cell.config.training.seq_len == 64
        assert cell.config.training.max_steps == (
            40 if cell.phase == "pretrain" else 80
        )
        assert cell.config.training.max_tokens == (
            10240 if cell.phase == "pretrain" else 20480
        )
    reopened = open_lock(publish_lock(resolved, tmp_path / "experiment"))
    assert reopened.plan_sha256 == resolved.plan_sha256
    assert reopened.comparisons == resolved.comparisons

    # Omission retains the old all-phases behavior and catches distinct parents.
    raw = plan.model_dump(mode="json")
    raw["comparisons"][0].pop("phases")
    with pytest.raises(
        ValueError, match="undeclared changed fields.*parent_checkpoint"
    ):
        resolve_plan(ExperimentPlan.model_validate(raw), bound_path)

    # General selection also works for fresh phases and the implicit main phase.
    raw["phases"] = [{"id": "first"}, {"id": "second"}]
    all_phases = resolve_plan(ExperimentPlan.model_validate(raw), bound_path)
    assert len(all_phases.comparisons) == 2
    raw["comparisons"][0]["phases"] = ["second"]
    selected = resolve_plan(ExperimentPlan.model_validate(raw), bound_path)
    assert len(selected.comparisons) == 1
    assert selected.comparisons[0].baseline.startswith("second:")
    raw["phases"] = []
    raw["comparisons"][0]["phases"] = ["main"]
    implicit = resolve_plan(ExperimentPlan.model_validate(raw), bound_path)
    assert len(implicit.comparisons) == 1
    assert implicit.comparisons[0].baseline.startswith("main:")
