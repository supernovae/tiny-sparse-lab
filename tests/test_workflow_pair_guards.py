"""Isolated paired-admission refusals; no artifact, runtime or model execution.

Start with an admitted metadata fixture, then vary one typed boundary value.
Native Campaign validation sometimes rejects the same malformed wiring earlier;
model_copy deliberately isolates the admission layer's own defense in depth.
No implementation-source pins, runtime receipts or artifact verifiers decide
the refusal assertions.
"""

from pathlib import Path

import pytest
from test_workflow_pair import admit, change, pair_fixture
from test_workflow_recipe import update

from sparselab.experiments import workflow_execution


@pytest.fixture
def context(tmp_path):
    source, bindings = pair_fixture(tmp_path / "recipe")
    return admit(source, bindings, tmp_path / "work")


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", {"train_max_tokens": 128}),
        ("tokenizer", {"path": Path("/different/tokenizer.json")}),
        ("runtime", {"backend": "cuda"}),
        ("runtime", {"engine": "mlx"}),
    ],
    ids=["shared-dataset", "shared-tokenizer", "cpu-backend", "pytorch-engine"],
)
def test_expanded_configuration_contract(context, monkeypatch, field, value):
    expand = workflow_execution.expand_axes

    def expanded(*args, **kwargs):
        cells = list(expand(*args, **kwargs))
        config, coordinate = cells[1]
        section = getattr(config, field)
        # Preserve the other admission conditions to isolate this compiler port.
        cells[1] = (
            config.model_copy(update={field: section.model_copy(update=value)}),
            coordinate,
        )
        return tuple(cells)

    monkeypatch.setattr(workflow_execution, "expand_axes", expanded)
    with pytest.raises(
        ValueError, match="^workflow supports snapshot CPU PyTorch only$"
    ):
        context._validate_workflow()
    assert not context.work_dir.exists()


def test_snapshot_profile_at_public_admission(tmp_path):
    source, bindings = pair_fixture(tmp_path / "recipe")

    def synthetic(raw):
        raw["dataset"]["source"] = "synthetic"
        for field in (
            "train_path",
            "validation_path",
            "source_manifest_path",
            "license",
        ):
            raw["dataset"].pop(field)

    change(source, "configs/run.json", synthetic)
    update(bindings, config_paths={})
    with pytest.raises(
        ValueError, match="^workflow supports snapshot CPU PyTorch only$"
    ):
        admit(source, bindings, tmp_path / "work")
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "field,value",
    [("interventions", ["optimizer.peak", "model.hidden_dim"]), ("phases", ["main"])],
    ids=["one-intervention", "no-phases"],
)
def test_comparison_contract_at_public_admission(tmp_path, field, value):
    source, bindings = pair_fixture(tmp_path / "recipe")
    change(
        source,
        "plans/plan.json",
        lambda raw: raw["comparisons"][0].update({field: value}),
    )
    with pytest.raises(
        ValueError,
        match="^paired workflow requires two choices and one controlled comparison$",
    ):
        admit(source, bindings, tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_recovery_is_not_a_paired_workflow(context):
    context.campaign = context.campaign.model_copy(update={"recovery": "recovery.json"})
    with pytest.raises(
        ValueError,
        match="^workflow requires one native plan and one run/collect/evaluation branch per cell$",
    ):
        context._validate_workflow()
    assert not context.work_dir.exists()


@pytest.mark.parametrize(
    "stage_id,updates",
    [
        ("plan", {"source": "other-plan.json"}),
        ("plan", {"mode": "bind"}),
        ("plan", {"tokenizer": "packed"}),
        ("plan", {"prepared": "token"}),
        ("plan", {"requires": ("token",)}),
        ("runtime", {"plan": "other-plan"}),
        ("runtime", {"requires": ("plan", "token")}),
        ("runtime", {"profile_id": "cpu-profile"}),
        ("runtime", {"worker": "worker"}),
        ("gate", {"bind": ("plan",)}),
        ("gate", {"bind": ("runtime",)}),
        ("gate", {"requires": ("plan",)}),
        ("gate", {"requires": ("runtime",)}),
        ("run-candidate", {"plan": "other-plan"}),
        ("run-candidate", {"runtime": "other-runtime"}),
        ("collect-candidate", {"plan": "other-plan"}),
        (
            "eval-candidate",
            {"collect": "collect-baseline", "requires": ("collect-baseline",)},
        ),
        ("eval-candidate", {"backend": "cuda"}),
        ("eval-candidate", {"runtime": "runtime"}),
        ("eval-candidate", {"requires": ("collect-candidate", "runtime")}),
    ],
    ids=[
        "plan-source",
        "plan-lock-mode",
        "tokenizer-port",
        "prepared-port",
        "plan-requirements",
        "runtime-plan",
        "runtime-requirements",
        "runtime-profile",
        "runtime-worker",
        "gate-bind-runtime",
        "gate-bind-plan",
        "gate-require-runtime",
        "gate-require-plan",
        "run-plan",
        "run-runtime",
        "collect-plan",
        "evaluation-collection",
        "evaluation-backend",
        "evaluation-runtime",
        "evaluation-requirements",
    ],
)
def test_stage_connection_contract(context, stage_id, updates):
    context.campaign = context.campaign.model_copy(
        update={
            "stages": tuple(
                stage.model_copy(update=updates) if stage.id == stage_id else stage
                for stage in context.campaign.stages
            )
        }
    )
    with pytest.raises(ValueError, match="^unsupported portable Campaign connections$"):
        context._validate_workflow()
    assert not context.work_dir.exists()


def test_shared_dataset_at_public_admission(tmp_path):
    source, bindings = pair_fixture(tmp_path / "recipe")
    change(
        source,
        "plans/plan.json",
        lambda raw: raw["axes"][0]["choices"][1]["set"].update(
            {"dataset.train_max_tokens": 128}
        ),
    )
    with pytest.raises(
        ValueError, match="^workflow supports snapshot CPU PyTorch only$"
    ):
        admit(source, bindings, tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_caller_allowance_accepts_full_horizon_without_launching(tmp_path):
    """The small documented example is not a built-in experiment ceiling."""
    source, bindings = pair_fixture(tmp_path / "recipe")
    change(
        source,
        "configs/run.json",
        lambda raw: raw["training"].update(max_steps=1_000_000, max_tokens=16_000_000),
    )
    context = admit(
        source,
        bindings,
        tmp_path / "work",
        limits={
            "max_training_updates": 2_000_014,
            "max_target_positions": 32_000_224,
            "max_eval_batches": 4,
        },
    )
    assert context.config.training.max_steps == 1_000_000
    assert not context.work_dir.exists()
