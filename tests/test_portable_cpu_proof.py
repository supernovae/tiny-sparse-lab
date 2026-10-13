"""External preparation reaches the native CPU admission boundary without dispatch.

The downstream native declaration is authored by this fixture, not an execution
adapter or an automatic conversion of the portable workflow profile.
"""

import json
from pathlib import Path

import pytest
from test_preparation_execution import git, rows, setup
from test_workflow_recipe import update, write_workflow

from sparselab.campaign.engine import CampaignEngine
from sparselab.config.models import RunConfig
from sparselab.experiments.preparation_execution import PreparationExecution


def commit(root):
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "CPU proof fixture",
    )


def native_campaign(root, config, artifacts):
    """Author ordinary native declarations, with explicit operational locations."""
    root.mkdir()
    suite = {
        "evaluation_suite_version": 1,
        "id": "synthetic-heldout",
        "evaluations": [{"id": "heldout", "role": "diagnostic", "kind": "heldout_lm"}],
    }
    plan = {
        "plan_version": 1,
        "id": "synthetic-proof",
        "base_run": "run.json",
        "evaluation_suite": "suite.json",
        "artifacts": artifacts,
        "inputs": {"tokenizer": "token", "prepared_data": "packed"},
    }
    stages = [
        {
            "id": name,
            "kind": "artifact_reference",
            "scope": scope,
            "artifact": artifacts[name],
        }
        for name, scope in (("token", "tokenizer"), ("packed", "model"))
    ]
    stages += [
        {
            "id": "plan",
            "kind": "experiment_plan",
            "scope": "model",
            "requires": ["token", "packed"],
            "source": "plan.json",
            "mode": "lock",
            "tokenizer": "token",
            "prepared": "packed",
        },
        {
            "id": "runtime",
            "kind": "runtime_acceptance",
            "scope": "runtime",
            "requires": ["plan"],
            "plan": "plan",
        },
        {
            "id": "gate",
            "kind": "approval",
            "scope": "model",
            "requires": ["runtime"],
            "bind": ["plan", "runtime"],
        },
        {
            "id": "run",
            "kind": "experiment_run",
            "scope": "model",
            "requires": ["plan", "runtime", "gate"],
            "plan": "plan",
            "runtime": "runtime",
            "cell": "main:single",
        },
        {
            "id": "collect",
            "kind": "experiment_collect",
            "scope": "evaluation",
            "requires": ["plan", "run"],
            "plan": "plan",
            "run": "run",
        },
        {
            "id": "evaluation",
            "kind": "evaluation",
            "scope": "evaluation",
            "requires": ["collect"],
            "collect": "collect",
            "suite": "suite.json",
        },
    ]
    for name, document in {
        "run.json": config.model_dump(mode="json"),
        "suite.json": suite,
        "plan.json": plan,
        "campaign.json": {
            "campaign_version": 1,
            "id": "synthetic-proof",
            "stages": stages,
        },
    }.items():
        (root / name).write_text(json.dumps(document))
    commit(root)
    return root / "campaign.json"


def test_external_preparation_reaches_native_execution_budget_boundary(
    tmp_path, monkeypatch
):
    from sparselab.experiments.cli import locked_cell_request
    from sparselab.experiments.lock import open_lock
    from sparselab.experiments.publication_report import (
        build_publication_report,
        write_publication_report,
    )
    from sparselab.experiments.workflow_recipe import publish_workflow
    from sparselab.workers.controller import Controller

    args = setup(tmp_path)
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(args[2]))
    prep = PreparationExecution(*args)
    prepared = prep.engine().apply()
    assert {row["state"] for row in prepared["stages"]} == {"COMPLETE"}
    artifacts = {
        name: {
            **rows(prepared)[stage]["outputs"][0],
            "version": 1,
            "producer": "native",
            "path": rows(prepared)[stage]["availability"]["path"],
        }
        for name, stage in (("token", "train"), ("packed", "pack"))
    }
    replay = prep.engine()
    monkeypatch.setattr(
        replay, "dispatch", lambda *a, **kw: pytest.fail("duplicate preparation")
    )
    assert rows(replay.apply())["pack"]["outputs"] == rows(prepared)["pack"]["outputs"]

    raw = prep.loaded.configs["pack"].model_dump(mode="json")
    raw["training"].update(max_steps=1, max_tokens=16)
    raw["optimizer"]["warmup_steps"] = 0
    raw["checkpoint"]["every_steps"] = 1
    raw["evaluation"].update(every_steps=1, max_batches=1)
    config = RunConfig.model_validate(raw)
    # The explicit fixture admission: fixed tiny existing CPU dimensions, one update.
    assert config.dataset.source == "synthetic"
    assert (
        config.runtime.engine,
        config.runtime.backend,
        config.runtime.precision,
    ) == ("pytorch", "cpu", "fp32")
    assert (
        config.model.num_layers,
        config.model.hidden_dim,
        config.model.ffn_dim,
        config.model.vocab_size,
    ) == (1, 16, 32, 264)
    assert (
        config.training.seq_len,
        config.training.micro_batch_size,
        config.training.gradient_accumulation,
    ) == (16, 1, 1)
    assert config.dataset.train_max_tokens + config.dataset.validation_max_tokens == 210

    # A portable lock over the same actual inputs is still not execution admission.
    source, bindings = write_workflow(
        tmp_path / "workflow",
        config.model_dump(mode="json"),
        {
            name: {key: value for key, value in artifact.items() if key != "path"}
            for name, artifact in artifacts.items()
        },
    )
    update(
        bindings,
        artifacts={name: artifact["path"] for name, artifact in artifacts.items()},
        logging_root=str(config.logging.root_dir),
    )
    portable_path = publish_workflow(source, bindings, tmp_path / "portable-lock")
    portable = open_lock(portable_path)
    with pytest.raises(ValueError, match="WORKFLOW_EXECUTION_UNAVAILABLE"):
        locked_cell_request(
            portable, portable.cells[0], "forbidden", tmp_path / "forbidden", None
        )
    assert not (tmp_path / "forbidden").exists()

    source = native_campaign(tmp_path / "native-recipe", config, artifacts)
    engine = CampaignEngine(source, args[2], cold_verify=True)
    monkeypatch.setattr(
        Controller,
        "submit_many",
        lambda *a, **kw: pytest.fail("model dispatch exceeds admission"),
    )
    stopped = engine.apply()
    assert rows(stopped)["gate"]["state"] == "AWAITING_APPROVAL"
    engine.approve(
        "gate", "approve", "Owner-authorized one-update synthetic CPU fixture only"
    )
    stopped = engine.apply()
    assert rows(stopped)["run"]["state"] == "READY"
    native = rows(stopped)
    workspace = Path(native["plan"]["availability"]["workspace"])
    lock_path = Path(native["plan"]["availability"]["path"])
    lock = open_lock(lock_path)
    assert not (workspace / "controller").exists()
    # Worker execution stages through warmup before training. Existing schema
    # cannot express zero pilot updates: do not dispatch under a one-update cap.
    from pydantic import ValidationError

    from sparselab.config.models import StagingConfig

    assert (config.staging.smoke_steps, config.staging.warmup_steps) == (2, 5)
    from sparselab.staging import pilot_config

    pilots = [
        pilot_config(config, purpose, tmp_path / "unused-stage")
        for purpose in ("smoke", "warmup")
    ]
    assert [pilot.training.max_steps for pilot in pilots] == [2, 5]
    assert [pilot.training.max_tokens for pilot in pilots] == [32, 80]
    assert (
        config.training.max_steps + sum(pilot.training.max_steps for pilot in pilots)
        == 8
    )
    assert not (tmp_path / "unused-stage").exists()
    for field in ("smoke_steps", "warmup_steps"):
        with pytest.raises(ValidationError):
            StagingConfig.model_validate({field: 0})
    minimum = StagingConfig(smoke_steps=1, warmup_steps=1)
    assert minimum.smoke_steps + minimum.warmup_steps + config.training.max_steps == 3
    report = build_publication_report(lock_path)
    assert report["readiness"]["execution_coverage"] == "incomplete"
    assert report["readiness"]["scientific_result"] == "not_assessed"
    assert report["cells"][0]["status"] == "pending"
    assert report["cells"][0]["attempts"] == []
    output = tmp_path / "planned-report.json"
    write_publication_report(lock_path, output)
    assert json.loads(output.read_text()) == report
    assert lock.cells[0].config.training.max_steps == 1
    # Re-entering native apply preserves completed preparation/locking receipts;
    # it cannot infer execution permission from the fixture's approval receipt.
    monkeypatch.setattr(
        engine, "dispatch", lambda *a, **kw: pytest.fail("duplicate native stage")
    )
    repeated = engine.apply(resume=True)
    assert rows(repeated)["run"]["state"] == "READY"
    assert {name: row["outputs"] for name, row in rows(repeated).items()} == {
        name: row["outputs"] for name, row in native.items()
    }
    assert not (workspace / "controller").exists()


@pytest.mark.parametrize("damage", ["recipe", "bindings", "inventory"])
def test_changed_preparation_context_never_dispatches(tmp_path, monkeypatch, damage):
    args = setup(tmp_path)
    context = PreparationExecution(*args)
    engine = context.engine()
    monkeypatch.setattr(
        engine, "dispatch", lambda *a, **kw: pytest.fail("changed input dispatched")
    )
    target = {
        "recipe": args[0],
        "bindings": args[1],
        "inventory": args[0].parent / "run.yaml",
    }[damage]
    target.write_text(target.read_text() + "\n")
    with pytest.raises(
        ValueError, match="CONTEXT_CHANGED|digest mismatch|unchanged clean declarations"
    ):
        engine.apply()
    assert not args[2].exists()


def test_portable_evaluation_declaration_is_an_explicit_remaining_gap(tmp_path):
    from test_workflow_recipe import repin

    from sparselab.experiments.workflow_recipe import load_workflow

    source, bindings = write_workflow(tmp_path / "workflow")
    update(source.parent / "plans/plan.json", evaluation_suite="suite.json")
    repin(source)
    with pytest.raises(
        ValueError, match="unavailable workflow profile feature: evaluation_suite"
    ):
        load_workflow(source, bindings)
    assert not (source.parent / "payload").exists()
