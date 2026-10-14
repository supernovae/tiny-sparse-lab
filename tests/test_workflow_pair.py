"""Portable paired admission and native planning/attempts, without model work."""

import copy
import json
import sys
from pathlib import Path

import pytest
from test_experiment_snapshot_inputs import snapshot_inputs  # noqa: F401
from test_preparation_execution import git, rows
from test_workflow_execution import execution_fixture
from test_workflow_recipe import FIXTURE, repin, update

from sparselab.experiments.workflow_execution import (
    WorkflowAdmission,
    WorkflowExecution,
)
from sparselab.experiments.workflow_recipe import load_workflow, resolve_workflow


def pair_fixture(root, config=None, artifacts=None, locations=None):
    config = copy.deepcopy(config or json.loads(FIXTURE.read_text()))
    paths = {}
    for key, member in (
        ("train_path", "train.jsonl"),
        ("validation_path", "validation.jsonl"),
        ("source_manifest_path", "manifest.json"),
    ):
        paths[f"dataset.{key}"] = config["dataset"].get(key, f"snapshot/{member}")
        config["dataset"][key] = f"snapshot/{member}"
    config["dataset"].update(source="snapshot", revision="a" * 40, license="MIT")
    source, bindings = execution_fixture(root, config, artifacts)
    update(bindings, config_paths=paths, **(locations or {}))
    update(
        root / "plans/plan.json",
        axes=[
            {
                "name": "arm",
                "choices": [
                    {"label": "baseline", "set": {"optimizer.peak": 0.003}},
                    {"label": "candidate", "set": {"optimizer.peak": 0.002}},
                ],
            }
        ],
        comparisons=[
            {
                "id": "paired",
                "baseline": {"arm": "baseline"},
                "variant": {"arm": "candidate"},
                "mode": "controlled",
                "interventions": ["optimizer.peak"],
                "invariants": ["seed", "training.max_steps", "evaluation.max_batches"],
            }
        ],
    )
    campaign = root / "campaign.json"
    document = json.loads(campaign.read_text())
    original = {stage["kind"]: stage for stage in document["stages"]}
    document["stages"] = document["stages"][:5]
    for arm in ("baseline", "candidate"):
        run = {
            **original["experiment_run"],
            "id": f"run-{arm}",
            "cell": f"main:arm={arm}",
            "requires": ["plan", "runtime", "gate"]
            + (["run-baseline"] if arm == "candidate" else []),
        }
        collect = {
            **original["experiment_collect"],
            "id": f"collect-{arm}",
            "run": run["id"],
            "requires": ["plan", "run-baseline", "run-candidate"],
        }
        evaluation = {
            **original["evaluation"],
            "id": f"eval-{arm}",
            "collect": collect["id"],
            "requires": [collect["id"]],
        }
        document["stages"].extend([run, collect, evaluation])
    campaign.write_text(json.dumps(document))
    repin(source)
    return source, bindings


def admit(source, bindings, work, **overrides):
    git(source.parent, "init", "-q")
    git(source.parent, "add", ".")
    git(
        source.parent,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "paired fixture",
    )
    loaded = load_workflow(source, bindings)
    payload = {
        "workflow_admission_version": 2,
        "reviewed_commit": git(source.parent, "rev-parse", "HEAD"),
        "recipe_sha256": loaded.recipe_sha256,
        "binding_sha256": loaded.binding_sha256,
        "cpu_python": str(Path(sys.executable)),
        "pair_limits": {
            "max_training_updates": 54,
            "max_target_positions": 1728,
            "max_eval_batches": 4,
        },
    }
    payload.update(overrides)
    return WorkflowExecution(
        source, bindings, work, WorkflowAdmission.model_validate(payload)
    )


def change(source, filename, transform):
    path = source.parent / filename
    document = json.loads(path.read_text())
    transform(document)
    path.write_text(json.dumps(document))
    repin(source)


@pytest.fixture
def prepared_pair(snapshot_inputs, tmp_path):  # noqa: F811
    plan, _, _ = snapshot_inputs
    # Find by kind, keeping the native artifact identities and payload paths.
    artifacts = {}
    locations = {}
    for name, kind in (("token", "tokenizer"), ("packed", "prepared_data")):
        artifact = next(a for a in plan.artifacts.values() if a.kind == kind)
        artifacts[name] = {
            key: value
            for key, value in artifact.model_dump(mode="json").items()
            if key in {"kind", "version", "producer", "identifier", "sha256"}
        }
        locations[name] = artifact.path
    return pair_fixture(
        tmp_path / "recipe",
        plan.base_run.model_dump(mode="json"),
        artifacts,
        {"artifacts": locations, "logging_root": str(tmp_path / "runs")},
    )


def test_metadata_admission_keeps_v1_shape_and_is_read_only(tmp_path):
    source, bindings = pair_fixture(tmp_path / "recipe")
    context = admit(source, bindings, tmp_path / "work")
    assert context.admission.workflow_admission_version == 2
    assert len(context.campaign.stages) == 11
    assert not context.work_dir.exists()
    old = context.admission.model_dump(mode="json")
    old.pop("pair_limits")
    old["workflow_admission_version"] = 1
    assert WorkflowAdmission.model_validate(old).model_dump(mode="json") == old


@pytest.mark.parametrize(
    "field,value",
    [
        ("workflow_admission_version", True),
        ("workflow_admission_version", "2"),
        ("pair_limits", None),
        (
            "pair_limits",
            {
                "max_training_updates": True,
                "max_target_positions": 1728,
                "max_eval_batches": 4,
            },
        ),
    ],
)
def test_strict_new_admission(tmp_path, field, value):
    source, bindings = pair_fixture(tmp_path / "recipe")
    with pytest.raises(ValueError):
        admit(source, bindings, tmp_path / "work", **{field: value})
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_training_updates", 53),
        ("max_target_positions", 863),
        ("max_eval_batches", 3),
    ],
)
def test_limits_include_both_arms_and_pilots(tmp_path, field, value):
    source, bindings = pair_fixture(tmp_path / "recipe")
    limits = {
        "max_training_updates": 54,
        "max_target_positions": 864,
        "max_eval_batches": 4,
    }
    limits[field] = value
    with pytest.raises(ValueError, match="declared .*limit"):
        admit(source, bindings, tmp_path / "work", pair_limits=limits)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "extra-choice",
        "missing-comparison",
        "wrong-selector",
        "descriptive",
        "seed-intervention",
        "cuda",
    ],
)
def test_pair_shape_refuses_without_payload_reads(tmp_path, damage):
    source, bindings = pair_fixture(tmp_path / "recipe")

    def mutate(plan):
        if damage == "extra-choice":
            plan["axes"][0]["choices"].append(
                {"label": "third", "set": {"optimizer.peak": 0.001}}
            )
        elif damage == "missing-comparison":
            plan["comparisons"] = []
        elif damage == "wrong-selector":
            plan["comparisons"][0]["variant"] = {"arm": "missing"}
        elif damage == "descriptive":
            plan["comparisons"][0]["mode"] = "descriptive"
        elif damage == "seed-intervention":
            plan["comparisons"][0]["interventions"] = ["seed"]
        else:
            plan["execution"]["backend"] = "cuda"

    change(source, "plans/plan.json", mutate)
    with pytest.raises(ValueError, match="paired"):
        admit(source, bindings, tmp_path / "work")
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "same-cell",
        "no-gate",
        "parallel",
        "early-collect",
        "same-run",
        "same-collect",
        "foreign-suite",
    ],
)
def test_connections_are_checked_by_stage_id(tmp_path, damage):
    source, bindings = pair_fixture(tmp_path / "recipe")

    def mutate(document):
        stages = {s["id"]: s for s in document["stages"]}
        if damage == "same-cell":
            stages["run-candidate"]["cell"] = stages["run-baseline"]["cell"]
        elif damage == "no-gate":
            stages["run-candidate"]["requires"].remove("gate")
        elif damage == "parallel":
            stages["run-candidate"]["requires"].remove("run-baseline")
        elif damage == "early-collect":
            stages["collect-baseline"]["requires"].remove("run-candidate")
        elif damage == "same-run":
            stages["collect-candidate"]["run"] = "run-baseline"
        elif damage == "same-collect":
            stages["eval-candidate"].update(
                collect="collect-baseline", requires=["collect-baseline"]
            )
        else:
            stages["eval-candidate"]["suite"] = "foreign.json"

    change(source, "campaign.json", mutate)
    with pytest.raises(ValueError, match="connections"):
        admit(source, bindings, tmp_path / "work")


@pytest.mark.parametrize("aftermath", ["cancel", "lost", "ambiguous"])
def test_native_pair_lock_and_zero_wait_attempt_replay(
    prepared_pair, tmp_path, monkeypatch, aftermath
):
    from sparselab.experiments.lock import open_lock
    from sparselab.workers.controller import Controller

    source, bindings = prepared_pair
    context = admit(source, bindings, tmp_path / "execution")
    engine = context.engine()
    # No worker tick means no runtime probes, pilot, training or scoring.
    monkeypatch.setattr(
        Controller, "tick", lambda *a, **kw: pytest.fail("model dispatch")
    )
    staged = rows(engine.apply())
    assert staged["gate"]["state"] == "AWAITING_APPROVAL"
    locked = open_lock(Path(staged["plan"]["availability"]["path"]))
    assert len(locked.cells) == 2
    assert locked.comparisons[0].interventions == ("optimizer.peak",)
    assert locked.comparisons[0].baseline == "main:arm=baseline"
    assert locked.comparisons[0].variant == "main:arm=candidate"
    engine.approve("gate", "approve", "fixture submission only; no worker execution")
    submitted = rows(engine.apply(execute_runs=True, max_wait_seconds=0))
    assert submitted["run-baseline"]["state"] == "RUNNING"
    assert submitted["run-candidate"]["state"] != "RUNNING"
    controller = Controller(
        Path(staged["plan"]["availability"]["workspace"]) / "controller", read_only=True
    )
    attempts = controller.list_experiments()
    assert len(attempts) == 1
    engine.apply(resume=True, execute_runs=False, max_wait_seconds=0)
    assert controller.list_experiments() == attempts

    from sparselab.experiments.evidence import collect_evidence, read_evidence

    workspace = Path(staged["plan"]["availability"]["workspace"])
    index = collect_evidence(locked, workspace)
    evidence = read_evidence(locked, workspace, Path(index["index_path"]))
    assert evidence["comparisons"][0]["status"] == "unavailable"
    queued = submitted["run-baseline"]
    if aftermath == "cancel":
        assert (
            Controller(workspace / "controller").cancel(
                queued["measurements"]["run_id"]
            )["status"]
            == "CANCELLED"
        )
        expected = "INTERRUPTED"
    elif aftermath == "lost":
        import shutil

        shutil.rmtree(workspace / "controller")
        expected = "FAILED"
    else:
        original = Controller.list_experiments
        monkeypatch.setattr(
            Controller, "list_experiments", lambda self: original(self) * 2
        )
        expected = "FAILED"
    monkeypatch.setattr(
        Controller, "submit_many", lambda *a, **kw: pytest.fail("replacement attempt")
    )
    refused = rows(engine.apply(resume=True, execute_runs=True, max_wait_seconds=0))
    assert refused["run-baseline"]["state"] == expected
    assert refused["run-candidate"]["state"] == "BLOCKED"
    assert refused["run-baseline"]["measurements"] == queued["measurements"]
    if aftermath == "lost":
        assert "LOST_SUBMISSION" in refused["run-baseline"]["reason"]
        assert not (workspace / "controller").exists()
    elif aftermath == "ambiguous":
        assert "ambiguous controller attempts" in refused["run-baseline"]["reason"]


@pytest.mark.parametrize("damage", ["undeclared", "invariant", "no-change"])
def test_native_comparability_refuses_before_runtime(
    prepared_pair, tmp_path, monkeypatch, damage
):
    source, bindings = prepared_pair

    def mutate(plan):
        if damage == "undeclared":
            plan["axes"][0]["choices"][1]["set"]["model.hidden_dim"] = 32
        elif damage == "invariant":
            plan["comparisons"][0]["invariants"].append("optimizer.peak")
        else:
            plan["axes"][0]["choices"][1]["set"]["optimizer.peak"] = 0.003

    change(source, "plans/plan.json", mutate)
    context = admit(source, bindings, tmp_path / "execution")
    with pytest.raises(ValueError, match="undeclared|invariant|unchanged|expected"):
        resolve_workflow(source, bindings)
    assert not context.work_dir.exists()


@pytest.mark.parametrize(
    "member", ["train.jsonl", "validation.jsonl", "manifest.json", "excluded.jsonl"]
)
def test_snapshot_identity_refuses_before_attempt(prepared_pair, tmp_path, member):
    source, bindings = prepared_pair
    context = admit(source, bindings, tmp_path / "execution")
    target = tmp_path / "snapshot" / member
    target.write_bytes(target.read_bytes() + b" ")
    result = rows(context.engine().apply(execute_runs=True))
    assert any(row["state"] in {"FAILED", "BLOCKED"} for row in result.values())
    assert result["plan"]["state"] != "COMPLETE"
    assert not list(context.work_dir.rglob("controller"))


@pytest.mark.parametrize("seed,expected", [(True, 1), ("7", 7)])
def test_pair_preserves_legacy_run_config_seed_acceptance(tmp_path, seed, expected):
    config = json.loads(FIXTURE.read_text())
    config["seed"] = seed
    source, bindings = pair_fixture(tmp_path / "recipe", config)
    assert admit(source, bindings, tmp_path / "work").config.seed == expected


def test_graph_names_and_serialization_order_are_not_workflow_roles(tmp_path):
    source, bindings = pair_fixture(tmp_path / "recipe")

    def rename(document):
        names = {
            stage["id"]: "renamed-" + stage["id"]
            for stage in document["stages"]
            if stage["kind"] != "artifact_reference"
        }
        for stage in document["stages"]:
            for key in ("id", "plan", "runtime", "run", "collect"):
                if key in stage:
                    stage[key] = names.get(stage[key], stage[key])
            for key in ("requires", "bind"):
                if key in stage:
                    stage[key] = [names.get(value, value) for value in stage[key]]
        document["stages"].reverse()

    change(source, "campaign.json", rename)
    context = admit(source, bindings, tmp_path / "work")
    assert {s.id for s in context.campaign.stages if s.kind == "experiment_run"} == {
        "renamed-run-baseline",
        "renamed-run-candidate",
    }


def test_pair_public_inspect_remains_nonexecuting(tmp_path):
    import os
    import subprocess

    from test_portable_recipe import _PROCESS_FENCE

    source, bindings = pair_fixture(tmp_path / "recipe")
    code = (
        _PROCESS_FENCE
        + '\nfrom pathlib import Path\nfrom sparselab.experiments.workflow_recipe import inspect_workflow\nresult = inspect_workflow(Path(sys.argv[1]), Path(sys.argv[2]))\nassert result["executable"] is False\n'
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(source), str(bindings)],
        capture_output=True,
        check=False,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("member", ["tokenizer", "prepared_data"])
def test_artifact_bytes_refuse_before_native_submission(
    prepared_pair, tmp_path, monkeypatch, member
):
    from sparselab.workers.controller import Controller

    source, bindings = prepared_pair
    context = admit(source, bindings, tmp_path / "execution")
    artifact = next(
        a for a in context.loaded.plan.artifacts.values() if a.kind == member
    )
    target = Path(artifact.path)
    if member == "prepared_data":
        target /= "train.npy"
    target.write_bytes(target.read_bytes() + b"tamper")
    monkeypatch.setattr(
        Controller,
        "submit_many",
        lambda *a, **kw: pytest.fail("tampered input submitted"),
    )
    result = rows(context.engine().apply(execute_runs=True))
    assert result["plan"]["state"] != "COMPLETE"
    assert result["run-baseline"]["state"] == "BLOCKED"


def test_pair_context_constructor_does_not_import_model_runtime(tmp_path):
    import subprocess

    source, bindings = pair_fixture(tmp_path / "recipe")
    context = admit(source, bindings, tmp_path / "work")
    document = tmp_path / "admission.json"
    document.write_text(context.admission.model_dump_json())
    code = """import sys
from pathlib import Path
from sparselab.experiments.workflow_execution import WorkflowAdmission, WorkflowExecution
source, bindings, work, admission = map(Path, sys.argv[1:])
WorkflowExecution(source, bindings, work, WorkflowAdmission.model_validate_json(admission.read_text()))
assert not work.exists()
assert not any(name in sys.modules for name in ('torch', 'numpy', 'tokenizers', 'datasets', 'sparselab.campaign.engine'))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(source),
            str(bindings),
            str(context.work_dir),
            str(document),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
