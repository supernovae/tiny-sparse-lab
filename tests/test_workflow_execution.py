"""Authenticated portable Campaign boundaries; fixtures do not dispatch models."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_preparation_execution import git
from test_workflow_recipe import (  # noqa: F401
    native_workflow,
    repin,
    update,
    write_workflow,
)

from sparselab.experiments.workflow_execution import (
    WorkflowAdmission,
    WorkflowExecution,
)
from sparselab.experiments.workflow_recipe import load_workflow


def execution_fixture(root, config=None, artifacts=None):
    source, bindings = write_workflow(root, config, artifacts)
    raw = json.loads(source.read_text())
    suite = {
        "evaluation_suite_version": 1,
        "id": "diagnostic",
        "evaluations": [{"id": "heldout", "kind": "heldout_lm", "role": "diagnostic"}],
    }
    (root / "suite.json").write_text(json.dumps(suite))
    update(root / "plans/plan.json", evaluation_suite="suite.json")
    stages = [
        {
            "id": name,
            "kind": "artifact_reference",
            "scope": scope,
            "artifact": raw["artifacts"][name],
        }
        for name, scope in (("token", "tokenizer"), ("packed", "model"))
    ]
    stages += [
        {
            "id": "plan",
            "kind": "experiment_plan",
            "scope": "model",
            "requires": ["token", "packed"],
            "source": "plans/plan.json",
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
            "requires": ["plan", "runtime"],
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
    (root / "campaign.json").write_text(
        json.dumps({"campaign_version": 1, "id": "portable", "stages": stages})
    )
    raw["campaign"] = "campaign.json"
    for name in ("campaign.json", "suite.json"):
        raw["files"].append(
            {
                "path": name,
                "role": "metadata",
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            }
        )
    source.write_text(json.dumps(raw))
    repin(source)
    return source, bindings


def admitted(source, bindings, work):
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
        "portable CPU fixture",
    )
    loaded = load_workflow(source, bindings)
    return WorkflowExecution(
        source,
        bindings,
        work,
        WorkflowAdmission(
            workflow_admission_version=1,
            reviewed_commit=git(source.parent, "rev-parse", "HEAD"),
            recipe_sha256=loaded.recipe_sha256,
            binding_sha256=loaded.binding_sha256,
            cpu_python=Path(sys.executable),
        ),
    )


@pytest.mark.parametrize(
    "damage", ["descriptor", "inventory", "bindings", "suite", "commit"]
)
def test_changed_context_refuses_before_dispatch(tmp_path, monkeypatch, damage):
    source, bindings = execution_fixture(tmp_path / "recipe")
    context = admitted(source, bindings, tmp_path / "work")
    engine = context.engine()
    monkeypatch.setattr(
        engine,
        "dispatch",
        lambda *a, **kw: pytest.fail("dispatch after changed context"),
    )
    target = {
        "descriptor": source,
        "inventory": source.parent / "configs/run.json",
        "bindings": bindings,
        "suite": source.parent / "suite.json",
        "commit": source,
    }[damage]
    if damage != "commit":
        target.write_text(target.read_text() + "\n")
    if damage == "commit":
        git(source.parent, "add", ".")
        git(
            source.parent,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "changed committed recipe",
            "--allow-empty",
        )
    with pytest.raises(
        ValueError, match="digest mismatch|CONTEXT_CHANGED|unchanged clean declarations"
    ):
        engine.apply(execute_runs=True)
    assert not context.work_dir.exists()


def test_authored_connections_require_approval_before_run(tmp_path):
    source, bindings = execution_fixture(tmp_path / "recipe")
    path = source.parent / "campaign.json"
    raw = json.loads(path.read_text())
    next(stage for stage in raw["stages"] if stage["kind"] == "experiment_run")[
        "requires"
    ].remove("gate")
    path.write_text(json.dumps(raw))
    repin(source)
    with pytest.raises(ValueError, match="portable Campaign connections"):
        admitted(source, bindings, tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_native_plan_approval_and_lock_authority(
    native_workflow,  # noqa: F811
    tmp_path,
):
    from sparselab.experiments.cli import locked_cell_request
    from sparselab.experiments.lock import open_lock

    old_source, old_bindings = native_workflow
    old = load_workflow(old_source, old_bindings)
    source, bindings = execution_fixture(
        tmp_path / "admitted",
        old.context.read_document(old.context.reference(old.source, old.plan.base_run)),
        {
            name: item.model_dump(mode="json")
            for name, item in old.descriptor.artifacts.items()
        },
    )
    bindings.write_bytes(old_bindings.read_bytes())
    context = admitted(source, bindings, tmp_path / "work")
    engine = context.engine()
    result = engine.apply()
    rows = {row["id"]: row for row in result["stages"]}
    assert rows["gate"]["state"] == "AWAITING_APPROVAL"
    assert rows["run"]["state"] != "COMPLETE"
    lock = open_lock(Path(rows["plan"]["availability"]["path"]))
    with pytest.raises(ValueError, match="WORKFLOW_EXECUTION_UNAVAILABLE"):
        locked_cell_request(lock, lock.cells[0], "worker", tmp_path, None)
    engine.approve("gate", "approve", "fixture approval; no model dispatch")
    assert engine.inspect("status")["stages"]
    context.authorize_lock(lock)
    bindings.write_text(bindings.read_text() + "\n")
    with pytest.raises(ValueError, match="WORKFLOW_CONTEXT_CHANGED"):
        locked_cell_request(
            lock, lock.cells[0], "worker", tmp_path, None, workflow_context=context
        )


def test_campaign_metadata_inspection_has_no_runtime_effects(tmp_path):
    import os
    import subprocess

    from test_portable_recipe import _PROCESS_FENCE

    source, bindings = execution_fixture(tmp_path / "recipe")
    code = (
        _PROCESS_FENCE
        + "\nfrom pathlib import Path\nfrom sparselab.experiments.workflow_recipe import inspect_workflow\ninspect_workflow(Path(sys.argv[1]), Path(sys.argv[2]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(source), str(bindings)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
