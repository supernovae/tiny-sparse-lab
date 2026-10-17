"""Native inventory generation and derivative connectivity, using neutral declarations."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_portable_recipe import _PROCESS_FENCE, update_member, write_recipe
from test_preparation_recipe import recipe as preparation_fixture
from test_workflow_recipe import write_workflow

from sparselab.documents import read_document
from sparselab.experiments.recipe import inspect_recipe
from sparselab.experiments.recipe_authoring import create_recipe


@pytest.mark.parametrize(
    "profile,factory",
    [
        ("matrix", write_recipe),
        ("preparation", preparation_fixture),
        ("workflow", write_workflow),
    ],
)
def test_create_without_existing_descriptor_and_relocate(tmp_path, profile, factory):
    built = factory(tmp_path / "authored")
    source = built[0] if isinstance(built, tuple) else built
    descriptor = read_document(source)
    source.unlink()  # Create must not depend on an existing hashed descriptor.
    identities = tmp_path / "identities.json"
    identities.write_text(json.dumps(descriptor.get("artifacts", {})))
    (source.parent / "not-selected.txt").write_text("not inventory")
    result = create_recipe(
        source.parent,
        tmp_path / "published",
        profile=profile,
        recipe_id=descriptor["id"],
        entrypoint=descriptor["entrypoint"],
        files=[f"{m['role']}:{m['path']}" for m in descriptor["files"]],
        artifacts=None if profile == "preparation" else identities,
        campaign=descriptor.get("campaign"),
    )
    output = Path(result["output"])
    assert not result["executable"]
    generated = read_document(output)
    assert generated["files"] == descriptor["files"]
    assert not (output.parent / "not-selected.txt").exists()
    relocated = tmp_path / "relocated"
    shutil.move(output.parent, relocated)
    if profile == "matrix":
        assert inspect_recipe(relocated / "recipe.json")["cells"]
    else:
        bindings = built[1]
        if profile == "preparation":
            from sparselab.experiments.preparation_recipe import (
                inspect_preparation_recipe,
            )

            preview = inspect_preparation_recipe(relocated / "recipe.json", bindings)
        else:
            from sparselab.experiments.workflow_recipe import inspect_workflow

            preview = inspect_workflow(relocated / "recipe.json", bindings)
        assert preview["executable"] is False
        from sparselab.experiments.recipe_export import export_recipe

        assert (
            export_recipe(relocated / "recipe.json", tmp_path / "again")["files"]
            == len(descriptor["files"]) + 1
        )


@pytest.mark.parametrize(
    "damage",
    ["escape", "symlink", "duplicate", "unknown-role", "self", "missing", "malformed"],
)
def test_create_refuses_before_output(tmp_path, damage):
    source = write_recipe(tmp_path / "source")
    files = ["experiment_plan:plans/plan.yaml", "run_config:configs/run.json"]
    if damage == "escape":
        files.append("metadata:../outside.json")
    elif damage == "symlink":
        (source.parent / "link.json").symlink_to(source.parent / "notes.json")
        files.append("metadata:link.json")
    elif damage == "duplicate":
        files.append(files[0])
    elif damage == "unknown-role":
        files.append("invalid:notes.json")
    elif damage == "self":
        (source.parent / "recipe.json").write_text("{}")
        files.append("metadata:recipe.json")
    elif damage == "missing":
        files.append("metadata:absent.json")
    else:
        (source.parent / "notes.json").write_text('{"x":1,"x":2}')
        files.append("metadata:notes.json")
    with pytest.raises((ValueError, OSError)):
        create_recipe(
            source.parent,
            tmp_path / "output",
            profile="matrix",
            recipe_id="test",
            entrypoint="plans/plan.yaml",
            files=files,
        )
    assert not (tmp_path / "output").exists()


def test_create_cli_unrelated_cwd_and_no_clobber(tmp_path):
    source = write_recipe(tmp_path / "source")
    descriptor = read_document(source)
    identities = tmp_path / "identities.json"
    identities.write_text(json.dumps(descriptor["artifacts"]))
    destination = tmp_path / "generated"
    args = [
        "experiment",
        "recipe",
        "create",
        str(source.parent),
        "--profile",
        "matrix",
        "--id",
        "new-study",
        "--entrypoint",
        descriptor["entrypoint"],
        "--output",
        str(destination),
        "--artifacts",
        str(identities),
    ]
    for member in descriptor["files"]:
        args += ["--file", f"{member['role']}:{member['path']}"]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "sparselab", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["executable"] is False
    snapshot = {
        p.relative_to(destination): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }
    result = subprocess.run(
        [sys.executable, "-m", "sparselab", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert snapshot == {
        p.relative_to(destination): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }


def derivative(root, **updates):
    artifact = {
        "kind": "checkpoint",
        "version": 1,
        "producer": "native",
        "identifier": "base-generation",
        "sha256": "b" * 64,
        "path": "base/checkpoints/step-10",
        "state": "full",
    }
    plan = {
        "comparisons": [],
        "artifacts": {"base": artifact},
        "phases": [
            {
                "id": "derivative",
                "transition": "extend_budget",
                "checkpoint": "base",
                "set": {"training.max_steps": 20},
            }
        ],
    }
    plan.update(updates)
    return write_recipe(root, plan_updates=plan)


def test_derivative_checkpoint_missing_then_unverified_and_relocated(
    tmp_path, monkeypatch
):
    source = derivative(tmp_path / "source")
    result = inspect_recipe(source)
    assert result["phases"][0]["consumption"] == "supplied_checkpoint"
    assert result["phases"][0]["validation"] == "declaration_connectivity_only"
    assert (
        next(x for x in result["requirements"] if x["kind"] == "checkpoint")["status"]
        == "unavailable"
    )
    checkpoint = source.parent / "base/checkpoints/step-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "not-a-real-model.bin").write_bytes(b"not loaded")
    assert (
        next(
            x
            for x in inspect_recipe(source)["requirements"]
            if x["kind"] == "checkpoint"
        )["status"]
        == "unverified"
    )
    relocated = tmp_path / "relocated"
    shutil.move(source.parent, relocated)
    monkeypatch.chdir(tmp_path)
    moved = inspect_recipe(relocated / source.name)
    assert moved["recipe_sha256"] == result["recipe_sha256"]
    assert moved["executable"] is False


@pytest.mark.parametrize(
    "damage,match",
    [
        ("missing-name", "pinned external"),
        ("weights", "full checkpoint"),
        ("cycle", "earlier declared"),
        ("phase-path", "path intervention"),
        ("unknown-producer", "unknown producer"),
    ],
)
def test_derivative_refusals_match_native_graph(tmp_path, damage, match):
    source = derivative(tmp_path / "source")
    raw = read_document(source.parent / "plans/plan.yaml")
    if damage == "missing-name":
        raw["phases"][0]["checkpoint"] = "unknown"
    elif damage == "weights":
        raw["artifacts"]["base"]["state"] = "weights"
    elif damage == "cycle":
        raw["phases"] = [
            {
                "id": "a",
                "transition": "resume",
                "parent": "a",
                "selector": "best_validation",
            }
        ]
    elif damage == "phase-path":
        raw["phases"][0]["set"] = {"tokenizer.path": "other.json"}
    else:
        raw["artifacts"]["future"] = {
            "kind": "checkpoint",
            "version": 1,
            "producer": "native",
            "identifier": "future",
            "from_phase": "absent",
            "selector": "terminal",
        }
    update_member(source, "plans/plan.yaml", json.dumps(raw))
    with pytest.raises(ValueError, match=match):
        inspect_recipe(source)
    if damage in {"missing-name", "weights", "cycle"}:
        from sparselab.experiments.lock import _phases
        from sparselab.experiments.plan import ExperimentPlan

        with pytest.raises(ValueError, match=match):
            _phases(ExperimentPlan.model_validate(raw))


def test_derivative_cli_is_metadata_only(tmp_path):
    source = derivative(tmp_path / "source")
    fence = _PROCESS_FENCE + "\nfrom sparselab.cli.entry import main\nmain()\n"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            fence,
            "experiment",
            "recipe",
            "inspect",
            str(source),
            "--json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["phases"][0]["checkpoint"] == "base"


def test_native_phase_projection_is_unchanged(tmp_path):
    from sparselab.experiments.lock import _phases
    from sparselab.experiments.phase_graph import validate_phase_graph
    from sparselab.experiments.plan import ExperimentPlan

    source = derivative(tmp_path / "source")
    plan = ExperimentPlan.model_validate(
        read_document(source.parent / "plans/plan.yaml")
    )
    assert [p.model_dump() for p in _phases(plan)] == [
        {key: value for key, value in p.model_dump().items() if key != "set"}
        for p in validate_phase_graph(plan)
    ]


def test_noncheckpoint_artifacts_and_phase_comparisons_refuse(tmp_path):
    source = derivative(tmp_path / "source")
    raw = read_document(source.parent / "plans/plan.yaml")
    raw["artifacts"]["base"]["kind"] = "tokenizer"
    update_member(source, "plans/plan.yaml", json.dumps(raw))
    with pytest.raises(ValueError, match="only checkpoint"):
        inspect_recipe(source)
    source = derivative(tmp_path / "other")
    raw = read_document(source.parent / "plans/plan.yaml")
    raw["comparisons"] = [
        {
            "id": "comparison",
            "baseline": {"width": "base"},
            "variant": {"width": "wide"},
            "interventions": ["model.ffn_dim"],
        }
    ]
    update_member(source, "plans/plan.yaml", json.dumps(raw))
    with pytest.raises(ValueError, match="phase comparisons"):
        inspect_recipe(source)
