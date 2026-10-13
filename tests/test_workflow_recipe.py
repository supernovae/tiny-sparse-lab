"""Bounded external prepared-input → native lock, with explicit operation fences."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_portable_recipe import _PROCESS_FENCE

from sparselab.experiments.plan import ExperimentPlan, recipe_run_config
from sparselab.experiments.workflow_recipe import (
    inspect_workflow,
    load_workflow,
    publish_workflow,
    resolve_workflow,
)

FIXTURE = Path(__file__).parent / "fixtures/contracts/run_config_v2.json"


def write_workflow(root, config=None, artifacts=None):
    root.mkdir(parents=True)
    (root / "plans").mkdir()
    (root / "configs").mkdir()
    config = config or json.loads(FIXTURE.read_text())
    config["tokenizer"]["path"] = "input/tokenizer.json"
    config["dataset"]["cache_dir"] = "cache"
    config["logging"]["root_dir"] = "runs"
    (root / "configs/run.json").write_text(json.dumps(config))
    plan = {
        "plan_version": 1,
        "id": "external",
        "base_run": "configs/run.json",
        "inputs": {"tokenizer": "token", "prepared_data": "packed"},
        "execution": {"backend": "cpu"},
    }
    (root / "plans/plan.json").write_text(json.dumps(plan))
    (root / "notes.json").write_text('{"plugin":{"path":"../opaque"}}')
    artifacts = artifacts or {
        "token": {
            "kind": "tokenizer",
            "version": 1,
            "producer": "native",
            "identifier": "tok",
            "sha256": "a" * 64,
        },
        "packed": {
            "kind": "prepared_data",
            "version": 1,
            "producer": "native",
            "identifier": "packed",
            "sha256": "b" * 64,
        },
    }
    descriptor = {
        "portable_workflow_version": 1,
        "id": "workflow",
        "entrypoint": "plans/plan.json",
        "contracts": {"experiment_plan": 1, "run_config": 2},
        "artifacts": artifacts,
        "files": [
            {
                "path": name,
                "role": role,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            }
            for name, role in [
                ("plans/plan.json", "experiment_plan"),
                ("configs/run.json", "run_config"),
                ("notes.json", "metadata"),
            ]
        ],
    }
    source = root / "workflow.json"
    source.write_text(json.dumps(descriptor))
    bindings = root / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "workflow_bindings_version": 1,
                "artifacts": {
                    "token": "payload/tok/tokenizer.json",
                    "packed": "payload/cache/packed",
                },
                "logging_root": "runs",
            }
        )
    )
    return source, bindings


def update(path, **changes):
    raw = json.loads(path.read_text())
    raw.update(changes)
    path.write_text(json.dumps(raw))


def repin(source):
    raw = json.loads(source.read_text())
    for member in raw["files"]:
        member["sha256"] = hashlib.sha256(
            (source.parent / member["path"]).read_bytes()
        ).hexdigest()
    source.write_text(json.dumps(raw))


def test_metadata_relocation_binding_identity_and_snapshot_context(
    tmp_path, monkeypatch
):
    source, bindings = write_workflow(tmp_path / "one")
    first = load_workflow(source, bindings)
    shutil.copytree(source.parent, tmp_path / "two")
    second = load_workflow(
        tmp_path / "two/workflow.json", tmp_path / "two/bindings.json"
    )
    assert first.recipe_sha256 == second.recipe_sha256
    assert first.binding_sha256 == second.binding_sha256
    one = recipe_run_config(first.plan, first.source, context=first.context)
    two = recipe_run_config(second.plan, second.source, context=second.context)
    assert one.tokenizer.path == source.parent / "payload/tok/tokenizer.json"
    assert one.tokenizer.path != two.tokenizer.path
    update(bindings, logging_root="other-runs")
    rebound = load_workflow(source, bindings)
    assert rebound.recipe_sha256 == first.recipe_sha256
    assert rebound.binding_sha256 != first.binding_sha256
    # No source-relative/CWD decoy or later disk mutation replaces the snapshot.
    (source.parent / "plans/configs").mkdir()
    (source.parent / "plans/configs/run.json").write_text('{"decoy":true}')
    (source.parent / "configs/run.json").write_text('{"decoy":true}')
    monkeypatch.chdir(tmp_path)
    assert recipe_run_config(first.plan, first.source, context=first.context) == one
    with pytest.raises(ValueError, match="digest mismatch"):
        load_workflow(source, bindings)
    with pytest.raises(TypeError):
        first.context.config_paths["configs/run.json"]["tokenizer.path"] = Path("evil")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("portable_workflow_version", True, "must be integer 1"),
        ("portable_workflow_version", 2, "must be integer 1"),
        ("contracts", {"experiment_plan": True, "run_config": 2}, "valid integer"),
        ("entrypoint", "../plan.json", "unsafe or ambiguous"),
        ("capabilities", {"cpu": "ready"}, "Extra inputs"),
    ],
)
def test_descriptor_refusals(tmp_path, field, value, match):
    source, bindings = write_workflow(tmp_path / "recipe")
    update(source, **{field: value})
    with pytest.raises(ValueError, match=match):
        inspect_workflow(source, bindings)


@pytest.mark.parametrize(
    "damage,match",
    [
        ("duplicate", "duplicate workflow file"),
        ("digest", "digest mismatch"),
        ("unknown_binding", "identity names"),
        ("traversal", "unsafe or ambiguous"),
        ("symlink", "symlinked workflow binding"),
        ("config_binding", "remaining native input paths"),
        ("false_version", "must be integer 1"),
        ("capability", "Extra inputs"),
        ("self_inventory", "cannot inventory itself"),
    ],
)
def test_inventory_and_local_binding_refusals(tmp_path, damage, match):
    source, bindings = write_workflow(tmp_path / "recipe")
    raw = json.loads(source.read_text())
    local = json.loads(bindings.read_text())
    if damage == "duplicate":
        raw["files"].append(raw["files"][-1])
    elif damage == "digest":
        (source.parent / "notes.json").write_text('{"changed":true}')
    elif damage == "unknown_binding":
        local["artifacts"]["unknown"] = "unknown"
    elif damage == "traversal":
        local["artifacts"]["token"] = "../tokenizer.json"
    elif damage == "symlink":
        (source.parent / "payload").symlink_to(tmp_path)
    elif damage == "config_binding":
        local["config_paths"] = {"plugin.path": "unknown"}
    elif damage == "false_version":
        local["workflow_bindings_version"] = True
    elif damage == "capability":
        local["capabilities"] = {"status": "ready"}
    else:
        raw["files"].append(
            {"path": "workflow.json", "role": "metadata", "sha256": "c" * 64}
        )
    source.write_text(json.dumps(raw))
    bindings.write_text(json.dumps(local))
    with pytest.raises(ValueError, match=match):
        load_workflow(source, bindings)


@pytest.mark.parametrize(
    "feature,value",
    [
        ("evaluation_suite", False),
        ("phases", [{"id": "phase"}]),
        ("corpus_variants", [{}]),
        ("execution", {"attempt_contract": False}),
    ],
)
def test_unsupported_workflow_refused_before_optional_imports(tmp_path, feature, value):
    source, bindings = write_workflow(tmp_path / "recipe")
    update(source.parent / "plans/plan.json", **{feature: value})
    repin(source)
    code = (
        _PROCESS_FENCE
        + "\nfrom sparselab.experiments.workflow_recipe import inspect_workflow\ntry:\n inspect_workflow(__import__('pathlib').Path(sys.argv[1]),__import__('pathlib').Path(sys.argv[2]))\nexcept ValueError as e:\n assert 'unavailable workflow profile feature' in str(e), str(e)\nelse:\n raise AssertionError('accepted')\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(source), str(bindings)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key", ["tokenizer.path", "tokenizer", "dataset"])
def test_path_axes_refused(tmp_path, key):
    source, bindings = write_workflow(tmp_path / "recipe")
    update(
        source.parent / "plans/plan.json",
        axes=[
            {"name": "path", "choices": [{"label": "bad", "set": {key: "elsewhere"}}]}
        ],
    )
    repin(source)
    with pytest.raises(ValueError, match="axes cannot change local path"):
        load_workflow(source, bindings)


def test_cli_metadata_and_help_have_no_payload_runtime_or_write_effects(tmp_path):
    source, bindings = write_workflow(tmp_path / "recipe")
    code = (
        _PROCESS_FENCE
        + "\ndef payload_guard(event,args):\n if event == 'open' and '/payload/' in str(args[0]):\n  raise AssertionError('payload opened')\nsys.addaudithook(payload_guard)\nfrom sparselab.cli.entry import main\nmain()\n"
    )
    for args in (
        ["--help"],
        ["inspect", str(source), "--bindings", str(bindings), "--json"],
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                code,
                "--work-dir",
                str(tmp_path / "forbidden-work"),
                "--hf-token-file",
                "/forbidden-token",
                "experiment",
                "recipe",
                "workflow",
                *args,
            ],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        if args[0] == "inspect":
            result = json.loads(result.stdout)
            assert (
                result["runtime_capability"] == "unknown"
                and result["executable"] is False
            )
    assert not (tmp_path / "forbidden-work").exists()


@pytest.fixture
def native_workflow(tmp_path):
    from test_campaign_preparation_plan import configs

    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
    from sparselab.training.manifest import sha256_file

    payload = tmp_path / "payload"
    run, tokenizer = configs(payload)
    path = train_tokenizer(tokenizer)
    prepared = prepare_data(run, load_tokenizer(path))
    manifest = prepared.manifest
    artifacts = {
        "token": {
            "kind": "tokenizer",
            "version": 1,
            "producer": "native",
            "identifier": path.parent.name,
            "sha256": sha256_file(path),
        },
        "packed": {
            "kind": "prepared_data",
            "version": 1,
            "producer": "native",
            "identifier": manifest["settings_sha256"],
            "sha256": manifest["manifest_sha256"],
        },
    }
    source, bindings = write_workflow(
        tmp_path / "recipe", run.model_dump(mode="json"), artifacts
    )
    update(
        bindings,
        artifacts={"token": str(path), "packed": str(prepared.root)},
        logging_root=str(run.logging.root_dir),
    )
    return source, bindings


def test_native_parity_explicit_publication_and_execution_refusal(
    native_workflow, tmp_path, monkeypatch
):
    from sparselab.experiments import cli
    from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan

    source, bindings = native_workflow

    def unexpected_work(*args, **kwargs):
        pytest.fail("resolution/publication attempted computation or runtime admission")

    monkeypatch.setattr("sparselab.data.tokenizer.train_tokenizer", unexpected_work)
    monkeypatch.setattr("sparselab.data.packing.prepare_data", unexpected_work)
    monkeypatch.setattr("sparselab.workers.controller.Controller", unexpected_work)
    monkeypatch.setattr(
        "sparselab.runtime_profile.require_authorization", unexpected_work
    )
    loaded = load_workflow(source, bindings)
    config = recipe_run_config(loaded.plan, loaded.source, context=loaded.context)
    # An ordinary native fixture uses exactly the same bound config and artifacts.
    native = ExperimentPlan.model_validate(
        {**loaded.plan.model_dump(mode="python"), "base_run": config}
    )
    native_source = tmp_path / "native.json"
    native_source.write_text(native.model_dump_json())
    ordinary = resolve_plan(native, native_source)
    lock = resolve_workflow(source, bindings)
    assert lock.scientific_sha256 == ordinary.scientific_sha256
    assert lock.plan_sha256 == ordinary.plan_sha256
    assert lock.availability["declaration_hashes"] == {
        name: hashlib.sha256(data).hexdigest()
        for name, data in loaded.context.documents.items()
    }
    workspace = tmp_path / "published"
    assert not workspace.exists()
    target = publish_workflow(source, bindings, workspace)
    opened = open_lock(target)
    assert opened.model_dump() == lock.model_dump()
    assert publish_lock(opened, workspace) == target
    monkeypatch.setattr(cli, "_verification", lambda args: {})
    monkeypatch.setattr(
        cli,
        "_run_cells",
        lambda *a: pytest.fail("execution selection before profile refusal"),
    )
    with pytest.raises(ValueError, match="^WORKFLOW_EXECUTION_UNAVAILABLE"):
        cli._run(SimpleNamespace(lock=target))
    with pytest.raises(ValueError, match="^WORKFLOW_EXECUTION_UNAVAILABLE"):
        cli.locked_cell_request(opened, opened.cells[0], "worker", workspace, None)


def test_changed_artifact_cannot_resolve_or_publish(native_workflow, tmp_path):
    source, bindings = native_workflow
    token = Path(json.loads(bindings.read_text())["artifacts"]["token"])
    token.write_bytes(token.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest|tokenizer|JSON"):
        publish_workflow(source, bindings, tmp_path / "published")
    assert not (tmp_path / "published").exists()


def test_inventory_order_and_changed_declarations_have_distinct_identities(tmp_path):
    source, bindings = write_workflow(tmp_path / "recipe")
    first = load_workflow(source, bindings)
    update(source, files=list(reversed(first.descriptor.model_dump()["files"])))
    reordered = load_workflow(source, bindings)
    assert reordered.recipe_sha256 == first.recipe_sha256
    assert reordered.descriptor_sha256 != first.descriptor_sha256
    (source.parent / "notes.json").write_text('{"plugin":{"path":"still-opaque"}}')
    repin(source)
    assert load_workflow(source, bindings).recipe_sha256 != first.recipe_sha256


@pytest.mark.parametrize("seed,expected", [(True, 1), ("7", 7), (7.0, 7)])
def test_workflow_preserves_native_seed_acceptance(tmp_path, seed, expected):
    source, bindings = write_workflow(tmp_path / "recipe")
    update(source.parent / "configs/run.json", seed=seed)
    repin(source)
    loaded = load_workflow(source, bindings)
    assert (
        recipe_run_config(loaded.plan, loaded.source, context=loaded.context).seed
        == expected
    )


def test_binding_relocation_preserves_native_science_and_does_not_overwrite_lock(
    native_workflow, tmp_path
):
    source, bindings = native_workflow
    first = resolve_workflow(source, bindings)
    workspace = tmp_path / "locks-work"
    target = publish_workflow(source, bindings, workspace)
    before = target.read_bytes()
    update(bindings, logging_root=str(tmp_path / "different-output"))
    second = resolve_workflow(source, bindings)
    assert first.scientific_sha256 == second.scientific_sha256
    assert first.plan_sha256 == second.plan_sha256
    assert first.availability != second.availability
    with pytest.raises(ValueError, match="different availability binding"):
        publish_workflow(source, bindings, workspace)
    assert target.read_bytes() == before
