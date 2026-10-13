"""External preparation intent stays metadata, using the native graph authority."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_campaign_preparation_plan import declaration
from test_portable_recipe import _PROCESS_FENCE
from test_workflow_recipe import repin, update

from sparselab.campaign.plan import CampaignPlan
from sparselab.campaign.preparation_plan import (
    BoundPreparationInput,
    PreparationArtifact,
    PreparationRequest,
    compile_preparation,
)
from sparselab.experiments.preparation_recipe import (
    inspect_preparation_recipe,
    load_preparation_recipe,
)

FIXTURE = Path(__file__).parent / "fixtures/contracts/run_config_v2.json"


def recipe(root, reuse=False):
    root.mkdir()
    run = json.loads(FIXTURE.read_text())
    run["tokenizer"]["path"] = "tok/tokenizer.json"
    tokenizer = {
        "schema_version": 1,
        "vocab_size": 512,
        "max_documents": 8,
        "output_dir": "tok",
        "dataset": run["dataset"],
    }
    docs = {
        "campaign.json": declaration(reuse),
        "run.yaml": run,
        "tokenizer.yaml": tokenizer,
        "notes.json": {"plugin": {"path": "../opaque"}},
    }
    roles = {
        "campaign.json": "campaign",
        "run.yaml": "run_config",
        "tokenizer.yaml": "tokenizer_config",
        "notes.json": "metadata",
    }
    for name, value in docs.items():
        (root / name).write_text(json.dumps(value))
    source = root / "recipe.json"
    source.write_text(
        json.dumps(
            {
                "portable_preparation_version": 1,
                "id": "prep",
                "entrypoint": "campaign.json",
                "contracts": {"campaign": 1, "run_config": 2, "tokenizer_config": 1},
                "files": [
                    {
                        "path": name,
                        "role": roles[name],
                        "sha256": hashlib.sha256(
                            (root / name).read_bytes()
                        ).hexdigest(),
                    }
                    for name in docs
                ],
            }
        )
    )
    paths = {
        "run.yaml": {
            "tokenizer.path": "payload/tok/tokenizer.json",
            "dataset.cache_dir": "payload/cache",
            "logging.root_dir": "payload/runs",
        }
    }
    if not reuse:
        paths["tokenizer.yaml"] = {
            "output_dir": "payload/tok",
            "dataset.cache_dir": "payload/cache",
        }
    bindings = root / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "preparation_bindings_version": 1,
                "config_paths": paths,
                "artifacts": {"reuse": "payload/tok/tokenizer.json"} if reuse else {},
            }
        )
    )
    return source, bindings


@pytest.mark.parametrize("reuse", [False, True])
def test_graph_context_matches_native_and_can_supply_native_request(
    tmp_path, monkeypatch, reuse
):
    source, bindings = recipe(tmp_path / "recipe", reuse)
    loaded = load_preparation_recipe(source, bindings)
    native = CampaignPlan.model_validate(
        json.loads((source.parent / "campaign.json").read_text())
    )
    assert loaded.plan == native
    assert loaded.compiled == compile_preparation(native)
    op = loaded.compiled.operation("pack")
    assert op.inputs[0].producer == ("reuse" if reuse else "train")
    assert loaded.compiled.ordered_stage_ids == (op.inputs[0].producer, "pack")
    # Future output identity enters only after an upstream verifier, never in preview.
    request = PreparationRequest(
        operation=op,
        source=source.parent / "campaign.json",
        config=loaded.configs["pack"],
        inputs=(
            BoundPreparationInput(
                reference=op.inputs[0],
                path=loaded.configs["pack"].tokenizer.path,
                artifact=PreparationArtifact(
                    kind="tokenizer", identifier="verified-fixture", sha256="a" * 64
                ),
            ),
        ),
    )
    assert request.operation == op and request.config == loaded.configs["pack"]
    from types import SimpleNamespace

    from sparselab.campaign import preparation
    from sparselab.data import packing, tokenizer

    calls = []
    sentinel = object()

    def load_tokenizer(path):
        assert path == request.inputs[0].path
        calls.append("tokenizer")
        return sentinel

    def pack(config, token, **options):
        assert config == request.config and token is sentinel
        calls.append("pack")
        return SimpleNamespace(root=config.dataset.cache_dir / "fixture")

    def verify(bound, path, **options):
        assert bound is request
        assert path == request.config.dataset.cache_dir / "fixture"
        calls.append("verify")
        return PreparationArtifact(
            kind="prepared_data", identifier="fixture", sha256="b" * 64
        )

    monkeypatch.setattr(tokenizer, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(
        tokenizer,
        "train_tokenizer",
        lambda *a, **kw: pytest.fail("unexpected downstream train"),
    )
    monkeypatch.setattr(packing, "prepare_data", pack)
    monkeypatch.setattr(preparation, "_artifact", verify)
    assert preparation.dispatch(request).artifact.kind == "prepared_data"
    assert calls == ["tokenizer", "pack", "verify"]
    preview = inspect_preparation_recipe(source, bindings)
    assert preview["committed_provenance"] == "unverified"
    assert preview["runtime_capability"] == "unknown" and not preview["executable"]
    assert "sha256" not in preview["plan"]["operations"][-1]["output"]
    assert not (source.parent / "payload").exists()


def test_relocation_inventory_order_snapshot_and_opaque_paths(tmp_path, monkeypatch):
    source, bindings = recipe(tmp_path / "one")
    loaded = load_preparation_recipe(source, bindings)
    shutil.copytree(source.parent, tmp_path / "two")
    moved = load_preparation_recipe(
        tmp_path / "two/recipe.json", tmp_path / "two/bindings.json"
    )
    assert moved.recipe_sha256 == loaded.recipe_sha256
    assert moved.configs["pack"].tokenizer.path != loaded.configs["pack"].tokenizer.path
    assert moved.compiled == loaded.compiled
    update(source, files=list(reversed(json.loads(source.read_text())["files"])))
    assert (
        load_preparation_recipe(source, bindings).recipe_sha256 == loaded.recipe_sha256
    )
    monkeypatch.chdir(tmp_path)
    (source.parent / "run.yaml").write_text("{}")
    assert loaded.context.read_document(source.parent / "run.yaml")["seed"] == 7
    assert (
        loaded.context.read_document(source.parent / "notes.json")["plugin"]["path"]
        == "../opaque"
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        load_preparation_recipe(source, bindings)


@pytest.mark.parametrize(
    "damage,match",
    [
        ("version", "less than or equal"),
        ("boolean", "valid integer"),
        ("operation", "unsupported preparation operation"),
        ("snapshot", "snapshot context"),
        ("recovery", "recovery context"),
        ("cycle", "cycle"),
        ("duplicate", "duplicate campaign"),
        ("edge", "must be in requires"),
        ("missing", "requires unknown"),
        ("producer", "wrong producer kind"),
        ("reference", "undeclared config"),
        ("role", "run_config member"),
        ("paths", "exactly the declared native paths"),
        ("tokenizer", "compiled producer location"),
        ("traversal", "unsafe"),
        ("artifact", "exactly the external producer"),
        ("contract", "unsupported preparation contracts"),
    ],
)
def test_refusals(tmp_path, damage, match):
    source, bindings = recipe(tmp_path / "recipe")
    campaign = source.parent / "campaign.json"
    raw = json.loads(campaign.read_text())
    local = json.loads(bindings.read_text())
    if damage == "version":
        update(source, portable_preparation_version=2)
    elif damage == "boolean":
        update(source, portable_preparation_version=True)
    elif damage == "operation":
        raw["stages"][0]["kind"] = "runtime_acceptance"
    elif damage == "snapshot":
        raw["stages"][0]["snapshot"] = "acquire"
    elif damage == "recovery":
        raw["recovery"] = "recovery.json"
    elif damage == "cycle":
        raw["stages"][1]["requires"] = ["pack"]
    elif damage == "duplicate":
        raw["stages"].append(raw["stages"][0])
    elif damage == "edge":
        raw["stages"][0]["requires"] = []
    elif damage == "missing":
        raw["stages"][0]["requires"] = ["absent"]
    elif damage == "producer":
        raw["stages"][0]["tokenizer"] = "other"
        raw["stages"][0]["requires"] = ["other"]
        raw["stages"].append(
            {
                **raw["stages"][0],
                "id": "other",
                "tokenizer": "train",
                "requires": ["train"],
            }
        )
    elif damage == "reference":
        raw["stages"][0]["config"] = "absent.json"
        local["config_paths"]["absent.json"] = local["config_paths"].pop("run.yaml")
    elif damage == "role":
        descriptor = json.loads(source.read_text())
        next(m for m in descriptor["files"] if m["path"] == "run.yaml")["role"] = (
            "metadata"
        )
        source.write_text(json.dumps(descriptor))
    elif damage == "paths":
        local["config_paths"]["run.yaml"]["plugin.path"] = "opaque"
    elif damage == "tokenizer":
        local["config_paths"]["run.yaml"]["tokenizer.path"] = (
            "payload/decoy/tokenizer.json"
        )
    elif damage == "traversal":
        local["config_paths"]["run.yaml"]["tokenizer.path"] = "../escape"
    elif damage == "artifact":
        local["artifacts"]["extra"] = "payload/tok/tokenizer.json"
    elif damage == "contract":
        update(source, contracts={"campaign": 2})
    campaign.write_text(json.dumps(raw))
    bindings.write_text(json.dumps(local))
    repin(source)
    with pytest.raises((ValueError, TypeError), match=match):
        load_preparation_recipe(source, bindings)
    assert not (source.parent / "payload").exists()


def test_symlink_overlap_and_duplicate_document_refusals(tmp_path):
    source, bindings = recipe(tmp_path / "recipe")
    with pytest.raises(ValueError, match="separate"):
        load_preparation_recipe(source, source)
    with pytest.raises(ValueError, match="separate"):
        load_preparation_recipe(source, source.parent / "run.yaml")
    (source.parent / "payload").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_preparation_recipe(source, bindings)
    (source.parent / "payload").unlink()
    (source.parent / "run.yaml").write_text('{"seed":1,"seed":2}')
    repin(source)
    with pytest.raises(ValueError, match="duplicate"):
        load_preparation_recipe(source, bindings)


@pytest.mark.parametrize("seed,expected", [(True, 1), ("7", 7), (7.0, 7)])
def test_legacy_seeds_preserved(tmp_path, seed, expected):
    source, bindings = recipe(tmp_path / "recipe")
    update(source.parent / "run.yaml", seed=seed)
    repin(source)
    assert load_preparation_recipe(source, bindings).configs["pack"].seed == expected


def test_cli_has_no_compute_import_payload_write_network_or_runtime(tmp_path):
    source, bindings = recipe(tmp_path / "recipe")
    fence = _PROCESS_FENCE.replace("'sparselab.campaign.plan', ", "")
    code = (
        fence
        + "\ndef payload_guard(event,args):\n if event == 'open' and '/payload/' in str(args[0]):\n  raise AssertionError('payload opened')\nsys.addaudithook(payload_guard)\nfrom sparselab.cli.entry import main\nmain()\n"
    )
    for args in [
        ["--help"],
        ["inspect", str(source), "--bindings", str(bindings), "--json"],
    ]:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                code,
                "--work-dir",
                str(tmp_path / "forbidden"),
                "--hf-token-file",
                "/forbidden-token",
                "experiment",
                "recipe",
                "preparation",
                *args,
            ],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert not (tmp_path / "forbidden").exists()


def test_content_identity_and_local_binding_identity_are_distinct(tmp_path):
    source, bindings = recipe(tmp_path / "recipe")
    original = load_preparation_recipe(source, bindings)
    local = json.loads(bindings.read_text())
    local["config_paths"]["run.yaml"]["logging.root_dir"] = "different-output"
    bindings.write_text(json.dumps(local))
    rebound = load_preparation_recipe(source, bindings)
    assert rebound.recipe_sha256 == original.recipe_sha256
    assert rebound.binding_sha256 != original.binding_sha256
    update(source.parent / "run.yaml", seed=8)
    repin(source)
    changed = load_preparation_recipe(source, bindings)
    assert changed.recipe_sha256 != original.recipe_sha256
    assert changed.binding_sha256 == rebound.binding_sha256


@pytest.mark.parametrize(
    "path", ["./tok/tokenizer.json", "tok//tokenizer.json", "/absolute/tokenizer.json"]
)
def test_authored_paths_are_checked_before_path_normalization(tmp_path, path):
    source, bindings = recipe(tmp_path / "recipe")
    update(source.parent / "run.yaml", tokenizer={"path": path})
    repin(source)
    with pytest.raises(ValueError, match="unsafe or ambiguous"):
        load_preparation_recipe(source, bindings)
