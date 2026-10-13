"""Tiny metadata-only portable recipe contract and native-compiler fences."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sparselab.config.models import RunConfig
from sparselab.experiments.compiler import expand_axes
from sparselab.experiments.plan import ExperimentPlan, base_run_config
from sparselab.experiments.recipe import inspect_recipe, load_recipe
from sparselab.recovery.provenance import declaration_reference
from sparselab.training.manifest import config_sha256, source_identity

FIXTURE = Path(__file__).parent / "fixtures/contracts/run_config_v2.json"


def write_recipe(root, *, seed=7, backend="cpu", plan_updates=None):
    root.mkdir(parents=True)
    (root / "plans").mkdir()
    (root / "configs").mkdir()
    config = json.loads(FIXTURE.read_text())
    config["seed"] = seed
    config["runtime"]["backend"] = backend
    (root / "configs/run.json").write_text(json.dumps(config))
    plan = {
        "plan_version": 1,
        "id": "native-matrix",
        "base_run": "configs/run.json",
        "axes": [
            {
                "name": "width",
                "choices": [
                    {"label": "base", "set": {"model.ffn_dim": 32}},
                    {"label": "wide", "set": {"model.ffn_dim": 64}},
                ],
            }
        ],
        "comparisons": [
            {
                "id": "width",
                "baseline": {"width": "base"},
                "variant": {"width": "wide"},
                "interventions": ["model.ffn_dim"],
            }
        ],
    }
    plan.update(plan_updates or {})
    (root / "plans/plan.yaml").write_text(yaml.safe_dump(plan))
    (root / "notes.json").write_text('{"plugin":{"path":"../opaque","future":true}}')
    descriptor = {
        "portable_recipe_version": 1,
        "id": "portable-width",
        "entrypoint": "plans/plan.yaml",
        "contracts": {"experiment_plan": 1, "run_config": 2},
        "files": [
            {
                "path": name,
                "role": role,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            }
            for name, role in [
                ("plans/plan.yaml", "experiment_plan"),
                ("configs/run.json", "run_config"),
                ("notes.json", "metadata"),
            ]
        ],
        "artifacts": {"tokenizer.path": "a" * 64},
    }
    path = root / "recipe.yaml"
    path.write_text(yaml.safe_dump(descriptor))
    return path


def update_member(source, name, text):
    target = source.parent / name
    target.write_text(text)
    descriptor = yaml.safe_load(source.read_text())
    for member in descriptor["files"]:
        if member["path"] == name:
            member["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    source.write_text(yaml.safe_dump(descriptor))


def test_relocation_preserves_recipe_lab_and_native_config_identities(tmp_path):
    source = write_recipe(tmp_path / "first")
    relocated = tmp_path / "second"
    shutil.copytree(source.parent, relocated)
    first = inspect_recipe(source)
    second = inspect_recipe(relocated / "recipe.yaml")
    assert first["recipe_sha256"] == second["recipe_sha256"]
    assert first["installed_lab"] == second["installed_lab"]
    assert first["installed_lab"]["sha256"] == source_identity()["sha256"]
    assert first["binding"] != second["binding"]
    assert [cell["requested_config_sha256"] for cell in first["cells"]] == [
        cell["requested_config_sha256"] for cell in second["cells"]
    ]
    assert first["comparisons"] == second["comparisons"]
    assert first["executable"] is False
    assert first["requirements"][0]["status"] == "unavailable"
    assert first["requirements"][-1]["status"] == "unknown"


def test_recipe_root_beats_cwd_local_and_git_decoys(tmp_path, monkeypatch):
    source = write_recipe(tmp_path / "repo/recipe")
    for root in (tmp_path, tmp_path / "repo", source.parent / "plans"):
        (root / "configs").mkdir(exist_ok=True)
        (root / "configs/run.json").write_text('{"seed":999}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sparselab.recovery.provenance.repository_root",
        lambda *_: pytest.fail("Git discovery reached"),
    )
    preview = inspect_recipe(source)
    assert preview["cells"][0]["config"]["seed"] == 7
    assert preview["binding"]["paths"]["tokenizer.path"] == str(
        source.parent / "configs/artifacts/tokenizer.json"
    )


def test_context_absent_preserves_legacy_lookup_and_seeds(tmp_path, monkeypatch):
    source = write_recipe(tmp_path / "recipe", seed=True)
    plan_path = source.parent / "plans/plan.yaml"
    monkeypatch.setattr(
        "sparselab.recovery.provenance.repository_root", lambda *_: source.parent
    )
    assert (
        declaration_reference(plan_path, "configs/run.json")
        == source.parent / "configs/run.json"
    )
    native = base_run_config(
        ExperimentPlan.model_validate(yaml.safe_load(plan_path.read_text())), plan_path
    )
    assert native.seed == 1
    assert inspect_recipe(source)["cells"][0]["config"]["seed"] == 1


@pytest.mark.parametrize("seed", [True, "7", 7.0, 0, 2**32 - 1])
def test_native_seed_and_compiler_parity(tmp_path, seed):
    source = write_recipe(tmp_path / "recipe", seed=seed)
    loaded = load_recipe(source)
    assert loaded.context.read_document(source.parent / "notes.json") == {
        "plugin": {"path": "../opaque", "future": True}
    }
    result = inspect_recipe(source)
    raw = json.loads((source.parent / "configs/run.json").read_text())
    native = RunConfig.model_validate(raw)
    plan = ExperimentPlan.model_validate(
        loaded.context.read_document(source.parent / "plans/plan.yaml")
    )
    cells = expand_axes(native, plan.axes, max_runs=10, base_dir=source.parent)
    assert [config_sha256(cell.model_dump(mode="json")) for cell, _ in cells] == [
        cell["requested_config_sha256"] for cell in result["cells"]
    ]


def test_backend_remains_scientific_and_recipe_distinction(tmp_path):
    cpu = inspect_recipe(write_recipe(tmp_path / "cpu", backend="cpu"))
    cuda = inspect_recipe(write_recipe(tmp_path / "cuda", backend="cuda"))
    assert cpu["recipe_sha256"] != cuda["recipe_sha256"]
    assert (
        cpu["cells"][0]["requested_config_sha256"]
        != cuda["cells"][0]["requested_config_sha256"]
    )
    # Existing v1 lock science, not a new recipe-derived equivalence domain.
    from sparselab.experiments.lock import _scientific_cell

    def cell(result):
        config = result["cells"][0]["config"]
        return {
            "id": "one",
            "coordinate": {},
            "phase": "main",
            "config": config,
            "requested_runtime": config["runtime"],
            "effective": {},
            "artifacts": {},
        }

    assert _scientific_cell(cell(cpu)) != _scientific_cell(cell(cuda))


def test_changed_bytes_fail_pin_and_resealing_changes_recipe_only(tmp_path):
    source = write_recipe(tmp_path / "recipe")
    first = inspect_recipe(source)
    notes = source.parent / "notes.json"
    notes.write_text(notes.read_text() + "\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        inspect_recipe(source)
    update_member(source, "notes.json", notes.read_text())
    second = inspect_recipe(source)
    assert first["recipe_sha256"] != second["recipe_sha256"]
    assert first["installed_lab"] == second["installed_lab"]


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "bool-version",
        "unknown",
        "duplicate",
        "duplicate-metadata",
        "contracts",
        "bool-contract",
        "undeclared",
        "absolute",
        "traversal",
        "alias-path",
        "bad-artifact",
        "missing-artifact",
        "wrong-role",
    ],
)
def test_descriptor_rejects_ambiguity_and_unavailable_contracts(tmp_path, mutation):
    source = write_recipe(tmp_path / "recipe")
    raw = yaml.safe_load(source.read_text())
    if mutation == "version":
        raw["portable_recipe_version"] = 2
    elif mutation == "bool-version":
        raw["portable_recipe_version"] = True
    elif mutation == "unknown":
        raw["plugin"] = "arbitrary.python:entry"
    elif mutation == "duplicate":
        raw["files"].append({**raw["files"][1], "role": "metadata"})
    elif mutation == "duplicate-metadata":
        raw["files"].append(dict(raw["files"][2]))
    elif mutation == "contracts":
        raw["contracts"]["new_backend"] = 1
    elif mutation == "bool-contract":
        raw["contracts"]["experiment_plan"] = True
    elif mutation == "undeclared":
        raw["files"] = [f for f in raw["files"] if f["role"] != "run_config"]
    elif mutation == "absolute":
        raw["files"][0]["path"] = "/tmp/outside.yaml"
    elif mutation == "traversal":
        raw["files"][0]["path"] = "../outside.yaml"
    elif mutation == "alias-path":
        raw["files"][0]["path"] = "plans//plan.yaml"
    elif mutation == "bad-artifact":
        raw["artifacts"]["plugin.path"] = "b" * 64
    elif mutation == "missing-artifact":
        raw["artifacts"] = {}
    elif mutation == "wrong-role":
        raw["files"][1]["role"] = "metadata"
    source.write_text(yaml.safe_dump(raw))
    expected = {
        "version": "portable_recipe_version must be integer 1",
        "bool-version": "portable_recipe_version must be integer 1",
        "unknown": "Extra inputs are not permitted",
        "duplicate": "duplicate recipe file path",
        "duplicate-metadata": "duplicate recipe file path",
        "contracts": "unavailable recipe contracts",
        "bool-contract": "Input should be a valid integer",
        "undeclared": "undeclared recipe reference",
        "absolute": "unsafe or ambiguous recipe reference",
        "traversal": "unsafe or ambiguous recipe reference",
        "alias-path": "unsafe or ambiguous recipe reference",
        "bad-artifact": "unsupported artifact field: plugin.path",
        "missing-artifact": "recipe artifacts must declare exactly the native input path fields",
        "wrong-role": "base_run must reference a declared run_config",
    }
    with pytest.raises(ValueError, match=re.escape(expected[mutation])):
        inspect_recipe(source)


@pytest.mark.parametrize(
    "field", ["phases", "corpus_variants", "artifacts", "evaluation_suite"]
)
def test_unsupported_composition_is_explicit_not_silently_omitted(tmp_path, field):
    source = write_recipe(tmp_path / "recipe", plan_updates={field: "unsupported"})
    with pytest.raises(
        ValueError, match=f"unavailable recipe preview feature: {field}"
    ):
        inspect_recipe(source)


@pytest.mark.parametrize("kind", ["member", "root", "input"])
def test_symlinks_are_rejected(tmp_path, kind):
    source = write_recipe(tmp_path / "recipe")
    if kind == "member":
        member = source.parent / "notes.json"
        outside = tmp_path / "notes.json"
        member.rename(outside)
        member.symlink_to(outside)
    elif kind == "root":
        link = tmp_path / "linked"
        link.symlink_to(source.parent, target_is_directory=True)
        source = link / "recipe.yaml"
    else:
        (source.parent / "configs/artifacts").symlink_to(
            tmp_path, target_is_directory=True
        )
    with pytest.raises(ValueError, match="symlink"):
        inspect_recipe(source)


@pytest.mark.parametrize("reference", ["../outside", "/tmp/outside"])
def test_native_input_escape_is_rejected(tmp_path, reference):
    source = write_recipe(tmp_path / "recipe")
    raw = json.loads((source.parent / "configs/run.json").read_text())
    raw["tokenizer"]["path"] = reference
    update_member(source, "configs/run.json", json.dumps(raw))
    with pytest.raises(ValueError, match="reference"):
        inspect_recipe(source)


def test_path_axis_cannot_bypass_binding(tmp_path):
    source = write_recipe(
        tmp_path / "recipe",
        plan_updates={
            "axes": [
                {
                    "name": "path",
                    "choices": [
                        {
                            "label": "bad",
                            "set": {"model.memory_package_path": "/outside"},
                        }
                    ],
                }
            ],
            "comparisons": [],
        },
    )
    with pytest.raises(ValueError, match="path intervention"):
        inspect_recipe(source)


def test_snapshot_parsed_is_exactly_the_hashed_bytes(tmp_path):
    source = write_recipe(tmp_path / "recipe")
    loaded = load_recipe(source)
    (source.parent / "configs/run.json").write_text('{"seed":999}')
    assert loaded.context.read_document(source.parent / "configs/run.json")["seed"] == 7
    with pytest.raises(TypeError):
        loaded.context.documents["configs/run.json"] = b"{}"


def test_metadata_read_limits(tmp_path, monkeypatch):
    from sparselab.experiments import recipe

    source = write_recipe(tmp_path / "recipe")
    monkeypatch.setattr(recipe, "MAX_DOCUMENT_BYTES", 16)
    with pytest.raises(ValueError, match="exceeds"):
        load_recipe(source)


@pytest.mark.parametrize("help_only", [False, True])
@pytest.mark.parametrize(
    "prefix",
    [[], ["--work-dir", "forbidden-work", "--hf-token-file", "/forbidden-token"]],
)
def test_real_entry_has_no_execution_heavy_imports_or_writes(
    tmp_path, prefix, help_only
):
    source = write_recipe(tmp_path / "recipe")
    # Existing payloads must remain unopened as well as unverified.
    assets = source.parent / "configs/artifacts"
    assets.mkdir()
    (assets / "tokenizer.json").write_text("payload must not be read")
    code = """
import importlib.abc
import os
import re
import sys
class Fence(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mlx', 'numpy', 'datasets', 'tokenizers', 'huggingface_hub'} or fullname in {'sparselab.runtime', 'sparselab.cli.main', 'sparselab.experiments.lock', 'sparselab.verification_proofs', 'sparselab.campaign.engine'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Fence())
def guard(event, args):
    if event in {'subprocess.Popen', 'os.system', 'socket.connect', 'os.mkdir', 'os.rename', 'os.remove'}:
        raise AssertionError('unexpected effect: ' + event)
    if event == 'open':
        path, mode, flags = args
        if str(path).endswith('/configs/artifacts/tokenizer.json') or str(path) == '/forbidden-token':
            raise AssertionError('payload or credential read')
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            raise AssertionError('write open: ' + str(path))
sys.addaudithook(guard)
from sparselab.cli.entry import main
sys.argv = ['sparselab', *sys.argv[1:]]
main()
"""
    args = [*prefix, "experiment", "recipe", "inspect", str(source), "--json"]
    if help_only:
        args = [*prefix, "experiment", "recipe", "--help"]
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, *args],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    if not help_only:
        output = json.loads(result.stdout)
        assert output["requirements"][0]["status"] == "unverified"
        assert output["executable"] is False
    assert not (tmp_path / "forbidden-work").exists()


@pytest.mark.parametrize(
    "spelling", ["./artifacts/tokenizer.json", "artifacts//tokenizer.json"]
)
def test_native_path_original_spelling_is_unambiguous(tmp_path, spelling):
    source = write_recipe(tmp_path / "recipe")
    config = json.loads((source.parent / "configs/run.json").read_text())
    config["tokenizer"]["path"] = spelling
    update_member(source, "configs/run.json", json.dumps(config))
    with pytest.raises(ValueError, match="ambiguous recipe reference"):
        inspect_recipe(source)


def test_lab_identity_does_not_enter_recipe_fingerprint(tmp_path, monkeypatch):
    source = write_recipe(tmp_path / "recipe")
    first = inspect_recipe(source)
    monkeypatch.setattr(
        "sparselab.experiments.recipe.source_identity",
        lambda: {
            "algorithm": first["installed_lab"]["algorithm"],
            "sha256": "b" * 64,
        },
    )
    second = inspect_recipe(source)
    assert first["recipe_sha256"] == second["recipe_sha256"]
    assert first["binding"] == second["binding"]
    assert first["installed_lab"] != second["installed_lab"]


def test_matrix_cap_uses_native_compiler(tmp_path):
    source = write_recipe(tmp_path / "recipe")
    with pytest.raises(ValueError, match="max_runs"):
        inspect_recipe(source, max_runs=1)


def test_inventory_order_does_not_change_recipe_fingerprint(tmp_path):
    source = write_recipe(tmp_path / "recipe")
    first = load_recipe(source)
    descriptor = yaml.safe_load(source.read_text())
    descriptor["files"].reverse()
    source.write_text(yaml.safe_dump(descriptor))
    reordered = load_recipe(source)
    assert first.fingerprint == reordered.fingerprint
    assert first.descriptor_sha256 != reordered.descriptor_sha256
    assert first.context.documents == reordered.context.documents


def test_total_metadata_limit_counts_descriptor_and_all_members(tmp_path, monkeypatch):
    from sparselab.experiments import recipe

    source = write_recipe(tmp_path / "recipe")
    descriptor = yaml.safe_load(source.read_text())
    sizes = [source.stat().st_size] + [
        (source.parent / member["path"]).stat().st_size
        for member in descriptor["files"]
    ]
    # Tiny fixture only: isolate aggregate accounting from the per-document cap.
    monkeypatch.setattr(recipe, "MAX_DOCUMENT_BYTES", max(sizes))
    monkeypatch.setattr(recipe, "MAX_RECIPE_BYTES", sum(sizes))
    load_recipe(source)
    monkeypatch.setattr(recipe, "MAX_RECIPE_BYTES", sum(sizes) - 1)
    with pytest.raises(
        ValueError, match=f"recipe metadata exceeds {sum(sizes) - 1} bytes"
    ):
        load_recipe(source)


def test_attempt_contract_is_explicitly_unavailable(tmp_path):
    source = write_recipe(
        tmp_path / "recipe",
        plan_updates={
            "execution": {
                "attempt_contract": {"path": "attempt.yaml", "sha256": "a" * 64}
            }
        },
    )
    with pytest.raises(
        ValueError,
        match=r"unavailable recipe preview feature: execution\.attempt_contract",
    ):
        inspect_recipe(source)


def test_descriptor_cannot_inventory_itself_before_digest_validation(tmp_path):
    source = write_recipe(tmp_path / "recipe")
    descriptor = yaml.safe_load(source.read_text())
    descriptor["files"].append(
        {
            "path": source.name,
            "role": "metadata",
            "sha256": "0" * 64,
        }
    )
    source.write_text(yaml.safe_dump(descriptor))
    # No self-referential digest is manufactured: the dedicated refusal must
    # precede the downstream digest mismatch that would otherwise reject this.
    with pytest.raises(ValueError, match="recipe descriptor cannot inventory itself"):
        load_recipe(source)


def test_context_reference_rejects_uninventoried_existing_document(tmp_path):
    source = write_recipe(tmp_path / "recipe")
    context = load_recipe(source).context
    (source.parent / "extra.yaml").write_text("{}")
    with pytest.raises(ValueError, match="undeclared recipe reference: 'extra.yaml'"):
        context.reference(source.parent / "plans/plan.yaml", "extra.yaml")
