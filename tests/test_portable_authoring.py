"""Neutral declaration validation/export; never model execution."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_portable_recipe import _PROCESS_FENCE, write_recipe
from test_preparation_recipe import recipe as preparation_fixture
from test_workflow_recipe import write_workflow

from sparselab.config.validation import validate_config
from sparselab.documents import read_document
from sparselab.experiments.recipe import inspect_recipe
from sparselab.experiments.recipe_export import export_recipe

FIXTURE = Path(__file__).parent / "fixtures/contracts/run_config_v2.json"


@pytest.mark.parametrize("kind", ["model", "optimizer", "run"])
def test_schema_validation_without_runtime_or_effects(tmp_path, kind):
    config = json.loads(FIXTURE.read_text())
    document = config if kind == "run" else config[kind]
    source = tmp_path / "section.json"
    source.write_text(json.dumps(document))
    command = (
        _PROCESS_FENCE
        + "\nfrom sparselab.config.validation import main\nmain("
        + repr([str(source), "--kind", kind, "--json"])
        + ")"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", command],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["schema_valid"] is True and report["executable"] is False
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["limitations"]


def test_validation_cli_and_malformed_config(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text("name: adamw\npeak: 0.001\n")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "sparselab",
            "config",
            "validate",
            str(source),
            "--kind",
            "optimizer",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    assert json.loads(result.stdout)["kind"] == "optimizer"
    source.write_text("name: adamw\nmisspelled_field: 7\n")
    with pytest.raises(ValueError, match="extra_forbidden"):
        validate_config(source, "optimizer")
    source.write_text('{"name":"adamw","name":"adafactor"}')
    with pytest.raises(ValueError, match="duplicate"):
        validate_config(source, "optimizer")


@pytest.mark.parametrize("factory", [write_recipe, preparation_fixture, write_workflow])
def test_export_exact_inventory_without_external_payloads(tmp_path, factory):
    built = factory(tmp_path / "authored")
    source = built[0] if isinstance(built, tuple) else built
    raw = read_document(source)
    (source.parent / "credential.secret").write_text("not an input")
    target = tmp_path / "relocated"
    result = export_recipe(source, target)
    output = Path(result["output"])
    assert output.read_bytes() == source.read_bytes()
    for member in raw["files"]:
        assert (target / member["path"]).read_bytes() == (
            source.parent / member["path"]
        ).read_bytes()
    assert not (target / "credential.secret").exists()
    with pytest.raises(FileExistsError):
        export_recipe(source, target)
    if factory == write_recipe:
        assert (
            inspect_recipe(output)["recipe_sha256"]
            == inspect_recipe(source)["recipe_sha256"]
        )


def test_export_tamper_symlink_and_partial_failure(tmp_path, monkeypatch):
    source = write_recipe(tmp_path / "authored")
    target = tmp_path / "new"
    original = (source.parent / "notes.json").read_bytes()
    (source.parent / "notes.json").write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        export_recipe(source, target)
    assert not target.exists()
    (source.parent / "notes.json").write_bytes(original)
    link = tmp_path / "linked"
    link.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_recipe(source, link / "new")
    opening = Path.open

    def fail_descriptor(path, *args, **kwargs):
        if path == target / source.name:
            raise OSError("fixture write interruption")
        return opening(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_descriptor)
    with pytest.raises(OSError, match="fixture write interruption"):
        export_recipe(source, target)
    assert target.exists() and not (target / source.name).exists()
