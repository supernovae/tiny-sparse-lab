"""Current preparation capability contract; no runtime or payload work."""

import argparse
import json
from pathlib import Path

import pytest
from test_preparation_execution import setup
from test_preparation_recipe import recipe

from sparselab.experiments import preparation_execution, preparation_recipe, recipe_cli


@pytest.mark.parametrize("version", [1, 2, 3, True, "3", None])
@pytest.mark.parametrize("command", ["apply", "resume"])
def test_obsolete_admission_refuses_before_context(
    tmp_path, monkeypatch, version, command
):
    path = tmp_path / "admission.json"
    path.write_text(json.dumps({"preparation_admission_version": version}))
    monkeypatch.setattr(
        preparation_execution,
        "PreparationExecution",
        lambda *a, **kw: pytest.fail("obsolete admission reached context"),
    )
    with pytest.raises(SystemExit, match="obsolete preparation admission; regenerate"):
        recipe_cli.handle_preparation_execution(
            argparse.Namespace(
                source=tmp_path / "missing-recipe",
                bindings=tmp_path / "missing-bindings",
                work_dir=tmp_path / "work",
                admission=path,
                preparation_command=command,
            )
        )
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("version", [1, 2, True, "2", None])
def test_obsolete_bindings_refuse_before_compilation(tmp_path, monkeypatch, version):
    source, bindings = recipe(tmp_path / "recipe")
    raw = json.loads(bindings.read_text())
    raw["preparation_bindings_version"] = version
    bindings.write_text(json.dumps(raw))
    monkeypatch.setattr(
        preparation_recipe,
        "compile_preparation",
        lambda *a, **kw: pytest.fail("obsolete bindings reached graph compilation"),
    )
    with pytest.raises(ValueError, match="obsolete preparation bindings; regenerate"):
        preparation_recipe.load_preparation_recipe(source, bindings)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_snapshot_bytes": 1},
        {"max_snapshot_records": 1},
        {"max_snapshot_bytes": True, "max_snapshot_records": 1},
        {"max_snapshot_bytes": 1, "max_snapshot_records": "1"},
        {"max_snapshot_bytes": 0, "max_snapshot_records": 1},
    ],
)
def test_snapshot_limits_are_paired_and_strict(tmp_path, limits):
    *_, admission = setup(tmp_path)
    with pytest.raises(ValueError):
        preparation_execution.preparation_admission(
            {**admission.model_dump(), **limits}
        )


@pytest.mark.parametrize(
    "model,name",
    [
        (preparation_execution.PreparationAdmission, "preparation-admission"),
        (preparation_recipe.PreparationBindings, "preparation-bindings"),
    ],
)
def test_current_schema_snapshot(model, name):
    path = Path(__file__).resolve().parents[1] / "schemas" / f"{name}.schema.json"
    assert json.loads(path.read_text()) == model.model_json_schema()
