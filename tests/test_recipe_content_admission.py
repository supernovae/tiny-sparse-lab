"""Exact-content provenance is distinct from committed Git provenance."""

import hashlib
import json

import pytest
from test_preparation_execution import setup
from test_workflow_execution import admitted, execution_fixture

from sparselab.experiments.preparation_execution import PreparationExecution
from sparselab.experiments.recipe_admission import RecipeAdmission
from sparselab.experiments.workflow_execution import WorkflowExecution


def content_admission(admission, source):
    document = admission.model_dump(mode="json")
    document.pop("reviewed_commit")
    document["reviewed_descriptor_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    return type(admission).model_validate(document)


@pytest.mark.parametrize(
    "value", [{}, {"reviewed_commit": "a" * 40, "reviewed_descriptor_sha256": "b" * 64}]
)
def test_exactly_one_source_mode(value):
    with pytest.raises(ValueError, match="exactly one"):
        RecipeAdmission.model_validate(value)


def test_legacy_serialization_unchanged():
    assert RecipeAdmission(reviewed_commit="a" * 40).model_dump(mode="json") == {
        "reviewed_commit": "a" * 40
    }


@pytest.mark.parametrize("kind", ["preparation", "workflow"])
@pytest.mark.parametrize("damage", ["descriptor", "inventory", "binding", "mode"])
def test_content_context_refresh_refuses_before_dispatch(
    tmp_path, monkeypatch, kind, damage
):
    if kind == "preparation":
        source, bindings, work, admission = setup(tmp_path)
        cls = PreparationExecution
    else:
        source, bindings = execution_fixture(tmp_path / "recipe")
        work = tmp_path / "work"
        admission = admitted(source, bindings, work).admission
        cls = WorkflowExecution
    admission = content_admission(admission, source)
    (source.parent / ".git").rename(tmp_path / "retained-git-metadata")
    context = cls(source, bindings, work, admission)
    provenance = context.refresh()
    assert provenance["status"] == "CONTENT_BOUND"
    assert provenance["source_commit"] is None
    assert all(row["status"] == "CONTENT_BOUND" for row in provenance["declarations"])
    engine = context.engine()
    monkeypatch.setattr(
        engine,
        "dispatch",
        lambda *a, **k: pytest.fail("dispatch after content mutation"),
    )
    if damage == "descriptor":
        source.write_bytes(source.read_bytes() + b"\n")
        message = (
            "descriptor bytes differ"
            if kind == "preparation"
            else "WORKFLOW_CONTEXT_CHANGED: recipe or bindings differ"
        )
    elif damage == "inventory":
        member = json.loads(source.read_text())["files"][0]["path"]
        path = source.parent / member
        path.write_bytes(path.read_bytes() + b"\n")
        message = "digest mismatch"
    elif damage == "binding":
        raw = json.loads(bindings.read_text())
        raw["unexpected"] = "change"
        bindings.write_text(json.dumps(raw))
        message = "extra_forbidden"
    else:
        # Switching modes invalidates existing receipt context even if closure
        # bytes agree; use the engine's ordinary identity guard before dispatch.
        prior = {
            "stages": [
                {
                    "id": "fixture",
                    "receipt": {},
                    "stage_input_sha256": "a",
                    "declaration_provenance": {
                        **provenance,
                        f"{kind}_context": {"different": "mode"},
                    },
                }
            ]
        }
        with pytest.raises(ValueError, match="CONTEXT_CHANGED"):
            engine._assert_declaration_identity(prior, provenance)
        return
    with pytest.raises(ValueError, match=message):
        engine.apply()
    assert not engine.store.root.exists()


def test_plain_directory_preparation_completes_and_replays(tmp_path, monkeypatch):
    source, bindings, work, admission = setup(tmp_path)
    admission = content_admission(admission, source)
    (source.parent / ".git").rename(tmp_path / "retained-git-metadata")
    context = PreparationExecution(source, bindings, work, admission)
    engine = context.engine()
    result = engine.apply()
    assert all(row["state"] == "COMPLETE" for row in result["stages"]), result
    for row in result["stages"]:
        assert row["declaration_provenance"]["status"] == "CONTENT_BOUND"
        assert row["declaration_provenance"]["source_commit"] is None
    monkeypatch.setattr(
        engine, "dispatch", lambda *a, **k: pytest.fail("replayed adapter")
    )
    assert all(
        row["state"] == "COMPLETE" for row in engine.apply(resume=True)["stages"]
    )
