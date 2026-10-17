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
@pytest.mark.parametrize("damage", ["descriptor", "inventory", "binding"])
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
    assert_content_inventory(provenance, source, admission)
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
        assert_content_inventory(row["declaration_provenance"], source, admission)
    monkeypatch.setattr(
        engine, "dispatch", lambda *a, **k: pytest.fail("replayed adapter")
    )
    assert all(
        row["state"] == "COMPLETE" for row in engine.apply(resume=True)["stages"]
    )


def assert_content_inventory(provenance, source, admission):
    names = [
        source.name,
        *(member["path"] for member in json.loads(source.read_text())["files"]),
    ]
    expected = {
        name: hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
        for name in names
    }
    declarations = provenance["declarations"]
    assert len(declarations) == len(expected)
    assert {row["path"]: row["sha256"] for row in declarations} == expected
    assert expected[source.name] == admission.reviewed_descriptor_sha256


@pytest.mark.parametrize("initial_content", [False, True])
def test_real_source_mode_switch_refuses_resume(tmp_path, initial_content):
    source, bindings, work, committed = setup(tmp_path)
    content = content_admission(committed, source)
    first, second = (content, committed) if initial_content else (committed, content)
    initial = PreparationExecution(source, bindings, work, first).engine().apply()
    assert all(row["state"] == "COMPLETE" for row in initial["stages"])
    engine = PreparationExecution(source, bindings, work, second).engine()
    with pytest.raises(
        ValueError, match="PREPARATION_CONTEXT_CHANGED: receipt context differs"
    ):
        engine.apply(resume=True)


def test_content_reread_refuses_racing_member_bytes(tmp_path, monkeypatch):
    from sparselab.experiments import recipe_admission

    source, bindings, work, committed = setup(tmp_path)
    admission = content_admission(committed, source)
    context = PreparationExecution(source, bindings, work, admission)
    member = source.parent / json.loads(source.read_text())["files"][0]["path"]
    original = recipe_admission._metadata_bytes

    def changed_on_provenance_reread(path):
        data = original(path)
        return data + b" " if path == member else data

    monkeypatch.setattr(
        recipe_admission, "_metadata_bytes", changed_on_provenance_reread
    )
    with pytest.raises(
        ValueError, match="CONTENT_CONTEXT_CHANGED: declaration bytes differ"
    ):
        context.refresh()
    assert not work.exists()
