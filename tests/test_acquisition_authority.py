"""Admission and ownership boundaries, without acquisition or model work."""

import dataclasses
import json

import pytest
from test_portable_acquisition import acquisition_recipe
from test_workflow_recipe import repin

from sparselab.experiments.preparation_execution import (
    PreparationExecution,
    preparation_admission,
)
from sparselab.experiments.preparation_recipe import load_preparation_recipe


@pytest.fixture
def context(tmp_path, monkeypatch):
    source, bindings, work, document, _ = acquisition_recipe(tmp_path, monkeypatch, {})
    return PreparationExecution(
        source, bindings, work, preparation_admission(json.loads(document.read_text()))
    )


def changed_paths(context, **changes):
    paths = {
        name: dict(fields) for name, fields in context.loaded.acquisition_paths.items()
    }
    paths["snapshot"].update(changes)
    return dataclasses.replace(context.loaded, acquisition_paths=paths)


@pytest.mark.parametrize(
    "location", ["outside", "root", "campaigns", "scratch", "execution", "ownership"]
)
def test_acquisition_location_authority(context, location):
    if location == "outside":
        path = context.work_dir.parent / "outside"
    elif location == "root":
        path = context.work_dir
    else:
        path = context.work_dir / location / "payload"
    loaded = changed_paths(context, cache_dir=path)
    match = (
        "children of the explicit work root"
        if location in {"outside", "root"}
        else "overlap native operational storage"
    )
    with pytest.raises(ValueError, match=match):
        context._admit_acquisitions(loaded)
    assert not context.work_dir.exists()


def test_acquisition_paths_distinct(context):
    loaded = changed_paths(
        context, cache_dir=context.loaded.acquisition_paths["snapshot"]["output"]
    )
    with pytest.raises(ValueError, match="paths must be distinct"):
        context._admit_acquisitions(loaded)


@pytest.mark.parametrize("empty", [False, True])
def test_admission_exact_stage_authority(context, empty):
    if empty:
        loaded = dataclasses.replace(context.loaded, acquisition_paths={})
        match = "admission has no matching stages"
    else:
        context.admission.acquisitions["extra"] = context.admission.acquisitions[
            "snapshot"
        ]
        loaded = context.loaded
        match = "admissions must name exactly"
    with pytest.raises(ValueError, match=match):
        context._admit_acquisitions(loaded)
    assert not context.work_dir.exists()


@pytest.mark.parametrize(
    "field,nested", [("cache", False), ("cache", True), ("logs", False), ("logs", True)]
)
def test_acquisition_overlaps_preparation_outputs(context, field, nested):
    raw = json.loads(context.bindings.read_text())
    config = context.loaded.configs["pack"]
    path = config.dataset.cache_dir if field == "cache" else config.logging.root_dir
    if nested:
        path /= "child"
    raw["acquisitions"]["snapshot"]["cache_dir"] = str(path)
    context.bindings.write_text(json.dumps(raw))
    loaded = load_preparation_recipe(context.source, context.bindings)
    admitted = context.admission.model_copy(
        update={"binding_sha256": loaded.binding_sha256}
    )
    with pytest.raises(
        ValueError, match="acquisition paths overlap preparation outputs"
    ):
        PreparationExecution(
            context.source, context.bindings, context.work_dir, admitted
        )
    assert not context.work_dir.exists()


@pytest.mark.parametrize(
    "damage,match",
    [
        ("bindings", "bindings must name exactly"),
        ("role", "inventoried source_lock member"),
        ("resume", "explicit Campaign resume"),
    ],
)
def test_recipe_acquisition_authority(context, damage, match):
    if damage == "bindings":
        raw = json.loads(context.bindings.read_text())
        raw["acquisitions"]["extra"] = raw["acquisitions"]["snapshot"]
        context.bindings.write_text(json.dumps(raw))
    elif damage == "role":
        raw = json.loads(context.source.read_text())
        next(member for member in raw["files"] if member["role"] == "source_lock")[
            "role"
        ] = "metadata"
        context.source.write_text(json.dumps(raw))
    else:
        path = context.source.parent / "campaign.json"
        raw = json.loads(path.read_text())
        raw["stages"][0]["resume"] = True
        path.write_text(json.dumps(raw))
        repin(context.source)
    with pytest.raises(ValueError, match=match):
        load_preparation_recipe(context.source, context.bindings)
    assert not context.work_dir.exists()


def test_independent_invocations_exclude_and_release(context):
    second = PreparationExecution(
        context.source, context.bindings, context.work_dir, context.admission
    )
    # Independent contexts open independent file descriptions, even in one process.
    # No scheduling or timeouts are needed to demonstrate LOCK_EX | LOCK_NB.
    with (
        context.acquisition_scope(),
        pytest.raises(BlockingIOError),
        second.acquisition_scope(),
    ):
        pytest.fail("concurrent acquisition admitted")
    with second.acquisition_scope():
        assert second._transports["snapshot"].receipt()["source_charged"] == 0
