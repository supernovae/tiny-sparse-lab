"""Native snapshot → tokenizer → pack, with acquisition replaced by local rows."""

import json
import sys

import pytest
from test_dataset_sources import acquire, inventory
from test_preparation_execution import git, rows, setup
from test_workflow_recipe import repin

from sparselab.cli.entry import main
from sparselab.data import sources
from sparselab.experiments.preparation_execution import (
    preparation_admission,
)
from sparselab.experiments.preparation_recipe import load_preparation_recipe


def snapshot_recipe(tmp_path, monkeypatch):
    # Real native lock/snapshot generation and verification; only acquisition
    # metadata/transport are substituted. No tokenizer/packer/verifier mocks.
    with monkeypatch.context() as acquisition:
        acquisition.setattr(sources, "_hub_identity", inventory)
        acquisition.setattr(
            sources,
            "_stream",
            lambda source, split, cache, **kw: iter(
                {"text": f"{split} local document {i}: a fox walks through the forest."}
                for i in range(4)
            ),
        )
        manifest = acquire(tmp_path / "input")
    source, bindings, work, admission = setup(tmp_path)
    raw_manifest = json.loads(manifest.read_text())
    campaign_path = source.parent / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["stages"].insert(
        0,
        {
            "id": "snapshot",
            "kind": "artifact_reference",
            "scope": "corpus",
            "artifact": {
                "kind": "dataset_snapshot",
                "version": 1,
                "producer": "native fixture",
                "identifier": raw_manifest["lock"]["lock_sha256"],
                "sha256": raw_manifest["manifest_sha256"],
                "path": "snapshot/manifest.json",
            },
        },
    )
    for stage in campaign["stages"][1:]:
        stage["snapshot"] = "snapshot"
        stage.setdefault("requires", []).append("snapshot")
    campaign_path.write_text(json.dumps(campaign))
    local = json.loads(bindings.read_text())
    local["artifacts"]["snapshot"] = str(manifest)
    for name in ("run.yaml", "tokenizer.yaml"):
        path = source.parent / name
        config = json.loads(path.read_text())
        config["dataset"].update(
            source="snapshot",
            train_max_tokens=512,
            validation_max_tokens=256,
            revision="a" * 40,
            license="MIT",
            dataset_config="default",
            train_path="snapshot/train.jsonl",
            validation_path="snapshot/validation.jsonl",
            source_manifest_path="snapshot/manifest.json",
        )
        for field, member in (
            ("train_path", "train.jsonl"),
            ("validation_path", "validation.jsonl"),
            ("source_manifest_path", "manifest.json"),
        ):
            local["config_paths"][name]["dataset." + field] = str(
                manifest.parent / member
            )
        path.write_text(json.dumps(config))
    bindings.write_text(json.dumps(local))
    repin(source)
    git(source.parent, "add", ".")
    git(
        source.parent,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "snapshot preparation",
    )
    loaded = load_preparation_recipe(source, bindings)
    raw = admission.model_dump(mode="json")
    raw.update(
        preparation_admission_version=2,
        reviewed_commit=git(source.parent, "rev-parse", "HEAD"),
        recipe_sha256=loaded.recipe_sha256,
        binding_sha256=loaded.binding_sha256,
        max_snapshot_bytes=1_000_000,
        max_snapshot_records=40,
    )
    document = tmp_path / "admission.json"
    document.write_text(json.dumps(raw))
    return source, bindings, work, document, manifest


def invoke(monkeypatch, capsys, action, args):
    source, bindings, work, admission, _ = args
    argv = [
        "sparselab",
        "experiment",
        "recipe",
        "preparation",
        action,
        str(source),
        "--bindings",
        str(bindings),
        "--json",
    ]
    if action != "inspect":
        argv += ["--work-dir", str(work), "--admission", str(admission)]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return json.loads(capsys.readouterr().out)


def test_public_snapshot_preparation_and_replay(tmp_path, monkeypatch, capsys):
    args = snapshot_recipe(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sources,
        "snapshot_source",
        lambda *a, **kw: pytest.fail("no acquisition in portable preparation"),
    )
    monkeypatch.setattr(
        sources, "_stream", lambda *a, **kw: pytest.fail("no transport")
    )
    inspected = invoke(monkeypatch, capsys, "inspect", args)
    assert inspected["runtime_capability"] == "unknown"
    assert not args[2].exists()
    result = invoke(monkeypatch, capsys, "apply", args)
    assert all(row["state"] == "COMPLETE" for row in result["stages"]), result
    assert rows(result)["snapshot"]["outputs"][0]["kind"] == "dataset_snapshot"
    from sparselab.campaign.engine import CampaignEngine

    monkeypatch.setattr(
        CampaignEngine, "dispatch", lambda *a, **kw: pytest.fail("replayed preparation")
    )
    replay = invoke(monkeypatch, capsys, "resume", args)
    assert {k: v["outputs"] for k, v in rows(replay).items()} == {
        k: v["outputs"] for k, v in rows(result).items()
    }


@pytest.mark.parametrize(
    "damage,match",
    [
        ("v1", "version 2"),
        ("bytes", "byte limit"),
        ("records", "source record limit"),
        ("missing", "missing"),
        ("payload", "digest|mismatch"),
        ("symlink", "symlink"),
        ("license", "revision/license mismatch"),
        ("revision", "revision/license mismatch"),
        ("config", "configuration mismatch"),
        ("location", "compiled snapshot location"),
        ("digest", "domain digest mismatch"),
        ("identifier", "logical identifier mismatch"),
    ],
)
def test_snapshot_admission_refuses_before_output(
    tmp_path, monkeypatch, capsys, damage, match
):
    args = snapshot_recipe(tmp_path, monkeypatch)
    source, bindings, work, admission, manifest = args
    raw = json.loads(admission.read_text())
    if damage == "v1":
        raw["preparation_admission_version"] = 1
        del raw["max_snapshot_bytes"], raw["max_snapshot_records"]
    elif damage in {"bytes", "records"}:
        raw["max_snapshot_" + damage] = 1
    elif damage == "missing":
        manifest.unlink()
    elif damage == "payload":
        with (manifest.parent / "train.jsonl").open("a") as out:
            out.write("{}\n")
    elif damage == "symlink":
        (manifest.parent / "unexpected").symlink_to(manifest)
    elif damage == "location":
        local = json.loads(bindings.read_text())
        local["config_paths"]["run.yaml"]["dataset.train_path"] = str(
            tmp_path / "other"
        )
        bindings.write_text(json.dumps(local))
    else:
        if damage in {"digest", "identifier"}:
            path = source.parent / "campaign.json"
            doc = json.loads(path.read_text())
            doc["stages"][0]["artifact"][
                "sha256" if damage == "digest" else "identifier"
            ] = "0" * 64
        else:
            path = source.parent / "run.yaml"
            doc = json.loads(path.read_text())
            doc["dataset"][
                {
                    "license": "license",
                    "revision": "revision",
                    "config": "dataset_config",
                }[damage]
            ] = "0" * 40 if damage == "revision" else "changed"
        path.write_text(json.dumps(doc))
        repin(source)
        git(source.parent, "add", ".")
        git(
            source.parent,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "changed declaration",
        )
        loaded = load_preparation_recipe(source, bindings)
        raw.update(
            recipe_sha256=loaded.recipe_sha256,
            reviewed_commit=git(source.parent, "rev-parse", "HEAD"),
        )
    admission.write_text(json.dumps(raw))
    monkeypatch.setattr(
        sources,
        "snapshot_source",
        lambda *a, **kw: pytest.fail("unexpected acquisition"),
    )
    with pytest.raises(SystemExit, match=match):
        invoke(monkeypatch, capsys, "apply", args)
    assert not work.exists()


def test_v1_admission_serialization_unchanged(tmp_path):
    _, _, _, old = setup(tmp_path)
    assert (
        preparation_admission(old.model_dump(mode="json")).model_dump_json()
        == old.model_dump_json()
    )
    assert "max_snapshot" not in old.model_dump_json()


def test_snapshot_inspect_does_not_open_payload(tmp_path, monkeypatch):
    import os
    import subprocess

    from test_portable_recipe import _PROCESS_FENCE

    source, bindings, work, _, manifest = snapshot_recipe(tmp_path, monkeypatch)
    code = _PROCESS_FENCE.replace("'sparselab.campaign.plan', ", "")
    code += f"""
def payload_guard(event, args):
    if event == 'open' and {str(manifest.parent)!r} in str(args[0]):
        raise AssertionError('snapshot payload opened during inspection')
sys.addaudithook(payload_guard)
from sparselab.cli.entry import main
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            "experiment",
            "recipe",
            "preparation",
            "inspect",
            str(source),
            "--bindings",
            str(bindings),
            "--json",
        ],
        capture_output=True,
        check=False,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert not work.exists()


def test_snapshot_changed_after_receipt_refuses_replay(tmp_path, monkeypatch, capsys):
    args = snapshot_recipe(tmp_path, monkeypatch)
    result = invoke(monkeypatch, capsys, "apply", args)
    assert all(row["state"] == "COMPLETE" for row in result["stages"])
    from sparselab.campaign.engine import CampaignEngine

    monkeypatch.setattr(
        CampaignEngine,
        "dispatch",
        lambda *a, **kw: pytest.fail("changed input dispatched"),
    )
    with (args[4].parent / "events.jsonl").open("a") as output:
        output.write("{}\n")
    with pytest.raises(SystemExit, match="digest mismatch"):
        invoke(monkeypatch, capsys, "resume", args)
    assert list(args[2].glob("campaigns/**/receipts/**/*.json"))


def test_snapshot_artifact_memo_tracks_sibling_bytes(tmp_path, monkeypatch):
    from sparselab.experiments.artifacts import (
        _reuse_verified_artifact,
        verify_artifact,
    )

    source, bindings, _, _, manifest = snapshot_recipe(tmp_path, monkeypatch)
    loaded = load_preparation_recipe(source, bindings)
    artifact = next(
        s.artifact for s in loaded.plan.stages if s.kind == "artifact_reference"
    )
    artifact = artifact.model_copy(update={"path": str(manifest)})
    memo = {}
    verify_artifact(artifact, source, memo=memo)
    with (manifest.parent / "train.jsonl").open("a") as output:
        output.write("{}\n")
    with pytest.raises(ValueError, match="changed since resolution"):
        _reuse_verified_artifact(artifact, source, memo)
    with pytest.raises(ValueError, match="mismatch"):
        verify_artifact(artifact, source, memo=memo)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_snapshot_bytes", True),
        ("max_snapshot_records", "40"),
        ("max_snapshot_records", 0),
    ],
)
def test_snapshot_limits_are_strict(tmp_path, monkeypatch, field, value):
    args = snapshot_recipe(tmp_path, monkeypatch)
    raw = json.loads(args[3].read_text())
    raw[field] = value
    with pytest.raises(ValueError):
        preparation_admission(raw)


def test_snapshot_rebound_admission_cannot_reuse_receipts(
    tmp_path, monkeypatch, capsys
):
    args = snapshot_recipe(tmp_path, monkeypatch)
    result = invoke(monkeypatch, capsys, "apply", args)
    assert all(row["state"] == "COMPLETE" for row in result["stages"])
    source, bindings, _, document, _ = args
    bindings.write_text(bindings.read_text() + "\n")
    raw = json.loads(document.read_text())
    raw["binding_sha256"] = load_preparation_recipe(source, bindings).binding_sha256
    document.write_text(json.dumps(raw))
    from sparselab.campaign.engine import CampaignEngine

    monkeypatch.setattr(
        CampaignEngine,
        "dispatch",
        lambda *a, **kw: pytest.fail("rebound input dispatched"),
    )
    with pytest.raises(
        SystemExit, match="PREPARATION_CONTEXT_CHANGED: receipt context differs"
    ):
        invoke(monkeypatch, capsys, "resume", args)


def rebind_snapshot(args, destination):
    _, bindings, _, _, _ = args
    local = json.loads(bindings.read_text())
    local["artifacts"]["snapshot"] = str(destination / "manifest.json")
    for fields in local["config_paths"].values():
        for field, name in (
            ("source_manifest_path", "manifest.json"),
            ("train_path", "train.jsonl"),
            ("validation_path", "validation.jsonl"),
        ):
            fields["dataset." + field] = str(destination / name)
    bindings.write_text(json.dumps(local))
    refresh_admission(args)


def refresh_admission(args):
    source, bindings, _, admission, _ = args
    loaded = load_preparation_recipe(source, bindings)
    raw = json.loads(admission.read_text())
    raw.update(recipe_sha256=loaded.recipe_sha256, binding_sha256=loaded.binding_sha256)
    admission.write_text(json.dumps(raw))


@pytest.mark.parametrize(
    "direction", ["output_contains_input", "input_contains_output"]
)
@pytest.mark.parametrize("retained", [False, True])
def test_snapshot_output_overlap_before_writes(
    tmp_path, monkeypatch, capsys, direction, retained
):
    import shutil
    from pathlib import Path

    args = snapshot_recipe(tmp_path, monkeypatch)
    _, bindings, work, _, manifest = args
    local = json.loads(bindings.read_text())
    if direction == "output_contains_input":
        destination = (
            Path(local["config_paths"]["run.yaml"]["logging.root_dir"]) / "snapshot"
        )
    else:
        destination = work / "inputs"
        for fields in local["config_paths"].values():
            fields["dataset.cache_dir"] = str(destination / "cache")
        bindings.write_text(json.dumps(local))
    if retained:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(manifest.parent, destination)
    rebind_snapshot(args, destination)
    before = {
        str(p.relative_to(work)): p.read_bytes() for p in work.rglob("*") if p.is_file()
    }
    with pytest.raises(SystemExit, match="outputs must not overlap snapshot inputs"):
        invoke(monkeypatch, capsys, "apply", args)
    assert {
        str(p.relative_to(work)): p.read_bytes() for p in work.rglob("*") if p.is_file()
    } == before
    if not retained:
        assert not work.exists()
    else:
        assert not (work / "campaigns").exists()
        assert not (work / "scratch").exists()


@pytest.mark.parametrize("direction", ["source_without_port", "port_without_source"])
@pytest.mark.parametrize("route", ["loader", "apply_v1"])
def test_snapshot_source_and_port_must_agree(
    tmp_path, monkeypatch, capsys, direction, route
):
    args = snapshot_recipe(tmp_path, monkeypatch)
    source, bindings, work, admission, _ = args
    if direction == "source_without_port":
        path = source.parent / "campaign.json"
        campaign = json.loads(path.read_text())
        local = json.loads(bindings.read_text())
        del local["artifacts"]["snapshot"]
        bindings.write_text(json.dumps(local))
        campaign["stages"] = campaign["stages"][1:]
        for stage in campaign["stages"]:
            del stage["snapshot"]
            stage["requires"].remove("snapshot")
        path.write_text(json.dumps(campaign))
    else:
        for name in ("run.yaml", "tokenizer.yaml"):
            path = source.parent / name
            config = json.loads(path.read_text())
            from sparselab.local_stories_pins import LICENSE, REVISION

            config["dataset"].update(
                source="local_stories", license=LICENSE, revision=REVISION
            )
            path.write_text(json.dumps(config))
    repin(source)
    git(source.parent, "add", ".")
    git(
        source.parent,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "source port mismatch",
    )
    # Use independently computed descriptor/binding identities so deleting the
    # guard cannot turn this into an unrelated stale-admission failure.
    import hashlib

    from sparselab.experiments.preparation_recipe import PreparationRecipe
    from sparselab.training.manifest import canonical_json

    descriptor = PreparationRecipe.model_validate_json(source.read_text()).model_dump(
        mode="json"
    )
    descriptor["files"] = sorted(descriptor["files"], key=lambda item: item["path"])
    raw = json.loads(admission.read_text())
    raw.update(
        preparation_admission_version=1,
        reviewed_commit=git(source.parent, "rev-parse", "HEAD"),
        recipe_sha256=hashlib.sha256(
            b"sparselab-portable-preparation-v1\0" + canonical_json(descriptor)
        ).hexdigest(),
        binding_sha256=hashlib.sha256(bindings.read_bytes()).hexdigest(),
    )
    del raw["max_snapshot_bytes"], raw["max_snapshot_records"]
    admission.write_text(json.dumps(raw))
    if route == "loader":
        with pytest.raises(ValueError, match="requires a compiled snapshot input"):
            load_preparation_recipe(source, bindings)
    else:
        with pytest.raises(SystemExit, match="requires a compiled snapshot input"):
            invoke(monkeypatch, capsys, "apply", args)
    assert not work.exists()


@pytest.mark.parametrize(
    "field", ["source_manifest_path", "train_path", "validation_path"]
)
def test_snapshot_all_compiled_locations(tmp_path, monkeypatch, field):
    source, bindings, work, _, _ = snapshot_recipe(tmp_path, monkeypatch)
    local = json.loads(bindings.read_text())
    local["config_paths"]["run.yaml"]["dataset." + field] = str(tmp_path / "other")
    bindings.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="compiled snapshot location"):
        load_preparation_recipe(source, bindings)
    assert not work.exists()


def test_snapshot_record_limit_counts_all_splits(tmp_path, monkeypatch, capsys):
    args = snapshot_recipe(tmp_path, monkeypatch)
    source = json.loads(args[4].read_text())["lock"]["source"]
    needed = source["resources"]["max_source_records"] * len(source["splits"])
    assert needed == 40
    raw = json.loads(args[3].read_text())
    raw["max_snapshot_records"] = needed - 1
    args[3].write_text(json.dumps(raw))
    with pytest.raises(SystemExit, match="source record limit"):
        invoke(monkeypatch, capsys, "apply", args)
    assert not args[2].exists()
    raw["max_snapshot_records"] = needed
    args[3].write_text(json.dumps(raw))
    from sparselab.experiments.preparation_execution import PreparationExecution

    PreparationExecution(args[0], args[1], args[2], preparation_admission(raw))
    assert not args[2].exists()


@pytest.mark.parametrize(
    "missing", ["lock", "source", "resources", "max_source_records", "splits"]
)
def test_snapshot_manifest_diagnostic_before_verification(
    tmp_path, monkeypatch, capsys, missing
):
    args = snapshot_recipe(tmp_path, monkeypatch)
    manifest = json.loads(args[4].read_text())
    parent = manifest
    for key in ("lock", "source", "resources", "max_source_records"):
        if missing == "splits" and key == "resources":
            del parent["splits"]
            break
        if key == missing:
            del parent[key]
            break
        parent = parent[key]
    args[4].write_text(json.dumps(manifest))
    from sparselab.experiments import artifacts

    monkeypatch.setattr(
        artifacts,
        "verify_artifact",
        lambda *a, **kw: pytest.fail("malformed manifest reached verification"),
    )
    with pytest.raises(SystemExit, match="snapshot manifest lacks .*lock.source"):
        invoke(monkeypatch, capsys, "apply", args)
    assert not args[2].exists()


@pytest.mark.parametrize(
    "namespace", ["scratch", "campaigns", "execution", "ownership"]
)
def test_snapshot_refuses_operational_storage(tmp_path, monkeypatch, capsys, namespace):
    args = snapshot_recipe(tmp_path, monkeypatch)
    rebind_snapshot(args, args[2] / namespace / "input")
    with pytest.raises(
        SystemExit, match="snapshot input overlaps native operational storage"
    ):
        invoke(monkeypatch, capsys, "apply", args)
    assert not args[2].exists()


def test_dataset_artifact_refuses_verified_legacy_import(tmp_path, monkeypatch):
    from sparselab.data import local_stories, snapshot_import
    from sparselab.experiments.artifacts import _verify_domain
    from sparselab.experiments.plan import Artifact

    monkeypatch.setattr(
        local_stories,
        "_hub_stream",
        lambda split, _: iter({"text": f"{split} row {i}"} for i in range(3)),
    )
    old = local_stories.snapshot(tmp_path / "old", train_count=2, validation_count=1)
    path = snapshot_import.import_legacy_snapshot(old, tmp_path / "imported")
    verified = sources.verify_snapshot(path)
    assert verified["format"] == snapshot_import.FORMAT
    artifact = Artifact(
        kind="dataset_snapshot",
        version=1,
        producer="fixture",
        identifier=verified["lock"]["lock_sha256"],
        sha256=verified["manifest_sha256"],
        path=str(path),
    )
    # Direct domain verification isolates the format guard from authority pins
    # and preparation's flat-directory admission (imports retain legacy/ bytes).
    with pytest.raises(
        ValueError, match="dataset snapshot requires a native source-lock manifest"
    ):
        _verify_domain(artifact, path)


@pytest.mark.parametrize("damage", ["version", "filename"])
def test_dataset_artifact_envelope(tmp_path, monkeypatch, damage):
    from sparselab.experiments.artifacts import _verify_domain

    source, bindings, _, _, manifest = snapshot_recipe(tmp_path, monkeypatch)
    artifact = load_preparation_recipe(source, bindings).plan.stages[0].artifact
    if damage == "version":
        artifact = artifact.model_copy(update={"version": 2})
    else:
        other = manifest.with_name("other.json")
        manifest.rename(other)
        manifest = other
    with pytest.raises(ValueError, match="version 1 and manifest.json"):
        _verify_domain(artifact, manifest)


def test_snapshot_retained_input_under_work_root_is_admitted(tmp_path, monkeypatch):
    import shutil

    from sparselab.experiments.preparation_execution import PreparationExecution

    args = snapshot_recipe(tmp_path, monkeypatch)
    destination = args[2] / "inputs" / "snapshot"
    destination.parent.mkdir(parents=True)
    shutil.copytree(args[4].parent, destination)
    rebind_snapshot(args, destination)
    PreparationExecution(
        args[0],
        args[1],
        args[2],
        preparation_admission(json.loads(args[3].read_text())),
    )
    assert not (args[2] / "scratch").exists()
    assert not (args[2] / "campaigns").exists()


def test_snapshot_import_format_refused_before_domain_verification(
    tmp_path, monkeypatch, capsys
):
    from sparselab.data.snapshot_import import FORMAT
    from sparselab.experiments import artifacts

    args = snapshot_recipe(tmp_path, monkeypatch)
    raw = json.loads(args[4].read_text())
    raw["format"] = FORMAT
    args[4].write_text(json.dumps(raw))
    monkeypatch.setattr(
        artifacts,
        "verify_artifact",
        lambda *a, **kw: pytest.fail("format reached domain verification"),
    )
    with pytest.raises(
        SystemExit, match="snapshot requires a native source-lock manifest"
    ):
        invoke(monkeypatch, capsys, "apply", args)
    assert not args[2].exists()


def test_preparation_cli_does_not_hide_internal_keyerror(tmp_path, monkeypatch, capsys):
    from sparselab.experiments.preparation_execution import PreparationExecution

    args = snapshot_recipe(tmp_path, monkeypatch)

    def broken(*args, **kwargs):
        raise KeyError("internal defect")

    monkeypatch.setattr(PreparationExecution, "engine", broken)
    with pytest.raises(KeyError, match="internal defect"):
        invoke(monkeypatch, capsys, "apply", args)
    assert not args[2].exists()
