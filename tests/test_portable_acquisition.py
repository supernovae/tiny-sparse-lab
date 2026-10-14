"""Public acquisition → native snapshot → tokenizer → pack with loopback HTTP."""

import json
import shutil

import pytest
from test_bounded_acquisition import admission as transport_admission
from test_bounded_acquisition import http_files as http_files  # noqa: PLC0414
from test_bounded_acquisition import pinned_files
from test_portable_snapshot_preparation import invoke
from test_preparation_execution import git, rows, setup
from test_workflow_recipe import repin

from sparselab.experiments.preparation_recipe import load_preparation_recipe


def acquisition_recipe(tmp_path, monkeypatch, http_files):
    lock_path, lock, files = pinned_files(tmp_path / "input", monkeypatch)
    http_files["files"] = files
    source, bindings, work, admission = setup(tmp_path)
    shutil.copyfile(lock_path, source.parent / "source.lock.json")
    descriptor = json.loads(source.read_text())
    descriptor["files"].append(
        {"path": "source.lock.json", "role": "source_lock", "sha256": "0" * 64}
    )
    source.write_text(json.dumps(descriptor))
    campaign_path = source.parent / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["stages"].insert(
        0,
        {
            "id": "snapshot",
            "kind": "dataset_snapshot",
            "scope": "corpus",
            "lock": "source.lock.json",
            "output": "snapshot",
            "cache_dir": "acquisition-cache",
        },
    )
    for stage in campaign["stages"][1:]:
        stage["snapshot"] = "snapshot"
        stage.setdefault("requires", []).append("snapshot")
    campaign_path.write_text(json.dumps(campaign))
    local = json.loads(bindings.read_text())
    local.update(
        acquisitions={
            "snapshot": {
                "output": str(work / "snapshot"),
                "cache_dir": str(work / "acquisition-cache"),
            }
        },
    )
    for name in ("run.yaml", "tokenizer.yaml"):
        path = source.parent / name
        config = json.loads(path.read_text())
        config["dataset"].update(
            source="snapshot",
            train_max_tokens=512,
            validation_max_tokens=256,
            revision=lock["source"]["revision"],
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
                work / "snapshot" / member
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
        "bounded acquisition",
    )
    loaded = load_preparation_recipe(source, bindings)
    raw = admission.model_dump(mode="json")
    raw.update(
        reviewed_commit=git(source.parent, "rev-parse", "HEAD"),
        recipe_sha256=loaded.recipe_sha256,
        binding_sha256=loaded.binding_sha256,
        max_snapshot_bytes=1_000_000,
        max_snapshot_records=40,
        acquisitions={"snapshot": transport_admission(lock).model_dump(mode="json")},
    )
    document = tmp_path / "admission.json"
    document.write_text(json.dumps(raw))
    return source, bindings, work, document, work / "snapshot" / "manifest.json"


def receipt(work):
    return json.loads(
        next(work.glob("execution/acquisition/*/receipt.json")).read_text()
    )


def test_public_acquisition_and_replay(tmp_path, monkeypatch, capsys, http_files):
    args = acquisition_recipe(tmp_path, monkeypatch, http_files)
    inspected = invoke(monkeypatch, capsys, "inspect", args)
    assert inspected["runtime_capability"] == "unknown"
    assert not args[2].exists()
    assert not http_files["requests"]
    import tempfile

    directories = []
    original = tempfile.TemporaryDirectory

    def owned_verification(*a, **kw):
        if kw.get("prefix") == "sparselab-verify-":
            directories.append(kw.get("dir"))
        return original(*a, **kw)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", owned_verification)
    result = invoke(monkeypatch, capsys, "apply", args)
    assert all(row["state"] == "COMPLETE" for row in result["stages"]), result
    assert directories and all(
        path is not None and path.is_relative_to(args[2] / "execution" / "acquisition")
        for path in directories
    )
    first = receipt(args[2])
    assert first["scanned_rows"] == first["scanned_rows_observed"] == 8
    assert len(http_files["requests"]) == 2
    from sparselab.campaign.engine import CampaignEngine

    monkeypatch.setattr(
        CampaignEngine, "dispatch", lambda *a, **kw: pytest.fail("redispatch")
    )
    replay = invoke(monkeypatch, capsys, "resume", args)
    assert {k: v["outputs"] for k, v in rows(replay).items()} == {
        k: v["outputs"] for k, v in rows(result).items()
    }
    final = receipt(args[2])
    assert final["source_charged"] == first["source_charged"]
    assert final["verification_bytes"] > first["verification_bytes"]
    assert len(http_files["requests"]) == 2


def test_public_interrupted_transfer_resume(tmp_path, monkeypatch, capsys, http_files):
    args = acquisition_recipe(tmp_path, monkeypatch, http_files)
    http_files["mode"] = "short"
    result = invoke(monkeypatch, capsys, "apply", args)
    assert rows(result)["snapshot"]["state"] == "INTERRUPTED", result
    first = receipt(args[2])
    assert first["source_charged"] > first["source_actual"]
    http_files["mode"] = "ok"
    resumed = invoke(monkeypatch, capsys, "resume", args)
    assert all(row["state"] == "COMPLETE" for row in resumed["stages"]), resumed
    final = receipt(args[2])
    assert final["source_charged"] > final["source_actual"]
    assert len(final["transfers"]) == 3


def test_bounded_snapshot_retains_native_identity(
    tmp_path, monkeypatch, capsys, http_files
):
    from sparselab.data import sources

    args = acquisition_recipe(tmp_path, monkeypatch, http_files)
    result = invoke(monkeypatch, capsys, "apply", args)
    assert rows(result)["snapshot"]["state"] == "COMPLETE"
    bounded = sources.verify_snapshot(args[4], verification_mode="cold")

    def local_stream(source, split, cache, **kw):
        return sources._stream_parquet(
            [
                str(tmp_path / "input" / row["path"])
                for row in kw["inventory"]["files"][split]
            ],
            kw["inventory"].get("loader_options", {}),
            source,
            cache,
            {},
        )

    monkeypatch.setattr(sources, "_stream", local_stream)
    control = sources.snapshot_source(
        args[0].parent / "source.lock.json",
        tmp_path / "control",
        tmp_path / "control-cache",
    )
    native = sources.verify_snapshot(control, verification_mode="cold")
    assert bounded == native
    assert len(http_files["requests"]) == 2


@pytest.mark.parametrize(
    "damage,match",
    [
        ("missing-authority", "admissions must name exactly"),
        (
            "missing-limits",
            "snapshot operations require explicit byte and record limits",
        ),
        ("authority", "authority differs"),
        ("records", "record admission"),
        ("overlap", "overlap"),
    ],
)
def test_acquisition_admission_before_output(
    tmp_path, monkeypatch, capsys, http_files, damage, match
):
    args = acquisition_recipe(tmp_path, monkeypatch, http_files)
    raw = json.loads(args[3].read_text())
    if damage == "missing-authority":
        raw.pop("acquisitions")
    elif damage == "missing-limits":
        raw.pop("max_snapshot_bytes")
        raw.pop("max_snapshot_records")
    elif damage == "authority":
        raw["acquisitions"]["snapshot"]["lock_sha256"] = "0" * 64
    elif damage == "records":
        raw["max_snapshot_records"] = 39
    else:
        bindings = json.loads(args[1].read_text())
        bindings["acquisitions"]["snapshot"]["cache_dir"] = str(
            args[2] / "snapshot" / "cache"
        )
        args[1].write_text(json.dumps(bindings))
        raw["binding_sha256"] = load_preparation_recipe(args[0], args[1]).binding_sha256
    args[3].write_text(json.dumps(raw))
    monkeypatch.setattr(
        "sparselab.experiments.preparation_execution.PreparationExecution.engine",
        lambda *a: pytest.fail("unexpected engine dispatch"),
    )
    with pytest.raises(SystemExit, match=match):
        invoke(monkeypatch, capsys, "apply", args)
    assert not args[2].exists()
    assert not http_files["requests"]


def test_public_cancel_retains_authority(tmp_path, monkeypatch, capsys, http_files):
    args = acquisition_recipe(tmp_path, monkeypatch, http_files)

    def cancel():
        next(args[2].glob("execution/acquisition/*/transport.sqlite")).with_name(
            "CANCEL"
        ).touch()

    http_files["on_request"] = cancel
    result = invoke(monkeypatch, capsys, "apply", args)
    assert rows(result)["snapshot"]["state"] == "INTERRUPTED"
    charged = receipt(args[2])["source_charged"]
    http_files.pop("on_request")
    next(args[2].glob("execution/acquisition/*/CANCEL")).unlink()
    result = invoke(monkeypatch, capsys, "resume", args)
    assert all(row["state"] == "COMPLETE" for row in result["stages"])
    assert receipt(args[2])["source_charged"] > charged


def test_public_publication_commit_crash_recovery(
    tmp_path, monkeypatch, capsys, http_files
):
    from sparselab.campaign.state import CampaignStore

    args = acquisition_recipe(tmp_path, monkeypatch, http_files)
    original = CampaignStore.commit

    def crash(self, stage, row):
        if stage.id == "snapshot":
            raise KeyboardInterrupt("published before receipt commit")
        return original(self, stage, row)

    monkeypatch.setattr(CampaignStore, "commit", crash)
    with pytest.raises(KeyboardInterrupt):
        invoke(monkeypatch, capsys, "apply", args)
    assert args[4].exists()
    first = receipt(args[2])
    monkeypatch.setattr(CampaignStore, "commit", original)
    result = invoke(monkeypatch, capsys, "resume", args)
    assert all(row["state"] == "COMPLETE" for row in result["stages"]), result
    assert receipt(args[2])["source_charged"] == first["source_charged"]
    assert len(http_files["requests"]) == 2


def test_acquisition_inspect_is_metadata_only(tmp_path, monkeypatch, http_files):
    import os
    import subprocess
    import sys

    from test_portable_recipe import _PROCESS_FENCE

    args = acquisition_recipe(tmp_path, monkeypatch, http_files)
    fence = _PROCESS_FENCE.replace("'sparselab.campaign.plan', ", "")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            fence + "\nfrom sparselab.cli.entry import main\nmain()\n",
            "--hf-token-file",
            "/forbidden-token",
            "experiment",
            "recipe",
            "preparation",
            "inspect",
            str(args[0]),
            "--bindings",
            str(args[1]),
            "--json",
        ],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not args[2].exists()
    assert not http_files["requests"]


def test_acquisition_grants_without_snapshot_stage_refuse_before_output(
    tmp_path, monkeypatch, capsys
):
    source, bindings, work, admission = setup(tmp_path)
    grant = transport_admission(
        {
            "source": {"repo_id": "fixture/data", "revision": "a" * 40},
            "lock_sha256": "b" * 64,
        }
    )
    admitted = admission.model_copy(update={"acquisitions": {"unexpected": grant}})
    document = tmp_path / "admission.json"
    document.write_text(admitted.model_dump_json())
    # Synthetic graph has no snapshot operation, valid provenance and bindings.
    assert not load_preparation_recipe(source, bindings).acquisition_paths
    monkeypatch.setattr("socket.socket.connect", lambda *a: pytest.fail("network"))
    monkeypatch.setattr(
        "sparselab.experiments.preparation_execution.PreparationExecution.engine",
        lambda *a: pytest.fail("unexpected engine dispatch"),
    )
    with pytest.raises(
        SystemExit, match="acquisition admission has no matching stages"
    ):
        invoke(monkeypatch, capsys, "apply", (source, bindings, work, document, None))
    assert not work.exists()


@pytest.mark.parametrize("damage", ["missing", "extra"])
def test_acquisition_bindings_exact_stage_names(tmp_path, monkeypatch, damage):
    if damage == "missing":
        source, bindings, work, _, _ = acquisition_recipe(tmp_path, monkeypatch, {})
        assert set(load_preparation_recipe(source, bindings).acquisition_paths) == {
            "snapshot"
        }
    else:
        source, bindings, work, _ = setup(tmp_path)
        assert not load_preparation_recipe(source, bindings).acquisition_paths
    local = json.loads(bindings.read_text())
    if damage == "missing":
        local.pop("acquisitions")
    else:
        local["acquisitions"] = {
            "unexpected": {
                "output": str(work / "snapshot"),
                "cache_dir": str(work / "acquisition-cache"),
            }
        }
    bindings.write_text(json.dumps(local))
    monkeypatch.setattr("socket.socket.connect", lambda *a: pytest.fail("network"))
    with pytest.raises(
        ValueError, match="^acquisition bindings must name exactly the snapshot stages$"
    ):
        load_preparation_recipe(source, bindings)
    assert not work.exists()
