"""Operation-local authentication of real generic snapshots."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
import test_campaign_snapshot_lifecycle as campaign_fixtures
import test_dataset_sources as dataset_fixtures
import yaml
from test_dataset_sources import acquire, declaration, runtime

from sparselab.data import sources


@pytest.fixture
def offline(monkeypatch):
    return dataset_fixtures.offline.__wrapped__(monkeypatch)


@pytest.fixture
def snapshot_campaign(tmp_path, monkeypatch):
    return campaign_fixtures.snapshot_campaign.__wrapped__(tmp_path, monkeypatch)


def instrument(monkeypatch):
    counts = {"cold": 0, "preflight": 0, "payload": 0}
    for name, counter in [
        ("_verify_snapshot_cold", "cold"),
        ("_snapshot_preflight", "preflight"),
        ("sha256_file", "payload"),
    ]:
        original = getattr(sources, name)

        def wrapped(*args, _original=original, _counter=counter, **kwargs):
            counts[_counter] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(sources, name, wrapped)
    return counts


def test_reuse_preflights_and_fresh_manifest(tmp_path, offline, monkeypatch):
    path = acquire(tmp_path)
    counts = instrument(monkeypatch)
    assert sources.snapshot_verification_statistics() is None
    with sources.snapshot_verification_operation():
        first = sources.verify_snapshot(path)
        identity = first["manifest_sha256"]
        payload_calls = counts["payload"]
        first["splits"]["train"]["sha256"] = "corrupted caller object"
        second = sources.verify_snapshot(runtime(path))
        assert second["manifest_sha256"] == identity
        assert second["splits"]["train"]["sha256"] != "corrupted caller object"
        assert counts == {"cold": 1, "preflight": 2, "payload": payload_calls}
        stats = sources.snapshot_verification_statistics()
        assert stats["calls"] == 2 and stats["reuse_hits"] == 1
        assert stats["cold_verifications"] == 1
        assert stats["verification_seconds"] >= stats["cold_verification_seconds"] >= 0
    assert sources.snapshot_verification_statistics() is None
    with sources.snapshot_verification_operation():
        sources.verify_snapshot(path)
    assert counts["cold"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "b" * 40),
        ("license", "other"),
        ("dataset_config", "other"),
        ("train_path", "train-copy.jsonl"),
        ("validation_path", "validation-copy.jsonl"),
    ],
)
def test_config_preflight_never_reused(tmp_path, offline, field, value):
    path = acquire(tmp_path)
    if field.endswith("path"):
        other = tmp_path / value
        other.write_bytes((path.parent / field.replace("_path", ".jsonl")).read_bytes())
        value = other
    with sources.snapshot_verification_operation():
        sources.verify_snapshot(path)
        with pytest.raises(ValueError, match="mismatch"):
            sources.verify_snapshot(runtime(path).model_copy(update={field: value}))
        sources.verify_snapshot(path)
        assert sources.snapshot_verification_statistics()["cold_verifications"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "same_size",
        "restored_mtime",
        "inode",
        "mode",
        "extra",
        "missing",
        "symlink",
        "manifest",
    ],
)
def test_changed_tree_not_authorized(tmp_path, offline, monkeypatch, mutation):
    path = acquire(tmp_path)
    counts = instrument(monkeypatch)
    target = path.parent / "train.jsonl"
    with sources.snapshot_verification_operation():
        sources.verify_snapshot(path)
        before = target.stat()
        if mutation in {"same_size", "restored_mtime"}:
            target.write_bytes(target.read_bytes().replace(b"train one", b"wrong one"))
            if mutation == "restored_mtime":
                os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif mutation == "inode":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(target.read_bytes())
            replacement.replace(target)
        elif mutation == "mode":
            target.chmod(0o600 if before.st_mode & 0o777 != 0o600 else 0o644)
        elif mutation == "extra":
            (path.parent / "extra").write_text("unexpected")
        elif mutation == "missing":
            target.unlink()
        elif mutation == "symlink":
            original = tmp_path / "original"
            target.replace(original)
            target.symlink_to(original)
        else:
            path.write_bytes(path.read_bytes() + b" ")
        if mutation in {"inode", "mode", "manifest"}:
            sources.verify_snapshot(path)
            assert counts["cold"] == 2
            assert sources.snapshot_verification_statistics()["invalidations"] == 1
        else:
            with pytest.raises((ValueError, OSError)):
                sources.verify_snapshot(path)


def test_unused_split_and_symlink_ancestor(tmp_path, offline):
    offline["test"] = [{"text": "extra split"}]
    path = acquire(
        tmp_path / "input",
        declaration(
            splits={"train": "training", "validation": "heldout", "test": "testing"},
            dedup={"priority": ["validation", "train", "test"]},
        ),
    )
    with sources.snapshot_verification_operation():
        sources.verify_snapshot(path)
        (path.parent / "test.jsonl").write_bytes(b"changed\n")
        with pytest.raises(ValueError):
            sources.verify_snapshot(runtime(path))
    alias = tmp_path / "alias"
    alias.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        sources.verify_snapshot(alias / path.name)


def test_cold_dominates_nested_scope(tmp_path, offline, monkeypatch):
    path = acquire(tmp_path)
    counts = instrument(monkeypatch)
    with sources.snapshot_verification_operation():
        sources.verify_snapshot(path)
        sources.verify_snapshot(path, verification_mode="cold")
        with (
            sources.snapshot_verification_operation(mode="cold"),
            sources.snapshot_verification_operation(),
        ):
            sources.verify_snapshot(path, verification_mode="verified_reuse")
            sources.verify_snapshot(path)
        sources.verify_snapshot(path)
        assert counts["cold"] == 4
        assert sources.snapshot_verification_statistics()["reuse_hits"] == 1
    sources.verify_snapshot(path, verification_mode="verified_reuse")
    sources.verify_snapshot(path)
    assert counts["cold"] == 6


def test_mutation_during_cold_verification_aborts(tmp_path, offline, monkeypatch):
    path = acquire(tmp_path)
    original = sources._verify_snapshot_cold

    def changing(*args):
        result = original(*args)
        target = path.parent / "train.jsonl"
        target.chmod(0o600)
        return result

    monkeypatch.setattr(sources, "_verify_snapshot_cold", changing)
    with sources.snapshot_verification_operation():
        with pytest.raises(ValueError, match="changed during"):
            sources.verify_snapshot(path)
        monkeypatch.setattr(sources, "_verify_snapshot_cold", original)
        sources.verify_snapshot(path)
        assert sources.snapshot_verification_statistics()["cold_verifications"] == 2


def test_exception_exit_and_inherited_pid(tmp_path, offline, monkeypatch):
    path = acquire(tmp_path)
    counts = instrument(monkeypatch)
    with pytest.raises(RuntimeError), sources.snapshot_verification_operation():
        sources.verify_snapshot(path)
        raise RuntimeError("operation fails")
    with sources.snapshot_verification_operation():
        sources.verify_snapshot(path)
        state = sources._snapshot_operation.get()
        state.pid = -1
        sources.verify_snapshot(path)
        assert sources.snapshot_verification_statistics() is None
        with sources.snapshot_verification_operation():
            sources.verify_snapshot(path)
            assert sources.snapshot_verification_statistics()["reuse_hits"] == 0
    assert counts["cold"] == 4


def test_acquisition_publication_remains_cold(tmp_path, offline, monkeypatch):
    calls = []
    original = sources.verify_snapshot

    def observing(config, **kwargs):
        calls.append(kwargs.get("verification_mode"))
        return original(config, **kwargs)

    monkeypatch.setattr(sources, "verify_snapshot", observing)
    with sources.snapshot_verification_operation():
        acquire(tmp_path)
    assert calls == ["cold"]


def test_campaign_preparation_cold_reuse_identities(
    snapshot_campaign, tmp_path, monkeypatch
):
    from sparselab.campaign.engine import CampaignEngine

    campaign, _ = snapshot_campaign
    document = yaml.safe_load(campaign.read_text())
    document["stages"] = document["stages"][:3]
    campaign.write_text(yaml.safe_dump(document))
    sources.snapshot_source(
        tmp_path / "source.lock.json", tmp_path / "snapshot", tmp_path / "hub"
    )
    # Only synthetic acquisition is substituted; all downstream verification is native.
    counts = instrument(monkeypatch)
    engine = CampaignEngine(campaign, tmp_path / "work")
    result = engine.apply(allow_uncommitted_declaration=True)
    assert all(row["state"] == "COMPLETE" for row in result["stages"]), result
    assert counts["preflight"] > counts["cold"]
    warm = engine.inspect("status")
    cold = CampaignEngine(campaign, tmp_path / "work", cold_verify=True).inspect(
        "status"
    )
    assert warm == cold
    target = tmp_path / "snapshot" / "train.jsonl"
    target.write_bytes(target.read_bytes().replace(b"small fox", b"wrong fox"))
    availability = engine.inspect("status")["recoverability"]
    assert any(row["classification"] == "ERROR" for row in availability)
    with pytest.raises(ValueError):
        engine.inspect("next")


def test_native_campaign_preparation_observations(
    snapshot_campaign, tmp_path, monkeypatch
):
    campaign, _ = snapshot_campaign
    document = yaml.safe_load(campaign.read_text())
    document["stages"] = document["stages"][:3]
    campaign.write_text(yaml.safe_dump(document))
    sources.snapshot_source(
        tmp_path / "source.lock.json", tmp_path / "snapshot", tmp_path / "hub"
    )
    # Restore acquisition substitutions before spawning the ordinary CLI.
    monkeypatch.undo()
    observations = tmp_path / "observations"
    observations.mkdir()
    results = []
    for verb, extra, name in [
        ("apply", ["--allow-uncommitted-declaration"], "apply"),
        ("status", [], "status"),
        ("status", ["--cold-verify"], "cold-status"),
    ]:
        command = [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(tmp_path / "work"),
            "campaign",
            verb,
            str(campaign),
            *extra,
            "--observations-output",
            str(observations / f"{name}.json"),
            "--json",
        ]
        process = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=120,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
        )
        (tmp_path / f"{name}.stdout.json").write_text(process.stdout)
        (tmp_path / f"{name}.stderr.log").write_text(process.stderr)
        assert process.returncode == 0, process.stderr + process.stdout
        payload = json.loads(process.stdout)
        assert all(row["state"] == "COMPLETE" for row in payload["stages"]), payload
        results.append(payload["stages"])
        report = json.loads((observations / f"{name}.json").read_text())
        assert report["status"] == "completed"
        assert report["runtime"]["observation_scope"] == "controller_host_process_tree"
        if name == "apply":
            assert "data_prepare" in {record["phase"] for record in report["records"]}
        assert report["snapshot_verification"]["cold_verifications"] >= 1
        if name == "cold-status":
            assert report["snapshot_verification"]["reuse_hits"] == 0
        else:
            assert report["snapshot_verification"]["reuse_hits"] >= 1
    assert results[1] == results[2]
    retained = {
        path: path.read_bytes()
        for path in (tmp_path / "work" / "campaigns").rglob("*.json")
    }
    plain = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(tmp_path / "work"),
            "campaign",
            "status",
            str(campaign),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert plain.returncode == 0, plain.stderr
    assert json.loads(plain.stdout)["stages"] == results[1]
    assert all(path.read_bytes() == content for path, content in retained.items())
    tampered = tmp_path / "tampered-snapshot"
    shutil.copytree(tmp_path / "snapshot", tampered)
    target = tampered / "train.jsonl"
    target.write_bytes(target.read_bytes().replace(b"small fox", b"wrong fox"))
    bad_document = {
        **document,
        "id": "tampered-snapshot",
        "stages": [{**document["stages"][0], "output": str(tampered)}],
    }
    bad_campaign = tmp_path / "tampered-campaign.yaml"
    bad_campaign.write_text(yaml.safe_dump(bad_document))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(tmp_path / "work"),
            "campaign",
            "apply",
            str(bad_campaign),
            "--allow-uncommitted-declaration",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "tampered.stdout.json").write_text(process.stdout)
    (tmp_path / "tampered.stderr.log").write_text(process.stderr)
    assert process.returncode != 0
    payload = json.loads(process.stdout)
    assert payload.get("state") == "FAILED" or any(
        row["state"] == "FAILED" for row in payload.get("stages", [])
    )


def test_imported_snapshots_never_memoized(tmp_path, monkeypatch):
    import test_data_legacy

    from sparselab.data.snapshot_import import import_legacy_snapshot

    dataset = test_data_legacy._old_snapshot(monkeypatch, tmp_path / "old")
    path = import_legacy_snapshot(dataset, tmp_path / "imported")
    with sources.snapshot_verification_operation():
        first = sources.verify_snapshot(path)
        assert sources.verify_snapshot(path) == first
        statistics = sources.snapshot_verification_statistics()
        assert statistics["cold_verifications"] == 2
        assert statistics["reuse_hits"] == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_cannot_inherit_parent_evidence(tmp_path, offline):
    path = acquire(tmp_path)
    read_fd, write_fd = os.pipe()
    with sources.snapshot_verification_operation():
        sources.verify_snapshot(path)
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                assert sources.snapshot_verification_statistics() is None
                with sources.snapshot_verification_operation():
                    sources.verify_snapshot(path)
                    statistics = sources.snapshot_verification_statistics()
                    assert statistics["cold_verifications"] == 1
                    assert statistics["reuse_hits"] == 0
                os.write(write_fd, b"authenticated cold")
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        message = os.read(read_fd, 1024)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        assert status == 0 and message == b"authenticated cold"
        assert sources.snapshot_verification_statistics()["cold_verifications"] == 1


@pytest.fixture
def prepared_snapshot_fixture(tmp_path, monkeypatch):
    import test_snapshot_preparation

    yield from test_snapshot_preparation.snapshot_fixture.__wrapped__(
        tmp_path, monkeypatch
    )


def test_explicit_cold_lock_reopen_dominates_snapshot_operation(
    tmp_path, prepared_snapshot_fixture
):
    from sparselab.data.packing import prepare_data
    from sparselab.experiments.direct_inputs import bind_direct_inputs
    from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan

    fixture = prepared_snapshot_fixture
    prepared = prepare_data(fixture.run, fixture.tokenizer)
    run = tmp_path / "run.yaml"
    run.write_text(yaml.safe_dump(fixture.run.model_dump(mode="json")))
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "plan_version": 1,
                "id": "cold-snapshot-lock",
                "base_run": "run.yaml",
                "phases": [{"id": "pretrain", "transition": "fresh"}],
                "execution": {"backend": "cpu"},
            }
        )
    )
    declaration = tmp_path / "experiment.yaml"
    plan = bind_direct_inputs(run, template, prepared.root, declaration)
    lock = publish_lock(resolve_plan(plan, declaration), tmp_path / "locked")
    with sources.snapshot_verification_operation():
        expected = sources.verify_snapshot(fixture.run.dataset)["manifest_sha256"]
        reopened = open_lock(lock, verification_mode="cold")
        assert reopened.cells[0].config.dataset == fixture.run.dataset
        stats = sources.snapshot_verification_statistics()
        assert stats["cold_verifications"] > 1
        assert stats["reuse_hits"] == 0
        assert (
            sources.verify_snapshot(fixture.run.dataset)["manifest_sha256"] == expected
        )
        assert sources.snapshot_verification_statistics()["reuse_hits"] == 1
