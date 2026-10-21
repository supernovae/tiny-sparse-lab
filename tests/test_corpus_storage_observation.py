"""Tiny pinned offline fixtures; no real corpus, Hub or training workloads."""

import json
import sqlite3
import sys
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import pytest
import yaml
from test_corpus_acquisition_steps import fixture, run

from sparselab.cli.main import main
from sparselab.corpus import acquisition_step, transport_budget
from sparselab.corpus.project import load_project
from sparselab.corpus.transport_budget import TransportBudget


def local_fixture(tmp_path, monkeypatch, *, network_cap=None):
    recipe, _, _, _, calls, content = fixture(tmp_path, monkeypatch)
    config = yaml.safe_load(recipe.read_text())
    config["transport_budget"]["max_disk_bytes"] = 1
    if network_cap is not None:
        config["transport_budget"]["max_source_body_bytes"] = network_cap
    recipe.write_text(yaml.safe_dump(config))
    project = load_project(recipe)
    root = tmp_path / "local-work"
    ledger = TransportBudget.initialize(
        root / "corpora/example/transport-budget.sqlite", project
    )
    return recipe, project, root, ledger, calls, content


def test_local_storage_has_no_write_or_occupancy_quota_and_preserves_network_history(
    tmp_path, monkeypatch
):
    _, project, root, ledger, calls, content = local_fixture(tmp_path, monkeypatch)
    before = ledger.receipt()
    ledger.observe_storage(
        owner="owner", reference="local-no-storage-allowance", min_free_bytes=1
    )
    for key, value in before.items():
        assert ledger.receipt()[key] == value
    reopened = TransportBudget(ledger.path, project)
    assert reopened.observes_storage
    plan = run(project, root, max_new_bytes=None, min_free_bytes=1, plan_only=True)
    assert (
        plan["admission"]["native_new_path_bytes"]
        > project.config.transport_budget.max_disk_bytes
    )
    assert plan["admission"]["storage_mode"] == "observational"
    assert plan["admission"]["max_new_bytes"] is None
    assert not calls
    result = run(project, root, max_new_bytes=None, min_free_bytes=1)
    assert result["status"] == "source_verified"
    assert ledger.receipt()["source_charged"] == len(content)
    assert ledger.receipt()["source_actual"] == len(content)
    with pytest.raises(ValueError, match="allowance exhausted"):
        ledger.reserve_metadata(
            project.config.transport_budget.max_metadata_body_bytes + 1
        )
    with pytest.raises(ValueError, match="already observational"):
        reopened.observe_storage(owner="owner", reference="again", min_free_bytes=1)


def test_actual_free_floor_is_kept_without_prediction_and_cannot_be_lowered(
    tmp_path, monkeypatch
):
    _, project, root, ledger, calls, _ = local_fixture(tmp_path, monkeypatch)
    ledger.observe_storage(owner="owner", reference="ref", min_free_bytes=100)
    monkeypatch.setattr(
        acquisition_step.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=100, f_frsize=1),
    )
    run(project, root, max_new_bytes=None, min_free_bytes=100, plan_only=True)
    with pytest.raises(ValueError, match="cannot lower"):
        run(project, root, max_new_bytes=None, min_free_bytes=99, plan_only=True)
    monkeypatch.setattr(
        acquisition_step.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=99, f_frsize=1),
    )
    with pytest.raises(ValueError, match="disk headroom"):
        run(project, root, max_new_bytes=None, min_free_bytes=100, plan_only=True)
    monkeypatch.setattr(
        transport_budget.shutil, "disk_usage", lambda _: SimpleNamespace(free=99)
    )
    with pytest.raises(ValueError, match="free-space floor"):
        ledger.check_disk(root, 10**12)
    assert not calls


def test_network_limit_and_explicit_default_policy_still_refuse_before_http(
    tmp_path, monkeypatch
):
    _, project, root, ledger, calls, _ = local_fixture(
        tmp_path, monkeypatch, network_cap=1
    )
    ledger.observe_storage(owner="owner", reference="ref", min_free_bytes=1)
    with pytest.raises(ValueError, match="source body allowance"):
        run(project, root, max_new_bytes=None, min_free_bytes=1)
    assert not calls
    (tmp_path / "default").mkdir()
    _, project, root, ledger, calls, _ = fixture(tmp_path / "default", monkeypatch)
    with pytest.raises(ValueError, match="positive remaining"):
        run(project, root, max_new_bytes=None)
    assert not calls and not ledger.observes_storage


def test_invalid_storage_selection_and_active_transfer_do_not_mutate(
    tmp_path, monkeypatch
):
    _, project, _, ledger, _, content = fixture(tmp_path, monkeypatch)
    for fields in (
        {"owner": "", "reference": "ref", "min_free_bytes": 1},
        {"owner": "owner", "reference": " ref", "min_free_bytes": 1},
        {"owner": "owner", "reference": "ref", "min_free_bytes": 0},
    ):
        before = ledger.path.read_bytes()
        with pytest.raises(ValueError, match="required"):
            ledger.observe_storage(**fields)
        assert ledger.path.read_bytes() == before
    shard = project.sources[0].acquisition.bounded_shards[0]
    ledger.reserve_transfer("one", shard.path, len(content))
    before = ledger.receipt()
    with pytest.raises(ValueError, match="active shard transfer"):
        ledger.observe_storage(owner="owner", reference="ref", min_free_bytes=1)
    assert ledger.receipt() == before


@pytest.mark.parametrize("change", ["mode", "floor", "planned", "provenance"])
def test_corrupt_storage_policy_refuses_on_read(tmp_path, monkeypatch, change):
    _, project, _, ledger, _, _ = fixture(tmp_path, monkeypatch)
    ledger.observe_storage(owner="owner", reference="ref", min_free_bytes=1)
    with sqlite3.connect(ledger.path) as db:
        body = json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
        if change == "mode":
            body["storage_mode"] = "unknown"
        elif change == "floor":
            body["storage_observation"]["min_free_bytes"] = 0
        elif change == "planned":
            body["storage_observation"]["planned_max_disk_bytes"] += 1
        else:
            body["storage_observation"]["owner"] = " owner "
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(body),))
    with pytest.raises(ValueError, match="counters are invalid"):
        TransportBudget(ledger.path, project)


def test_public_storage_command_and_dataset_transport_boundary(tmp_path, monkeypatch):
    recipe, _, root, ledger, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(root),
            "corpus",
            "budget-observe-storage",
            str(recipe),
            "--owner",
            "owner",
            "--approval-reference",
            "ref",
            "--min-free-bytes",
            "1",
        ],
    )
    output = StringIO()
    with redirect_stdout(output):
        assert main() is None
    assert json.loads(output.getvalue())["storage_mode"] == "observational"
    assert ledger.observes_storage
    from test_bounded_acquisition import pinned_files, transport

    _, lock, _ = pinned_files(tmp_path / "dataset-input", monkeypatch)
    dataset = transport(tmp_path / "dataset", lock)
    dataset.initialize()
    with pytest.raises(ValueError, match="only for a corpus"):
        dataset.observe_storage(owner="owner", reference="ref", min_free_bytes=1)
