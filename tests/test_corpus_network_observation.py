"""Local network observation on tiny offline fixtures; no real downloads."""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import redirect_stdout
from io import StringIO

import pytest
from test_corpus_acquisition_steps import fixture, run

from sparselab.cli.main import main
from sparselab.corpus import transport_budget
from sparselab.corpus.transport_budget import TransportBudget


def test_network_observation_preserves_history_and_measures_explicit_retries(
    tmp_path, monkeypatch
):
    _, project, root, ledger, calls, content = fixture(
        tmp_path, monkeypatch, second=False
    )
    before = ledger.receipt()
    # No original counters, binding or planned values are reset.
    selected = ledger.observe_network(
        owner="owner", reference="approved-local-assignment"
    )
    observed = ledger.receipt()
    assert {key: observed[key] for key in before} == before
    assert observed["network_observation"] == selected
    assert observed["network_mode"] == "observational"
    spec = project.config.transport_budget
    assert spec is not None and spec.max_transfers < 100
    attempts = spec.max_transfers + 1
    now = [before["created_at"] + 1]
    monkeypatch.setattr(transport_budget.time, "time", lambda: now[0])
    monkeypatch.setattr(transport_budget.time, "monotonic", lambda: now[0])
    shard = project.sources[0].acquisition.bounded_shards[0]
    for _ in range(attempts):
        index = ledger.reserve_transfer("one", shard.path, len(content))
        now[0] += 1
        ledger.charge_transfer_actual(index, len(content))
        now[0] += 1
        ledger.finish_transfer(index, success=False, interrupted=True)
    metadata_bytes = (
        max(spec.max_metadata_body_bytes, spec.max_total_body_bytes or 0) + 1
    )
    ledger.reserve_metadata(metadata_bytes)
    ledger.charge_metadata_actual(metadata_bytes)
    saved = TransportBudget(ledger.path, project).receipt()
    assert len(saved["transfers"]) == attempts > spec.max_transfers
    assert attempts > spec.max_retries_per_shard + 1
    assert saved["source_actual"] == saved["source_charged"] == attempts * len(content)
    assert saved["source_charged"] > spec.max_source_body_bytes
    assert saved["metadata_actual"] > spec.max_metadata_body_bytes
    if spec.max_total_body_bytes is not None:
        assert (
            saved["source_charged"] + saved["metadata_charged"]
            > spec.max_total_body_bytes
        )
    last = saved["transfers"][-1]
    assert last["elapsed_seconds"] == 2
    assert last["body_bytes_per_second"] == len(content) / 2
    assert (
        last["started_at_unix"] < last["last_progress_at_unix"] < last["ended_at_unix"]
    )
    assert not calls  # Reservations/measurements never retry automatically.
    result = run(project, root)
    assert result["status"] == "source_verified"
    assert (
        TransportBudget(ledger.path, project).receipt()["binding"] == before["binding"]
    )
    with pytest.raises(ValueError, match="complete or terminal"):
        ledger.reserve_transfer("one", shard.path, len(content))
    with pytest.raises(ValueError, match="already observational"):
        ledger.observe_network(owner="owner", reference="repeat")


def test_network_policy_refuses_active_transfer_bad_provenance_and_subclasses(
    tmp_path, monkeypatch
):
    _, project, _, ledger, calls, content = fixture(tmp_path, monkeypatch, second=False)
    for owner, reference in (("", "ref"), ("owner", " "), ("owner\0", "ref")):
        before = ledger.path.read_bytes()
        with pytest.raises(ValueError, match="owner and approval"):
            ledger.observe_network(owner=owner, reference=reference)
        assert ledger.path.read_bytes() == before

    class OtherLedger(TransportBudget):
        pass

    with pytest.raises(ValueError, match="corpus transport ledger"):
        OtherLedger(ledger.path, project).observe_network(
            owner="owner", reference="ref"
        )
    shard = project.sources[0].acquisition.bounded_shards[0]
    ledger.reserve_transfer("one", shard.path, len(content))
    before = ledger.receipt()
    with pytest.raises(ValueError, match="active shard transfer"):
        ledger.observe_network(owner="owner", reference="ref")
    assert ledger.receipt() == before and not calls


@pytest.mark.parametrize(
    "tamper", ["unknown_mode", "changed_limit", "nonfinite_time", "missing_reference"]
)
def test_invalid_network_policy_refuses_on_read(tmp_path, monkeypatch, tamper):
    _, project, _, ledger, calls, _ = fixture(tmp_path, monkeypatch, second=False)
    ledger.observe_network(owner="owner", reference="ref")
    with sqlite3.connect(ledger.path) as db:
        body = json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
        if tamper == "unknown_mode":
            body["network_mode"] = "unknown"
        elif tamper == "changed_limit":
            body["network_observation"]["planned_limits"]["max_transfers"] += 1
        elif tamper == "nonfinite_time":
            body["network_observation"]["selected_at"] = float("inf")
        else:
            body["network_observation"].pop("reference")
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(body),))
    with pytest.raises(ValueError, match="counters are invalid"):
        TransportBudget(ledger.path, project)
    assert not calls


def test_public_network_policy_command_and_default_quota(tmp_path, monkeypatch):
    recipe, _, root, ledger, calls, _ = fixture(tmp_path, monkeypatch, second=False)
    with pytest.raises(ValueError, match="allowance exhausted"):
        ledger.reserve_metadata(ledger.spec.max_metadata_body_bytes + 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(root),
            "corpus",
            "budget-observe-network",
            str(recipe),
            "--owner",
            "owner",
            "--approval-reference",
            "ref",
        ],
    )
    out = StringIO()
    with redirect_stdout(out):
        main()
    assert json.loads(out.getvalue())["network_mode"] == "observational"
    ledger.reserve_metadata(ledger.spec.max_metadata_body_bytes + 1)
    assert ledger.receipt()["metadata_charged"] > ledger.spec.max_metadata_body_bytes
    assert not calls
