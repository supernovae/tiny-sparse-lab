"""Tiny offline corpus timing tests; no network, corpus or model workloads."""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import pytest
from test_corpus_acquisition_steps import fixture, run

from sparselab.cli.main import main
from sparselab.corpus import acquisition, transport_budget
from sparselab.corpus.transport_budget import TransportBudget


def test_observation_preserves_binding_charges_and_survives_reopening(
    tmp_path, monkeypatch
):
    _recipe, project, root, ledger, _calls, content = fixture(tmp_path, monkeypatch)
    run(project, root)
    initial = ledger.receipt()
    monkeypatch.setattr(transport_budget.time, "time", lambda: initial["deadline"] + 1)
    with pytest.raises(ValueError, match="deadline exceeded"):
        ledger.remaining_seconds()
    selected = ledger.observe_time(owner="owner", reference="approved-local-workflow")
    observed = ledger.receipt()
    assert observed["wall_clock_mode"] == "observational"
    assert observed["wall_clock_observation"] == selected
    for key in initial.keys() - {"wall_clock_mode"}:
        assert observed[key] == initial[key]
    reopened = TransportBudget(ledger.path, project)
    assert reopened.remaining_seconds() == 30
    result = run(project, root, "two")
    assert result["status"] == "source_verified"
    calls_before = list(_calls)
    complete = acquisition.acquire(project, root)
    assert set(complete["sources"]) == {"one", "two"}
    assert _calls == calls_before
    final = reopened.receipt()
    assert final["source_actual"] == initial["source_actual"] + len(content)
    assert final["source_charged"] > initial["source_charged"]
    assert final["wall_clock_observation"] == selected
    with pytest.raises(ValueError, match="already observational"):
        reopened.observe_time(owner="owner", reference="repeat")
    with pytest.raises(ValueError, match="allowance exhausted"):
        reopened.reserve_metadata(
            project.config.transport_budget.max_metadata_body_bytes
        )


def test_active_transfer_and_bad_provenance_refuse_without_mutation(
    tmp_path, monkeypatch
):
    _recipe, project, _root, ledger, _, content = fixture(
        tmp_path, monkeypatch, second=False
    )
    for owner, reference in (("", "ref"), ("owner", " "), ("owner\0", "ref")):
        before = ledger.path.read_bytes()
        with pytest.raises(ValueError, match="owner and approval reference"):
            ledger.observe_time(owner=owner, reference=reference)
        assert ledger.path.read_bytes() == before
    shard = project.sources[0].acquisition.bounded_shards[0]
    ledger.reserve_transfer("one", shard.path, len(content))
    before = ledger.receipt()
    with pytest.raises(ValueError, match="active shard transfer"):
        ledger.observe_time(owner="owner", reference="ref")
    assert ledger.receipt() == before


def test_observation_keeps_finite_socket_read_timeout(tmp_path, monkeypatch):
    _recipe, _project, _root, ledger, _, _ = fixture(
        tmp_path, monkeypatch, second=False
    )
    initial = ledger.receipt()
    ledger.observe_time(owner="owner", reference="ref")
    monkeypatch.setattr(
        transport_budget.time, "time", lambda: initial["deadline"] + 100000
    )
    timeouts = []
    response = SimpleNamespace(_sock=SimpleNamespace(settimeout=timeouts.append))
    acquisition._set_response_deadline(response, ledger)
    assert timeouts == [30]


def test_legacy_ledger_stays_enforced_and_tampered_policy_refuses(
    tmp_path, monkeypatch
):
    _recipe, project, _root, ledger, _, _ = fixture(tmp_path, monkeypatch, second=False)
    initial = ledger.receipt()
    with sqlite3.connect(ledger.path) as db:
        body = json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
        body.pop("wall_clock_mode")
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(body),))
    monkeypatch.setattr(transport_budget.time, "time", lambda: initial["deadline"] + 1)
    reopened = TransportBudget(ledger.path, project)
    with pytest.raises(ValueError, match="deadline exceeded"):
        reopened.remaining_seconds()
    reopened.observe_time(owner="owner", reference="ref")
    with sqlite3.connect(ledger.path) as db:
        body = json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
        body["wall_clock_observation"]["planned_deadline"] += 1
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(body),))
    with pytest.raises(ValueError, match="counters are invalid"):
        TransportBudget(ledger.path, project)


@pytest.mark.parametrize(
    "tamper",
    [
        "unknown_mode",
        "enforced_with_observation",
        "unexpected_key",
        "missing_reference",
        "padded_owner",
        "nul_reference",
        "boolean_selected_at",
        "nonfinite_selected_at",
        "early_selected_at",
        "changed_planned_deadline",
    ],
)
def test_tampered_observation_policy_refuses_on_read(tmp_path, monkeypatch, tamper):
    _recipe, project, _root, ledger, _, _ = fixture(tmp_path, monkeypatch, second=False)
    ledger.observe_time(owner="owner", reference="ref")
    with sqlite3.connect(ledger.path) as db:
        body = json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
        observation = body["wall_clock_observation"]
        if tamper == "unknown_mode":
            body["wall_clock_mode"] = "unknown"
        elif tamper == "enforced_with_observation":
            body["wall_clock_mode"] = "enforced"
        elif tamper == "unexpected_key":
            observation["unexpected"] = True
        elif tamper == "missing_reference":
            observation.pop("reference")
        elif tamper == "padded_owner":
            observation["owner"] = " owner "
        elif tamper == "nul_reference":
            observation["reference"] = "ref\u0000"
        elif tamper == "boolean_selected_at":
            observation["selected_at"] = True
        elif tamper == "nonfinite_selected_at":
            observation["selected_at"] = float("nan")
        elif tamper == "early_selected_at":
            observation["selected_at"] = body["created_at"] - 1
        elif tamper == "changed_planned_deadline":
            observation["planned_deadline"] += 1
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(body),))
    with pytest.raises(ValueError, match="counters are invalid"):
        TransportBudget(ledger.path, project)


def test_public_command_selects_policy_on_same_ledger(tmp_path, monkeypatch):
    recipe, _project, root, ledger, _, _ = fixture(tmp_path, monkeypatch, second=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(root),
            "corpus",
            "budget-observe-time",
            str(recipe),
            "--owner",
            "owner",
            "--approval-reference",
            "ref",
        ],
    )
    out = StringIO()
    with redirect_stdout(out):
        assert main() is None
    assert json.loads(out.getvalue())["wall_clock_mode"] == "observational"
    assert ledger.receipt()["wall_clock_observation"]["reference"] == "ref"
