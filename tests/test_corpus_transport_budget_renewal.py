"""Offline tests for explicit, nonrefundable transport deadline renewal."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from io import StringIO

import pytest
from test_corpus_acquisition_steps import fixture

from sparselab.cli.main import main
from sparselab.corpus import transport_budget
from sparselab.corpus.transport_budget import TransportBudget


def utc(seconds: float) -> str:
    return datetime.fromtimestamp(math.floor(seconds), UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def complete_one_transfer(ledger: TransportBudget, project, content: bytes) -> None:
    shard = project.sources[0].acquisition.bounded_shards[0]
    index = ledger.reserve_transfer("one", shard.path, len(content))
    ledger.charge_transfer_actual(index, len(content))
    ledger.finish_transfer(index, success=True)


def test_renewal_preserves_binding_and_charges_and_is_single_use(
    tmp_path, monkeypatch
):
    _recipe, project, _root, ledger, _, content = fixture(
        tmp_path, monkeypatch, second=False
    )
    complete_one_transfer(ledger, project, content)
    initial = ledger.receipt()
    target = utc(initial["deadline"] + 86400)
    target_seconds = transport_budget._utc_deadline_epoch(target)
    assert initial["deadline"] < target_seconds <= initial["deadline"] + 86400

    monkeypatch.setattr(
        transport_budget.time, "time", lambda: initial["deadline"] + 1
    )
    returned = ledger.renew_deadline(
        target, owner="owner", reference="owner-approved-extension"
    )
    renewed = ledger.receipt()
    assert returned == renewed["renewals"][0]
    assert renewed["deadline"] == target_seconds
    assert renewed["renewals"][0]["previous_deadline"] == initial["deadline"]
    assert renewed["binding"] == initial["binding"]
    for key in (
        "created_at",
        "source_charged",
        "source_actual",
        "metadata_charged",
        "metadata_actual",
        "transfers",
        "failures",
    ):
        assert renewed[key] == initial[key]
    assert ledger.remaining_seconds() == target_seconds - (initial["deadline"] + 1)

    before_second_renewal = ledger.receipt()
    monkeypatch.setattr(transport_budget.time, "time", lambda: target_seconds + 1)
    with pytest.raises(ValueError, match="renewal already used"):
        ledger.renew_deadline(
            utc(target_seconds + 86400),
            owner="owner",
            reference="another-approval",
        )
    assert ledger.receipt() == before_second_renewal


def test_renewal_refuses_before_expiry_active_transfer_and_overlong_window(
    tmp_path, monkeypatch
):
    _recipe, project, _root, ledger, _, content = fixture(
        tmp_path, monkeypatch, second=False
    )
    initial = ledger.receipt()
    target = utc(initial["deadline"] + 86400)
    with pytest.raises(ValueError, match="has not expired"):
        ledger.renew_deadline(
            target, owner="owner", reference="owner-approved-extension"
        )

    before = ledger.receipt()
    with monkeypatch.context() as patch:
        patch.setattr(
            transport_budget.time, "time", lambda: initial["deadline"] + 1
        )
        with pytest.raises(ValueError, match="one-day bound"):
            ledger.renew_deadline(
                utc(initial["deadline"] + 86401),
                owner="owner",
                reference="owner-approved-extension",
            )
    assert ledger.receipt() == before

    shard = project.sources[0].acquisition.bounded_shards[0]
    ledger.reserve_transfer("one", shard.path, len(content))
    before = ledger.receipt()
    with monkeypatch.context() as patch:
        patch.setattr(
            transport_budget.time, "time", lambda: initial["deadline"] + 1
        )
        with pytest.raises(ValueError, match="active shard transfer"):
            ledger.renew_deadline(
                target, owner="owner", reference="owner-approved-extension"
            )
    assert ledger.receipt() == before


def test_public_budget_renew_command_records_exact_utc_and_provenance(
    tmp_path, monkeypatch
):
    recipe, _project, root, ledger, _, _ = fixture(
        tmp_path, monkeypatch, second=False
    )
    initial = ledger.receipt()
    target = utc(initial["deadline"] + 86400)
    monkeypatch.setattr(
        transport_budget.time, "time", lambda: initial["deadline"] + 1
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(root),
            "corpus",
            "budget-renew",
            str(recipe),
            "--deadline",
            target,
            "--owner",
            "owner",
            "--approval-reference",
            "owner-approved-extension",
        ],
    )
    output = StringIO()
    with redirect_stdout(output):
        main()
    receipt = json.loads(output.getvalue())
    assert receipt["deadline"] == transport_budget._utc_deadline_epoch(target)
    assert receipt["renewals"] == [
        {
            "previous_deadline": initial["deadline"],
            "deadline": receipt["deadline"],
            "deadline_utc": target,
            "owner": "owner",
            "reference": "owner-approved-extension",
        }
    ]
    assert receipt["source_charged"] == initial["source_charged"] == 0
    assert receipt["binding"] == initial["binding"]



def test_legacy_ledger_without_renewal_field_remains_readable(tmp_path, monkeypatch):
    _, _, _, ledger, _, _ = fixture(tmp_path, monkeypatch, second=False)
    initial = ledger.receipt()
    with sqlite3.connect(ledger.path) as db:
        body = json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
        body.pop("renewals")
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(body),))
    assert ledger.receipt()["deadline"] == initial["deadline"]


def test_renewal_chain_tampering_refuses_on_read(tmp_path, monkeypatch):
    _, _, _, ledger, _, _ = fixture(tmp_path, monkeypatch, second=False)
    initial = ledger.receipt()
    target = utc(initial["deadline"] + 86400)
    with monkeypatch.context() as patch:
        patch.setattr(
            transport_budget.time, "time", lambda: initial["deadline"] + 1
        )
        ledger.renew_deadline(
            target, owner="owner", reference="owner-approved-extension"
        )
    with sqlite3.connect(ledger.path) as db:
        body = json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
        body["deadline"] += 1
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(body),))
    with pytest.raises(ValueError, match="counters are invalid"):
        ledger.receipt()
