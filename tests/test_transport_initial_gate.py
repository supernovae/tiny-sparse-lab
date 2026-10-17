"""Bounded ledger fixtures only: no HTTP transport or corpus acquisition."""

import gzip
import hashlib
import json

import pytest
import yaml
from test_corpus_acquisition import _budgeted_hf_fixture

from sparselab.cli.main import build_parser
from sparselab.corpus.project import load_project
from sparselab.corpus.transport_budget import TransportBudget
from sparselab.training.attempt_commands import classify_attempt_command


def fixture(tmp_path):
    recipe = _budgeted_hf_fixture(
        tmp_path, gzip.compress(b'{"text":"fixture"}\n', mtime=0)
    )
    config = yaml.safe_load(recipe.read_text())
    config["transport_budget"].update(
        max_source_body_bytes=1000,
        max_metadata_body_bytes=1000,
        max_total_body_bytes=100,
        initial_body_bytes=40,
        max_transfers=10,
    )
    recipe.write_text(yaml.safe_dump(config))
    project = load_project(recipe)
    return TransportBudget.initialize(tmp_path / "ledger.sqlite", project), recipe


def evidence(tmp_path, ledger, *, status="ready"):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "ledger_receipt": ledger.receipt(),
                "findings": {"status": status, "scope": "Synthetic fixture only"},
            }
        )
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "decision": "continue",
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "approval_reference": "fixture-owner-review-only",
            }
        )
    )
    return report, review


def pilot(ledger):
    ledger.reserve_metadata(10)
    ledger.charge_metadata_actual(4)
    index = ledger.reserve_transfer("one", "pilot", 30)
    ledger.charge_transfer_actual(index, 20)
    ledger.finish_transfer(index, success=True)


def test_gate_combines_metadata_and_source_and_keeps_spend_deadline(tmp_path):
    ledger, recipe = fixture(tmp_path)
    pilot(ledger)
    before = ledger.receipt()
    for action in (
        lambda: ledger.reserve_metadata(1),
        lambda: ledger.reserve_transfer("one", "next", 1),
    ):
        with pytest.raises(ValueError, match="initial feasibility gate"):
            action()
    report, review = evidence(tmp_path, ledger)
    receipt = ledger.release_gate(report, review)
    assert receipt["approval_reference"] == "fixture-owner-review-only"
    after = TransportBudget(ledger.path, load_project(recipe)).receipt()
    assert {
        key: value for key, value in after.items() if key != "gate_review"
    } == before
    ledger.reserve_transfer("one", "next", 60)
    with pytest.raises(ValueError, match="cumulative"):
        ledger.reserve_metadata(1)
    with pytest.raises(ValueError, match="already released"):
        ledger.release_gate(report, review)
    with pytest.raises(ValueError, match="already exists"):
        TransportBudget.initialize(ledger.path, load_project(recipe))


@pytest.mark.parametrize(
    "changed", ["stale", "report", "approval", "blocked", "active", "empty"]
)
def test_gate_refuses_unbound_or_incomplete_review(tmp_path, changed):
    ledger, _ = fixture(tmp_path)
    if changed != "empty":
        pilot(ledger)
    report, review = evidence(
        tmp_path, ledger, status="blocked" if changed == "blocked" else "ready"
    )
    if changed == "stale":
        ledger.record_failure("one", ValueError("retained failure"))
    elif changed == "report":
        report.write_text(report.read_text() + " ")
    elif changed == "approval":
        value = json.loads(review.read_text())
        value["approval_reference"] = ""
        review.write_text(json.dumps(value))
    elif changed == "active":
        # Use an unfinished transfer in a separate clean bounded ledger.
        fresh = tmp_path / "active"
        fresh.mkdir()
        ledger, _ = fixture(fresh)
        ledger.reserve_transfer("one", "pilot", 20)
        ledger.charge_transfer_actual(0, 10)
        report, review = evidence(tmp_path, ledger)
    before = ledger.receipt()
    with pytest.raises(ValueError):
        ledger.release_gate(report, review)
    assert ledger.receipt() == before


def test_gate_release_is_public_but_not_automatic_attempt_dispatch(tmp_path):
    parsed = build_parser().parse_args(
        [
            "corpus",
            "budget-release-gate",
            "project.yaml",
            "--report",
            "report.json",
            "--review",
            "review.json",
        ]
    )
    assert parsed.report == "report.json"
    with pytest.raises(ValueError, match="unreviewed native"):
        classify_attempt_command(
            [
                "sparselab",
                "corpus",
                "budget-release-gate",
                "project.yaml",
                "--report",
                "report.json",
                "--review",
                "review.json",
            ]
        )


def test_pilot_reuse_materializes_without_full_origin_lock(tmp_path, monkeypatch):
    import shutil

    from test_kml_50m_preparation_sequence import _sources

    from sparselab.corpus import acquisition
    from sparselab.corpus.split_inventory import write_split_inventory

    origin, _ = _sources(tmp_path, "full-origin")
    origin_project = load_project(origin / "acquire.yaml")
    lock = acquisition.verify_acquisition(origin_project, tmp_path)
    lock_path = tmp_path / "corpora/full-origin/acquisition.json"
    preserved = lock_path.with_suffix(".fixture-evidence")
    lock_path.rename(preserved)
    before = preserved.read_bytes()
    pilot_root = tmp_path / "pilot-view"
    shutil.copytree(origin, pilot_root)
    config = yaml.safe_load((pilot_root / "acquire.yaml").read_text())
    config["id"] = "pilot-view"
    config["source_effects"] = [
        {
            "source_id": source_id,
            "effect": "reuse_only",
            "origin_project_id": "full-origin",
            "declaration_sha256": entry["declaration_sha256"],
            "snapshot_sha256": entry["snapshot_sha256"],
        }
        for source_id, entry in lock["sources"].items()
    ]
    recipe = pilot_root / "acquire.yaml"
    recipe.write_text(yaml.safe_dump(config))

    def refuse_transport(*args, **kwargs):
        raise AssertionError("pilot reuse must never acquire new bytes")

    for name in ("_acquire_git", "_acquire_http", "_acquire_hf", "_acquire_wikimedia"):
        monkeypatch.setattr(acquisition, name, refuse_transport)
    project = load_project(recipe)
    result = acquisition.acquire(project, tmp_path)
    assert acquisition.verify_acquisition(project, tmp_path) == result
    assert not lock_path.exists()
    assert preserved.read_bytes() == before
    assert not (tmp_path / "corpora/pilot-view/transport-budget.sqlite").exists()
    inventory = tmp_path / "pilot-inventory.jsonl"
    write_split_inventory(project, tmp_path, inventory)
    assert inventory.read_text().strip()


def test_empty_metadata_body_preserves_gate_charge(tmp_path):
    ledger, _ = fixture(tmp_path)
    pilot(ledger)
    before = ledger.receipt()
    ledger.reserve_metadata(0)
    ledger.charge_metadata_actual(0)
    assert ledger.receipt() == before
    with pytest.raises(ValueError, match="initial feasibility gate"):
        ledger.reserve_metadata(1)


@pytest.mark.parametrize("amount", [-1, True, 1.5])
@pytest.mark.parametrize("kind", ["metadata", "source"])
def test_invalid_reservation_cannot_change_charges(tmp_path, amount, kind):
    ledger, _ = fixture(tmp_path)
    before = ledger.receipt()
    with pytest.raises(ValueError, match="nonnegative integer"):
        if kind == "metadata":
            ledger.reserve_metadata(amount)
        else:
            ledger.reserve_transfer("one", "invalid", amount)
    assert ledger.receipt() == before
