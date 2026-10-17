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


def mutate_ledger(ledger, change):
    import sqlite3

    with sqlite3.connect(ledger.path) as db:
        state = json.loads(
            db.execute("SELECT body FROM state WHERE id=1").fetchone()[0]
        )
        change(state)
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(state),))


@pytest.mark.parametrize(
    "change", ["total", "initial", "ungated", "keys", "blank", "stop_refs"]
)
def test_tampered_gate_ledger_refuses_read(tmp_path, change):
    ledger, _ = fixture(tmp_path)
    pilot(ledger)
    if change == "ungated":
        ledger.spec = ledger.spec.model_copy(update={"initial_body_bytes": None})

    def tamper(state):
        if change in {"total", "initial"}:
            state["metadata_charged"] = 90 if change == "total" else 20
            if change == "total":
                state["gate_review"] = {
                    "receipt_sha256": "a",
                    "report_sha256": "b",
                    "approval_reference": "fixture",
                }
        else:
            state["gate_review"] = {
                "receipt_sha256": "a",
                "report_sha256": "b",
                "approval_reference": "fixture",
            }
            if change == "keys":
                state["gate_review"]["extra"] = "x"
            elif change == "blank":
                state["gate_review"]["approval_reference"] = "  "
            elif change == "stop_refs":
                state["gate_review"]["stopped_transfers"] = {"one": " "}

    mutate_ledger(ledger, tamper)
    with pytest.raises(
        ValueError, match="total exceeded|initial gate exceeded|gate review is invalid"
    ):
        ledger.receipt()


@pytest.mark.parametrize(
    "change",
    [
        "report_link",
        "review_link",
        "large_report",
        "large_review",
        "report_keys",
        "decision_keys",
        "stop",
        "blank",
        "ungated",
        "deadline",
    ],
)
def test_release_review_guards_preserve_ledger(tmp_path, monkeypatch, change):
    import time

    ledger, _ = fixture(tmp_path)
    pilot(ledger)
    report, review = evidence(tmp_path, ledger)
    before = ledger.receipt()
    if change.endswith("_link"):
        original = report if change == "report_link" else review
        link = tmp_path / "link.json"
        link.symlink_to(original)
        if change == "report_link":
            report = link
        else:
            review = link
    elif change.startswith("large_"):
        target = report if change == "large_report" else review
        target.write_text(target.read_text() + " " * (1024 * 1024))
        if change == "large_report":
            decision = json.loads(review.read_text())
            decision["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
            review.write_text(json.dumps(decision))
    elif change == "report_keys":
        value = json.loads(report.read_text())
        value["extra"] = True
        report.write_text(json.dumps(value))
        value = json.loads(review.read_text())
        value["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
        review.write_text(json.dumps(value))
    elif change in {"decision_keys", "stop", "blank"}:
        value = json.loads(review.read_text())
        if change == "decision_keys":
            value["extra"] = True
        elif change == "stop":
            value["decision"] = "stop"
        else:
            value["approval_reference"] = "  "
        review.write_text(json.dumps(value))
    elif change == "ungated":
        ledger.spec = ledger.spec.model_copy(update={"initial_body_bytes": None})
    else:
        monkeypatch.setattr(time, "time", lambda: before["deadline"] + 1)
    with pytest.raises(ValueError):
        ledger.release_gate(report, review)
    assert ledger.receipt() == before


def test_gate_records_exact_reviewed_receipt_hash(tmp_path):
    from sparselab.training.manifest import canonical_json

    ledger, _ = fixture(tmp_path)
    pilot(ledger)
    report, review = evidence(tmp_path, ledger)
    expected = hashlib.sha256(canonical_json(ledger.receipt())).hexdigest()
    result = ledger.release_gate(report, review)
    assert result["receipt_sha256"] == expected
    assert result["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ["initial_body_bytes", "max_total_body_bytes"])
@pytest.mark.parametrize(
    "value", [True, 1.0, 1.5, "1", 0, -1, float("nan"), float("inf")]
)
def test_gate_spec_strict_positive_caps(tmp_path, field, value):
    from sparselab.corpus.project import TransportBudgetSpec

    ledger, _ = fixture(tmp_path)
    payload = ledger.spec.model_dump(mode="json")
    payload[field] = value
    if field == "max_total_body_bytes":
        payload.pop("initial_body_bytes")
    with pytest.raises(ValueError):
        TransportBudgetSpec.model_validate(payload)


@pytest.mark.parametrize("total", [None, 40, 39])
def test_gate_requires_larger_total(tmp_path, total):
    from sparselab.corpus.project import TransportBudgetSpec

    ledger, _ = fixture(tmp_path)
    payload = ledger.spec.model_dump(mode="json")
    payload["max_total_body_bytes"] = total
    with pytest.raises(ValueError, match="larger cumulative"):
        TransportBudgetSpec.model_validate(payload)


def test_legacy_caps_serialize_without_new_fields_and_missing_fields_fail_closed(
    tmp_path,
):
    from sparselab.corpus.project import TransportBudgetSpec

    ledger, _ = fixture(tmp_path)
    old = ledger.spec.model_dump(mode="json")
    old.pop("initial_body_bytes")
    old.pop("max_total_body_bytes")
    assert TransportBudgetSpec.model_validate(old).model_dump(mode="json") == old
    for field in ("initial_body_bytes", "max_total_body_bytes"):
        original = ledger.spec
        ledger.spec = original.model_copy()
        ledger.spec.__dict__.pop(field)
        with pytest.raises(AttributeError):
            ledger.receipt()
        ledger.spec = original


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit, "uncatchable"])
def test_interrupted_pilot_report_release_and_continuation(
    tmp_path, monkeypatch, interruption
):
    from test_corpus_acquisition import _mock_budgeted_hf_http

    from sparselab.corpus import acquisition
    from sparselab.training.manifest import canonical_json

    content = gzip.compress(
        json.dumps({"text": bytes(range(256)).hex() * 3}).encode() + b"\n", mtime=0
    )
    recipe = _budgeted_hf_fixture(tmp_path, content)
    config = yaml.safe_load(recipe.read_text())
    # Enough for one source plus metadata, but not its retry after more metadata.
    config["transport_budget"].update(
        initial_body_bytes=len(content) + 400, max_total_body_bytes=10000
    )
    recipe.write_text(yaml.safe_dump(config))
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger = TransportBudget.initialize(
        root / "corpora/example/transport-budget.sqlite", project
    )
    _mock_budgeted_hf_http(monkeypatch, content)
    original = acquisition._copy_hf_body

    def interrupt(response, output, limit, budget, index, resources):
        budget.charge_transfer_actual(index, 4)
        raise (
            KeyboardInterrupt("fixture stop")
            if interruption == "uncatchable"
            else interruption("fixture stop")
        )

    monkeypatch.setattr(acquisition, "_copy_hf_body", interrupt)
    with pytest.raises(
        KeyboardInterrupt if interruption == "uncatchable" else interruption
    ):
        acquisition.acquire(project, root)
    assert ledger.receipt()["transfers"][0]["status"] == "interrupted"
    if interruption == "uncatchable":
        mutate_ledger(
            ledger, lambda state: state["transfers"][0].update(status="started")
        )
    monkeypatch.setattr(acquisition, "_copy_hf_body", original)
    with pytest.raises(ValueError, match="initial feasibility gate"):
        acquisition.acquire(project, root)
    report, review = evidence(tmp_path, ledger)
    before = ledger.receipt()
    if interruption == "uncatchable":
        with pytest.raises(ValueError, match="explicit stop evidence"):
            ledger.release_gate(report, review)
        decision = json.loads(review.read_text())
        decision["stopped_transfers"] = {
            before["transfers"][0]["key"]: "fixture supervisor confirmed stopped"
        }
        review.write_text(json.dumps(decision))
    released = ledger.release_gate(report, review)
    assert (
        released["receipt_sha256"] == hashlib.sha256(canonical_json(before)).hexdigest()
    )
    after = ledger.receipt()
    for key in (
        "source_charged",
        "source_actual",
        "metadata_charged",
        "deadline",
        "failures",
    ):
        assert after[key] == before[key]
    lock = acquisition.acquire(project, root)
    assert acquisition.verify_acquisition(project, root) == lock
    after = ledger.receipt()
    assert after["source_charged"] == 2 * len(content)
    assert [t["status"] for t in after["transfers"]] == ["interrupted", "complete"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_git_transfer_interrupt_is_recorded_and_reraised(
    tmp_path, monkeypatch, interruption
):
    from test_corpus_acquisition import _bounded_git_fixture, _mock_bounded_git_http

    from sparselab.corpus import acquisition

    content = b"tiny fixture blob"
    recipe, oid = _bounded_git_fixture(tmp_path, content)
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger = TransportBudget.initialize(
        root / "corpora/example/transport-budget.sqlite", project
    )
    _mock_bounded_git_http(monkeypatch, content, oid)
    original = acquisition._copy_hf_body
    error = interruption("fixture interrupted Git")

    def interrupt(*args, **kwargs):
        raise error

    monkeypatch.setattr(acquisition, "_copy_hf_body", interrupt)
    with pytest.raises(interruption) as caught:
        acquisition.acquire(project, root)
    assert caught.value is error
    before = ledger.receipt()
    assert before["transfers"][0]["status"] == "interrupted"
    assert before["source_charged"] == len(content)
    monkeypatch.setattr(acquisition, "_copy_hf_body", original)
    acquisition.acquire(project, root)
    after = ledger.receipt()
    assert after["source_charged"] == 2 * len(content)
    assert after["deadline"] == before["deadline"]


@pytest.mark.parametrize(
    "stops",
    [
        {"one:pilot": " "},
        {"other": "fixture stop"},
        {"one:pilot": "fixture stop", "extra": "fixture"},
        [],
    ],
)
def test_stop_evidence_must_exactly_cover_started_transfers(tmp_path, stops):
    ledger, _ = fixture(tmp_path)
    ledger.reserve_transfer("one", "pilot", 40)
    report, review = evidence(tmp_path, ledger)
    value = json.loads(review.read_text())
    value["stopped_transfers"] = stops
    review.write_text(json.dumps(value))
    before = ledger.receipt()
    with pytest.raises(ValueError):
        ledger.release_gate(report, review)
    assert ledger.receipt() == before


def test_explicit_stopped_transfer_recovery_before_first_byte(tmp_path):
    ledger, _ = fixture(tmp_path)
    ledger.reserve_transfer("one", "pilot", 40)
    report, review = evidence(tmp_path, ledger)
    value = json.loads(review.read_text())
    value["stopped_transfers"] = {
        "one:pilot": "fixture supervisor verified process tree stopped"
    }
    review.write_text(json.dumps(value))
    before = ledger.receipt()
    ledger.release_gate(report, review)
    after = ledger.receipt()
    assert after["source_actual"] == before["source_actual"] == 0
    assert after["source_charged"] == before["source_charged"] == 40
    assert after["transfers"][0]["status"] == "interrupted"
    assert after["gate_review"]["stopped_transfers"] == value["stopped_transfers"]
    ledger.reserve_transfer("one", "pilot", 40)
    assert ledger.receipt()["source_charged"] == 80
