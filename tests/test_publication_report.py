"""Tiny native identity tests plus explicitly synthetic rendering contract fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from test_experiment_direct_inputs import inputs
from test_experiment_evidence import _row, _terminal

from sparselab.experiments import evidence
from sparselab.experiments import publication_report as publication
from sparselab.experiments.direct_inputs import bind_direct_inputs
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
from sparselab.experiments.retrospective import retrospective_views

FIXTURES = Path(__file__).parent / "fixtures/publication_report"
CASES = json.loads((FIXTURES / "observations-v1.json").read_text())["cases"]


@pytest.fixture
def native(tmp_path, monkeypatch):
    config, packed, run, template = inputs(tmp_path, monkeypatch)
    raw = config.model_dump(mode="json")
    raw["training"].update(max_steps=1, max_tokens=32)
    raw["optimizer"]["warmup_steps"] = 0
    run.write_text(yaml.safe_dump(raw))
    bound = tmp_path / "bound.yaml"
    plan = bind_direct_inputs(run, template, packed, bound)
    lock = resolve_plan(plan, bound)
    workspace = tmp_path / "experiment"
    path = publish_lock(lock, workspace)
    return path, open_lock(path), workspace


def snapshot(lock, workspace, rows, monkeypatch):
    # Supply a controller's recorded rows without executing a worker/controller tick.
    monkeypatch.setattr(evidence.Controller, "list_experiments", lambda self: rows)
    return Path(evidence.collect_evidence(lock, workspace)["index_path"])


def inventory(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_native_pending_reopens_verifies_and_does_not_mutate(native, monkeypatch):
    path, lock, workspace = native
    index = snapshot(lock, workspace, [], monkeypatch)
    before = inventory(workspace.parent)
    expected = retrospective_views(lock, workspace=workspace, index_path=index)
    report = publication.build_publication_report(
        path, workspace=workspace, index_path=index
    )
    assert (
        report["provenance"]["evidence_index_sha256"]
        == expected["views"]["evidence_coverage"]["facts"]["evidence_index_sha256"]
    )
    assert report["cells"][0]["status"] == "pending"
    assert report["cells"][0]["attempts"] == []
    assert report["readiness"]["execution_coverage"] == "incomplete"
    assert inventory(workspace.parent) == before
    assert str(workspace.parent) not in json.dumps(report)


@pytest.mark.parametrize(
    "state,expected",
    [
        ("FAILED", "failed"),
        ("INTERRUPTED", "censored"),
        ("QUEUED", "pending"),
        ("UNKNOWN", "unavailable"),
        ("COMPLETE", "invalid"),
    ],
)
def test_native_noncomplete_evidence_parity(native, monkeypatch, state, expected):
    path, lock, workspace = native
    row = _row(lock, state=state)
    if state in ("FAILED", "INTERRUPTED", "COMPLETE"):
        row["terminal_receipt"] = _terminal(row, state)
    if state == "COMPLETE":
        row["spec"]["plan"]["config_sha256"] = "b" * 64
    index = snapshot(lock, workspace, [row], monkeypatch)
    report = publication.build_publication_report(
        path, workspace=workspace, index_path=index
    )
    assert report["cells"][0]["status"] == expected
    assert report["cells"][0]["attempts"][0]["status"] == expected
    assert report["cells"][0]["attempts"][0]["quality_observations"] == []
    assert report["readiness"]["scientific_result"] == "not_assessed"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_synthetic_rendering_contract(native, monkeypatch, case):
    path, lock, _ = native
    views = retrospective_views(lock)
    cell = views["views"]["evidence_coverage"]["facts"]["cells"][0]
    cell["status"] = case["status"]
    attempt = {
        "attempt_id": "attempt-one",
        "run_id": "run-one",
        "status": case["status"],
        "reason": "secret=/private/diagnostic",
        "api_key": "secret-value",
        "checkpoints": [],
        "quality_observations": [],
        "missing_reports": ["/private/missing"],
        "rejected_reports": [{"reason": "secret"}],
    }
    if case["loss"] is not None:
        attempt["checkpoints"] = [
            {
                "digest": "a" * 64,
                "step": 1,
                "tokens_seen": 32,
                "validation_loss": case["loss"],
                "verification_scope": "weights",
                "path": "/private/checkpoint",
                "token": "secret",
            }
        ]
        attempt["quality_observations"] = [
            {
                "checkpoint_sha256": "a" * 64,
                "sha256": "b" * 64,
                "loss": case["loss"],
                "kind": "held_out_validation_v2",
                "identities": {
                    "protocol": "next-token-cross-entropy-v1",
                    "tokenizer.json": "c" * 64,
                    "api_key": "secret",
                },
                "metadata": {"secret": "never-export"},
            }
        ]
    cell["evidence"] = {
        "selected_run_id": "run-one" if case["status"] == "complete" else None,
        "attempts": [attempt],
    }
    views["views"]["readiness_decision"]["facts"]["execution_coverage"] = (
        "complete" if case["status"] == "complete" else "incomplete"
    )
    # These fixtures exercise projection only. Native verifier behavior is tested above
    # and separately with a complete checkpoint below; no fabricated acceptance claim.
    monkeypatch.setattr(
        publication, "retrospective_views", lambda *a, **kw: copy.deepcopy(views)
    )
    report = publication.build_publication_report(path)
    observed = report["cells"][0]["attempts"][0]
    assert report["cells"][0]["status"] == case["status"]
    assert observed["missing_report_count"] == observed["rejected_report_count"] == 1
    if case["loss"] is not None:
        assert observed["quality_observations"][0]["loss"] == case["loss"]
        assert observed["checkpoints"][0]["verification_scope"] == "weights"
    else:
        assert observed["quality_observations"] == []
    assert all(value == "unknown" for value in report["availability"].values())
    for encoded in (publication._json(report), publication._markdown(report)):
        assert b"secret" not in encoded and b"/private" not in encoded
        assert b"no_automatic_promotion" in encoded.replace(b"\\", b"")
        assert b"does not establish scientific comparability or a winner" in encoded


def test_tampered_lock_snapshot_and_changed_rows_refuse(native, monkeypatch):
    path, lock, workspace = native
    row = _row(lock)
    index = snapshot(lock, workspace, [row], monkeypatch)
    original = index.read_bytes()
    index.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="digest"):
        publication.build_publication_report(
            path, workspace=workspace, index_path=index
        )
    index.write_bytes(original)
    row["spec"]["plan"]["config_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="changed or is unavailable"):
        publication.build_publication_report(
            path, workspace=workspace, index_path=index
        )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical"):
        publication.build_publication_report(path)


def test_explicit_inputs_and_no_dictionary_bypass(native):
    path, lock, workspace = native
    with pytest.raises(ValueError, match="together"):
        publication.build_publication_report(path, workspace=workspace)
    with pytest.raises(TypeError):
        publication.build_publication_report(lock.model_dump(mode="json"))
    with pytest.raises(ValueError, match="format"):
        publication.write_publication_report(path, workspace / "bad", format="html")


@pytest.mark.parametrize("format", ["json", "markdown"])
def test_determinism_durable_replay_collision_and_symlink(native, format):
    path, _, workspace = native
    output = workspace / "reports" / format
    first = publication.write_publication_report(path, output, format=format)
    content, inode = first.read_bytes(), first.stat().st_ino
    assert publication.write_publication_report(path, output, format=format) == first
    assert (first.read_bytes(), first.stat().st_ino) == (content, inode)
    first.write_bytes(b"unrelated retained bytes")
    with pytest.raises(ValueError, match="different bytes"):
        publication.write_publication_report(path, output, format=format)
    assert first.read_bytes() == b"unrelated retained bytes"
    target = output.with_name("symlink-target")
    target.write_bytes(content)
    link = output.with_name("link")
    link.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe"):
        publication.write_publication_report(path, link, format=format)
    assert target.read_bytes() == content


def test_available_descriptive_comparison_does_not_rank(native, monkeypatch):
    from sparselab.experiments.plan import ExperimentPlan, load_plan

    _, original, workspace = native
    source = workspace.parent / "bound.yaml"
    raw = load_plan(source).model_dump(mode="json")
    batches = original.cells[0].config.evaluation.max_batches
    raw["axes"] = [
        {
            "name": "protocol",
            "choices": [
                {"label": "base", "set": {"evaluation.max_batches": batches}},
                {"label": "changed", "set": {"evaluation.max_batches": batches + 1}},
            ],
        }
    ]
    raw["comparisons"] = [
        {
            "id": "incompatible",
            "baseline": {"protocol": "base"},
            "variant": {"protocol": "changed"},
            "mode": "descriptive",
            "interventions": ["evaluation.max_batches"],
            "confounders": ["different held-out protocol"],
        }
    ]
    lock = resolve_plan(ExperimentPlan.model_validate(raw), source)
    path = publish_lock(lock, workspace)
    pair = lock.comparisons[0]
    assert pair.baseline != pair.variant
    views = retrospective_views(lock)
    # Only observed availability is synthetic here; the mismatched protocol lock
    # is compiled and reopened by the real native path.
    views["views"]["evidence_coverage"]["facts"]["comparisons"] = [
        {
            "id": pair.id,
            "status": "available",
            "baseline_run_id": "base",
            "variant_run_id": "candidate",
        }
    ]
    monkeypatch.setattr(publication, "retrospective_views", lambda *a, **kw: views)
    report = publication.build_publication_report(path)
    assert report["comparisons"][0]["recorded_status"] == "available"
    assert report["comparisons"][0]["scientific_comparability"] == "not_assessed"
    assert report["comparisons"][0]["winner"] is None
    assert report["comparisons"][0]["declared_confounder_count"] == 1
    assert b"secret" not in publication._json(report)


@pytest.mark.slow
def test_native_complete_checkpoint_metrics_and_read_only_rendering(
    native, monkeypatch
):
    from sparselab.training import trainer
    from sparselab.workers.models import ExperimentSpec

    path, lock, workspace = native
    row = _row(lock, state="COMPLETE")
    spec = ExperimentSpec.model_validate(row["spec"])
    config = lock.cells[0].config
    # Exactly one tiny synthetic CPU update creates native evidence before rendering.
    trainer.train(
        config,
        run_id=row["run_id"],
        worker_id=row["worker_id"],
        experiment_id=row["experiment_id"],
        attempt_id=row["attempt_id"],
        dispatch_metadata={
            "spec_digest": spec.digest(),
            "bundle_digest": spec.dispatch_bundle_digest,
            "plan": spec.plan.model_dump(mode="json"),
        },
    )
    run = workspace / "controller" / row["run_id"]
    shutil.copytree(config.logging.root_dir / row["run_id"], run)
    receipt = _terminal(row, "COMPLETE")
    receipt["artifacts"] = [
        {
            "relative_path": "run/" + str(file.relative_to(run)),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "size_bytes": file.stat().st_size,
        }
        for file in sorted(run.rglob("*"))
        if file.is_file()
    ]
    row.update(ingestion_status="COMPLETE", terminal_receipt=receipt)
    index = snapshot(lock, workspace, [row], monkeypatch)
    native_views = retrospective_views(lock, workspace=workspace, index_path=index)
    assert (
        native_views["views"]["evidence_coverage"]["facts"]["cells"][0]["status"]
        == "complete"
    )
    before = inventory(workspace.parent)

    def prohibited(*args, **kwargs):
        pytest.fail("rendering attempted execution or mutation")

    monkeypatch.setattr(trainer, "train", prohibited)
    from sparselab.evaluation import suite

    monkeypatch.setattr(suite, "run_suite", prohibited)
    monkeypatch.setattr(evidence, "collect_evidence", prohibited)
    monkeypatch.setattr(evidence.Controller, "submit", prohibited)
    from sparselab.data import packing, sources, tokenizer

    monkeypatch.setattr(tokenizer, "train_tokenizer", prohibited)
    monkeypatch.setattr(packing, "prepare_data", prohibited)
    monkeypatch.setattr(sources, "snapshot_source", prohibited)
    report = publication.build_publication_report(
        path, workspace=workspace, index_path=index
    )
    observed = report["cells"][0]["attempts"][0]
    expected = native_views["views"]["evidence_coverage"]["facts"]["cells"][0][
        "evidence"
    ]["attempts"][0]
    assert report["readiness"]["execution_coverage"] == "complete"
    assert (
        observed["quality_observations"][0]["loss"]
        == expected["quality_observations"][0]["loss"]
    )
    assert (
        observed["checkpoints"][0]["verification_scope"]
        == expected["checkpoints"][0]["verification_scope"]
    )
    assert report["availability"]["full_state_resume"] == "unknown"
    assert inventory(workspace.parent) == before
    assert publication._json(report) == publication._json(
        publication.build_publication_report(
            path, workspace=workspace, index_path=index
        )
    )
    # A changed receipt-bound artifact cannot silently become an incomplete report.
    (run / "resolved_config.yaml").write_text("{}")
    with pytest.raises(ValueError, match="changed or is unavailable"):
        publication.build_publication_report(
            path, workspace=workspace, index_path=index
        )


def test_replay_requires_successful_directory_sync(native, monkeypatch):
    path, _, workspace = native
    output = workspace / "report.json"
    publication.write_publication_report(path, output)
    original = output.read_bytes()
    primitive = publication.write_bytes_exclusive

    def broken_sync(_directory):
        raise OSError("injected directory sync failure")

    monkeypatch.setattr(
        publication,
        "write_bytes_exclusive",
        lambda path, encoded: primitive(path, encoded, sync_directory=broken_sync),
    )
    with pytest.raises(OSError, match="directory sync failure"):
        publication.write_publication_report(path, output)
    assert output.read_bytes() == original


def test_forged_plan_identity_and_wrong_evidence_location_refuse(native, monkeypatch):
    from sparselab.training.manifest import canonical_json

    path, lock, workspace = native
    index = snapshot(lock, workspace, [], monkeypatch)
    with pytest.raises(ValueError, match="outside plan workspace"):
        publication.build_publication_report(
            path, workspace=workspace / "other", index_path=index
        )
    payload = json.loads(index.read_bytes())
    payload["plan_sha256"] = "d" * 64
    raw = canonical_json(payload)
    forged = index.with_name(hashlib.sha256(raw).hexdigest() + ".json")
    forged.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="different locked plan"):
        publication.build_publication_report(
            path, workspace=workspace, index_path=forged
        )


@pytest.mark.parametrize("nested", [{"secret": "must-not-export"}, ["must-not-export"]])
def test_allowlisted_names_drop_nested_values(native, monkeypatch, nested):
    path, lock, _ = native
    views = retrospective_views(lock)
    cell = views["views"]["evidence_coverage"]["facts"]["cells"][0]
    cell["evidence"] = {
        "attempts": [
            {
                "status": nested,
                "run_id": nested,
                "receipt_sha256": None,
                "checkpoints": [
                    {
                        "digest": nested,
                        "verification_scope": nested,
                        "step": 0,
                        "validation_loss": 1.25,
                    }
                ],
                "quality_observations": [
                    {
                        "kind": nested,
                        "loss": nested,
                        "step": 0,
                        "identities": {"protocol": nested, "tokenizer.json": "a" * 64},
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(publication, "retrospective_views", lambda *a, **kw: views)
    report = publication.build_publication_report(path)
    attempt = report["cells"][0]["attempts"][0]
    assert "status" not in attempt and "run_id" not in attempt
    assert attempt["receipt_sha256"] is None
    assert attempt["checkpoints"] == [{"step": 0, "validation_loss": 1.25}]
    assert attempt["quality_observations"] == [
        {"step": 0, "identities": {"tokenizer.json": "a" * 64}}
    ]
    for encoded in (publication._json(report), publication._markdown(report)):
        assert b"must-not-export" not in encoded
        assert b'"secret"' not in encoded


def test_markdown_escapes_prose_and_preserves_fenced_json_values(native, monkeypatch):
    import re

    path, lock, _ = native
    views = retrospective_views(lock)
    cell = views["views"]["evidence_coverage"]["facts"]["cells"][0]
    run_id = "run\n```<script>|](target)\r"
    status = "state|](target)"
    observation = "</script>```|](target)"
    cell["status"] = status
    cell["evidence"] = {
        "selected_run_id": run_id,
        "attempts": [
            {
                "run_id": run_id,
                "status": status,
                "quality_observations": [
                    {"kind": observation, "identities": {"protocol": observation}}
                ],
            }
        ],
    }
    monkeypatch.setattr(publication, "retrospective_views", lambda *a, **kw: views)
    report = publication.build_publication_report(path)
    markdown = publication._markdown(report).decode()
    escaped_run = r"run \`\`\`&lt;script&gt;\|\]\(target\) "
    escaped_status = r"state\|\]\(target\)"
    # Literal prose must preserve one table row and escape link/fence delimiters.
    assert f"| {cell['cell_id']} | {escaped_status} | {escaped_run} |" in markdown
    assert f"Recorded status: {escaped_status}\n" in markdown
    blocks = re.findall(r"```json\n(.*?)\n```", markdown, flags=re.DOTALL)
    assert len(blocks) == len(report["cells"])
    assert markdown.count("```") == 2 * len(blocks)
    for block, expected in zip(blocks, report["cells"], strict=True):
        # Pipes/link punctuation are inert inside JSON; do not corrupt their values.
        assert json.loads(block) == expected
        assert "`" not in block and "<" not in block and ">" not in block
    assert "|](target)" in blocks[0]
    assert r"\u003c/script\u003e\u0060\u0060\u0060" in blocks[0]
    prose = re.sub(r"```json\n.*?\n```", "", markdown, flags=re.DOTALL)
    assert "<script>" not in prose and "](target)" not in prose
