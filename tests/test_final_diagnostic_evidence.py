"""Recorded byte fixtures, native suite verification, and zero computation fences."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_experiment_evidence import _lock
from test_workflow_pair import admit, pair_fixture
from test_workflow_recipe import repin

from sparselab.evaluation import suite
from sparselab.experiments import evidence
from sparselab.experiments.diagnostic_evidence import (
    bind_diagnostics,
    report_diagnostics,
)
from sparselab.training.manifest import canonical_json


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def seal_index(record, directory):
    record["index_sha256"] = digest(
        suite._index_binding(
            record["suite_sha256"],
            record["checkpoint_sha256"],
            record["evaluations"],
            record["evaluation_runtime"],
        )
    )
    record["record_sha256"] = digest(
        {k: v for k, v in record.items() if k != "record_sha256"}
    )
    path = directory / f"suite-{record['index_sha256']}.json"
    path.write_bytes(canonical_json(record) + b"\n")
    return path


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    lock = _lock()
    run = tmp_path / "controller/run-one"
    directory = run / "evaluations"
    directory.mkdir(parents=True)
    source = tmp_path / "suite.json"
    source.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "diagnostic",
                "evaluations": [
                    {"id": "heldout", "kind": "heldout_lm", "role": "diagnostic"}
                ],
            }
        )
    )
    runtime = {
        "engine": "pytorch",
        "backend": "cpu",
        "precision": "fp32",
        "device_index": 0,
    }
    metrics = {"loss": 2.5, "perplexity": 12.18, "batches": 1, "valid_targets": 16}
    result = {
        "identity": {
            "checkpoint_sha256": "a" * 64,
            "checkpoint_relative_path": "checkpoints/step_00000002_gen_000002",
            "runtime": runtime,
        },
        "metrics": metrics,
    }
    result_path = directory / "heldout.json"
    result_path.write_bytes(canonical_json(result) + b"\n")
    row = {
        "id": "heldout",
        "kind": "heldout_lm",
        "role": "diagnostic",
        "status": "COMPLETED",
        "path": str(result_path),
        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "source_sha256": None,
        "result": metrics,
        "reason": None,
    }
    record = {
        "format": "evaluation-index-v1",
        "run": str(run),
        "run_id": run.name,
        "suite": str(source),
        "suite_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint": result["identity"]["checkpoint_relative_path"],
        "checkpoint_sha256": "a" * 64,
        "evaluation_runtime": {**runtime, "observed": runtime},
        "evaluations": [row],
    }
    index = seal_index(record, directory)
    reference = {
        "stage": "eval-one",
        "cell_id": "main:single",
        "run_id": run.name,
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": "a" * 64,
        "suite": str(source),
        "suite_sha256": record["suite_sha256"],
        "index": str(index),
        "index_sha256": record["index_sha256"],
    }
    lock = lock.model_copy(
        update={
            "evaluation_suite": {"id": "diagnostic", "sha256": record["suite_sha256"]},
            "availability": {"evaluation_suite": str(source)},
        }
    )
    from sparselab.experiments.lock import ResolvedExperimentPlan, _identities

    raw = lock.model_dump(mode="json")
    raw["scientific_sha256"], raw["plan_sha256"] = _identities(raw)
    lock = ResolvedExperimentPlan.model_validate(raw)
    payload = evidence._build_index(lock, [], tmp_path / "controller")
    payload["cells"]["main:single"].update(
        status="complete",
        selected_run_id=run.name,
        attempts=[
            {
                "attempt_id": "attempt-one",
                "run_id": run.name,
                "status": "complete",
                "checkpoints": [
                    {"path": "step_00000002_gen_000002", "digest": "a" * 64}
                ],
            }
        ],
    )
    payload["complete"] = True
    # Only checkpoint payload verification is substituted: no model/checkpoint is
    # created. Native suite/index/result hashing and binding checks remain real.
    selected = []

    def checkpoint(run_path, generation, expected, **kwargs):
        selected.append((run_path, generation, expected))
        assert (run_path, generation, expected) == (run, record["checkpoint"], "a" * 64)

    monkeypatch.setattr(suite, "_selected_checkpoint", checkpoint)

    def no_compute(*args, **kwargs):
        pytest.fail("evidence verification/reporting attempted computation")

    monkeypatch.setattr(suite, "run_suite", no_compute)
    monkeypatch.setattr(evidence.Controller, "tick", no_compute)
    return SimpleNamespace(
        root=tmp_path,
        lock=lock,
        payload=payload,
        ref=reference,
        record=record,
        index=index,
        source=source,
        result=result_path,
        selected=selected,
    )


def test_verified_binding_roundtrip_and_report_projection(recorded):
    f = recorded
    result = bind_diagnostics(f.lock, f.payload, f.root, [f.ref])
    assert f.selected
    assert result["version"] == 2 and f.payload["version"] == 1
    assert result["diagnostic_evaluations"][0]["result"]["loss"] == 2.5
    projected = report_diagnostics(result)
    assert projected[0]["status"] == "COMPLETED"
    assert str(f.root) not in json.dumps(projected)
    assert projected[0]["interpretation"] == "diagnostic_only"


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "other"),
        ("cell_id", "other"),
        ("checkpoint", "checkpoints/other"),
        ("checkpoint_sha256", "b" * 64),
        ("suite_sha256", "b" * 64),
        ("index_sha256", "b" * 64),
    ],
)
def test_wrong_expected_identity_refuses(recorded, field, value):
    ref = {**recorded.ref, field: value}
    with pytest.raises(ValueError):
        bind_diagnostics(recorded.lock, recorded.payload, recorded.root, [ref])


@pytest.mark.parametrize("target", ["index", "source", "result"])
def test_tampered_native_bytes_refuse(recorded, target):
    getattr(recorded, target).write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        bind_diagnostics(recorded.lock, recorded.payload, recorded.root, [recorded.ref])


@pytest.mark.parametrize("target", ["index", "result"])
def test_missing_record_is_not_computed(recorded, target):
    getattr(recorded, target).unlink()
    with pytest.raises((ValueError, OSError)):
        bind_diagnostics(recorded.lock, recorded.payload, recorded.root, [recorded.ref])


def test_unavailable_result_remains_unavailable(recorded):
    f = recorded
    f.record["evaluations"][0].update(
        status="UNAVAILABLE",
        path=None,
        sha256=None,
        result=None,
        reason="private failed evaluation",
    )
    path = seal_index(f.record, f.index.parent)
    ref = {**f.ref, "index": str(path), "index_sha256": f.record["index_sha256"]}
    projected = report_diagnostics(bind_diagnostics(f.lock, f.payload, f.root, [ref]))[
        0
    ]
    assert projected["status"] == "UNAVAILABLE" and projected["metrics"] == {}
    assert projected["reason_present"] and projected["result_sha256"] is None
    assert "private" not in json.dumps(projected)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "incomplete", "checkpoint", "symlink"]
)
def test_coverage_and_checkpoint_membership(recorded, mutation):
    f = recorded
    refs = [f.ref]
    if mutation == "missing":
        refs = []
    if mutation == "duplicate":
        refs *= 2
    if mutation == "incomplete":
        f.payload["cells"]["main:single"]["status"] = "pending"
    if mutation == "checkpoint":
        f.payload["cells"]["main:single"]["attempts"][0]["checkpoints"] = []
    if mutation == "symlink":
        link = f.index.parent / "alias.json"
        link.symlink_to(f.index)
        refs = [{**f.ref, "index": str(link)}]
    with pytest.raises(ValueError):
        bind_diagnostics(f.lock, f.payload, f.root, refs)


def test_versioned_snapshot_replay_and_reverification(recorded, monkeypatch):
    f = recorded
    monkeypatch.setattr(
        evidence, "_build_index", lambda *a, **k: copy.deepcopy(f.payload)
    )
    monkeypatch.setattr(evidence.Controller, "list_experiments", lambda self: [])
    old = evidence.collect_evidence(f.lock, f.root)
    old_bytes = Path(old["index_path"]).read_bytes()
    new = evidence.collect_evidence(f.lock, f.root, diagnostic_references=[f.ref])
    assert (
        old["version"] == 1
        and new["version"] == 2
        and old["index_path"] != new["index_path"]
    )
    assert (
        evidence.collect_evidence(f.lock, f.root, diagnostic_references=[f.ref]) == new
    )
    assert evidence.read_evidence(f.lock, f.root, Path(new["index_path"])) == new
    assert evidence.read_evidence(f.lock, f.root, Path(old["index_path"])) == old
    assert Path(old["index_path"]).read_bytes() == old_bytes
    f.result.write_text("{}")
    with pytest.raises(ValueError):
        evidence.read_evidence(f.lock, f.root, Path(new["index_path"]))
    assert evidence.read_evidence(f.lock, f.root, Path(old["index_path"])) == old


def final_fixture(root):
    source, bindings = pair_fixture(root)
    path = root / "campaign.json"
    raw = json.loads(path.read_text())
    raw["stages"].append(
        {
            "id": "final",
            "kind": "experiment_collect",
            "scope": "evaluation",
            "plan": "plan",
            "run": "run-baseline",
            "evaluations": ["eval-baseline", "eval-candidate"],
            "requires": [
                "plan",
                "run-baseline",
                "run-candidate",
                "eval-baseline",
                "eval-candidate",
            ],
        }
    )
    path.write_text(json.dumps(raw))
    repin(source)
    return source, bindings


def test_v3_admission_is_metadata_only(tmp_path):
    source, bindings = final_fixture(tmp_path / "recipe")
    context = admit(source, bindings, tmp_path / "work", workflow_admission_version=3)
    assert len(context.campaign.stages) == 12 and not context.work_dir.exists()
    assert context.campaign.stages[-1].evaluations == (
        "eval-baseline",
        "eval-candidate",
    )


@pytest.mark.parametrize(
    "damage",
    [
        "v2",
        "missing-final",
        "missing-eval",
        "duplicate",
        "wrong-producer",
        "extra-dependency",
        "not-terminal",
    ],
)
def test_final_graph_refuses_before_store(tmp_path, damage):
    source, bindings = final_fixture(tmp_path / "recipe")
    path = source.parent / "campaign.json"
    raw = json.loads(path.read_text())
    final = raw["stages"][-1]
    if damage == "missing-final":
        raw["stages"].pop()
    if damage == "missing-eval":
        final["evaluations"].pop()
    if damage == "duplicate":
        final["evaluations"].append("eval-baseline")
    if damage == "wrong-producer":
        final["evaluations"][0] = "run-baseline"
    if damage == "extra-dependency":
        final["requires"].append("gate")
    if damage == "not-terminal":
        raw["stages"].append(
            {
                "id": "later",
                "kind": "approval",
                "scope": "model",
                "requires": ["final"],
                "bind": ["final"],
            }
        )
    path.write_text(json.dumps(raw))
    repin(source)
    with pytest.raises(ValueError):
        admit(
            source,
            bindings,
            tmp_path / "work",
            workflow_admission_version=2 if damage == "v2" else 3,
        )
    assert not (tmp_path / "work").exists()


def test_report_v2_includes_only_verified_qualified_diagnostics(recorded, monkeypatch):
    from sparselab.experiments import publication_report

    f = recorded
    monkeypatch.setattr(
        evidence, "_build_index", lambda *a, **k: copy.deepcopy(f.payload)
    )
    monkeypatch.setattr(evidence.Controller, "list_experiments", lambda self: [])
    monkeypatch.setattr(publication_report, "open_lock", lambda path: f.lock)
    index = evidence.collect_evidence(f.lock, f.root, diagnostic_references=[f.ref])
    before = {p: p.read_bytes() for p in f.root.rglob("*") if p.is_file()}
    report = publication_report.build_publication_report(
        f.root / "lock.json", workspace=f.root, index_path=Path(index["index_path"])
    )
    assert report["format"] == "sparselab-publication-report-v2"
    assert report["diagnostic_evaluations"][0]["metrics"]["loss"] == 2.5
    assert report["readiness"]["scientific_result"] == "not_assessed"
    assert str(f.root) not in publication_report._json(report).decode()
    assert b"Recorded diagnostic evaluations" in publication_report._markdown(report)
    assert before == {p: p.read_bytes() for p in f.root.rglob("*") if p.is_file()}
    legacy = evidence.collect_evidence(f.lock, f.root)
    old = publication_report.build_publication_report(
        f.root / "lock.json", workspace=f.root, index_path=Path(legacy["index_path"])
    )
    assert (
        old["format"] == "sparselab-publication-report-v1"
        and "diagnostic_evaluations" not in old
    )
    f.result.write_text("{}")
    with pytest.raises(ValueError):
        publication_report.build_publication_report(
            f.root / "lock.json", workspace=f.root, index_path=Path(index["index_path"])
        )


def test_native_final_collection_dispatch_and_receipt_reopen(recorded, monkeypatch):
    from sparselab.campaign.engine import CampaignEngine
    from sparselab.campaign.plan import ExperimentCollect

    f = recorded
    f.payload["cells"]["main:single"]["attempts"][0]["checkpoints"][0]["step"] = 2
    monkeypatch.setattr(
        evidence, "_build_index", lambda *a, **k: copy.deepcopy(f.payload)
    )
    monkeypatch.setattr(evidence.Controller, "list_experiments", lambda self: [])
    engine = object.__new__(CampaignEngine)
    engine.observer = None
    engine._verification_options = {"proof_store": None, "verification_mode": "cold"}
    final = ExperimentCollect(
        id="final",
        kind="experiment_collect",
        scope="evaluation",
        plan="plan",
        run="run",
        evaluations=["eval-one"],
        requires=["plan", "run", "eval-one"],
    )
    engine.stages = {
        "run": SimpleNamespace(id="run", cell="main:single"),
        "collect": SimpleNamespace(id="collect", run="run"),
        "eval-one": SimpleNamespace(
            id="eval-one", collect="collect", suite="suite.json"
        ),
    }
    monkeypatch.setattr(engine, "_lock", lambda *a: f.lock)
    monkeypatch.setattr(engine, "_path", lambda ref: f.root / ref)
    monkeypatch.setattr(engine, "_upstream", lambda rows, name: rows[name])
    rows = {
        "final": {"stage_input_sha256": "input"},
        "plan": {"availability": {"workspace": str(f.root)}},
        "run": {"outputs": [{"identifier": "run-one"}]},
        "collect": {
            "measurements": {
                "generation": "step_00000002_gen_000002",
                "sha256": "a" * 64,
            }
        },
        "eval-one": {
            "availability": {"path": str(f.index)},
            "outputs": [{"sha256": f.ref["index_sha256"]}],
        },
    }
    first = CampaignEngine.dispatch.__wrapped__(
        engine, final, rows, 0, execute_runs=False
    )
    assert first["state"] == "COMPLETE"
    assert (
        evidence.read_evidence(f.lock, f.root, Path(first["availability"]["path"]))[
            "version"
        ]
        == 2
    )
    second = CampaignEngine.dispatch.__wrapped__(
        engine, final, rows, 0, execute_runs=False
    )
    assert second == first
    # Replay verifies the exact declared upstream references, not merely a valid
    # index chosen from the same run. Neither verification path dispatches work.
    engine.store = SimpleNamespace(input_sha=lambda *a: "input")
    monkeypatch.setattr(engine, "_projection_verification_key", lambda *a: "key")
    row = {**first, "stage_input_sha256": "input"}
    engine._verify(final, row, rows)
    rows["eval-one"]["outputs"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="references differ"):
        engine._verify(final, row, rows)


def test_absent_final_fields_preserve_native_serialization():
    from sparselab.campaign.plan import ExperimentCollect

    raw = {
        "id": "collect",
        "kind": "experiment_collect",
        "scope": "evaluation",
        "plan": "plan",
        "run": "run",
        "requires": ["plan", "run"],
    }
    assert ExperimentCollect.model_validate(raw).model_dump(mode="json") == raw


def test_same_native_index_identity_changed_record_refuses_replay(
    recorded, monkeypatch
):
    f = recorded
    monkeypatch.setattr(
        evidence, "_build_index", lambda *a, **k: copy.deepcopy(f.payload)
    )
    monkeypatch.setattr(evidence.Controller, "list_experiments", lambda self: [])
    saved = evidence.collect_evidence(f.lock, f.root, diagnostic_references=[f.ref])
    f.record["created_at_utc"] = "changed-but-self-consistent-record"
    changed = seal_index(f.record, f.index.parent)
    assert changed == f.index
    suite.verify_evaluation_index(changed)
    with pytest.raises(ValueError, match="evidence changed"):
        evidence.read_evidence(f.lock, f.root, Path(saved["index_path"]))


def test_valid_but_different_suite_is_not_the_locked_suite(recorded):
    f = recorded
    other = f.root / "other-suite.json"
    other.write_bytes(f.source.read_bytes())
    f.record["suite"] = str(other)
    seal_index(f.record, f.index.parent)
    with pytest.raises(ValueError, match="locked declaration"):
        bind_diagnostics(f.lock, f.payload, f.root, [{**f.ref, "suite": str(other)}])


@pytest.mark.parametrize(
    "field,value",
    [("loss", "private data"), ("loss", True), ("batches", 0), ("valid_targets", True)],
)
def test_invalid_completed_metrics_do_not_become_report_claims(recorded, field, value):
    f = recorded
    result = json.loads(f.result.read_text())
    result["metrics"][field] = value
    f.result.write_bytes(canonical_json(result) + b"\n")
    f.record["evaluations"][0]["result"] = result["metrics"]
    f.record["evaluations"][0]["sha256"] = hashlib.sha256(
        f.result.read_bytes()
    ).hexdigest()
    path = seal_index(f.record, f.index.parent)
    with pytest.raises(ValueError, match="diagnostic metrics|diagnostic counters"):
        bind_diagnostics(
            f.lock,
            f.payload,
            f.root,
            [{**f.ref, "index": str(path), "index_sha256": f.record["index_sha256"]}],
        )


@pytest.mark.parametrize("producer", ["missing", "run-baseline"])
def test_unordered_final_graph_rejects_invalid_evaluation_collection(
    tmp_path, producer
):
    from sparselab.campaign.plan import CampaignPlan

    source, _ = final_fixture(tmp_path / "recipe")
    raw = json.loads((source.parent / "campaign.json").read_text())
    final = raw["stages"].pop()
    raw["stages"].insert(0, final)
    evaluation = next(s for s in raw["stages"] if s["id"] == "eval-baseline")
    evaluation["collect"] = producer
    with pytest.raises(ValueError):
        CampaignPlan.model_validate(raw)


def test_paired_results_keep_each_cell_binding_and_unavailable_observation(
    recorded, monkeypatch
):
    f = recorded
    second_run = f.root / "controller/run-two"
    directory = second_run / "evaluations"
    directory.mkdir(parents=True)
    record = copy.deepcopy(f.record)
    record.update(run=str(second_run), run_id=second_run.name)
    record["evaluations"][0].update(
        status="UNAVAILABLE",
        path=None,
        sha256=None,
        result=None,
        reason="private failure detail",
    )
    index = seal_index(record, directory)
    second = {
        **f.ref,
        "stage": "eval-two",
        "cell_id": "main:second",
        "run_id": second_run.name,
        "index": str(index),
        "index_sha256": record["index_sha256"],
    }
    cell = copy.deepcopy(f.payload["cells"][f.ref["cell_id"]])
    cell["selected_run_id"] = second_run.name
    cell["attempts"][0].update(run_id=second_run.name, attempt_id="attempt-two")
    f.payload["cells"][second["cell_id"]] = cell
    checked = []

    def checkpoint(run, generation, expected, **kwargs):
        assert run.name in {"run-one", "run-two"}
        assert generation == f.ref["checkpoint"] and expected == "a" * 64
        checked.append(run.name)

    monkeypatch.setattr(suite, "_selected_checkpoint", checkpoint)
    result = bind_diagnostics(f.lock, f.payload, f.root, [second, f.ref])
    report = report_diagnostics(result)
    assert checked == ["run-one", "run-two"]
    assert [row["status"] for row in report] == ["COMPLETED", "UNAVAILABLE"]
    assert report[1]["metrics"] == {} and report[1]["reason_present"]
    assert "private failure" not in json.dumps(report)
    with pytest.raises(ValueError, match="run differs"):
        bind_diagnostics(
            f.lock,
            f.payload,
            f.root,
            [
                {**f.ref, "cell_id": second["cell_id"]},
                {**second, "cell_id": f.ref["cell_id"]},
            ],
        )
