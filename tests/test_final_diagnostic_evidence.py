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
    assert result["diagnostic_status"] == "collected"
    assert f.payload["diagnostic_status"] == "not_collected"
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


@pytest.mark.parametrize("unavailable_result", [None, {"loss": 1.2, "batches": 1}])
def test_unavailable_result_remains_unavailable(recorded, unavailable_result):
    f = recorded
    f.record["evaluations"][0].update(
        status="UNAVAILABLE",
        path=None,
        sha256=None,
        result=unavailable_result,
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


def test_interim_and_final_snapshot_replay_and_reverification(recorded, monkeypatch):
    f = recorded
    monkeypatch.setattr(
        evidence, "_build_index", lambda *a, **k: copy.deepcopy(f.payload)
    )
    monkeypatch.setattr(evidence.Controller, "list_experiments", lambda self: [])
    old = evidence.collect_evidence(f.lock, f.root)
    old_bytes = Path(old["index_path"]).read_bytes()
    new = evidence.collect_evidence(f.lock, f.root, diagnostic_references=[f.ref])
    assert (
        old["diagnostic_status"] == "not_collected"
        and new["diagnostic_status"] == "collected"
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
    return pair_fixture(root)


def test_current_admission_is_metadata_only(tmp_path):
    source, bindings = final_fixture(tmp_path / "recipe")
    context = admit(source, bindings, tmp_path / "work")
    assert len(context.campaign.stages) == 12 and not context.work_dir.exists()
    assert context.campaign.stages[-1].evaluations == (
        "eval-baseline",
        "eval-candidate",
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing-final",
        "two-finals",
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
    if damage == "two-finals":
        raw["stages"].append({**copy.deepcopy(final), "id": "second-final"})
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
        )
    assert not (tmp_path / "work").exists()


def test_current_report_includes_only_verified_qualified_diagnostics(
    recorded, monkeypatch
):
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
    assert report["format"] == "sparselab-publication-report"
    assert report["diagnostic_evaluations"][0]["metrics"]["loss"] == 2.5
    assert report["readiness"]["scientific_result"] == "not_assessed"
    assert str(f.root) not in publication_report._json(report).decode()
    markdown = publication_report._markdown(report)
    assert f"Format: {report['format']}".encode() in markdown
    assert b"Coverage: collected" in markdown
    assert b"Recorded diagnostic evaluations" in markdown
    assert rb"metrics \{" in markdown
    assert rb"2\.5" in markdown
    assert before == {p: p.read_bytes() for p in f.root.rglob("*") if p.is_file()}
    legacy = evidence.collect_evidence(f.lock, f.root)
    old = publication_report.build_publication_report(
        f.root / "lock.json", workspace=f.root, index_path=Path(legacy["index_path"])
    )
    assert (
        old["format"] == "sparselab-publication-report"
        and old["diagnostic_evaluations"] == []
        and old["diagnostic_status"] == "not_collected"
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
            "diagnostic_status"
        ]
        == "collected"
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

    source, bindings = final_fixture(tmp_path / "recipe")
    context = admit(source, bindings, tmp_path / "work")
    raw = context.campaign.model_dump(mode="json")
    final = raw["stages"].pop()
    raw["stages"].insert(0, final)
    evaluation = next(s for s in raw["stages"] if s["id"] == "eval-baseline")
    CampaignPlan.model_validate(raw)  # The unordered, bound graph is valid first.
    evaluation["collect"] = producer
    with pytest.raises(ValueError, match="final collection evaluation belongs"):
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
    with pytest.raises(ValueError, match="exactly one diagnostic per cell"):
        bind_diagnostics(
            f.lock, f.payload, f.root, [f.ref, {**second, "stage": f.ref["stage"]}]
        )
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


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"limits": None},
            "valid dictionary",
        ),
        ({"workflow_admission_version": 4}, "obsolete workflow admission"),
    ],
)
def test_admission_version_and_limits_fail_as_value_errors(
    tmp_path, overrides, message
):
    # An otherwise valid current pair must reject obsolete selectors.
    source, bindings = pair_fixture(tmp_path / "recipe")
    with pytest.raises(ValueError, match=message):
        admit(source, bindings, tmp_path / "work", **overrides)
    assert not (tmp_path / "work").exists()


def test_two_distinct_stages_cannot_bind_the_same_cell(recorded):
    f = recorded
    bind_diagnostics(f.lock, f.payload, f.root, [f.ref])
    with pytest.raises(ValueError, match="exactly one diagnostic per cell"):
        bind_diagnostics(
            f.lock, f.payload, f.root, [f.ref, {**f.ref, "stage": "another"}]
        )


@pytest.mark.parametrize(
    "field,value", [("path", "other-generation"), ("digest", "b" * 64)]
)
def test_checkpoint_membership_axes_are_independent(recorded, field, value):
    f = recorded
    suite.verify_evaluation_index(f.index)  # Ref and native index still agree.
    checkpoint = f.payload["cells"][f.ref["cell_id"]]["attempts"][0]["checkpoints"][0]
    checkpoint[field] = value
    with pytest.raises(ValueError, match="checkpoint is absent"):
        bind_diagnostics(f.lock, f.payload, f.root, [f.ref])


def test_index_checkpoint_must_match_selected_inventoried_generation(recorded):
    f = recorded
    # The real index/result describe gen 2. Gen 3 is also inventoried and selected
    # by the ref, so membership cannot mask the index/ref identity check.
    selected = {
        **f.ref,
        "checkpoint": "checkpoints/step_00000003_gen_000003",
        "checkpoint_sha256": "b" * 64,
    }
    f.payload["cells"][f.ref["cell_id"]]["attempts"][0]["checkpoints"].append(
        {"path": "step_00000003_gen_000003", "digest": "b" * 64}
    )
    suite.verify_evaluation_index(f.index)
    with pytest.raises(ValueError, match="index differs from declared"):
        bind_diagnostics(f.lock, f.payload, f.root, [selected])


@pytest.mark.parametrize("changed_bytes", [False, True])
def test_index_suite_must_match_locked_reference(recorded, changed_bytes):
    f = recorded
    other = f.root / "other-suite.json"
    declaration = json.loads(f.source.read_text())
    if changed_bytes:
        declaration["id"] = "other-diagnostic"
    other.write_bytes(
        canonical_json(declaration) if changed_bytes else f.source.read_bytes()
    )
    f.record.update(
        suite=str(other), suite_sha256=hashlib.sha256(other.read_bytes()).hexdigest()
    )
    index = seal_index(f.record, f.index.parent)
    ref = {**f.ref, "index": str(index), "index_sha256": f.record["index_sha256"]}
    suite.verify_evaluation_index(index)
    with pytest.raises(ValueError, match="index differs from declared"):
        bind_diagnostics(f.lock, f.payload, f.root, [ref])


def test_resealed_suite_changed_after_lock_is_refused(recorded):
    f = recorded
    declaration = json.loads(f.source.read_text())
    declaration["id"] = "edited-after-lock"
    f.source.write_bytes(canonical_json(declaration))
    changed_sha = hashlib.sha256(f.source.read_bytes()).hexdigest()
    f.record["suite_sha256"] = changed_sha
    index = seal_index(f.record, f.index.parent)
    ref = {
        **f.ref,
        "suite_sha256": changed_sha,
        "index": str(index),
        "index_sha256": f.record["index_sha256"],
    }
    suite.verify_evaluation_index(index)
    with pytest.raises(ValueError, match="locked declaration"):
        bind_diagnostics(f.lock, f.payload, f.root, [ref])


def test_absent_locked_suite_raises_value_error(recorded):
    f = recorded
    lock = f.lock.model_copy(update={"evaluation_suite": None})
    with pytest.raises(ValueError, match="locked declaration"):
        bind_diagnostics(lock, f.payload, f.root, [f.ref])


def test_completed_null_perplexity_is_preserved(recorded):
    f = recorded
    result = json.loads(f.result.read_text())
    result["metrics"]["perplexity"] = None
    f.result.write_bytes(canonical_json(result) + b"\n")
    f.record["evaluations"][0].update(
        result=result["metrics"],
        sha256=hashlib.sha256(f.result.read_bytes()).hexdigest(),
    )
    index = seal_index(f.record, f.index.parent)
    ref = {**f.ref, "index": str(index), "index_sha256": f.record["index_sha256"]}
    report = report_diagnostics(bind_diagnostics(f.lock, f.payload, f.root, [ref]))
    assert report[0]["metrics"]["perplexity"] is None
    assert report[0]["metrics"]["loss"] == 2.5


@pytest.mark.parametrize("version", [1, 2])
def test_obsolete_evidence_refuses_without_controller_or_rewrite(
    recorded, monkeypatch, version
):
    f = recorded
    payload = {**f.payload, "version": version}
    payload.pop("diagnostic_status")
    payload.pop("diagnostic_evaluations")
    directory = f.root / "evidence"
    directory.mkdir()
    path = directory / f"{digest(payload)}.json"
    original = canonical_json(payload) + b"\n"
    path.write_bytes(original)
    monkeypatch.setattr(
        evidence,
        "Controller",
        lambda *a, **k: pytest.fail("obsolete evidence opened controller"),
    )
    with pytest.raises(
        ValueError, match="obsolete or unsupported evidence contract; collect fresh"
    ):
        evidence.read_evidence(f.lock, f.root, path)
    assert path.read_bytes() == original
