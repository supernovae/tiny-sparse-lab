"""An admitted native tokenizer -> paired model try -> report, offline CPU only."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import test_lab_job
import yaml

from sparselab import lab_job as jobs
from sparselab.lab_mode import read_record
from sparselab.training.manifest import sha256_file

pytestmark = test_lab_job.pytestmark
job_factory = test_lab_job.job_factory


@pytest.fixture
def native_job(job_factory):
    job = job_factory(synthetic=False)
    return job.model_copy(
        update={
            "policy": job.policy.model_copy(
                update={"max_wall_seconds": 90.0, "termination_grace_seconds": 5.0}
            ),
            "max_evidence_bytes": 64 * 1024 * 1024,
        }
    )


def test_native_model_workflow_and_collection_never_retrain(
    native_job, monkeypatch, tmp_path
):
    # Preserve the former ambient-pytest contamination regression on the real
    # native workflow, without a public pytest workload or another model run.
    monkeypatch.setenv("PYTEST_ADDOPTS", "--lab70-nonexistent-option")
    monkeypatch.setenv("PYTEST_PLUGINS", "lab70_nonexistent_host_plugin")
    publish = jobs.publish_immutable

    def fail_publication(path, payload):
        if path.name == "summary.json":
            raise OSError("final publication unavailable")
        return publish(path, payload)

    monkeypatch.setattr(jobs, "publish_immutable", fail_publication)
    previous = os.umask(0o002)
    try:
        with pytest.raises(OSError, match="publication unavailable"):
            jobs.admit(native_job)
    finally:
        observed = os.umask(previous)
    assert observed == 0o002
    assert jobs.admit(native_job)["status"] == "COLLECTABLE"
    monkeypatch.setattr(jobs, "publish_immutable", publish)
    result = jobs.collect(native_job)
    summary = result["summary"]
    assert summary["controller_completion"] == "PASS", summary
    assert [gate["name"] for gate in summary["gates"]] == [
        "source",
        "runtime",
        "inputs",
        "tokenizer",
        "try",
        "report",
    ]
    assert all(
        gate["executed"] and gate["returncode"] == 0 for gate in summary["gates"]
    )
    native = summary["native_workflow"]
    assert native["comparison"]["verdict"] in {
        "CANDIDATE_LOWER_LOSS",
        "CANDIDATE_HIGHER_LOSS",
        "TIE",
    }
    for arm in native["arms"].values():
        assert arm["step"] == 3 and arm["tokens_seen"] == 96
        assert (
            math.isfinite(arm["heldout"]["loss"])
            and arm["heldout"]["valid_targets"] > 0
        )
    evidence, _, _ = jobs._paths(native_job)
    record = read_record(evidence / "native/lab/tries" / native["try_id"] / "try.json")
    assert record["seed"]["baseline"] == record["seed"]["candidate"] == 7
    assert record["probe"]["tiers_run"] == ["fast"]
    import torch

    from sparselab.training.checkpoints import CheckpointManager

    for name, arm in record["arms"].items():
        assert arm["data"]["train_sha256"] != arm["data"]["validation_sha256"]
        assert (
            arm["data"]["eval_tokenizer_sha256"]
            == record[f"{name}_inputs"]["tokenizer_file_sha256"]
        )
        run = Path(arm["run_dir"])
        manager = CheckpointManager(run)
        generations = sorted((run / "checkpoints").glob("step_*"))
        initial, terminal = (
            manager.load(generations[0]),
            manager.load(run / arm["checkpoint"]),
        )
        assert initial.step == 0 and terminal.step == 3
        assert any(
            not torch.equal(initial.model[key], tensor)
            for key, tensor in terminal.model.items()
        )
        assert all(torch.isfinite(tensor).all() for tensor in terminal.model.values())
    before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (evidence / "native").rglob("*")
        if path.is_file()
    }
    assert jobs.admit(native_job) == result
    assert jobs.collect(native_job) == result
    after = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (evidence / "native").rglob("*")
        if path.is_file()
    }
    assert after == before
    assert len(list(native_job.host.state.glob("*/pending.json"))) == 1
    observation = Path(native["arms"]["baseline"]["heldout"]["observation"])
    retained = tmp_path / "retained-observation.json"
    observation.rename(retained)
    try:
        with pytest.raises(FileNotFoundError):
            jobs.collect(native_job)
        assert len(list(native_job.host.state.glob("*/pending.json"))) == 1
    finally:
        retained.rename(observation)
    assert jobs.collect(native_job) == result
    # Reuse this completed workflow for verifier fault injection. Match the
    # specific refusal so a later report/hash check cannot mask a missing guard.
    import copy

    from sparselab import lab_mode

    for damage, message in (
        ("job-binding", "native record identity mismatch"),
        ("reused-arm", "native arm did not execute"),
        ("short-steps", "native optimizer did not reach declared"),
        ("short-tokens", "native optimizer did not reach declared"),
    ):
        altered = copy.deepcopy(record)
        if damage == "job-binding":
            altered["request"]["job_binding"]["manifest_sha256"] = "0" * 64
        elif damage == "reused-arm":
            altered["arms"]["baseline"]["reused"] = True
        elif damage == "short-steps":
            altered["arms"]["baseline"]["step"] -= 1
        else:
            altered["arms"]["baseline"]["tokens_seen"] -= 1
        with monkeypatch.context() as patch:
            patch.setattr(lab_mode, "read_record", lambda path, value=altered: value)
            with pytest.raises(ValueError, match=message):
                jobs.collect(native_job)
        assert len(list(native_job.host.state.glob("*/pending.json"))) == 1
    report = evidence / "report.stdout"
    original_report = report.read_bytes()
    try:
        report.write_text("{}\n")
        with pytest.raises(ValueError, match="report did not consume the exact"):
            jobs.collect(native_job)
        assert len(list(native_job.host.state.glob("*/pending.json"))) == 1
    finally:
        report.write_bytes(original_report)
    assert jobs.collect(native_job) == result
    print(
        json.dumps(
            {
                "elapsed_seconds": summary["elapsed_seconds"],
                "evidence_bytes": summary["evidence_bytes"],
                "peak_tree_rss_bytes": summary["peak_tree_rss_bytes"],
                "native_workflow": native,
            },
            sort_keys=True,
        )
    )


def test_changed_native_input_refuses_before_admission(native_job):
    (native_job.workload.root / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="identity changed"):
        jobs.admit(native_job)
    assert not (native_job.host.evidence / native_job.job_id).exists()


def test_native_cancellation_is_retained_without_relaunch(native_job, tmp_path):
    import os
    import subprocess
    import sys
    import time

    from sparselab.campaign.state import publish_immutable

    manifest = tmp_path / "native-handoff.json"
    publish_immutable(manifest, native_job.payload())
    process = subprocess.Popen(
        [
            sys.executable,
            "-P",
            "-B",
            "-m",
            "sparselab",
            "lab-job",
            "admit",
            str(manifest),
        ],
        env={**os.environ, "PYTHONPATH": str(native_job.host.repository / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    evidence, _, _ = jobs._paths(native_job)
    record_path = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and process.poll() is None:
        records = list((evidence / "native/lab/tries").glob("*/try.json"))
        if records:
            record_path = records[0]
            record = read_record(record_path)
            assert (
                record["request"]["job_binding"]["manifest_sha256"]
                == native_job.identity()
            )
            (record_path.parent / "CANCEL").touch()
            break
        time.sleep(0.01)
    stdout, stderr = process.communicate(timeout=40)
    assert record_path is not None, (stdout, stderr)
    assert process.returncode == 1, (stdout, stderr)
    record = read_record(record_path)
    assert record["status"] == "interrupted"
    result = jobs.collect(native_job)
    assert result["summary"]["controller_completion"] == "FAIL"
    assert result["summary"]["cleanup_verified"] is True
    assert result["summary"]["gates"][-1]["executed"] is False
    assert jobs.admit(native_job) == result
    assert len(list(native_job.host.state.glob("*/pending.json"))) == 1
    assert len(list((evidence / "native/lab/tries").glob("*/try.json"))) == 1


def test_seal_binds_native_inputs_and_runtime_without_execution(
    native_job, tmp_path, capsys
):
    draft = native_job.payload()
    for key in ("runtime_sha256", "python_version", "torch_version"):
        draft.pop(key)
    draft["workload"]["files"] = list(draft["workload"]["files"])
    source, sealed = tmp_path / "draft.json", tmp_path / "sealed.json"
    source.write_text(json.dumps(draft))
    jobs.main(["seal", str(source), "--output", str(sealed)])
    assert jobs.load_job(sealed) == native_job
    assert json.loads(capsys.readouterr().out)["status"] == "SEALED_NOT_ADMITTED"
    assert not list(native_job.host.state.iterdir())
    with pytest.raises(FileExistsError):
        jobs.main(["seal", str(source), "--output", str(sealed)])


@pytest.mark.parametrize(
    "change", ["oversized", "external-data", "pruning", "unbounded"]
)
def test_native_profile_refuses_unsafe_declarations_before_admission(
    native_job, change
):
    from sparselab.lab_job_workload import declarations

    root = native_job.workload.root
    name = "bounds.yaml" if change == "unbounded" else "baseline.yaml"
    path = root / name
    config = yaml.safe_load(path.read_text())
    if change == "oversized":
        config["model"]["hidden_dim"] = 64
    elif change == "external-data":
        config["dataset"]["source"] = "tinystories"
    elif change == "pruning":
        config["checkpoint"]["keep_periodic"] = False
    else:
        config["lab_try"]["max_wall_seconds_per_arm"] = None
    path.write_text(yaml.safe_dump(config))
    workload = native_job.workload.model_copy(
        update={"files": {**native_job.workload.files, name: sha256_file(path)}}
    )
    with pytest.raises(ValueError):
        declarations(workload, root)
    assert not list(native_job.host.state.iterdir())


@pytest.mark.parametrize("same_path", [True, False])
def test_identical_native_splits_refuse_before_admission(native_job, same_path):
    from sparselab.lab_job_workload import declarations

    root = native_job.workload.root
    if same_path:
        for name in ("baseline.yaml", "tokenizer.yaml"):
            path = root / name
            data = yaml.safe_load(path.read_text())
            data["dataset"]["validation_path"] = "train.jsonl"
            path.write_text(yaml.safe_dump(data))
    else:
        (root / "validation.jsonl").write_bytes((root / "train.jsonl").read_bytes())
    workload = native_job.workload.model_copy(
        update={
            "files": {
                name: sha256_file(root / name) for name in native_job.workload.files
            }
        }
    )
    message = (
        "local_text training and validation must be separate files"
        if same_path
        else "training and held-out inputs must be distinct"
    )
    with pytest.raises(ValueError, match=message):
        declarations(workload, root)
    assert not list(native_job.host.state.glob("*/pending.json"))
