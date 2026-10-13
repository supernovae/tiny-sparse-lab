"""An admitted native tokenizer -> paired model try -> report, offline CPU only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import test_lab_job
import yaml

from sparselab import lab_job as jobs
from sparselab.lab_mode import read_record
from sparselab.training.manifest import sha256_file

pytestmark = test_lab_job.pytestmark
job = test_lab_job.job


@pytest.fixture
def native_job(job, tmp_path):
    root = tmp_path / "declarations"
    root.mkdir()
    for name, sentence in (
        ("train.jsonl", "The small red robot learns to stack a blue block."),
        ("validation.jsonl", "A green robot carefully stacks the small yellow blocks."),
    ):
        (root / name).write_text(
            "".join(
                json.dumps({"text": sentence + f" Example {index}."}) + "\n"
                for index in range(80)
            )
        )
    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repository / "configs/smoke_cpu.yaml").read_text())
    config["tokenizer"]["path"] = "../outputs/tokenizer/tokenizer.json"
    config["dataset"].update(
        source="local_text",
        license="synthetic fixture (locally generated)",
        train_path="train.jsonl",
        validation_path="validation.jsonl",
        cache_dir="../outputs/data",
        train_max_documents=8,
        validation_max_documents=4,
        train_max_tokens=2048,
        validation_max_tokens=512,
    )
    config["logging"]["root_dir"] = "../outputs/unused-runs"
    config["model"].update(
        vocab_size=260,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=32,
    )
    config["training"].update(
        seq_len=16, max_steps=3, max_tokens=96, micro_batch_size=2
    )
    config["evaluation"].update(every_steps=3, max_batches=2)
    config["checkpoint"].update(every_steps=3, keep_periodic=True)
    config["optimizer"]["warmup_steps"] = 1
    (root / "baseline.yaml").write_text(yaml.safe_dump(config))
    token = {
        "schema_version": 1,
        "vocab_size": 260,
        "min_frequency": 1,
        "max_documents": 8,
        "output_dir": "../outputs/tokenizer",
        "dataset": config["dataset"],
    }
    (root / "tokenizer.yaml").write_text(yaml.safe_dump(token))
    (root / "delta.yaml").write_text(
        "question: Does FFN width change held-out loss after the same tiny training budget?\nset:\n  model.ffn_dim: 48\n"
    )
    (root / "try.yaml").write_text(
        "try_config_version: 1\nseed: 7\nfresh_baseline: true\nprobe_tier: fast\nlab_dir: ../native/lab\nresource_envelope: bounds.yaml\n"
    )
    (root / "bounds.yaml").write_text(
        "resource_envelope_version: 1\nlab_try:\n  max_wall_seconds_per_arm: 30\n  tokenizer_batch_documents: 8\n  tokenizer_batch_source_bytes: 1024\n"
    )
    workload = jobs.TryWorkload(
        kind="lab_try",
        profile="tiny-cpu-lab",
        root=root,
        tokenizer="tokenizer.yaml",
        baseline="baseline.yaml",
        candidate="delta.yaml",
        declaration="try.yaml",
        files={path.name: sha256_file(path) for path in root.iterdir()},
    )
    return job.model_copy(
        update={
            "workload": workload,
            "policy": job.policy.model_copy(
                update={"max_wall_seconds": 90.0, "termination_grace_seconds": 5.0}
            ),
            "max_evidence_bytes": 64 * 1024 * 1024,
        }
    )


def test_native_model_workflow_and_collection_never_retrain(
    native_job, monkeypatch, tmp_path
):
    publish = jobs.publish_immutable

    def fail_publication(path, payload):
        if path.name == "summary.json":
            raise OSError("final publication unavailable")
        return publish(path, payload)

    monkeypatch.setattr(jobs, "publish_immutable", fail_publication)
    with pytest.raises(OSError, match="publication unavailable"):
        jobs.admit(native_job)
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
