"""Tiny opt-in post-run observation, with no source-corpus replay."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.corpus import tokenizer_observation as observed
from sparselab.corpus.mixture import TokenizerObservation
from sparselab.training.manifest import canonical_json


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else canonical_json(value) + b"\n"
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _case(tmp_path, monkeypatch):
    release = tmp_path / ("b" * 64)
    release.mkdir()
    config_path = tmp_path / "fit.json"
    output = tmp_path / "tokenizer"
    token = output / "tokenizer.json"
    request = observed._export_request("a" * 64, "lm", "0" * 64, 260, 1)
    export_id = hashlib.sha256(canonical_json(request)).hexdigest()
    export_dir = (
        tmp_path / "corpora" / "example" / "exports" / ("a" * 64) / "lm" / export_id
    )
    data = SimpleNamespace(
        source="local_text",
        revision="a" * 64,
        corpus_export_path=export_dir,
        corpus_release_path=release,
        train_path=release / "lm/train.jsonl",
        validation_path=release / "lm/validation.jsonl",
        train_max_documents=1,
        validation_max_documents=1,
        train_max_tokens=7,
        validation_max_tokens=7,
    )
    sampling = SimpleNamespace(model_dump=lambda **kwargs: {"max_fit_bytes": 8})
    config = SimpleNamespace(
        dataset=data,
        output_dir=output,
        vocab_size=260,
        min_frequency=1,
        max_documents=1,
        sampling=sampling,
    )
    monkeypatch.setattr(
        observed, "load_config", lambda path: SimpleNamespace(dataset=data)
    )
    monkeypatch.setattr(observed, "load_tokenizer_config", lambda path: config)
    from sparselab.corpus import completed_release

    monkeypatch.setattr(completed_release, "in_place", lambda path, manifest: True)
    report_bytes = b"report\n"
    license_bytes = b"license\n"
    files = {
        "report.json": {"sha256": hashlib.sha256(report_bytes).hexdigest()},
        "license-report.json": {"sha256": hashlib.sha256(license_bytes).hexdigest()},
        "lm/train.jsonl": {"sha256": "1" * 64},
        "lm/validation.jsonl": {"sha256": "2" * 64},
    }
    manifest = {"release_id": "a" * 64, "corpus_id": "example", "files": files}
    release_file = _put(release / "manifest.json", manifest)
    run = _put(export_dir / "run.yaml", b"run\n")
    token_cfg = _put(export_dir / "tokenizer.yaml", b"tokenizer config\n")
    export = {
        "release_id": manifest["release_id"],
        "schema_version": 1,
        "base_config_sha256": "0" * 64,
        "view": "lm",
        "vocab_size": 260,
        "release_manifest_sha256": release_file["sha256"],
        "report_sha256": files["report.json"]["sha256"],
        "license_report_sha256": files["license-report.json"]["sha256"],
        "run_config_sha256": run["sha256"],
        "tokenizer_config_sha256": token_cfg["sha256"],
        "splits": {
            "train": {
                "path": "lm/train.jsonl",
                "sha256": "1" * 64,
                "records": 1,
                "rendered_bytes": 5,
            },
            "validation": {
                "path": "lm/validation.jsonl",
                "sha256": "2" * 64,
                "records": 1,
                "rendered_bytes": 5,
            },
        },
    }
    binding = observed._export_binding(export)
    token_entry = _put(token, b"tokenizer bytes")
    token_manifest = {
        "sha256": token_entry["sha256"],
        "source": "local_text",
        "revision": "a" * 64,
        "vocab_size": 260,
        "corpus_export": binding,
        "training_contract": {
            "corpus_export": binding,
            "sampling": {"max_fit_bytes": 8},
            "vocab_size": 260,
            "min_frequency": 1,
            "max_documents": 1,
            "source": "local_text",
            "revision": "a" * 64,
        },
        "sampling_audit": {"complete": True, "disjoint_from_fit": True},
    }
    token_manifest_entry = _put(output / "tokenizer_manifest.json", token_manifest)
    config_entry = _put(config_path, b"fit config")
    session_id = "s1"
    job_id = "j1"
    job_root = tmp_path / "receipts" / session_id / "A"
    argv = ["python", "tokenizer", "train", str(config_path)]
    session_spec = {
        "session_id": session_id,
        "receipt_root": str(tmp_path / "receipts"),
        "slots": [{"id": "slot", "root": "A", "argv": argv}],
    }
    session_spec_entry = _put(tmp_path / "session.json", session_spec)
    job_spec = {
        "job_id": job_id,
        "source": {"commit": "c" * 40},
        "output_root": str(job_root),
        "argv": argv,
        "required_files": {str(config_path): config_entry["sha256"]},
    }
    job_spec_entry = _put(tmp_path / "job.json", job_spec)
    job_report = {
        "job_id": job_id,
        "source": job_spec["source"],
        "spec_sha256": job_spec_entry["sha256"],
        "reason": "exit",
        "returncode": 0,
        "process_success": True,
        "cleanup_verified": True,
        "evidence_complete": True,
        "logs_complete": True,
        "errors": [],
    }
    job_entry = _put(job_root / job_id / "report.json", job_report)
    stdout = _put(job_root / job_id / "stdout.log", (str(token) + "\n").encode())
    stderr = _put(job_root / job_id / "stderr.log", b"")
    session_report = {
        "session_id": session_id,
        "reason": "complete",
        "slots_completed": 1,
        "slots_total": 1,
        "pending_failure": None,
        "incomplete_jobs": [],
        "inventory_error": None,
        "cleanup_verified": True,
        "evidence_complete": True,
        "progress": {
            "jobs": [
                {
                    "job_id": job_id,
                    "slot": "slot",
                    "root": "A",
                    "launched": True,
                    "returncode": 0,
                    "spec_sha256": job_spec_entry["sha256"],
                    "report_sha256": job_entry["sha256"],
                    "stdout_sha256": stdout["sha256"],
                    "stderr_sha256": stderr["sha256"],
                }
            ]
        },
    }
    session_entry = _put(job_root.parent / "shared/report.json", session_report)
    outer = {
        "session_spec_sha256": session_spec_entry["sha256"],
        "reason": "exit",
        "engine_exit": 0,
        "container_exit": 0,
        "oom_killed": False,
        "cleanup_verified": True,
        "logs_complete": True,
        "errors": [],
        "ended_at_unix": 1,
    }
    outer_entry = _put(job_root.parent / "shared/container-report.json", outer)
    entries = {
        "job_spec": job_spec_entry,
        "session_spec": session_spec_entry,
        "job_report": job_entry,
        "session_report": session_entry,
        "container_report": outer_entry,
        "stdout": stdout,
        "stderr": stderr,
        "tokenizer_config": config_entry,
        "tokenizer": token_entry,
        "tokenizer_manifest": token_manifest_entry,
        "export": _put(export_dir / "export.json", export),
    }
    return (
        TokenizerObservation(
            method="post_run_observation",
            observed_at_utc=datetime.now(UTC),
            files=entries,
        ),
        config_path,
        config,
        release,
        manifest,
    )


def test_observation_binds_current_bytes_to_closed_job(tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    result = observed.verify_observed_tokenizer(*args)
    assert result["method"] == "post_run_observation"
    assert "did not attest" in result["limitation"]
    Path(args[0].files["tokenizer"].path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        observed.verify_observed_tokenizer(*args)


def test_observation_rejects_failed_or_inconsistent_session(tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    entry = args[0].files["session_report"]
    path = Path(entry.path)
    data = json.loads(path.read_bytes())
    data["pending_failure"] = "j1"
    replacement = _put(path, data)
    entry.sha256 = replacement["sha256"]
    with pytest.raises(ValueError, match="successful closed execution"):
        observed.verify_observed_tokenizer(*args)


def test_observation_rejects_audit_claim_even_when_manifest_digest_is_updated(
    tmp_path, monkeypatch
):
    args = _case(tmp_path, monkeypatch)
    entry = args[0].files["tokenizer_manifest"]
    path = Path(entry.path)
    data = json.loads(path.read_bytes())
    data["sampling_audit"]["disjoint_from_fit"] = False
    entry.sha256 = _put(path, data)["sha256"]
    with pytest.raises(ValueError, match="provenance differs"):
        observed.verify_observed_tokenizer(*args)
