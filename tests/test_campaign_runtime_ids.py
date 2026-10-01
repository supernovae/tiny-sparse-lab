"""End-to-end Campaign execution with machine-local logical runtime selection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_campaign import by_id, invoke_cli, make_full_campaign

from sparselab.evaluation.readiness import verify_readiness_result
from sparselab.evaluation.suite import verify_evaluation_index
from sparselab.experiments.binding import open_runtime_binding
from sparselab.experiments.lock import open_lock
from sparselab.runtime_environments import RuntimeEntry, register_runtime
from sparselab.workers.controller import Controller


def test_campaign_logical_cpu_runtime_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    source = make_full_campaign(tmp_path, monkeypatch)
    declaration = yaml.safe_load(source.read_text())
    next(stage for stage in declaration["stages"] if stage["id"] == "runtime")[
        "profile_id"
    ] = "cpu-py314"
    source.write_text(yaml.safe_dump(declaration))

    cpu = RuntimeEntry(
        python=Path(sys.executable).absolute(),
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    register_runtime("cpu-py314", cpu)
    register_runtime("wrong-cpu", cpu)
    register_runtime(
        "not-a-rocm-wheel",
        RuntimeEntry(
            python=Path(sys.executable).absolute(),
            engine="pytorch",
            backend="rocm",
            device_index=0,
        ),
    )

    # A registered profile whose ID differs from the declared acceptance is not
    # sufficient, even when it names the very same executable and backend.
    wrong_work = tmp_path / "wrong-work"
    wrong = by_id(invoke_cli(source, wrong_work, "apply", "--runtime", "wrong-cpu"))
    assert wrong["runtime"]["state"] == "BLOCKED"
    assert "matching profile_id" in wrong["runtime"]["reason"]
    wrong_lock = open_lock(Path(wrong["plan"]["availability"]["path"]))
    assert wrong_lock.id == "fixture"
    assert not list(wrong_work.rglob("runtime-bindings/*.json"))

    # A CPU wheel declared as ROCm must fail before producing any runtime binding.
    mismatched_source = tmp_path / "mismatched-campaign.yaml"
    mismatched_declaration = yaml.safe_load(source.read_text())
    next(
        stage for stage in mismatched_declaration["stages"] if stage["id"] == "runtime"
    )["profile_id"] = "not-a-rocm-wheel"
    mismatched_source.write_text(yaml.safe_dump(mismatched_declaration))
    mismatch_work = tmp_path / "mismatch-work"
    mismatch = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(mismatch_work),
            "campaign",
            "apply",
            str(mismatched_source),
            "--runtime",
            "not-a-rocm-wheel",
            "--allow-uncommitted-declaration",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if mismatch.returncode == 0:
        mismatch_rows = by_id(json.loads(mismatch.stdout))
        assert mismatch_rows["runtime"]["state"] == "BLOCKED"
        assert any(
            word in mismatch_rows["runtime"]["reason"].lower()
            for word in ("selection", "unavailable", "backend")
        )
    else:
        assert "runtime" in (mismatch.stdout + mismatch.stderr).lower()
    assert not list(mismatch_work.rglob("runtime-bindings/*.json"))

    work = tmp_path / "work"
    waiting = by_id(invoke_cli(source, work, "apply", "--runtime", "cpu-py314"))
    assert waiting["runtime"]["state"] == "COMPLETE"
    assert waiting["gate"]["state"] == "AWAITING_APPROVAL"
    lock_path = Path(waiting["plan"]["availability"]["path"])
    lock_bytes = lock_path.read_bytes()
    locked = open_lock(lock_path)
    science_sha = locked.scientific_sha256
    assert science_sha == wrong_lock.scientific_sha256
    assert locked.id == "fixture"
    assert [cell.id for cell in locked.cells] == ["main:single"]
    binding_path = Path(
        waiting["runtime"]["availability"]["runtime_bindings"]["main:single"]["path"]
    )
    binding = open_runtime_binding(binding_path, locked, locked.cells[0])
    assert binding["source_kind"] == "profile"
    assert binding["descriptor"]["id"] == "cpu-py314"
    assert (
        Path(binding["descriptor"]["python"]).resolve()
        == Path(sys.executable).resolve()
    )
    assert binding["descriptor"]["backend"] == "cpu"
    assert binding["tested_runtime"]["backend"] == "cpu"
    assert binding["scientific_sha256"] == science_sha
    assert binding["plan_sha256"] == locked.plan_sha256
    # Registry-only changes do not enter the already resolved scientific lock.
    register_runtime("unrelated-cpu", cpu)
    assert open_lock(lock_path).scientific_sha256 == science_sha

    invoke_cli(
        source, work, "approve", "gate", "--note", "logical CPU runtime acceptance"
    )
    result = by_id(
        invoke_cli(
            source,
            work,
            "apply",
            "--runtime",
            "cpu-py314",
            "--execute-runs",
            "--max-wait-seconds",
            "600",
        )
    )
    if result["run"]["state"] == "RUNNING":
        result = by_id(
            invoke_cli(
                source,
                work,
                "resume",
                "--runtime",
                "cpu-py314",
                "--max-wait-seconds",
                "600",
            )
        )
    assert all(
        result[stage]["state"] == "COMPLETE"
        for stage in ("runtime", "gate", "run", "collect", "evaluation", "model")
    )
    assert result["model"]["outcome"] == "READY_FOR_NEXT_STAGE"
    controller = Controller(
        Path(result["run"]["availability"]["workspace"]) / "controller",
        read_only=True,
    )
    (attempt,) = controller.list_experiments()
    assert attempt["status"] == attempt["ingestion_status"] == "COMPLETE"
    assert (
        attempt["spec"]["plan"]["runtime_binding_sha256"] == binding["binding_sha256"]
    )
    index = verify_evaluation_index(Path(result["evaluation"]["availability"]["path"]))
    readiness = verify_readiness_result(Path(result["model"]["availability"]["path"]))
    assert index["checkpoint_sha256"] == readiness["checkpoint_sha256"]
    assert index["index_sha256"] == readiness["index_sha256"]
    assert index["evaluation_runtime"]["backend"] == "cpu"
    assert lock_path.read_bytes() == lock_bytes
    assert open_lock(lock_path).scientific_sha256 == science_sha
