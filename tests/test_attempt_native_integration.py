"""One explicitly budgeted CPU integration node; never include in zero-update globs."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from test_training import config as small_cpu_config

from sparselab.config.models import RunConfig
from sparselab.operational_monitor import load_workspace_baseline
from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError
from sparselab.training.attempt_receipts import verify_native_phase_counters
from sparselab.training.manifest import sha256_file

RUN_ID = "kml-c05-native-cpu-q1"
GIB = 1024**3


def _save(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256_file(path)


def _invoke(path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    _save(
        path,
        {
            "argv": args,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    return result


def test_bounded_cpu_final_mask_and_ledger() -> None:
    root = Path(os.environ["KML_QUAL_ROOT"]).resolve(strict=True)
    assert root.is_absolute() and root.is_dir()
    baseline_path = root / "baseline.json"
    baseline = load_workspace_baseline(baseline_path, root)
    fixture = root / "fixture"
    fixture.mkdir(exist_ok=False)

    # This helper fits only a tiny synthetic tokenizer; it does not train a model.
    base = small_cpu_config(fixture)
    payload = base.model_dump(mode="json")
    payload["training"].update(
        micro_batch_size=2,
        gradient_accumulation=1,
        seq_len=16,
        max_steps=3,
        max_tokens=77,
    )
    payload["logging"]["checkpoint_every_steps"] = 1
    bounded = RunConfig.model_validate(payload)
    assert (
        bounded.training.max_steps,
        bounded.training.max_tokens,
        bounded.training.micro_batch_size,
        bounded.training.gradient_accumulation,
        bounded.training.seq_len,
    ) == (3, 77, 2, 1, 16)
    assert [min(32, 77 - 32 * step) for step in range(3)] == [32, 32, 13]
    runs = bounded.logging.root_dir
    runs.mkdir(parents=True, exist_ok=True)
    run = runs / RUN_ID
    assert not run.exists()
    config_path = root / "bounded-config.json"
    config_sha = _save(config_path, bounded.model_dump(mode="json"))

    inner_policy_path = root / "inner-policy.json"
    inner_policy_sha = _save(
        inner_policy_path,
        {
            "monitor_policy_version": 1,
            "max_tree_rss_bytes": 3584 * 1024**2,
            "max_added_workspace_bytes": GIB,
            "max_added_workspace_inodes": 1000,
            "max_wall_seconds": 470,
            "termination_grace_seconds": 3,
            "interval_seconds": 0.2,
        },
    )
    contract_path = root / "contract.json"
    contract_sha = _save(
        contract_path,
        {
            "contract_version": 1,
            "max_optimizer_updates": 3,
            "max_actual_target_positions": 77,
            "max_generation_calls": 0,
            "max_generated_tokens": 0,
            "max_wall_seconds": 480,
            "content_identity_sha256": config_sha,
            "monitor_policy_sha256": inner_policy_sha,
            "workspace_baseline_sha256": baseline.sha256,
        },
    )
    ledger = root / "ledger.sqlite"
    assert not ledger.exists()
    init = _invoke(
        root / "attempt-init.json",
        [
            sys.executable,
            "-m",
            "sparselab",
            "attempt",
            "init",
            "--ledger",
            str(ledger),
            "--contract",
            str(contract_path),
            "--contract-sha256",
            contract_sha,
        ],
    )
    assert init.returncode == 0, init.stderr

    completion = root / "train-completion.json"
    training = _invoke(
        root / "attempt-run.json",
        [
            sys.executable,
            "-m",
            "sparselab",
            "attempt",
            "run",
            "--ledger",
            str(ledger),
            "--label",
            "train_77",
            "--activity",
            "train",
            "--content-identity-sha256",
            config_sha,
            "--policy",
            str(inner_policy_path),
            "--baseline",
            str(baseline_path),
            "--workspace",
            str(root),
            "--completion",
            str(completion),
            "--updates",
            "3",
            "--target-positions",
            "77",
            "--generation-calls",
            "0",
            "--generated-tokens",
            "0",
            "--receipt-kind",
            "train",
            "--receipt-path",
            str(run),
            "--",
            sys.executable,
            "-m",
            "sparselab",
            "train",
            str(config_path),
            "--backend",
            "cpu",
            "--run-id",
            RUN_ID,
            "--runs-dir",
            str(runs),
        ],
    )
    assert training.returncode == 0, training.stderr

    status = AttemptBudget(ledger).status()
    reservation = status["reservations"]
    assert len(reservation) == 1
    assert status["charged_updates"] == 3
    assert status["charged_actual_target_positions"] == 77
    assert status["charged_generation_calls"] == 0
    assert status["charged_generated_tokens"] == 0
    assert (
        reservation[0]["updates"],
        reservation[0]["target_positions"],
        reservation[0]["actual_updates"],
        reservation[0]["actual_target_positions"],
        reservation[0]["actual_generation_calls"],
        reservation[0]["actual_generated_tokens"],
    ) == (3, 77, 3, 77, 0, 0)
    assert verify_native_phase_counters(
        "train", run, expected_content_sha256=config_sha
    ) == (3, 77, 0, 0)
    progress = json.loads((run / "progress.json").read_text())
    assert (progress["status"], progress["step"], progress["tokens_seen"]) == (
        "completed",
        3,
        77,
    )
    with sqlite3.connect(runs / "experiments.sqlite3") as db:
        counters = db.execute(
            "SELECT DISTINCT step,tokens_seen FROM metrics "
            "WHERE run_id=? AND name='performance/step_seconds' ORDER BY step",
            (RUN_ID,),
        ).fetchall()
    assert counters == [(1, 32), (2, 64), (3, 77)]
    assert [
        current - prior
        for prior, current in zip((0, 32, 64), (32, 64, 77), strict=True)
    ] == [32, 32, 13]
    inner = json.loads(completion.read_text())
    assert inner["living_descendants"] == 0
    phase = json.loads(
        completion.with_name(completion.name + ".attempt.json").read_text()
    )
    assert phase["status"] == "WITHIN_CAP"
    assert phase["native_monitor_status"] == "COMPLETE"
    assert phase["baseline_sha256"] == baseline.sha256
    assert phase["final_added_workspace_bytes"] <= GIB
    assert phase["final_added_workspace_inodes"] <= 1000
    assert sha256_file(config_path) == config_sha

    # These failure checks are observational: no second model or reservation.
    with pytest.raises(AttemptBudgetError, match="duplicate phase"):
        AttemptBudget(ledger).reserve_vector(
            "train_77",
            updates=3,
            target_positions=77,
            content_identity_sha256=config_sha,
        )
    with pytest.raises(AttemptBudgetError, match="differ from recorded"):
        AttemptBudget(ledger)._record_verified_actual(
            "train_77",
            updates=2,
            target_positions=77,
            generation_calls=0,
            generated_tokens=0,
            content_identity_sha256=config_sha,
        )
    _save(
        root / "qualification-summary.json",
        {
            "format": "c05-native-cpu-qualification-v1",
            "run_id": RUN_ID,
            "run_path": str(run),
            "config_sha256": config_sha,
            "contract_sha256": contract_sha,
            "baseline_sha256": baseline.sha256,
            "actual_updates": 3,
            "actual_target_positions": 77,
            "per_update_target_positions": [32, 32, 13],
            "actual_generation_calls": 0,
            "actual_generated_tokens": 0,
            "native_completion": str(completion),
            "ledger": str(ledger),
        },
    )
