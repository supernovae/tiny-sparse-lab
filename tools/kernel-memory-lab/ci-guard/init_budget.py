"""Create one exclusive native v2 budget for a reviewed CI job."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from sparselab.training.attempt_budget import AttemptBudget, AttemptContract

QUOTAS = {
    "serving": (20, 640, 113, 355),
    "archive": (2, 32, 0, 0),
    "zero": (0, 0, 0, 0),
}


def initialize(root: Path, kind: str, commit: str, deadline_ns: int) -> dict[str, str]:
    if kind not in QUOTAS or len(commit) != 40 or int(commit, 16) < 0:
        raise ValueError("reviewed job kind and exact commit SHA are required")
    remaining = (deadline_ns - time.time_ns()) / 1e9
    if remaining <= 0:
        raise RuntimeError("aggregate CI deadline reached before ledger creation")
    identity = hashlib.sha256(commit.encode("ascii")).hexdigest()
    baseline = root / "storage-baseline.json"
    policy = root / "quota-policy.json"
    policy.write_text(
        json.dumps({"kind": kind, "quotas": QUOTAS[kind]}, sort_keys=True) + "\n"
    )
    contract = AttemptContract(
        contract_version=1,
        max_optimizer_updates=QUOTAS[kind][0],
        max_actual_target_positions=QUOTAS[kind][1],
        max_generation_calls=QUOTAS[kind][2],
        max_generated_tokens=QUOTAS[kind][3],
        max_wall_seconds=remaining,
        content_identity_sha256=identity,
        monitor_policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        workspace_baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    )
    contract_path = root / "attempt-contract.json"
    raw = contract.model_dump_json().encode() + b"\n"
    contract_path.write_bytes(raw)
    ledger = root / "attempt-budget.sqlite"
    AttemptBudget.create_contract(
        ledger,
        contract_path=contract_path,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    events = root / "events"
    events.mkdir()
    return {
        "KML_CI_BUDGET_LEDGER": str(ledger),
        "KML_CI_CONTENT_SHA256": identity,
        "KML_CI_JOB_KIND": kind,
        "KML_CI_EVENTS_DIR": str(events),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=QUOTAS, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--deadline-ns", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(initialize(args.root, args.kind, args.commit, args.deadline_ns)))


if __name__ == "__main__":
    main()
