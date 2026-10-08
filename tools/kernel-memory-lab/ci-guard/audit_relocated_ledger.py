"""Read-only forensic check of an archived CI v2 ledger at a different path.

Live AttemptBudget validation deliberately still requires its original contract
path. This tool never changes that binding or authorizes additional work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

from init_budget import QUOTAS

from sparselab.training.attempt_budget import AttemptContract


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    root: Path,
    *,
    kind: str,
    commit: str,
    expected_ledger_sha256: str,
    expected_contract_sha256: str,
) -> dict:
    if kind not in QUOTAS or len(commit) != 40 or int(commit, 16) < 0:
        raise ValueError("reviewed job kind and exact commit SHA required")
    ledger = root / "attempt-budget.sqlite"
    contract_path = root / "attempt-contract.json"
    policy = root / "quota-policy.json"
    baseline = root / "storage-baseline.json"
    for path in (ledger, contract_path, policy, baseline):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or symlinked archived input: {path}")
    if any(
        Path(str(ledger) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")
    ):
        raise ValueError("archived ledger sidecar requires a consistent frozen copy")
    ledger_sha256 = digest(ledger)
    contract_sha256 = digest(contract_path)
    if ledger_sha256 != expected_ledger_sha256:
        raise ValueError("archived ledger digest differs from evidence inventory")
    if contract_sha256 != expected_contract_sha256:
        raise ValueError("archived contract digest differs from evidence inventory")
    archived_contract = AttemptContract.model_validate_json(contract_path.read_bytes())
    if (
        archived_contract.content_identity_sha256
        != hashlib.sha256(commit.encode()).hexdigest()
    ):
        raise ValueError("archived contract content identity differs from commit")
    if archived_contract.monitor_policy_sha256 != digest(policy):
        raise ValueError("archived policy digest differs from contract")
    if archived_contract.workspace_baseline_sha256 != digest(baseline):
        raise ValueError("archived baseline digest differs from contract")
    uri = f"file:{quote(str(ledger))}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("archived SQLite integrity check failed")
        row = connection.execute(
            "SELECT version, max_updates, used_updates, started_ns, deadline_ns, "
            "last_checked_ns, max_targets, used_targets, max_calls, used_calls, "
            "max_tokens, used_tokens, contract_path, contract_sha256, contract_json "
            "FROM budget WHERE id=1"
        ).fetchone()
        reservations = connection.execute(
            "SELECT label, updates, target_positions, generation_calls, generated_tokens, "
            "content_identity_sha256, actual_updates, actual_targets, actual_calls, "
            "actual_tokens, completed_ns FROM reservations ORDER BY id"
        ).fetchall()
    if row is None or row[0] != 2:
        raise ValueError("archived ledger is not v2")
    embedded = row[14].encode("utf-8")
    if hashlib.sha256(embedded).hexdigest() != row[13] or row[13] != contract_sha256:
        raise ValueError("embedded contract digest differs from archived contract")
    if AttemptContract.model_validate_json(embedded) != archived_contract:
        raise ValueError("embedded contract differs from archived contract")
    if not isinstance(row[12], str) or not Path(row[12]).is_absolute():
        raise ValueError("original contract location is invalid")
    maximum = (row[1], row[6], row[8], row[10])
    charged = (row[2], row[7], row[9], row[11])
    if (
        maximum != QUOTAS[kind]
        or any(type(value) is not int or value < 0 for value in (*maximum, *charged))
        or any(used > limit for used, limit in zip(charged, maximum))
    ):
        raise ValueError("archived quota or charged counters are invalid")
    if (
        any(type(row[index]) is not int for index in (3, 4, 5))
        or not row[3] <= row[5] < row[4]
        or row[4] - row[3] != int(archived_contract.max_wall_seconds * 1e9)
    ):
        raise ValueError("archived deadline counters are invalid")
    totals = [0, 0, 0, 0]
    actual = [0, 0, 0, 0]
    for reservation in reservations:
        label, *values = reservation
        requested = values[:4]
        identity = values[4]
        observed = values[5:9]
        completed_ns = values[9]
        if (
            not isinstance(label, str)
            or identity != archived_contract.content_identity_sha256
            or any(type(value) is not int or value < 0 for value in requested)
            or (
                any(value is None for value in observed)
                and not all(value is None for value in observed)
            )
            or (all(value is None for value in observed) and completed_ns is not None)
        ):
            raise ValueError("archived reservation identity or shape is invalid")
        if all(value is not None for value in observed):
            if type(completed_ns) is not int or any(
                type(value) is not int or not 0 <= value <= limit
                for value, limit in zip(observed, requested)
            ):
                raise ValueError("archived actual counters are invalid")
            actual = [left + right for left, right in zip(actual, observed)]
        totals = [left + right for left, right in zip(totals, requested)]
    if tuple(totals) != charged:
        raise ValueError("archived reservation totals differ from charged counters")
    if digest(ledger) != ledger_sha256 or digest(contract_path) != contract_sha256:
        raise ValueError("archived inputs changed during read-only audit")
    return {
        "audit_kind": "relocated-read-only-not-live-authorization",
        "sqlite_integrity": "ok",
        "ledger_sha256": ledger_sha256,
        "contract_sha256": contract_sha256,
        "original_contract_path": row[12],
        "quota": maximum,
        "charged": charged,
        "actual_from_completed_reservations": tuple(actual),
        "reservation_count": len(reservations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=QUOTAS, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.root,
                kind=args.kind,
                commit=args.commit,
                expected_ledger_sha256=args.ledger_sha256,
                expected_contract_sha256=args.contract_sha256,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
