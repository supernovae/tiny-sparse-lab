"""Read native charges and separate observed CI work without refunding failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from init_budget import QUOTAS

from sparselab.training.attempt_budget import AttemptBudget


def summarize(root: Path, kind: str) -> dict[str, object]:
    status = AttemptBudget(root / "attempt-budget.sqlite").status()
    events = [
        json.loads(path.read_text())
        for path in sorted((root / "events").glob("*.json"))
    ]
    actual = {
        "updates": sum(int(row.get("actual_updates", 0)) for row in events),
        "targets": sum(int(row.get("actual_targets", 0)) for row in events),
        "generation_requests": sum(
            row["label"].startswith("generation:") for row in events
        ),
        "completed_generation_calls": sum(
            row["label"].startswith("generation:") and row.get("completed") is True
            for row in events
        ),
        "sampled_generation_calls": sum(
            row["label"].startswith("generation:")
            and row.get("actual_sampled_tokens", 0) > 0
            for row in events
        ),
        "sampled_tokens": sum(
            int(row.get("actual_sampled_tokens", 0)) for row in events
        ),
    }
    charged = (
        status["charged_updates"],
        status["charged_actual_target_positions"],
        status["charged_generation_calls"],
        status["charged_generated_tokens"],
    )
    return {
        "kind": kind,
        "quota": QUOTAS[kind],
        "charged": charged,
        "actual": actual,
        "invocations": {
            name: sum(
                row["label"].startswith(f"train:{name}:")
                for row in status["reservations"]
            )
            for name in ("original", "snapshot-model", "archive-model")
        },
        "ledger": status,
        "event_count": len(events),
    }


def check_complete(result: dict[str, object]) -> None:
    kind = result["kind"]
    if tuple(result["charged"]) != QUOTAS[kind]:
        raise RuntimeError("CI job did not exercise its exact reviewed charge vector")
    expected = {
        "serving": {"original": 1, "snapshot-model": 9, "archive-model": 0},
        "archive": {"original": 0, "snapshot-model": 0, "archive-model": 1},
        "zero": {"original": 0, "snapshot-model": 0, "archive-model": 0},
    }[kind]
    if result["invocations"] != expected:
        raise RuntimeError("CI training fixture inventory differs from review")
    actual = result["actual"]
    if actual["updates"] != QUOTAS[kind][0] or actual["targets"] != QUOTAS[kind][1]:
        raise RuntimeError("CI observed training differs from charged reservation")
    if actual["generation_requests"] != QUOTAS[kind][2]:
        raise RuntimeError("CI observed generation-call count differs from reservation")
    if actual["sampled_tokens"] > QUOTAS[kind][3]:
        raise RuntimeError("CI observed sampled tokens exceed charged allowance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=QUOTAS, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = summarize(args.root, args.kind)
    print(json.dumps(result, sort_keys=True))
    if args.require_complete:
        check_complete(result)


if __name__ == "__main__":
    main()
