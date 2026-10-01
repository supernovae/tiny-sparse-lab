"""Parser integration for checkpoint-bound evaluation and model readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparselab.campaign.state import publish_immutable
from sparselab.evaluation.readiness import (
    assess_readiness,
    issue_review,
    verify_readiness_result,
)
from sparselab.evaluation.suite import run_suite, verify_evaluation_index


def _print(payload: dict, json_output: bool) -> None:
    print(
        json.dumps(payload, sort_keys=True)
        if json_output
        else json.dumps(payload, indent=2, sort_keys=True)
    )


def _suite_run(args: argparse.Namespace) -> None:
    output = run_suite(
        Path(args.suite),
        args.run_id,
        args.checkpoint,
        Path(args.runs_dir),
        args.backend,
    )
    index = verify_evaluation_index(output)
    if args.evidence_output:
        publish_immutable(
            Path(args.evidence_output),
            {
                "format": "evaluation-index-reference-v1",
                "index_sha256": index["index_sha256"],
                "checkpoint_sha256": index["checkpoint_sha256"],
                "index": str(output.resolve()),
                "evaluations": [
                    {key: row[key] for key in ("id", "status", "sha256", "reason")}
                    for row in index["evaluations"]
                ],
            },
        )
    _print({"index": str(output), **index}, args.json)


def _model(args: argparse.Namespace) -> None:
    path = assess_readiness(
        Path(args.policy), Path(args.index), Path(args.review) if args.review else None
    )
    result = verify_readiness_result(path)
    if args.evidence_output:
        publish_immutable(
            Path(args.evidence_output),
            {
                "format": "model-readiness-reference-v1",
                "result_sha256": result["result_sha256"],
                "result": str(path.resolve()),
                "state": result["state"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            },
        )
    _print({"output": str(path), **result}, args.json)


def _review(args: argparse.Namespace) -> None:
    result = issue_review(
        Path(args.index),
        args.reviewer,
        args.decision,
        args.note,
        Path(args.output),
        Path(args.surface_bundle) if args.surface_bundle else None,
    )
    _print({"output": args.output, **result}, args.json)


def register_evaluation_parser(
    subparsers: argparse._SubParsersAction,
    *,
    runs_dir_default: str | Path | None = None,
) -> None:
    """Add `evaluation suite run` beneath the existing top-level subparsers."""
    evaluation = subparsers.add_parser("evaluation")
    commands = evaluation.add_subparsers(dest="evaluation_command", required=True)
    suite = commands.add_parser("suite")
    actions = suite.add_subparsers(dest="suite_command", required=True)
    run = actions.add_parser("run")
    run.add_argument("suite")
    run.add_argument("run_id")
    run.add_argument("--checkpoint", required=True)
    run.add_argument(
        "--runs-dir",
        default=str(runs_dir_default) if runs_dir_default else None,
        required=runs_dir_default is None,
    )
    run.add_argument("--backend")
    run.add_argument("--json", action="store_true")
    run.add_argument("--evidence-output")
    run.set_defaults(handler=_suite_run)


def register_readiness_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add `readiness model|review` beneath existing readiness subparsers."""
    model = subparsers.add_parser("model")
    model.add_argument("policy")
    model.add_argument("index")
    model.add_argument("--review")
    model.add_argument("--json", action="store_true")
    model.add_argument("--evidence-output")
    model.set_defaults(handler=_model)
    review = subparsers.add_parser("review")
    review.add_argument("index")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", choices=("approve", "reject"), required=True)
    review.add_argument("--note", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--surface-bundle")
    review.add_argument("--json", action="store_true")
    review.set_defaults(handler=_review)
