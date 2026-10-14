"""Read-only checks for the bounded single-cell Workshop acceptance plan.

Inputs are native lock/workspace/index triples, not caller-authored reports.
This checks selected snapshots; the operator must additionally establish that
those snapshots cover all attempts in each isolated root. No work is launched.
"""

import argparse
import json
import math
from pathlib import Path


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _completed(report):
    _require(
        report["readiness"]["evidence_integrity"] == "verified_snapshot",
        "unverified evidence",
    )
    _require(
        report["readiness"]["execution_coverage"] == "complete", "incomplete coverage"
    )
    _require(len(report["cells"]) == 1, "expected one cell")
    cell = report["cells"][0]
    _require(
        cell["status"] == "complete" and cell["seed"] == 7,
        "incomplete or unexpected seed",
    )
    _require(
        cell["planned_budget"]
        == {
            "max_steps": 1,
            "max_tokens": 16,
            "seq_len": 16,
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
        },
        "unexpected training budget",
    )
    _require(
        cell["planned_evaluation"] == {"max_batches": 1}, "unexpected evaluation budget"
    )
    _require(len(cell["attempts"]) == 1, "expected exactly one attempt")
    attempt = cell["attempts"][0]
    _require(attempt["status"] == "complete", "attempt incomplete")
    _require(
        attempt.get("run_id") == cell["selected_run_id"]
        and bool(attempt.get("attempt_id")),
        "missing selected attempt",
    )
    _require(
        attempt["missing_report_count"] == attempt["rejected_report_count"] == 0,
        "missing/rejected reports",
    )
    _require(
        report["diagnostic_status"] == "collected"
        and len(report["diagnostic_evaluations"]) == 1,
        "missing diagnostic",
    )
    diagnostic = report["diagnostic_evaluations"][0]
    _require(
        diagnostic["status"] == "COMPLETED"
        and diagnostic["cell_id"] == cell["cell_id"]
        and diagnostic["run_id"] == cell["selected_run_id"],
        "diagnostic incomplete or mismatched",
    )
    metrics = diagnostic["metrics"]
    _require(
        metrics.get("batches") == 1 and metrics.get("valid_targets") == 16,
        "unexpected diagnostic work",
    )
    for key in ("loss", "perplexity"):
        value = metrics.get(key)
        _require(
            type(value) in (int, float) and math.isfinite(value),
            "missing/nonfinite diagnostic",
        )
    return cell, attempt, diagnostic


def compare_reports(left, right, *, mode):
    """Compare native verified projections; deliberately not a verifier itself."""
    _require(mode in ("replay", "repeat"), "unknown comparison mode")
    a, attempt_a, diagnostic_a = _completed(left)
    b, attempt_b, diagnostic_b = _completed(right)
    for field in ("scientific_sha256", "source_identity_sha256"):
        _require(
            bool(left["provenance"].get(field))
            and left["provenance"][field] == right["provenance"][field],
            f"changed {field}",
        )
    for field in (
        "cell_id",
        "config_sha256",
        "seed",
        "planned_budget",
        "planned_evaluation",
        "inputs",
    ):
        _require(a[field] == b[field], f"changed {field}")
    for field in ("stage", "suite_sha256"):
        _require(
            diagnostic_a[field] == diagnostic_b[field], f"changed diagnostic {field}"
        )
    if mode == "replay":
        _require(
            a == b
            and left["diagnostic_evaluations"] == right["diagnostic_evaluations"],
            "replay changed attempts or evidence",
        )
        _require(left["provenance"] == right["provenance"], "replay changed provenance")
    else:
        _require(
            attempt_a["attempt_id"] != attempt_b["attempt_id"]
            and attempt_a["run_id"] != attempt_b["run_id"],
            "repeat reused attempt",
        )
        for key in ("loss", "perplexity"):
            _require(
                math.isclose(
                    diagnostic_a["metrics"][key],
                    diagnostic_b["metrics"][key],
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ),
                f"repeat differs: {key}",
            )
    return {
        "mode": mode,
        "status": "PASS",
        "scope": "selected_verified_snapshots",
        "scientific_result": "not_assessed",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "repeat"))
    parser.add_argument(
        "--left",
        nargs=3,
        type=Path,
        required=True,
        metavar=("LOCK", "WORKSPACE", "INDEX"),
    )
    parser.add_argument(
        "--right",
        nargs=3,
        type=Path,
        required=True,
        metavar=("LOCK", "WORKSPACE", "INDEX"),
    )
    args = parser.parse_args(argv)
    # Lazy native verifier import: --help never imports optional execution code.
    from sparselab.experiments.publication_report import build_publication_report

    try:
        reports = [
            build_publication_report(lock, workspace=workspace, index_path=index)
            for lock, workspace, index in (args.left, args.right)
        ]
        result = compare_reports(*reports, mode=args.mode)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"acceptance check refused: {error}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
