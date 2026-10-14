"""Synthetic report projections test checks only, never scientific acceptance."""

import copy
import runpy
from pathlib import Path

import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/check_workshop_repeat.py")
)


def report():
    return {
        "provenance": {
            "scientific_sha256": "science",
            "source_identity_sha256": "source",
            "evidence_index_sha256": "index",
        },
        "readiness": {
            "evidence_integrity": "verified_snapshot",
            "execution_coverage": "complete",
        },
        "cells": [
            {
                "cell_id": "main:single",
                "config_sha256": "config",
                "seed": 7,
                "planned_budget": {
                    "max_steps": 1,
                    "max_tokens": 16,
                    "seq_len": 16,
                    "micro_batch_size": 1,
                    "gradient_accumulation": 1,
                },
                "planned_evaluation": {"max_batches": 1},
                "inputs": ["token", "data"],
                "status": "complete",
                "selected_run_id": "run-a",
                "attempts": [
                    {
                        "attempt_id": "attempt-a",
                        "run_id": "run-a",
                        "status": "complete",
                        "missing_report_count": 0,
                        "rejected_report_count": 0,
                    }
                ],
            }
        ],
        "diagnostic_status": "collected",
        "diagnostic_evaluations": [
            {
                "status": "COMPLETED",
                "cell_id": "main:single",
                "run_id": "run-a",
                "stage": "evaluate-heldout",
                "suite_sha256": "suite",
                "metrics": {
                    "loss": 5.4,
                    "perplexity": 221.4,
                    "batches": 1,
                    "valid_targets": 16,
                },
            }
        ],
    }


def repeat():
    value = report()
    value["cells"][0]["selected_run_id"] = "run-b"
    value["cells"][0]["attempts"][0].update(attempt_id="attempt-b", run_id="run-b")
    value["diagnostic_evaluations"][0]["run_id"] = "run-b"
    value["provenance"]["evidence_index_sha256"] = "other-index"
    return value


def test_replay_and_repeat_checks_do_not_mutate():
    a, b = report(), repeat()
    before = copy.deepcopy((a, b))
    assert TOOL["compare_reports"](a, a, mode="replay")["status"] == "PASS"
    assert (
        TOOL["compare_reports"](a, b, mode="repeat")["scientific_result"]
        == "not_assessed"
    )
    assert (a, b) == before
    with pytest.raises(ValueError, match="reused attempt"):
        TOOL["compare_reports"](a, a, mode="repeat")
    with pytest.raises(ValueError, match="replay changed"):
        TOOL["compare_reports"](a, b, mode="replay")


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "data",
        "attempt",
        "missing",
        "loss",
        "nan",
        "suite",
        "pending",
        "budget",
    ],
)
def test_repeat_refuses_changed_or_incomplete_evidence(damage):
    a, b = report(), repeat()
    cell = b["cells"][0]
    diagnostic = b["diagnostic_evaluations"][0]
    if damage == "source":
        b["provenance"]["source_identity_sha256"] = "changed"
    elif damage == "data":
        cell["inputs"] = ["other"]
    elif damage == "attempt":
        cell["attempts"] *= 2
    elif damage == "missing":
        cell["attempts"][0]["missing_report_count"] = 1
    elif damage in ("loss", "nan"):
        diagnostic["metrics"]["loss"] = 7.0 if damage == "loss" else float("nan")
    elif damage == "suite":
        diagnostic["suite_sha256"] = "changed"
    elif damage == "pending":
        cell["status"] = "pending"
    else:
        cell["planned_budget"]["max_steps"] = 2
    with pytest.raises(ValueError):
        TOOL["compare_reports"](a, b, mode="repeat")


def test_cli_reopens_both_native_inputs_and_propagates_verifier_refusal(
    tmp_path, monkeypatch, capsys
):
    from sparselab.experiments import publication_report

    calls = []

    def verify(lock, *, workspace, index_path):
        calls.append((lock, workspace, index_path))
        if len(calls) == 2:
            raise ValueError("tampered native index")
        return report()

    monkeypatch.setattr(publication_report, "build_publication_report", verify)
    with pytest.raises(SystemExit) as error:
        TOOL["main"](["repeat", "--left", "a", "b", "c", "--right", "d", "e", "f"])
    assert error.value.code == 1
    assert calls == [
        (Path("a"), Path("b"), Path("c")),
        (Path("d"), Path("e"), Path("f")),
    ]
    assert "tampered native index" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
