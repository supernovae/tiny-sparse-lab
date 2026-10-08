"""Static routing and allowlist checks; this file never launches CI or a model."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NORMAL_JOBS = ("lint", "fast", "evidence-macos-arm64", "serving-smoke")
DRAFT_JOBS = ("kml-draft-lint", "kml-draft-zero-update")
NODES = (
    "tests/test_research_lint.py::test_actual_tracked_research_records",
    "tests/test_research_lint.py::test_exact_pinned_legacy_records_are_retained",
    "tests/test_research_lint.py::test_changed_legacy_bytes_are_rejected",
    "tests/test_research_lint.py::test_same_legacy_bytes_at_different_path_are_rejected",
    "tests/test_research_lint.py::test_additional_unapproved_legacy_artifact_is_rejected",
    "tests/test_attempt_native_integration_optin.py::test_ordinary_pytest_skips_optimizer_node_before_setup",
    "tests/test_attempt_native_integration_optin.py::test_harness_context_selects_node_without_running_it",
    "tests/test_attempt_native_receipts.py::test_ledger_bound_train_suppresses_unaccounted_diagnostic_generation",
    "tests/test_kml_draft_ci.py::test_draft_workflow_routes_only_selected_zero_update_nodes",
)


def _workflow(name: str) -> dict:
    return yaml.load(
        (ROOT / ".github" / "workflows" / name).read_text(),
        Loader=yaml.BaseLoader,
    )


def _condition(expression: str, values: dict[str, str | bool]) -> bool:
    """Evaluate only the equality/AND/OR grammar used by these job conditions."""

    def atom(clause: str) -> bool:
        match = re.fullmatch(
            r"([a-zA-Z_.]+)\s*(==|!=)\s*('[^']+'|true|[a-zA-Z_.]+)", clause.strip()
        )
        assert match is not None, clause
        key, operator, raw = match.groups()
        assert key in values
        expected: str | bool = (
            True if raw == "true" else raw[1:-1] if raw.startswith("'") else values[raw]
        )
        equal = values[key] == expected
        return equal if operator == "==" else not equal

    return any(
        all(atom(part) for part in disjunction.split("&&"))
        for disjunction in expression.split("||")
    )


def test_draft_workflow_routes_only_selected_zero_update_nodes() -> None:
    normal = _workflow("ci.yml")
    draft = _workflow("kml-draft-integration.yml")
    assert normal["on"]["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in normal["on"]
    assert set(normal["on"]["pull_request"]["types"]) == {
        "opened",
        "synchronize",
        "reopened",
        "converted_to_draft",
        "ready_for_review",
    }
    assert draft["on"] == {
        "pull_request": {
            "types": ["opened", "synchronize", "reopened", "converted_to_draft"],
            "branches": ["main"],
        }
    }
    assert set(draft["jobs"]) == set(DRAFT_JOBS)
    assert normal["jobs"]["test"]["if"] == (
        "github.event_name == 'workflow_dispatch' && !inputs.serving_only"
    )
    assert normal["jobs"]["test-macos-arm64"]["if"] == (
        "github.event_name == 'workflow_dispatch' && !inputs.serving_only"
    )

    base = {
        "github.event_name": "pull_request",
        "github.event.pull_request.draft": True,
        "github.event.pull_request.head.repo.full_name": "owner/repo",
        "github.repository": "owner/repo",
        "github.event.pull_request.head.ref": "codex/kernel-memory-lab",
        "github.event.pull_request.base.ref": "main",
    }
    scenarios = (
        ({}, False, True),
        ({"github.event.pull_request.draft": False}, True, False),
        ({"github.event.pull_request.head.ref": "other"}, True, False),
        ({"github.event.pull_request.head.repo.full_name": "fork/repo"}, True, False),
        ({"github.event.pull_request.base.ref": "other"}, True, False),
        ({"github.event_name": "push"}, True, False),
        ({"github.event_name": "workflow_dispatch"}, True, False),
    )
    for changes, normal_expected, draft_expected in scenarios:
        values = base | changes
        for job in NORMAL_JOBS:
            assert _condition(normal["jobs"][job]["if"], values) is normal_expected
        for job in DRAFT_JOBS:
            assert _condition(draft["jobs"][job]["if"], values) is draft_expected
            assert draft["jobs"][job]["runs-on"] == "ubuntu-24.04"
    assert draft["jobs"]["kml-draft-lint"]["timeout-minutes"] == "5"
    assert draft["jobs"]["kml-draft-zero-update"]["timeout-minutes"] == "10"
    assert draft["permissions"] == {"contents": "read"}
    lint_command = draft["jobs"]["kml-draft-lint"]["steps"][-1]["run"]
    assert "ruff check ." in lint_command
    assert "ruff format --check ." not in lint_command
    assert "sparselab.research.lint --json" in lint_command

    check = draft["jobs"]["kml-draft-zero-update"]["steps"][-1]
    assert check["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert check["env"]["UV_OFFLINE"] == "1"
    command = shlex.split(check["run"].replace("\\\n", " "))
    assert command[:7] == ["uv", "run", "--locked", "--no-sync", "pytest", "-q", "-p"]
    assert "no:cacheprovider" in command
    assert tuple(token for token in command if token.startswith("tests/")) == NODES
    assert not any(token in command for token in ("-k", "-m", "-n", "--pyargs"))
