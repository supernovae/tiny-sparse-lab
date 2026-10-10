"""Static CI routing checks; this file never launches a model or workflow."""

from __future__ import annotations

import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NODES = (
    "tests/test_research_lint.py::test_actual_tracked_research_records",
    "tests/test_research_lint.py::test_exact_pinned_legacy_records_are_retained",
    "tests/test_research_lint.py::test_changed_legacy_bytes_are_rejected",
    "tests/test_research_lint.py::test_same_legacy_bytes_at_different_path_are_rejected",
    "tests/test_research_lint.py::test_additional_unapproved_legacy_artifact_is_rejected",
    "tests/test_preparation_commands.py",
    "tests/test_attempt_contract.py::test_zero_update_vector_reservations_persist_without_refund",
    "tests/test_operational_monitor_safety.py::test_absent_optional_fields_preserve_legacy_policy_and_receipt_serialization",
    "tests/test_corpus_mixture.py::test_native_packing_supervises_exact_exported_quota",
    "tests/test_panel_decoder_controls.py::test_absent_controls_preserve_legacy_declaration_and_hash",
    "tests/test_reviewed_score_readiness.py::test_renderer_and_exact_frozen_question_prompt_binding",
    "tests/test_reviewed_score_readiness.py::test_threshold_failure_and_missing_scores_cannot_promote",
    "tests/test_attempt_native_integration_optin.py::test_ordinary_pytest_skips_optimizer_node_before_setup",
    "tests/test_ci_routing.py::test_ordinary_ci_selects_only_explicit_zero_model_nodes",
)


def test_ordinary_ci_selects_only_explicit_zero_model_nodes() -> None:
    workflow_path = ROOT / ".github/workflows/ci.yml"
    workflow = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["push"] == {"branches": ["main"]}
    assert set(workflow["on"]["pull_request"]["types"]) == {
        "opened",
        "synchronize",
        "reopened",
        "converted_to_draft",
        "ready_for_review",
    }
    assert workflow["on"]["workflow_dispatch"]["inputs"]["suite"]["options"] == [
        "safe",
        "integration-linux",
        "platform-macos",
        "release-candidate",
    ]
    assert not (ROOT / ".github/workflows/kml-draft-integration.yml").exists()
    assert "tags" not in workflow["on"]["push"]
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    assert set(jobs) == {
        "safe",
        "integration-linux",
        "platform-macos",
        "release-candidate",
    }
    assert jobs["safe"]["if"] == (
        "github.event_name != 'workflow_dispatch' || inputs.suite == 'safe'"
    )
    assert jobs["safe"]["runs-on"] == "ubuntu-24.04"
    assert jobs["safe"]["timeout-minutes"] == "12"
    safe_steps = jobs["safe"]["steps"]
    assert safe_steps[3]["run"] == "uv sync --locked --extra cpu --dev"
    assert "ruff check ." in safe_steps[4]["run"]
    assert "ruff format --check ." in safe_steps[4]["run"]
    assert "sparselab.research.lint --json" in safe_steps[4]["run"]
    assert safe_steps[5]["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert safe_steps[5]["env"]["UV_OFFLINE"] == "1"
    command = shlex.split(safe_steps[5]["run"].replace("\\\n", " "))
    assert command[:7] == ["uv", "run", "--locked", "--no-sync", "pytest", "-q", "-p"]
    assert "no:cacheprovider" in command
    assert tuple(token for token in command if token.startswith("tests/")) == NODES
    assert not any(token in command for token in ("-k", "-m", "-n", "--pyargs"))

    for suite in ("integration-linux", "platform-macos", "release-candidate"):
        assert jobs[suite]["if"] == (
            f"github.event_name == 'workflow_dispatch' && inputs.suite == '{suite}'"
        )
    assert jobs["release-candidate"]["strategy"]["matrix"]["os"] == [
        "ubuntu-24.04",
        "macos-15",
    ]
    assert "tools/kernel-memory-lab/ci-guard/" not in workflow_path.read_text()
