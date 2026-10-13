"""Static CI routing checks; this file never launches a model or workflow."""

from __future__ import annotations

import shlex
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NODES = (
    "tests/test_research_lint.py::test_actual_tracked_research_records",
    "tests/test_research_lint.py::test_exact_pinned_legacy_records_are_retained",
    "tests/test_research_lint.py::test_changed_legacy_bytes_are_rejected",
    "tests/test_research_lint.py::test_same_legacy_bytes_at_different_path_are_rejected",
    "tests/test_research_lint.py::test_additional_unapproved_legacy_artifact_is_rejected",
    "tests/test_research_launch_surface.py",
    "tests/test_preparation_commands.py",
    "tests/test_verification_adapters.py",
    "tests/test_spot_safety.py",
    "tests/test_snapshot_warm_reuse.py",
    "tests/test_snapshot_inheritance.py",
    "tests/test_study_recovery.py",
    "tests/test_lab_job.py",
    "tests/test_corpus_acquisition.py::test_bounded_resource_readings_exact_expansion_cap",
    "tests/test_corpus_acquisition.py::test_bounded_overlong_line_retains_measured_prefix",
    "tests/test_corpus_acquisition.py::test_required_staging_reading_failure_is_closed",
    "tests/test_corpus_acquisition.py::test_parquet_required_expansion_reading_is_unavailable",
    "tests/test_corpus_acquisition.py::test_staging_counters_include_empty_interrupted_input",
    "tests/test_corpus_acquisition.py::test_wikimedia_interruption_retains_distinct_resource_receipts",
    "tests/test_corpus_acquisition.py::test_wikimedia_resource_journal_rejects_symlink",
    "tests/test_corpus_acquisition.py::test_hf_without_transport_ledger_retains_interrupted_shard",
    "tests/test_attempt_contract.py::test_zero_update_vector_reservations_persist_without_refund",
    "tests/test_operational_monitor_safety.py::test_absent_optional_fields_preserve_legacy_policy_and_receipt_serialization",
    "tests/test_corpus_mixture.py::test_native_packing_supervises_exact_exported_quota",
    "tests/test_panel_decoder_controls.py::test_absent_controls_preserve_legacy_declaration_and_hash",
    "tests/test_reviewed_score_readiness.py::test_renderer_and_exact_frozen_question_prompt_binding",
    "tests/test_reviewed_score_readiness.py::test_threshold_failure_and_missing_scores_cannot_promote",
    "tests/test_attempt_native_integration_optin.py::test_ordinary_pytest_skips_optimizer_node_before_setup",
    "tests/test_ci_routing.py::test_ordinary_ci_selects_only_explicit_zero_model_nodes",
)


LAB_NODES = (
    "tests/test_lab_job_native.py",
    "tests/test_preparation_execution.py",
    "tests/test_lab_mode.py::test_cpu_smoke_loop_yaml_to_report_within_budget",
    "tests/test_probes.py",
    "tests/test_references.py",
    "tests/test_probe_followups.py",
    "tests/test_explorer.py",
    "tests/test_dashboard_lab.py",
)
CPU_ENV = {
    **dict.fromkeys(
        (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "PYTHONDONTWRITEBYTECODE",
        ),
        "1",
    ),
    **dict.fromkeys(
        ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"), ""
    ),
}
EXCLUSIONS = "not mps and not mlx and not cuda and not rocm and not xpu and not network"
SYNC = "uv sync --locked --extra cpu --dev"


def _load(name):
    return yaml.load(
        (ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader
    )


def _step(job, **identity):
    """Require one semantic match, including when an action/command is unnamed."""
    matches = [
        step
        for step in job["steps"]
        if all(step.get(key) == value for key, value in identity.items())
    ]
    assert len(matches) == 1, (
        f"Expected one step matching {identity}, got {len(matches)}"
    )
    return matches[0]


def _tokens(step):
    return shlex.split(step["run"].replace("\\\n", " "))


def _before(job, first, second):
    assert job["steps"].index(first) < job["steps"].index(second)


def _test_job(workflow, key, name, timeout, *, nodes=None, full=False):
    job = workflow["jobs"][key]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == str(timeout)
    assert job.get("permissions", workflow["permissions"]) == {"contents": "read"}
    checkout = _step(job, uses="actions/checkout@v7")
    assert checkout["with"]["persist-credentials"] == "false"
    sync = _step(job, run=SYNC)
    step = _step(job, name=name)
    _before(job, checkout, sync)
    _before(job, sync, step)
    for item in (job, *job["steps"]):
        effective = workflow["env"] | job.get("env", {}) | item.get("env", {})
        assert all(effective.get(k) == v for k, v in CPU_ENV.items())
    assert step.get("env", {}).get("UV_OFFLINE") == "1"
    assert "if" not in step and "if" not in sync
    assert step.get("continue-on-error", "false") == "false"
    assert job.get("continue-on-error", "false") == "false"
    command = _tokens(step)
    prefix = ["uv", "run", "--locked", "--no-sync", "pytest", "-q"]
    if key == "lab-loop":
        prefix += ["-s"]
    prefix += [
        "-p",
        "no:cacheprovider",
        "--basetemp",
        f"$RUNNER_TEMP/{key if key != 'full-suite' else 'nightly'}-pytest",
    ]
    if key == "safe":
        prefix[-1] = "$RUNNER_TEMP/pr-safe-pytest"
    if nodes is not None:
        assert step["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert command[: len(prefix)] == prefix
        # Compare the entire remainder: duplicates, globs, options and paths fail.
        assert Counter(command[len(prefix) :]) == Counter(nodes)
    else:
        assert command == prefix + [
            "-n",
            "auto",
            "--maxprocesses=4",
            "--dist",
            "worksteal" if full else "loadfile",
            "--durations=50" if full else "--durations=25",
            "-m",
            EXCLUSIONS if full else f"not slow and {EXCLUSIONS}",
        ]
    return step


def _assert_ci(workflow):
    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert workflow["on"]["push"] == {"branches": ["main"]}
    assert workflow["on"]["pull_request"] == {
        "types": [
            "opened",
            "synchronize",
            "reopened",
            "converted_to_draft",
            "ready_for_review",
        ]
    }
    suite = workflow["on"]["workflow_dispatch"]["inputs"]["suite"]
    assert suite["options"] == [
        "safe",
        "fast",
        "integration-linux",
        "platform-macos",
        "release-candidate",
    ]
    assert suite["default"] == "safe" and suite["type"] == "choice"
    assert workflow["permissions"] == {"contents": "read"}
    assert all(workflow["env"].get(k) == v for k, v in CPU_ENV.items())
    jobs = workflow["jobs"]
    assert set(jobs) == {
        "safe",
        "lab-loop",
        "fast",
        "integration-linux",
        "platform-macos",
        "release-candidate",
    }
    for key in ("safe", "lab-loop", "fast"):
        choice = "safe" if key == "lab-loop" else key
        assert (
            jobs[key]["if"]
            == f"github.event_name != 'workflow_dispatch' || inputs.suite == '{choice}'"
        )
    for key in ("integration-linux", "platform-macos", "release-candidate"):
        assert (
            jobs[key]["if"]
            == f"github.event_name == 'workflow_dispatch' && inputs.suite == '{key}'"
        )
        assert jobs[key].get("permissions", workflow["permissions"]) == {
            "contents": "read"
        }
        assert jobs[key]["timeout-minutes"] == "30"
    assert jobs["release-candidate"]["strategy"]["matrix"]["os"] == [
        "ubuntu-24.04",
        "macos-15",
    ]
    safe = _test_job(
        workflow, "safe", "Explicit zero-model-work regressions", 12, nodes=NODES
    )
    static = _step(jobs["safe"], name="Static format and research records")
    assert static["env"]["UV_OFFLINE"] == "1"
    assert static["run"].splitlines() == [
        "uv run --locked --no-sync ruff check .",
        "uv run --locked --no-sync ruff format --check .",
        "uv run --locked --no-sync python -m sparselab.research.lint --json",
    ]
    _before(jobs["safe"], _step(jobs["safe"], run=SYNC), static)
    _before(jobs["safe"], static, safe)
    lab = _test_job(
        workflow,
        "lab-loop",
        "Timed lab loop (tokenizer -> try + probe -> report, references)",
        20,
        nodes=LAB_NODES,
    )
    assert lab["env"]["SPARSELAB_LAB_LOOP_BUDGET_SECONDS"] == "900"
    _test_job(workflow, "fast", "Fast CPU suite (not slow)", 20)
    assert "tools/kernel-memory-lab/ci-guard/" not in str(workflow)


def _assert_nightly(workflow):
    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["on"]["schedule"] == [{"cron": "17 8 * * *"}]
    assert workflow["permissions"] == {"contents": "read"}
    assert all(workflow["env"].get(k) == v for k, v in CPU_ENV.items())
    jobs = workflow["jobs"]
    assert set(jobs) == {"full-suite", "report-failure"}
    full = jobs["full-suite"]
    assert "if" not in full
    assert _step(full, uses="actions/checkout@v7")["with"]["ref"] == "main"
    _test_job(workflow, "full-suite", "Full CPU suite (slow included)", 60, full=True)
    report = jobs["report-failure"]
    assert report["needs"] == "full-suite"
    assert report["if"] == "failure()"
    assert report["permissions"] == {"issues": "write"}
    assert report["timeout-minutes"] == "5"
    script = _step(report, name="Open or update the nightly-failure issue")["run"]
    assert "gh issue comment" in script and "gh issue create" in script


def test_ordinary_ci_selects_only_explicit_zero_model_nodes():
    _assert_ci(_load("ci"))
    assert not (ROOT / ".github/workflows/kml-draft-integration.yml").exists()


def test_nightly_runs_full_suite_on_main_and_reports_failures():
    _assert_nightly(_load("nightly"))


@pytest.mark.parametrize(
    "name,check", [("ci", _assert_ci), ("nightly", _assert_nightly)]
)
def test_harmless_step_insertion_preserves_contract(name, check):
    workflow = _load(name)
    for job in workflow["jobs"].values():
        original = list(job["steps"])
        job["steps"] = [{"name": "Before setup", "run": "echo diagnostic"}]
        for step in original:
            job["steps"].extend(
                [step, {"name": "Diagnostic", "run": "echo diagnostic"}]
            )
    check(workflow)


@pytest.mark.parametrize("key,nodes", [("safe", NODES), ("lab-loop", LAB_NODES)])
@pytest.mark.parametrize(
    "change", ["reorder", "omit", "add", "duplicate", "wildcard", "broaden", "option"]
)
@pytest.mark.parametrize("node_order", ["original", "module-first"])
def test_exact_node_membership(key, nodes, change, node_order):
    workflow = _load("ci")
    job = workflow["jobs"][key]
    step = _step(
        job,
        name="Explicit zero-model-work regressions"
        if key == "safe"
        else "Timed lab loop (tokenizer -> try + probe -> report, references)",
    )
    command = _tokens(step)
    start = next(i for i, token in enumerate(command) if token.startswith("tests/"))
    selected = list(nodes)
    if node_order == "module-first":
        selected.sort(key=lambda node: "::" in node)
        assert "::" not in selected[0]
    # Establish that either ordering is valid before applying a regression.
    step["run"] = shlex.join(command[:start] + selected)
    _assert_ci(workflow)
    if change == "reorder":
        selected.reverse()
    elif change == "omit":
        selected.pop()
    elif change == "add":
        selected.append("tests/test_training.py")
    elif change == "duplicate":
        selected.append(nodes[0])
    elif change == "wildcard":
        selected[0] = "tests/test_*.py"
    elif change == "broaden":
        explicit = next(node for node in selected if "::" in node)
        selected[selected.index(explicit)] = explicit.split("::")[0]
    else:
        selected += ["-k", "smoke"]
    step["run"] = shlex.join(command[:start] + selected)
    if change == "reorder":
        _assert_ci(workflow)
    else:
        with pytest.raises(AssertionError):
            _assert_ci(workflow)


@pytest.mark.parametrize(
    "name,check", [("ci", _assert_ci), ("nightly", _assert_nightly)]
)
@pytest.mark.parametrize("kind", ["name", "uses", "run"])
@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_step_identity_must_be_unique(name, check, kind, change):
    workflow = _load(name)
    job = workflow["jobs"]["safe" if name == "ci" else "full-suite"]
    identity = {
        "name": "Explicit zero-model-work regressions"
        if name == "ci"
        else "Full CPU suite (slow included)",
        "uses": "actions/checkout@v7",
        "run": SYNC,
    }
    step = _step(job, **{kind: identity[kind]})
    if change == "missing":
        job["steps"].remove(step)
    else:
        job["steps"].append(deepcopy(step))
    with pytest.raises(AssertionError, match="Expected one step"):
        check(workflow)


@pytest.mark.parametrize(
    "name,check,key,test_name",
    [
        ("ci", _assert_ci, "fast", "Fast CPU suite (not slow)"),
        ("nightly", _assert_nightly, "full-suite", "Full CPU suite (slow included)"),
    ],
)
@pytest.mark.parametrize(
    "regression",
    [
        "new-route",
        "new-job",
        "write-permission",
        "job-write-permission",
        "credentials",
        "timeout",
        "threads",
        "device",
        "network",
        "job-env",
        "step-env",
        "online",
        "unlocked",
        "syncing",
        "worker-cap",
        "exclusions",
        "slow-selection",
        "sync-order",
        "skip-test",
        "ignore-failure",
    ],
)
def test_route_security_and_resource_regressions(
    name, check, key, test_name, regression
):
    workflow = _load(name)
    job = workflow["jobs"][key]
    step = _step(job, name=test_name)
    if regression == "new-route":
        workflow["on"]["pull_request_target"] = {}
    elif regression == "new-job":
        workflow["jobs"]["unreviewed"] = deepcopy(job)
    elif regression == "write-permission":
        workflow["permissions"]["contents"] = "write"
    elif regression == "job-write-permission":
        job["permissions"] = {"contents": "write"}
    elif regression == "credentials":
        _step(job, uses="actions/checkout@v7")["with"]["persist-credentials"] = "true"
    elif regression == "timeout":
        job["timeout-minutes"] = "120"
    elif regression in ("threads", "device", "network"):
        variable = {
            "threads": "OMP_NUM_THREADS",
            "device": "CUDA_VISIBLE_DEVICES",
            "network": "HF_HUB_OFFLINE",
        }[regression]
        workflow["env"][variable] = "8"
    elif regression == "job-env":
        job["env"] = {"MKL_NUM_THREADS": "8"}
    elif regression == "step-env":
        step["env"]["HIP_VISIBLE_DEVICES"] = "0"
    elif regression == "online":
        step["env"]["UV_OFFLINE"] = "0"
    elif regression == "sync-order":
        sync = _step(job, run=SYNC)
        job["steps"].remove(sync)
        job["steps"].append(sync)
    elif regression == "skip-test":
        step["if"] = "false"
    elif regression == "ignore-failure":
        job["continue-on-error"] = "true"
    else:
        old, new = {
            "unlocked": ("--locked", ""),
            "syncing": ("--no-sync", "--extra cpu"),
            "worker-cap": ("--maxprocesses=4", "--maxprocesses=8"),
            "exclusions": (" and not network", ""),
            "slow-selection": ("not slow and ", "")
            if name == "ci"
            else ("not mps", "not slow and not mps"),
        }[regression]
        assert old in step["run"]
        step["run"] = step["run"].replace(old, new)
    with pytest.raises(AssertionError):
        check(workflow)


@pytest.mark.parametrize(
    "key",
    [
        "safe",
        "lab-loop",
        "fast",
        "integration-linux",
        "platform-macos",
        "release-candidate",
    ],
)
def test_job_routes_cannot_be_broadened_or_disabled(key):
    workflow = _load("ci")
    workflow["jobs"][key]["if"] = "always()"
    with pytest.raises(AssertionError):
        _assert_ci(workflow)


@pytest.mark.parametrize(
    "regression",
    ["budget", "lab-offline", "lab-no-sync", "safe-plugin", "static-offline"],
)
def test_explicit_suite_bounds(regression):
    workflow = _load("ci")
    lab = _step(
        workflow["jobs"]["lab-loop"],
        name="Timed lab loop (tokenizer -> try + probe -> report, references)",
    )
    if regression == "budget":
        lab["env"]["SPARSELAB_LAB_LOOP_BUDGET_SECONDS"] = "1800"
    elif regression == "lab-offline":
        del lab["env"]["UV_OFFLINE"]
    elif regression == "lab-no-sync":
        lab["run"] = lab["run"].replace("--no-sync", "")
    elif regression == "safe-plugin":
        _step(workflow["jobs"]["safe"], name="Explicit zero-model-work regressions")[
            "env"
        ]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "0"
    else:
        _step(workflow["jobs"]["safe"], name="Static format and research records")[
            "env"
        ]["UV_OFFLINE"] = "0"
    with pytest.raises(AssertionError):
        _assert_ci(workflow)


@pytest.mark.parametrize(
    "regression",
    ["ref", "schedule", "dependency", "condition", "permissions", "report-command"],
)
def test_nightly_main_and_failure_reporting_contract(regression):
    workflow = _load("nightly")
    report = workflow["jobs"]["report-failure"]
    if regression == "ref":
        _step(workflow["jobs"]["full-suite"], uses="actions/checkout@v7")["with"][
            "ref"
        ] = "${{ github.ref }}"
    elif regression == "schedule":
        workflow["on"]["schedule"] = []
    elif regression == "dependency":
        report["needs"] = "unrelated"
    elif regression == "condition":
        report["if"] = "always()"
    elif regression == "permissions":
        report["permissions"]["contents"] = "write"
    else:
        _step(report, name="Open or update the nightly-failure issue")["run"] = (
            "echo ignored"
        )
    with pytest.raises(AssertionError):
        _assert_nightly(workflow)
