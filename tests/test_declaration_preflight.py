from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sparselab.recovery.provenance import (
    declaration_paths,
    declaration_preflight,
    git_provenance,
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def committed_plan(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Recovery fixture")
    (root / "run.yaml").write_text("seed: 7\n")
    (root / "plan.yaml").write_text(
        "plan_version: 1\nid: committed\nbase_run: run.yaml\n"
    )
    git(root, "add", ".")
    git(root, "commit", "-qm", "Declare scientific inputs")
    return root / "plan.yaml"


def test_only_exact_closure_is_gated(committed_plan: Path) -> None:
    source = committed_plan
    (source.parent / "unrelated.md").write_text("Not a scientific input\n")
    paths = declaration_paths(source, "experiment")
    assert {path.name for path in paths} == {"plan.yaml", "run.yaml"}
    before = git(source.parent, "rev-parse", "HEAD")
    assert (
        declaration_preflight(source, "experiment")["status"] == "CLEAN_AND_COMMITTED"
    )
    source.write_text(source.read_text().replace("committed", "changed"))
    with pytest.raises(ValueError, match="DIRTY_DECLARATION"):
        declaration_preflight(source, "experiment")
    override = declaration_preflight(source, "experiment", True)
    assert override["allow_uncommitted_declaration"] is True
    assert (
        override["declarations"][0]["sha256"]
        != override["declarations"][0]["head_sha256"]
    )
    git(source.parent, "add", "plan.yaml")
    assert git_provenance(paths)["status"] == "DIRTY_DECLARATION"
    assert git(source.parent, "rev-parse", "HEAD") == before


def test_head_not_index_and_deleted_input(committed_plan: Path) -> None:
    root = committed_plan.parent
    novel = root / "suite.yaml"
    novel.write_text("evaluation_suite_version: 1\n")
    git(root, "add", "suite.yaml")
    assert git_provenance((novel,))["status"] == "UNTRACKED_DECLARATION"
    run = root / "run.yaml"
    run.unlink()
    status = git_provenance((run,))
    assert status["status"] == "DIRTY_DECLARATION"
    assert status["declarations"][0]["sha256"] is None
    assert status["declarations"][0]["head_sha256"] is not None


def test_unknown_git_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "plan.yaml"
    source.write_text("plan_version: 1\nid: no-git\nbase_run: run.yaml\n")
    (tmp_path / "run.yaml").write_text("seed: 7\n")
    with pytest.raises(ValueError, match="UNKNOWN"):
        declaration_preflight(source, "experiment")
    overridden = declaration_preflight(source, "experiment", True)
    assert {row["path"] for row in overridden["declarations"]} == {
        "plan.yaml",
        "run.yaml",
    }
    assert all(row["sha256"] is not None for row in overridden["declarations"])
    git(tmp_path, "init", "-q")
    assert git_provenance((source,))["status"] == "UNKNOWN"


def test_closure_rejects_traversal_and_symlinks(committed_plan: Path) -> None:
    source = committed_plan
    source.write_text("plan_version: 1\nid: unsafe\nbase_run: ../run.yaml\n")
    with pytest.raises(ValueError, match="unsafe declaration-relative"):
        declaration_paths(source, "experiment")
    source.write_text("plan_version: 1\nid: unsafe\nbase_run: linked.yaml\n")
    (source.parent / "linked.yaml").symlink_to(source.parent / "run.yaml")
    with pytest.raises(ValueError, match="symlinked declaration"):
        declaration_paths(source, "experiment")


def test_checkout_location_does_not_change_provenance(
    committed_plan: Path, tmp_path: Path
) -> None:
    first = declaration_preflight(committed_plan, "experiment")
    clone = tmp_path / "second-checkout"
    subprocess.run(
        ["git", "clone", "--quiet", str(committed_plan.parent), str(clone)], check=True
    )
    git(clone, "switch", "-qc", "different-branch")
    second = declaration_preflight(clone / "plan.yaml", "experiment")
    assert second == first


def test_suite_is_in_the_committed_closure(committed_plan: Path) -> None:
    root = committed_plan.parent
    committed_plan.write_text(
        committed_plan.read_text() + "evaluation_suite: suite.yaml\n"
    )
    (root / "suite.yaml").write_text(
        "evaluation_suite_version: 1\nid: heldout\n"
        "evaluations:\n  - id: loss\n    role: gate\n    kind: heldout_lm\n"
    )
    git(root, "add", "plan.yaml")
    git(root, "commit", "-qm", "Declare suite reference")
    with pytest.raises(ValueError, match="UNTRACKED_DECLARATION"):
        declaration_preflight(committed_plan, "experiment")
    git(root, "add", "suite.yaml")
    git(root, "commit", "-qm", "Publish evaluation declaration")
    assert (
        declaration_preflight(committed_plan, "experiment")["status"]
        == "CLEAN_AND_COMMITTED"
    )


def test_dirty_cli_prepare_has_no_workspace_and_override_is_recorded(
    committed_plan: Path, tmp_path: Path
) -> None:
    import json
    import os
    import sys

    root = committed_plan.parent
    state = tmp_path / "persistent-state"
    repository = Path(__file__).resolve().parents[1]
    (root / "run.yaml").write_text(
        (repository / "configs/runtime_smoke_cpu.yaml").read_text()
    )
    (root / "suite.yaml").write_text(
        "evaluation_suite_version: 1\nid: heldout\n"
        "evaluations:\n  - id: loss\n    role: gate\n    kind: heldout_lm\n"
    )
    committed_plan.write_text(
        committed_plan.read_text() + "evaluation_suite: suite.yaml\n"
    )
    git(root, "add", ".")
    git(root, "commit", "-qm", "Bind evaluable preparation inputs")
    committed_plan.write_text(
        committed_plan.read_text().replace("id: committed", "id: changed")
    )
    command = [
        sys.executable,
        "-c",
        "from sparselab.cli.main import main; main()",
        "--work-dir",
        str(state),
        "experiment",
        "prepare",
        str(committed_plan),
        "--json",
    ]
    blocked = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=repository,
        env=os.environ.copy(),
        check=False,
    )
    assert blocked.returncode != 0
    assert "DIRTY_DECLARATION" in json.loads(blocked.stdout)["error"]
    assert not state.exists()
    allowed = subprocess.run(
        command + ["--allow-uncommitted-declaration"],
        capture_output=True,
        text=True,
        cwd=repository,
        env=os.environ.copy(),
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    preparation = json.loads(
        (state / "experiments/changed/preparation.json").read_text()
    )
    proof = preparation["declaration_provenance"]
    assert proof["status"] == "DIRTY_DECLARATION"
    assert proof["allow_uncommitted_declaration"] is True
    assert proof["source_commit"] == git(root, "rev-parse", "HEAD")
    assert not (state / "experiments/changed/controller").exists()


def test_later_published_evidence_does_not_repin_earlier_inputs(
    committed_plan: Path, tmp_path: Path
) -> None:
    import json

    from sparselab.recovery.evidence import export_evidence
    from sparselab.recovery.provenance import verify_source_commit

    root = committed_plan.parent
    input_commit = git(root, "rev-parse", "HEAD")
    observation = tmp_path / "runtime-observation.json"
    observation.write_text(json.dumps({"format_version": 1, "ok": True}))
    reference = root / "runtime-reference.json"
    export_evidence(
        "runtime_probe",
        observation,
        reference,
        source_commit=input_commit,
        declaration_hashes=declaration_preflight(committed_plan, "experiment")[
            "declarations"
        ],
    )
    recipe = root / "recovery.json"
    recipe.write_text(
        json.dumps(
            {
                "recovery_version": 1,
                "id": "published",
                "source_commit": input_commit,
                "evidence": ["runtime-reference.json"],
                "steps": [
                    {"id": "lock", "kind": "experiment_lock", "plan": "plan.yaml"},
                    {
                        "id": "family-choice",
                        "kind": "external_required",
                        "role": "family",
                        "reason": "No reviewed model-family declaration is supplied.",
                    },
                ],
            }
        )
    )
    git(root, "add", ".")
    git(root, "commit", "-qm", "Publish later compact evidence")
    closure = declaration_preflight(recipe, "recovery")
    assert "runtime-reference.json" in {row["path"] for row in closure["declarations"]}
    verified = verify_source_commit(recipe, input_commit)
    assert set(verified["verified_paths"]) == {"plan.yaml", "run.yaml"}
    reference.write_bytes(reference.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="DIRTY_DECLARATION"):
        declaration_preflight(recipe, "recovery")
    git(root, "checkout", "--", "runtime-reference.json")
    (root / "run.yaml").write_text("seed: 8\n")
    git(root, "add", "run.yaml")
    git(root, "commit", "-qm", "Change scientific input")
    with pytest.raises(ValueError, match="SOURCE_COMMIT_MISMATCH: run.yaml"):
        verify_source_commit(recipe, input_commit)


def test_snapshot_honors_explicit_cpu_selection_for_auto_runtime(
    committed_plan: Path, tmp_path: Path
) -> None:
    import yaml

    from sparselab.recovery.snapshot import snapshot

    root = committed_plan.parent
    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repository / "configs/runtime_smoke_cpu.yaml").read_text())
    config["runtime"]["backend"] = "auto"
    (root / "run.yaml").write_text(yaml.safe_dump(config))
    (root / "suite.yaml").write_text(
        "evaluation_suite_version: 1\nid: heldout\n"
        "evaluations:\n  - id: loss\n    role: gate\n    kind: heldout_lm\n"
    )
    authored = {
        "plan_version": 1,
        "id": "committed",
        "base_run": "run.yaml",
        "evaluation_suite": "suite.yaml",
        "execution": {"backend": "cpu"},
    }
    committed_plan.write_text(yaml.safe_dump(authored))
    git(root, "add", ".")
    git(root, "commit", "-qm", "Select CPU for automatic base runtime")
    state = tmp_path / "selected-state"
    ready = snapshot(committed_plan, state)
    assert ready["state"] == "BLOCKED"
    assert ready["reason_codes"] == ["REQUIRED_ARTIFACT_UNVERIFIED"]
    assert ready["runtime_requirement"]["backend"] == "cpu"
    assert not state.exists()
    authored.pop("execution")
    committed_plan.write_text(yaml.safe_dump(authored))
    git(root, "add", "plan.yaml")
    git(root, "commit", "-qm", "Leave runtime genuinely undefined")
    blocked = snapshot(committed_plan, state)
    assert "RUNTIME_REQUIREMENT_UNDEFINED" in blocked["reason_codes"]
    assert blocked["state"] == "BLOCKED"
    assert not state.exists()


def test_future_review_and_evidence_outputs_do_not_block_committed_execution(
    committed_plan: Path,
) -> None:
    root = committed_plan.parent
    committed_plan.write_text(
        committed_plan.read_text() + "evaluation_suite: suite.yaml\n"
    )
    (root / "suite.yaml").write_text(
        "evaluation_suite_version: 1\nid: outputs\nevaluations:\n"
        "- {id: loss, role: gate, kind: heldout_lm}\n"
        "- {id: human, role: blinded_surface, kind: surface_review, source: pending-review}\n"
        "- {id: evidence, role: descriptive, kind: evidence_reference, source: later.json}\n"
    )
    git(root, "add", "plan.yaml", "suite.yaml")
    git(root, "commit", "-qm", "Declare evaluations before model outputs exist")
    assert (
        declaration_preflight(committed_plan, "experiment")["status"]
        == "CLEAN_AND_COMMITTED"
    )
    (root / "pending-review").mkdir()
    (root / "pending-review/progress.txt").write_text("Review is not sealed.\n")
    assert (
        declaration_preflight(committed_plan, "experiment")["status"]
        == "CLEAN_AND_COMMITTED"
    )
