"""Tiny checked-in corpus recipe survives total loss of its sibling state root."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sparselab.campaign.state import publish_immutable
from sparselab.config.loading import load_tokenizer_config
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.export import export_release
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze
from sparselab.data.tokenizer import train_tokenizer
from sparselab.evaluation.readiness import verify_readiness_result
from sparselab.evaluation.suite import verify_evaluation_index
from sparselab.experiments.lock import publish_lock, resolve_plan
from sparselab.experiments.plan import load_plan
from sparselab.experiments.prepare import prepare_plan
from sparselab.recovery.engine import (
    inspect_manifest,
    plan_manifest,
)
from sparselab.training.manifest import sha256_file


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _cli(repo: Path, root: Path, *args: str) -> dict:
    env = {
        **os.environ,
        "SPARSELAB_WORK_DIR": str(root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "sparselab", "--work-dir", str(root), *args, "--json"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def lost_state(tmp_path_factory: pytest.TempPathFactory):
    parent = tmp_path_factory.mktemp("lifecycle-recovery")
    repo, persistent = parent / "repo", parent / "persistent-state"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture Reviewer")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "examples/tiny-campaign", repo / "recipe"
    )
    splits = yaml.safe_load((repo / "recipe/splits.yaml").read_text())
    splits["assignments"]["tiny_docs_family"] = "validation"
    (repo / "recipe/splits.yaml").write_text(yaml.safe_dump(splits))
    base = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
        ).read_text()
    )
    base["name"] = "tiny-reconstructable-model"
    base["tokenizer"]["path"] = str(persistent / "tokenizer-not-chosen-yet.json")
    base["dataset"]["cache_dir"] = str(persistent / "cache/explicit")
    base["logging"]["root_dir"] = str(persistent / "runs")
    base["training"].update(max_steps=2, max_tokens=32, gradient_accumulation=1)
    base["optimizer"]["warmup_steps"] = 1
    base["checkpoint"]["every_steps"] = 1
    base["evaluation"]["every_steps"] = 1
    (repo / "base.yaml").write_text(yaml.safe_dump(base))
    (repo / "suite.yaml").write_text(
        yaml.safe_dump(
            {
                "evaluation_suite_version": 1,
                "id": "tiny-heldout",
                "evaluations": [
                    {"id": "heldout", "role": "gate", "kind": "heldout_lm"}
                ],
            }
        )
    )
    (repo / "readiness.yaml").write_text(
        yaml.safe_dump(
            {
                "readiness_version": 1,
                "id": "tiny-readiness",
                "require_verified_checkpoint": True,
                "required_gate_ids": ["heldout"],
                "min_completed_evaluations": 1,
                "max_heldout_loss": {"evaluation_id": "heldout", "value": 100.0},
                "require_human_review": False,
            }
        )
    )
    _commit(repo, "Declare tiny corpus, model base, suite and policy")
    project = load_project(repo / "recipe/corpus.yaml")
    acquire(project, persistent)
    release = freeze(build(project, persistent, offline=True), persistent)
    export = export_release(release, "lm", repo / "base.yaml", 300, persistent)
    tokenizer = train_tokenizer(load_tokenizer_config(export / "tokenizer.yaml"))
    tokenizer_sha = sha256_file(tokenizer)
    plan = {
        "plan_version": 1,
        "id": "tiny-replay",
        "base_run": "base.yaml",
        "evaluation_suite": "suite.yaml",
        "execution": {"backend": "cpu"},
        "artifacts": {
            "selected_tokenizer": {
                "kind": "tokenizer",
                "version": 1,
                "producer": "sparselab",
                "identifier": tokenizer.parent.name,
                "sha256": tokenizer_sha,
                "path": str(tokenizer),
            }
        },
        "corpus_variants": [
            {
                "id": "selected",
                "project": "recipe/corpus.yaml",
                "tokenizer_artifact": "selected_tokenizer",
            }
        ],
    }
    (repo / "plan.yaml").write_text(yaml.safe_dump(plan))
    plan_commit = _commit(
        repo, "Pin authenticated tokenizer artifact and corpus-bound ExperimentPlan"
    )
    recipe = {
        "recovery_version": 1,
        "id": "tiny-replay",
        "source_commit": plan_commit,
        "runtime_requirement": {
            "engine": "pytorch",
            "backend": "cpu",
            "device_index": 0,
            "requirements": {},
        },
        "evaluation_suite": "suite.yaml",
        "readiness_policy": "readiness.yaml",
        "steps": [
            {
                "id": "release",
                "kind": "corpus_release",
                "project": "recipe/corpus.yaml",
                "expected_release_sha256": release.name,
            },
            {
                "id": "export",
                "kind": "corpus_export",
                "corpus": "release",
                "base_run": "base.yaml",
                "view": "lm",
                "vocab_size": 300,
                "expected_export_sha256": export.name,
            },
            {
                "id": "tokenizer",
                "kind": "tokenizer_train",
                "export": "export",
                "expected_tokenizer_sha256": tokenizer_sha,
            },
            {
                "id": "prepared",
                "kind": "prepared_data",
                "plan": "plan.yaml",
                "variant_id": "selected",
            },
            {
                "id": "lock",
                "kind": "experiment_lock",
                "plan": "plan.yaml",
                "prepared": "prepared",
            },
            {
                "id": "checkpoint",
                "kind": "checkpoint",
                "run": "tiny-replay-run",
                "cell": "step_00000002",
            },
            {
                "id": "publication",
                "kind": "external_required",
                "role": "family",
                "reason": "human family declaration is not inferred from a model run",
            },
        ],
    }
    manifest = repo / "recovery.yaml"
    manifest.write_text(yaml.safe_dump(recipe))
    _commit(repo, "Declare pinned recovery recipe without future outcome digests")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SPARSELAB_WORK_DIR", str(persistent))
        preparation = prepare_plan(
            load_plan(repo / "plan.yaml"),
            repo / "plan.yaml",
            persistent / "experiments/tiny-replay",
        )
    variant = preparation["variants"][0]
    publish_immutable(
        persistent / "experiments/tiny-replay/preparation.json", preparation
    )
    locked = publish_lock(
        resolve_plan(
            load_plan(repo / "plan.yaml"), repo / "plan.yaml", prepared=preparation
        ),
        persistent / "experiments/tiny-replay",
    )
    expected = {
        "release": release.name,
        "export": export.name,
        "tokenizer": tokenizer_sha,
        "prepared": json.loads(
            (Path(variant["prepared_path"]) / "manifest.json").read_text()
        )["manifest_sha256"],
        "lock": locked.stem,
    }
    recipe["steps"][3]["expected_manifest_sha256"] = expected["prepared"]
    recipe["steps"][4]["expected_plan_sha256"] = expected["lock"]
    manifest.write_text(yaml.safe_dump(recipe))
    _commit(repo, "Publish prepared and locked compact identities")
    packed = json.loads((Path(variant["prepared_path"]) / "manifest.json").read_text())
    tokenizer_artifact = {
        "kind": "tokenizer",
        "version": 1,
        "producer": "sparselab",
        "identifier": tokenizer.parent.name,
        "sha256": tokenizer_sha,
        "path": str(tokenizer),
    }
    prepared_artifact = {
        "kind": "prepared_data",
        "version": 1,
        "producer": "sparselab",
        "identifier": packed["settings_sha256"],
        "sha256": packed["manifest_sha256"],
        "path": variant["prepared_path"],
    }
    campaign = repo / "campaign.yaml"
    campaign.write_text(
        yaml.safe_dump(
            {
                "campaign_version": 1,
                "id": "tiny-corpus-bound",
                "recovery": "recovery.yaml",
                "stages": [
                    {
                        "id": "corpus",
                        "kind": "corpus_release",
                        "scope": "corpus",
                        "project": "recipe/corpus.yaml",
                    },
                    {
                        "id": "tokenizer",
                        "kind": "tokenizer_reference",
                        "scope": "tokenizer",
                        "artifact": tokenizer_artifact,
                    },
                    {
                        "id": "prepared",
                        "kind": "artifact_reference",
                        "scope": "model",
                        "artifact": prepared_artifact,
                    },
                    {
                        "id": "plan",
                        "kind": "experiment_plan",
                        "scope": "model",
                        "requires": ["corpus", "tokenizer", "prepared"],
                        "source": "plan.yaml",
                        "mode": "reference",
                        "lock": str(locked),
                        "corpus": "corpus",
                        "tokenizer": "tokenizer",
                        "prepared": "prepared",
                    },
                    {
                        "id": "runtime",
                        "kind": "runtime_acceptance",
                        "scope": "runtime",
                        "requires": ["plan"],
                        "plan": "plan",
                    },
                    {
                        "id": "gate",
                        "kind": "approval",
                        "scope": "model",
                        "requires": ["runtime"],
                        "bind": ["corpus", "tokenizer", "prepared", "plan", "runtime"],
                    },
                    {
                        "id": "run",
                        "kind": "experiment_run",
                        "scope": "model",
                        "requires": ["plan", "runtime", "gate"],
                        "plan": "plan",
                        "runtime": "runtime",
                        "cell": "main:single",
                    },
                    {
                        "id": "collect",
                        "kind": "experiment_collect",
                        "scope": "evaluation",
                        "requires": ["plan", "run"],
                        "plan": "plan",
                        "run": "run",
                    },
                    {
                        "id": "evaluation",
                        "kind": "evaluation",
                        "scope": "evaluation",
                        "requires": ["collect"],
                        "collect": "collect",
                        "suite": "suite.yaml",
                    },
                    {
                        "id": "model",
                        "kind": "model_readiness",
                        "scope": "model",
                        "requires": ["evaluation"],
                        "evaluation": "evaluation",
                        "policy": "readiness.yaml",
                    },
                ],
            },
            sort_keys=False,
        )
    )
    _commit(repo, "Declare explicit approved corpus-bound Campaign")
    shutil.rmtree(persistent)
    yield repo, persistent, manifest, expected


def test_tiny_committed_recipe_survives_total_state_loss_then_explicit_run(lost_state):
    repo, root, manifest, expected = lost_state
    assert not root.exists()
    inspected = _cli(repo, root, "recovery", "inspect", str(manifest))
    assert not root.exists()
    rows = {row["id"]: row for row in inspected["steps"]}
    assert {
        rows[key]["classification"]
        for key in ("release", "export", "tokenizer", "prepared", "lock")
    } == {"MISSING_RECONSTRUCTABLE"}
    assert rows["checkpoint"]["classification"] == "NOT_CREATED"
    assert rows["release"]["expected_sha256"] == expected["release"]
    planned = _cli(repo, root, "recovery", "plan", str(manifest))
    assert not root.exists()
    assert planned["commands"]
    evidence_output = repo / "reconstruction-evidence.json"
    result = _cli(
        repo,
        root,
        "recovery",
        "reconstruct",
        str(manifest),
        "--evidence-output",
        str(evidence_output),
    )
    assert [row["id"] for row in result["outcomes"]] == [
        "release",
        "export",
        "tokenizer",
        "prepared",
        "lock",
    ][: len(result["outcomes"])]
    assert all(row["status"] == "PRESENT" for row in result["outcomes"])
    present = {row["id"]: row for row in inspect_manifest(manifest, root)["steps"]}
    for name, sha in expected.items():
        assert present[name]["actual_sha256"] == sha
        assert present[name]["classification"] == "PRESENT"
    assert present["checkpoint"]["classification"] == "NOT_CREATED"
    assert not (root / "runs/tiny-replay-run").exists()
    assert not (root / "workers").exists()
    rows = plan_manifest(manifest, root)["steps"]
    assert {row["actual_sha256"] for row in rows if row["id"] == "release"} == {
        expected["release"]
    }
    snapshot = _cli(repo, root, "research", "snapshot", str(repo / "plan.yaml"))
    assert snapshot["state"] == "READY_FOR_EXECUTION"
    assert snapshot["provenance"]["status"] == "CLEAN_AND_COMMITTED"
    original_plan = (repo / "plan.yaml").read_bytes()
    (repo / "unrelated-log.txt").write_text("Not a scientific declaration.\n")
    assert (
        _cli(repo, root, "research", "snapshot", str(repo / "plan.yaml"))["state"]
        == "READY_FOR_EXECUTION"
    )
    changed = yaml.safe_load(original_plan)
    changed["phases"] = [{"id": "main", "set": {"training.max_steps": 3}}]
    (repo / "plan.yaml").write_text(yaml.safe_dump(changed))
    absent_root = root.parent / "blocked-state"
    blocked = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(absent_root),
            "research",
            "snapshot",
            str(repo / "plan.yaml"),
            "--json",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked.returncode == 1
    assert "DIRTY_DECLARATION" in json.loads(blocked.stdout)["reason_codes"]
    assert not absent_root.exists()
    (repo / "plan.yaml").write_bytes(original_plan)
    _commit(repo, "Publish compact reconstruction evidence and unrelated observation")
    replay = _cli(
        repo,
        root,
        "recovery",
        "reconstruct",
        str(manifest),
        "--evidence-output",
        str(evidence_output),
    )
    assert replay["receipt"] == result["receipt"]
    assert (
        replay["provenance"]["source_commit"] != result["provenance"]["source_commit"]
    )
    assert not (root / "workers").exists()

    campaign = repo / "campaign.yaml"
    linked = _cli(repo, root, "campaign", "status", str(campaign))
    linked_rows = {row["id"]: row for row in linked["recoverability"]}
    for step_id in ("release", "export", "tokenizer", "prepared", "lock"):
        assert linked_rows[step_id]["classification"] == "PRESENT"
    assert linked_rows["checkpoint"]["classification"] == "NOT_CREATED"
    linked_replay = _cli(repo, root, "campaign", "reconstruct", str(campaign))
    replay_rows = {row["id"]: row for row in linked_replay["recoverability"]}
    for step_id in ("release", "export", "tokenizer", "prepared", "lock"):
        assert replay_rows[step_id]["classification"] == "PRESENT"
    assert not (root / "workers").exists()

    awaiting = _cli(repo, root, "campaign", "apply", str(campaign))
    assert awaiting["next_action"]["action"] == "approve"
    assert not list((root / "runs").glob("*/checkpoints/step_*"))
    _cli(
        repo,
        root,
        "campaign",
        "approve",
        str(campaign),
        "gate",
        "--note",
        "Authorized tiny CPU cell",
    )
    no_run = _cli(repo, root, "campaign", "apply", str(campaign))
    assert no_run["next_action"]["action"] == "execute_run"
    assert not list((root / "runs").glob("*/checkpoints/step_*"))
    submitted = _cli(
        repo,
        root,
        "campaign",
        "apply",
        str(campaign),
        "--execute-runs",
        "--max-wait-seconds",
        "0",
    )
    assert (
        next(row for row in submitted["stages"] if row["id"] == "run")["state"]
        == "RUNNING"
    )
    # Reconciliation of this explicit submission needs no replacement authorization.
    completed = _cli(
        repo,
        root,
        "campaign",
        "resume",
        str(campaign),
        "--max-wait-seconds",
        "600",
    )
    stages = {row["id"]: row for row in completed["stages"]}
    assert stages["run"]["state"] == "COMPLETE"
    assert stages["evaluation"]["state"] == "COMPLETE"
    assert stages["model"]["state"] == "COMPLETE"
    assert stages["model"]["outcome"] == "READY_FOR_NEXT_STAGE"
    assert (
        stages["corpus"]["outputs"][0]["sha256"]
        == inspect_manifest(manifest, root)["steps"][0]["actual_sha256"]
    )
    assert all(row["kind"] != "promotion" for row in completed["stages"])
    run_path = Path(stages["run"]["availability"]["path"])
    index_path = Path(stages["evaluation"]["availability"]["path"])
    generation = verify_evaluation_index(index_path)["checkpoint"].split("/")[-1]
    fresh = _cli(repo, root, "campaign", "status", str(campaign))
    fresh_rows = {row["id"]: row for row in fresh["recoverability"]}
    assert fresh_rows["observed_checkpoint:collect"]["classification"] == "PRESENT"
    assert (
        fresh_rows["observed_checkpoint:collect"]["actual_sha256"]
        == stages["collect"]["measurements"]["sha256"]
    )
    assert fresh_rows["observed_checkpoint:collect"]["generation"] == generation
    assert fresh_rows["checkpoint"]["classification"] == "NOT_CREATED"
    suite_result = _cli(
        repo,
        root,
        "evaluation",
        "suite",
        "run",
        str(repo / "suite.yaml"),
        run_path.name,
        "--checkpoint",
        generation,
        "--runs-dir",
        str(run_path.parent),
    )
    assert (
        suite_result["checkpoint_sha256"]
        == verify_evaluation_index(index_path)["checkpoint_sha256"]
    )
    readiness_result = _cli(
        repo, root, "readiness", "model", str(repo / "readiness.yaml"), str(index_path)
    )
    assert readiness_result["state"] == "READY_FOR_NEXT_STAGE"
    assert (
        verify_readiness_result(Path(readiness_result["output"]))["checkpoint_sha256"]
        == suite_result["checkpoint_sha256"]
    )
