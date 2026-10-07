"""Check Card 04 launch inputs using native budget and policy readers; no launch."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from sparselab.operational_monitor import MonitorCompletion, load_monitor_policy
from sparselab.resource_envelope import load_resource_envelope
from sparselab.training.attempt_budget import AttemptBudget


def validate(phase: str, root: Path, task_root: Path, checkout: Path, ledger: Path):
    if phase not in {"stage", "train"}:
        raise ValueError("invalid phase")
    budget = AttemptBudget(ledger)
    budget.remaining_seconds()
    status = budget.status()
    updates, charged = (7, 7) if phase == "stage" else (113, 120)
    reservations = status["reservations"]
    if (
        status["max_updates"] != 120
        or status["max_wall_seconds"] > 1800
        or status["charged_updates"] != charged
        or len(reservations) != (1 if phase == "stage" else 2)
        or reservations[-1]["updates"] != updates
        or not reservations[-1]["label"].endswith(f"run-profile-phase.sh {phase}")
        or reservations[0]["updates"] != 7
        or not reservations[0]["label"].endswith("run-profile-phase.sh stage")
    ):
        raise ValueError("phase requires its exact fresh attempt-budget reservation")
    policy = load_monitor_policy(root / "profile-monitor-policy.yaml")
    if (
        policy.max_tree_rss_bytes is None
        or policy.max_tree_rss_bytes > 25769803776
        or policy.min_projected_disk_free_bytes is None
        or policy.min_projected_disk_free_bytes < 2147483648
        or policy.min_projected_disk_free_inodes is None
        or policy.min_projected_disk_free_inodes < 1000
        or not 0 < policy.interval_seconds <= 1
    ):
        raise ValueError("native monitor policy omits or loosens Card 04 caps")
    load_resource_envelope(root / "profile-resource-envelope.yaml")
    config = yaml.safe_load(
        (
            checkout
            / "experiments/research/kernel-memory-lab/card04-synthetic-profile.yaml"
        ).read_text()
    )
    for path in (config["dataset"]["cache_dir"], config["logging"]["root_dir"]):
        if not Path(path).is_absolute() or not Path(path).resolve().is_relative_to(
            task_root.resolve()
        ):
            raise ValueError("configured outputs are outside the monitored task root")
    if phase == "train":
        if (root / "stage-exit-code.txt").read_text().strip() != "0":
            raise ValueError("train requires successful stage supervision")
        completion = MonitorCompletion.model_validate_json(
            (root / "monitor-stage/completion.json").read_text()
        )
        if completion.status != "COMPLETE" or completion.returncode != 0:
            raise ValueError("train requires a completed native stage monitor")
        # Native train independently authenticates the stage bundle and inputs.


if __name__ == "__main__":
    validate(
        sys.argv[1],
        Path(os.environ["KML_PROFILE_ROOT"]),
        Path(os.environ["KML_TASK_ROOT"]),
        Path(os.environ["KML_CHECKOUT"]),
        Path(os.environ["SPARSELAB_ATTEMPT_BUDGET_LEDGER"]),
    )
