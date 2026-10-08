"""Fail closed on Card 05 measurement identity, limits, and phase order; no launch."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from sparselab.config.loading import load_config
from sparselab.config.models import RunConfig
from sparselab.operational_monitor import MonitorCompletion, load_monitor_policy
from sparselab.resource_envelope import load_resource_envelope
from sparselab.staging import _prepared_config_matches
from sparselab.training.attempt_budget import AttemptBudget
from sparselab.training.manifest import config_sha256, sha256_file, source_identity

_CONFIG_SHA256 = "9ab55b0e31e9486ebce9e4f342213ac5e1914396523742f99e9bdbd033979023"
_INPUTS_SHA256 = "aaa260d41ead37c6b347e4ec49cf57b16b6fc8c532820cc9a588f5d11bd47852"
_RUN_ID = "kml-card05-realdata-measurement-p1"
_DISK_CAP = 21_474_836_480
_RSS_CAP = 25_769_803_776


def validate(phase: str, root: Path, task_root: Path, checkout: Path, ledger: Path):
    if phase not in {"stage", "train"}:
        raise ValueError("invalid Card 05 measurement phase")
    root = root.resolve(strict=True)
    task_root = task_root.resolve(strict=True)
    checkout = checkout.resolve(strict=True)
    if not root.is_relative_to(task_root) or not ledger.resolve(
        strict=True
    ).is_relative_to(root):
        raise ValueError("attempt and ledger must be inside monitored task root")

    budget = AttemptBudget(ledger)
    budget.remaining_seconds()
    status = budget.status()
    reservations = status["reservations"]
    required = [("stage", 0)] if phase == "stage" else [("stage", 0), ("train", 32)]
    if (
        status["max_updates"] != 32
        or status["max_wall_seconds"] > 1800
        or status["charged_updates"] != (0 if phase == "stage" else 32)
        or len(reservations) != len(required)
        or any(
            item["updates"] != updates
            or not item["label"].endswith(f"run-real-data-measurement.sh {name}")
            for item, (name, updates) in zip(reservations, required, strict=True)
        )
    ):
        raise ValueError("phase requires exact fresh Card 05 budget reservations")

    baseline_bytes = (root / "profile-baseline-bytes.txt").read_text().strip()
    baseline_inodes = (root / "profile-baseline-inodes.txt").read_text().strip()
    if not baseline_bytes.isdecimal() or not baseline_inodes.isdecimal():
        raise ValueError("invalid common-root storage baseline")
    digest = hashlib.sha256(
        f"{baseline_bytes}\n{baseline_inodes}\n".encode()
    ).hexdigest()
    if (root / "profile-baseline-sha256.txt").read_text().strip() != digest:
        raise ValueError("common-root storage baseline changed")

    policy = load_monitor_policy(root / "profile-monitor-policy.yaml")
    if (
        policy.max_tree_rss_bytes is None
        or policy.max_tree_rss_bytes > _RSS_CAP
        or policy.min_projected_disk_free_bytes is None
        or policy.min_projected_disk_free_bytes < 2_147_483_648
        or policy.min_projected_disk_free_inodes is None
        or policy.min_projected_disk_free_inodes < 1000
        or not 0 < policy.interval_seconds <= 1
    ):
        raise ValueError("native monitor policy omits or loosens Card 05 caps")
    envelope = load_resource_envelope(root / "profile-resource-envelope.yaml")
    if (
        envelope.max_rss_bytes is None
        or envelope.max_rss_bytes > _RSS_CAP
        or envelope.min_disk_bytes is None
        or envelope.min_disk_bytes < _DISK_CAP + 2_147_483_648
        or envelope.min_inodes is None
        or envelope.min_inodes < 2000
    ):
        raise ValueError("resource envelope omits Card 05 headroom")

    config_path = (
        checkout
        / "experiments/research/kernel-memory-lab/card05-real-data-measurement.yaml"
    )
    config = load_config(config_path)
    if config_sha256(config.model_dump(mode="json")) != _CONFIG_SHA256:
        raise ValueError("Card 05 measurement config changed")
    prepared = (
        task_root
        / "corpora/kernel-memory-lab-card03-scale-retry1/prep/prepared-bundle-v1"
    )
    inputs_path = prepared / "inputs.json"
    if sha256_file(inputs_path) != _INPUTS_SHA256:
        raise ValueError("accepted prepared input identity changed")
    inputs = json.loads(inputs_path.read_text())
    if not _prepared_config_matches(
        RunConfig.model_validate(inputs["requested_config"]), config
    ):
        raise ValueError("sealed prepared config differs from Card 05 measurement")
    if inputs["source_identity_sha256"] != source_identity()["sha256"]:
        raise ValueError("prepared executable source identity changed")
    for path in (config.dataset.cache_dir, config.logging.root_dir, prepared):
        if not path.resolve().is_relative_to(task_root):
            raise ValueError("configured outputs or inputs leave monitored task root")
    run = config.logging.root_dir / _RUN_ID
    if run.exists() or run.is_symlink():
        raise FileExistsError("unique Card 05 measurement run already exists")
    if (
        config.training.seq_len
        * config.training.micro_batch_size
        * config.training.gradient_accumulation
        * 32
        > 32_768
    ):
        raise ValueError("Card 05 actual target-position ceiling could be exceeded")
    if config.seed != 17 or config.evaluation.max_batches != 1:
        raise ValueError("Card 05 seed or validation batch cap changed")

    if phase == "train":
        if (root / "stage-exit-code.txt").read_text().strip() != "0":
            raise ValueError("train requires successful zero-update stage supervision")
        completion = MonitorCompletion.model_validate_json(
            (root / "monitor-stage/completion.json").read_text()
        )
        if completion.status != "COMPLETE" or completion.returncode != 0:
            raise ValueError("train requires completed native stage monitor")
        # Native train authenticates the full stage bundle and prepared assets.


if __name__ == "__main__":
    validate(
        sys.argv[1],
        Path(os.environ["KML_PROFILE_ROOT"]),
        Path(os.environ["KML_TASK_ROOT"]),
        Path(os.environ["KML_CHECKOUT"]),
        Path(os.environ["SPARSELAB_ATTEMPT_BUDGET_LEDGER"]),
    )
