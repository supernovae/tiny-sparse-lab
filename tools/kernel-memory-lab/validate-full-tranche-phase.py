"""Fail closed on the one approved Card 05 full-tranche phase; no model work."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from sparselab.config.loading import load_config
from sparselab.operational_monitor import MonitorCompletion, load_monitor_policy
from sparselab.resource_envelope import load_resource_envelope
from sparselab.training.attempt_budget import AttemptBudget
from sparselab.training.manifest import config_sha256, sha256_file, source_identity

CONFIG_SHA256 = "4f22d18d80ba308b7358fc3d7375da5ecf4470f0d3bbddbb8cbc7e2e3d9e6a53"
FROZEN_SHA256 = "7b28f37de2c22121b27626462aafaa65921185fdabce606bb3016ac2a250bd55"
PANEL_SHA256 = "435dcb339c2e33bb623dc1bfbd9163a505521993f55b27640c0521b03caef0ef"
ORDER_SHA256 = "9a066c72f1ff500779f81e361458b1082cc84b4662713ed972a1b2edfed91cb4"
SUITE_SHA256 = "e0c75ec4ad3a6cbbdeb7e962f39366182ae5e42e16592ae587c6648208c47b92"
RUN_ID = "kml-card05-full-tranche-v2"
DISK_CAP = 68_719_476_736
RSS_CAP = 25_769_803_776


def validate(phase: str, root: Path, task_root: Path, checkout: Path, ledger: Path) -> None:
    if phase not in {"stage", "train", "evaluate"}:
        raise ValueError("invalid Card 05 full-tranche phase")
    root, task_root, checkout = (
        root.resolve(strict=True),
        task_root.resolve(strict=True),
        checkout.resolve(strict=True),
    )
    if not root.is_relative_to(task_root) or not ledger.resolve(strict=True).is_relative_to(root):
        raise ValueError("attempt and ledger must be inside monitored task root")
    budget = AttemptBudget(ledger)
    budget.remaining_seconds()
    status = budget.status()
    required = [("stage", 0)]
    if phase in {"train", "evaluate"}:
        required.append(("train", 4883))
    if phase == "evaluate":
        required.append(("evaluate", 0))
    reservations = status["reservations"]
    if (
        status["max_updates"] != 4883
        or status["max_wall_seconds"] > 10800
        or status["charged_updates"] != (0 if phase == "stage" else 4883)
        or len(reservations) != len(required)
        or any(
            row["updates"] != updates
            or not row["label"].endswith(f"run-full-tranche-phase.sh {name}")
            for row, (name, updates) in zip(reservations, required, strict=True)
        )
    ):
        raise ValueError("phase requires exact one-shot full-tranche reservations")
    started = datetime.fromisoformat(status["started_at_utc"]).timestamp()
    if phase in {"stage", "train"} and datetime.now(UTC).timestamp() >= started + 3000:
        raise ValueError("training phase deadline reached")
    if phase == "evaluate":
        marker = root / "evaluation-start-ns.txt"
        if not marker.is_file() or not marker.read_text().strip().isdecimal():
            raise ValueError("evaluation deadline marker missing")
        if int(marker.read_text().strip()) / 1e9 < started:
            raise ValueError("evaluation marker predates ledger")
        if datetime.now(UTC).timestamp() >= int(marker.read_text().strip()) / 1e9 + 7200:
            raise ValueError("evaluation phase deadline reached")

    baseline = [
        (root / f"profile-baseline-{name}.txt").read_text().strip()
        for name in ("bytes", "inodes")
    ]
    if any(not item.isdecimal() for item in baseline):
        raise ValueError("invalid common-root baseline")
    digest = hashlib.sha256(f"{baseline[0]}\n{baseline[1]}\n".encode()).hexdigest()
    if (root / "profile-baseline-sha256.txt").read_text().strip() != digest:
        raise ValueError("common-root baseline changed")
    if (root / "profile-baseline-sampler.txt").read_text().strip() != "sample-task-root-v1":
        raise ValueError("common-root baseline was not captured by the live-tree sampler")
    policy = load_monitor_policy(root / "profile-monitor-policy.yaml")
    envelope = load_resource_envelope(root / "profile-resource-envelope.yaml")
    if (
        policy.max_tree_rss_bytes is None
        or policy.max_tree_rss_bytes > RSS_CAP
        or policy.min_projected_disk_free_bytes is None
        or policy.min_projected_disk_free_bytes < 2_147_483_648
        or policy.min_projected_disk_free_inodes is None
        or policy.min_projected_disk_free_inodes < 2000
        or not 0 < policy.interval_seconds <= 1
        or envelope.max_rss_bytes is None
        or envelope.max_rss_bytes > RSS_CAP
        or envelope.min_disk_bytes is None
        or envelope.min_disk_bytes < DISK_CAP + 2_147_483_648
        or envelope.min_inodes is None
        or envelope.min_inodes < 4000
    ):
        raise ValueError("monitor or resource envelope loosens full-tranche caps")
    project = checkout / "experiments/research/kernel-memory-lab"
    config = load_config(project / "card05-full-tranche-v1.yaml")
    if config_sha256(config.model_dump(mode="json")) != CONFIG_SHA256:
        raise ValueError("full-tranche config changed")
    if source_identity()["sha256"] != "5fa9c2be187bf85467255febecc01d1e02e5a65215e2cb3ddc122f2778996a45":
        raise ValueError("native executable source identity changed")
    if (
        config.seed != 17
        or config.training.max_steps != 4883
        or config.training.max_tokens != 5_000_000
        or config.training.seq_len * config.training.micro_batch_size * config.training.gradient_accumulation != 1024
        or config.optimizer.warmup_steps != 100
        or config.optimizer.decay_steps != 4883
        or config.checkpoint.every_steps != 500
        or not config.checkpoint.keep_periodic
        or config.evaluation.every_steps != 500
        or config.evaluation.max_batches != 1
    ):
        raise ValueError("full-tranche scientific settings changed")
    for name, expected in (
        ("card05-language-panel-v1.json", PANEL_SHA256),
        ("card05-language-order-v1.json", ORDER_SHA256),
        ("card05-selected-evaluation-suite-v1.json", SUITE_SHA256),
    ):
        if sha256_file(project / name) != expected:
            raise ValueError(f"evaluation declaration changed: {name}")
    frozen = task_root / "card03-scale-operations/continuation2/eval-work/frozen-card03-evaluation-v1.json"
    if sha256_file(frozen) != FROZEN_SHA256:
        raise ValueError("frozen Card 03 evaluation changed")
    for path in (
        config.dataset.cache_dir,
        config.dataset.train_path,
        config.dataset.validation_path,
        config.dataset.mixture_output_path,
        config.tokenizer.path,
        config.logging.root_dir,
    ):
        if not path.resolve(strict=True).is_relative_to(task_root):
            raise ValueError("input or output escapes monitored task root")
    run = config.logging.root_dir / RUN_ID
    if phase in {"stage", "train"} and (run.exists() or run.is_symlink()):
        raise FileExistsError("unique full-tranche run already exists")
    if phase in {"train", "evaluate"}:
        if (root / "stage-exit-code.txt").read_text().strip() != "0":
            raise ValueError("training/evaluation requires completed stage")
        stage = MonitorCompletion.model_validate_json((root / "monitor-stage/completion.json").read_text())
        if stage.status != "COMPLETE" or stage.returncode != 0:
            raise ValueError("stage monitor did not complete")
    if phase == "evaluate":
        if (root / "train-exit-code.txt").read_text().strip() != "0":
            raise ValueError("evaluation requires completed training")
        train = MonitorCompletion.model_validate_json((root / "monitor-train/completion.json").read_text())
        if train.status != "COMPLETE" or train.returncode != 0:
            raise ValueError("train monitor did not complete")
        if not run.is_dir() or (root / "selected-checkpoint.json").is_file() is False:
            raise ValueError("verified selected checkpoint is missing")
        progress = json.loads((run / "progress.json").read_text())
        if progress.get("status") != "completed" or progress.get("step") != 4883 or progress.get("tokens_seen") != 5_000_000:
            raise ValueError("training did not reach exact target")


if __name__ == "__main__":
    validate(
        sys.argv[1],
        Path(os.environ["KML_PROFILE_ROOT"]),
        Path(os.environ["KML_TASK_ROOT"]),
        Path(os.environ["KML_CHECKOUT"]),
        Path(os.environ["SPARSELAB_ATTEMPT_BUDGET_LEDGER"]),
    )
