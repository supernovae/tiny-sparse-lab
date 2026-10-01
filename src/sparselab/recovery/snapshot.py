"""Fresh execution preflight without creating storage or modifying Git."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sparselab.experiments.plan import base_run_config, load_plan, read_document
from sparselab.recovery.provenance import (
    declaration_paths,
    declaration_reference,
    git_provenance,
)
from sparselab.workdir import resolve_work_dir, storage_checks
from sparselab.workspace_preflight import training_storage_checks


def snapshot(source: Path, work_dir: Path | None = None) -> dict[str, Any]:
    source = Path(source).absolute()
    root = resolve_work_dir(work_dir)
    raw = read_document(source)
    kinds = [
        (key, kind)
        for key, kind in (
            ("plan_version", "experiment"),
            ("campaign_version", "campaign"),
            ("recovery_version", "recovery"),
        )
        if key in raw
    ]
    if len(kinds) != 1:
        raise ValueError("UNKNOWN_DECLARATION: require exactly one strict version key")
    _, kind = kinds[0]
    provenance = git_provenance(declaration_paths(source, kind))
    reasons: list[str] = []
    if provenance["status"] != "CLEAN_AND_COMMITTED":
        reasons.append(provenance["status"])
    inputs: list[dict[str, Any]] = []
    recovery: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    storage: Any = {"status": "UNAVAILABLE"}
    evaluation: Any = None
    warnings = storage_checks(root)

    def inspect_plan(plan_source: Path) -> None:
        nonlocal runtime, storage, evaluation
        from sparselab.experiments.artifacts import verify_artifact
        from sparselab.experiments.lock import _runtime, resolve_plan

        plan = load_plan(plan_source)
        try:
            config = base_run_config(plan, plan_source)
        except (OSError, ValueError) as error:
            reasons.append("CONFIG_UNAVAILABLE")
            inputs.append(
                {
                    "id": plan.id,
                    "kind": "base_run",
                    "classification": "MISSING_EXTERNAL",
                    "reason": str(error),
                }
            )
            return
        try:
            config, _ = _runtime(config, plan.execution.backend)
        except ValueError as error:
            reasons.append(
                "RUNTIME_REQUIREMENT_UNDEFINED"
                if config.runtime.backend == "auto" and plan.execution.backend is None
                else "RUNTIME_REQUIREMENT_UNSUPPORTED"
            )
            inputs.append(
                {
                    "id": plan.id,
                    "kind": "runtime",
                    "classification": "MISSING_EXTERNAL",
                    "reason": str(error),
                }
            )
        else:
            candidate = {
                "engine": config.runtime.engine,
                "backend": config.runtime.backend,
                "device_index": config.runtime.device_index,
                "requirements": {},
            }
            if runtime is None:
                runtime = candidate
            elif any(
                runtime[key] != candidate[key]
                for key in ("engine", "backend", "device_index")
            ):
                reasons.append("RUNTIME_REQUIREMENT_MISMATCH")
        if plan.evaluation_suite:
            from sparselab.evaluation.suite import load_suite

            suite_source = declaration_reference(plan_source, plan.evaluation_suite)
            try:
                load_suite(suite_source)
                evaluation = {"suite": plan.evaluation_suite}
            except (OSError, ValueError) as error:
                reasons.append("EVALUATION_UNAVAILABLE")
                inputs.append(
                    {
                        "id": plan.evaluation_suite,
                        "kind": "evaluation_suite",
                        "classification": "MISSING_EXTERNAL",
                        "reason": str(error),
                    }
                )
        elif plan.evaluations:
            evaluation = [item.model_dump(mode="json") for item in plan.evaluations]
        else:
            reasons.append("EVALUATION_NOT_DECLARED")
        storage = [
            asdict(check)
            for check in training_storage_checks(
                config, work_dir=root, run_dir=root / "runs"
            )
        ]
        if any(check["status"] != "adequate" for check in storage):
            reasons.append("STORAGE_INADEQUATE")
        warnings.extend(storage_checks(config.dataset.cache_dir, kind="dataset_cache"))
        warnings.extend(storage_checks(config.logging.root_dir, kind="logging_root"))
        for name, artifact in plan.artifacts.items():
            if artifact.from_phase is not None:
                continue
            item = {
                "id": name,
                "kind": artifact.kind,
                "expected_sha256": artifact.sha256,
            }
            try:
                verified = verify_artifact(artifact, plan_source)
                item.update(classification="PRESENT", actual_sha256=verified["sha256"])
            except (OSError, ValueError) as error:
                item.update(classification="MISSING_EXTERNAL", reason=str(error))
                reasons.append("REQUIRED_ARTIFACT_UNVERIFIED")
            inputs.append(item)
        preparation = root / "experiments" / plan.id / "preparation.json"
        try:
            prepared = read_document(preparation) if plan.corpus_variants else None
            resolved = resolve_plan(plan, plan_source, prepared=prepared)
            inputs.append(
                {
                    "id": plan.id,
                    "kind": "experiment_lock",
                    "classification": "PRESENT",
                    "actual_sha256": resolved.plan_sha256,
                    "verification_scope": "resolved inputs; publication not required",
                }
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            reasons.append("REQUIRED_ARTIFACT_UNVERIFIED")
            inputs.append(
                {
                    "id": plan.id,
                    "kind": "experiment_lock",
                    "classification": "MISSING_RECONSTRUCTABLE",
                    "reason": str(error),
                }
            )

    if kind == "experiment":
        inspect_plan(source)
    elif kind == "campaign":
        from sparselab.campaign.engine import CampaignEngine

        engine = CampaignEngine(source, root)
        fresh = engine.inspect("plan")
        inputs = fresh["stages"]
        if engine.plan.recovery:
            from sparselab.recovery.engine import inspect_manifest
            from sparselab.recovery.manifest import load_manifest

            manifest_source = declaration_reference(source, engine.plan.recovery)
            manifest = load_manifest(manifest_source)
            recovery = inspect_manifest(manifest_source, engine.store.root)
            runtime = (
                manifest.runtime_requirement.model_dump(mode="json")
                if manifest.runtime_requirement
                else None
            )
        for stage in engine.plan.stages:
            if stage.kind == "experiment_plan":
                inspect_plan(declaration_reference(source, stage.source))
            if stage.kind == "evaluation":
                evaluation = {"suite": stage.suite}
        if any(
            row["state"] != "COMPLETE"
            for row in fresh["stages"]
            if row["kind"]
            in {
                "artifact_reference",
                "tokenizer_reference",
                "corpus_release",
                "experiment_plan",
                "runtime_acceptance",
            }
        ):
            reasons.append("REQUIRED_ARTIFACT_UNVERIFIED")
    else:
        from sparselab.recovery.engine import inspect_manifest
        from sparselab.recovery.manifest import load_manifest

        manifest = load_manifest(source)
        recovery = inspect_manifest(source, root)
        runtime = (
            manifest.runtime_requirement.model_dump(mode="json")
            if manifest.runtime_requirement
            else None
        )
        evaluation = manifest.evaluation_suite
        for step in manifest.steps:
            if step.kind == "experiment_lock":
                inspect_plan(declaration_reference(source, step.plan))
        rows = recovery.get("steps", recovery.get("artifacts", []))
        if any(
            row.get("classification") not in {"PRESENT", "NOT_CREATED"}
            and row.get("kind") != "optional_cache"
            for row in rows
        ):
            reasons.append("REQUIRED_ARTIFACT_UNVERIFIED")
    if runtime is None:
        reasons.append("RUNTIME_REQUIREMENT_UNDEFINED")
    if not evaluation:
        reasons.append("EVALUATION_NOT_DECLARED")
    if isinstance(storage, dict):
        reasons.append("STORAGE_UNAVAILABLE")
    return {
        "format": "sparselab-research-snapshot-v1",
        "state": "BLOCKED" if reasons else "READY_FOR_EXECUTION",
        "reason_codes": list(dict.fromkeys(reasons)),
        "source_commit": provenance["source_commit"],
        "provenance": provenance,
        "expected_inputs": inputs,
        "recoverability": recovery,
        "runtime_requirement": runtime,
        "evaluation": evaluation,
        "storage": storage,
        "storage_checks": warnings,
        "work_root": str(root),
    }


def _handle(args: argparse.Namespace) -> None:
    try:
        result = snapshot(Path(args.source), args.work_dir)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = {
            "format": "sparselab-research-snapshot-v1",
            "state": "BLOCKED",
            "reason_codes": ["INVALID_DECLARATION"],
            "reason": str(error),
        }
    print(
        json.dumps(result, sort_keys=True)
        if args.json
        else json.dumps(result, indent=2, sort_keys=True)
    )
    if result["state"] == "BLOCKED":
        raise SystemExit(1)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "snapshot", help="Read-only committed declaration and fresh execution preflight"
    )
    command.add_argument("source", type=Path)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
