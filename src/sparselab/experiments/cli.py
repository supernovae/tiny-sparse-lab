"""Authored experiment and immutable locked-plan CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sparselab.experiments.plan import base_run_config, load_plan
from sparselab.workdir import resolve_work_dir


def _emit(args: argparse.Namespace, data: dict[str, Any]) -> None:
    envelope = {
        "format": "sparselab-experiment-command-v1",
        "command": args.experiment_command,
        **data,
    }
    if args.json:
        print(json.dumps(envelope, sort_keys=True, allow_nan=False, default=str))
    else:
        for key, value in envelope.items():
            print(f"{key}: {json.dumps(value, sort_keys=True, default=str)}")


def _workspace(plan_id: str) -> Path:
    return resolve_work_dir() / "experiments" / plan_id


def _declaration(args: argparse.Namespace) -> tuple[Any, Path]:
    source = Path(args.source).absolute()
    return load_plan(source), source


def _preparation(workspace: Path) -> dict[str, Any] | None:
    path = workspace / "preparation.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("format") != "experiment-preparation-v1"
    ):
        raise ValueError(f"invalid preparation record: {path}")
    return value


def _inspect_declaration(args: argparse.Namespace) -> None:
    from sparselab.experiments.compiler import expand_axes, flatten_config

    plan, source = _declaration(args)
    if args.experiment_command == "diff" and plan.corpus_variants:
        from sparselab.experiments.lock import resolve_plan

        prepared = _preparation(_workspace(plan.id))
        if prepared is None:
            raise ValueError("prepare corpus variants before resolving comparisons")
        locked = resolve_plan(plan, source, prepared=prepared, max_runs=args.max_runs)
        _emit(
            args,
            {
                "id": plan.id,
                "scientific_sha256": locked.scientific_sha256,
                "comparisons": [
                    comparison.model_dump(mode="json")
                    for comparison in locked.comparisons
                ],
                "executable": False,
            },
        )
        return
    config = base_run_config(plan, source)
    axes = plan.axes
    if any(
        key.startswith("inputs.")
        for axis in axes
        for choice in axis.choices
        for key in choice.set
    ):
        # Typed selections are resolved by the lock compiler, never patched as RunConfig fields.
        axes = tuple(
            axis.model_copy(
                update={
                    "choices": tuple(
                        choice.model_copy(
                            update={
                                "set": {
                                    key: value
                                    for key, value in choice.set.items()
                                    if not key.startswith("inputs.")
                                }
                            }
                        )
                        for choice in axis.choices
                    )
                }
            )
            for axis in axes
        )
    cells = expand_axes(config, axes, max_runs=args.max_runs, base_dir=source.parent)
    payload: dict[str, Any] = {
        "id": plan.id,
        "plan_version": plan.plan_version,
        "cells": [
            {
                "coordinate": coordinate,
                "effective": flatten_config(cell, include_derived=True),
            }
            for cell, coordinate in cells
        ],
        "inputs": sorted(plan.inputs),
        "unresolved": ["external artifact identities", "selected runtime"]
        if plan.inputs
        or config.runtime.backend == "auto"
        or config.runtime.precision == "auto"
        else [],
        "executable": False,
    }
    if args.experiment_command == "inspect":
        from sparselab.experiments.lock import storage_preview

        payload["estimates"] = [
            storage_preview(cell, plan.retention.model_dump(mode="json"))
            for cell, _ in cells
        ]
    if args.experiment_command == "diff":
        from sparselab.experiments.compiler import compare_configs

        comparisons = []
        for comparison in plan.comparisons:
            base = [
                (cell, coordinate)
                for cell, coordinate in cells
                if all(coordinate.get(k) == v for k, v in comparison.baseline.items())
            ]
            variant = [
                (cell, coordinate)
                for cell, coordinate in cells
                if all(coordinate.get(k) == v for k, v in comparison.variant.items())
            ]
            if len(base) != 1 or len(variant) != 1:
                raise ValueError(
                    f"comparison {comparison.id} selectors must identify one cell on each side"
                )
            differences = compare_configs(
                base[0][0],
                variant[0][0],
                expected=comparison.interventions,
                invariants=comparison.invariants,
                include_derived=True,
            )
            comparisons.append(
                {
                    "id": comparison.id,
                    "mode": comparison.mode,
                    "differences": differences,
                    "invariants": "passed",
                }
            )
        payload["comparisons"] = comparisons
    _emit(args, payload)


def _prepare(args: argparse.Namespace) -> None:
    from sparselab.experiments.prepare import prepare_plan
    from sparselab.training.manifest import canonical_json

    plan, source = _declaration(args)
    from sparselab.recovery.provenance import declaration_preflight
    from sparselab.workdir import ensure_work_dir

    if not plan.evaluations and not plan.evaluation_suite:
        raise ValueError(
            "EVALUATION_NOT_DECLARED: prepare requires an evaluation declaration"
        )
    provenance = declaration_preflight(
        source, "experiment", args.allow_uncommitted_declaration
    )
    ensure_work_dir(args.work_dir)
    workspace = _workspace(plan.id)
    resource_envelope = args.resource_envelope_value
    record = prepare_plan(
        plan,
        source,
        workspace,
        resource_envelope=resource_envelope,
        tokenizer_batch_documents=args.tokenizer_batch_documents,
        tokenizer_batch_source_bytes=args.tokenizer_batch_source_bytes,
    )
    record["declaration_provenance"] = provenance
    record["storage_checks"] = getattr(args, "storage_checks", [])
    path = workspace / "preparation.json"
    encoded = canonical_json(record) + b"\n"
    if path.exists():
        from sparselab.experiments.plan import read_document

        previous = read_document(path)
        operational = {"declaration_provenance", "storage_checks"}
        if {
            key: value for key, value in previous.items() if key not in operational
        } != {key: value for key, value in record.items() if key not in operational}:
            raise ValueError(
                f"existing preparation record changed: {path}; use a new plan ID"
            )
    else:
        with path.open("xb") as handle:
            handle.write(encoded)
    _emit(
        args,
        {
            "id": plan.id,
            "preparation": str(path),
            "variants": [
                {key: value for key, value in variant.items() if key != "config"}
                for variant in record["variants"]
            ],
            "training_started": False,
        },
    )


def _lock(args: argparse.Namespace) -> None:
    from sparselab.experiments.lock import publish_lock, resolve_plan

    plan, source = _declaration(args)
    workspace = _workspace(plan.id)
    resolved = resolve_plan(
        plan, source, prepared=_preparation(workspace), max_runs=args.max_runs
    )
    path = publish_lock(resolved, workspace)
    _emit(
        args,
        {
            "id": plan.id,
            "lock": str(path),
            "scientific_sha256": resolved.scientific_sha256,
            "plan_sha256": resolved.plan_sha256,
        },
    )


def locked_cell_request(
    locked: Any,
    cell: Any,
    worker: str,
    workspace: Path,
    controller: Any,
    *,
    read_only: bool = False,
) -> dict[str, Any]:
    """Construct the same lock-bound worker request for CLI and campaign runs."""
    phase = next(phase for phase in locked.phases if phase.id == cell.phase)
    parent_path: Path | None = None
    parent_digest: str | None = None
    binding_digest: str | None = None
    if phase.parent is not None:
        from sparselab.experiments.binding import bind_generation
        from sparselab.training.manifest import sha256_file

        parent_id = f"{phase.parent}:{cell.id.split(':', 1)[1]}"
        matches = [
            row
            for row in controller.list_experiments()
            if (row["spec"].get("plan") or {}).get("plan_sha256") == locked.plan_sha256
            and (row["spec"].get("plan") or {}).get("cell_id") == parent_id
            and row["status"] == "COMPLETE"
            and row["ingestion_status"] == "COMPLETE"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"phase {cell.phase} needs one ingested parent {parent_id}"
            )
        binding, selected_parent = bind_generation(
            workspace,
            plan_sha256=locked.plan_sha256,
            cell_id=cell.id,
            parent_cell_id=parent_id,
            parent_run=controller.root / matches[0]["run_id"],
            selector=phase.selector,
            at_step=phase.at_step,
            full_state=phase.transition != "promote",
            read_only=read_only,
        )
        parent_path = Path(selected_parent["checkpoint_path"])
        parent_digest = selected_parent["checkpoint_sha256"]
        binding_digest = sha256_file(binding)
    elif phase.checkpoint is not None:
        parent_path = Path(locked.availability["artifacts"][phase.checkpoint])
        parent_digest = locked.artifacts[phase.checkpoint]["sha256"]
    request = {
        "config": cell.config,
        "worker": worker,
        "plan": {
            "plan_id": locked.id,
            "plan_sha256": locked.plan_sha256,
            "scientific_sha256": locked.scientific_sha256,
            "cell_id": cell.id,
            "phase_id": cell.phase,
            "coordinate": cell.coordinate,
            "config_sha256": cell.config_sha256,
            "parent_checkpoint_sha256": parent_digest,
            "execution_binding_sha256": binding_digest,
        },
    }
    if phase.transition in {"resume", "extend_budget"}:
        request[
            "extend_budget" if phase.transition == "extend_budget" else "resume"
        ] = parent_path
    elif phase.transition == "promote":
        request["promote"] = parent_path
    return request


def _run(args: argparse.Namespace) -> None:
    from sparselab.experiments.lock import open_lock
    from sparselab.workers.controller import Controller
    from sparselab.workers.models import WorkerDefinition

    locked = open_lock(Path(args.lock))
    from sparselab.recovery.provenance import declaration_preflight
    from sparselab.workdir import ensure_work_dir

    if not locked.evaluations and not locked.evaluation_suite:
        raise ValueError(
            "EVALUATION_NOT_DECLARED: run requires an evaluation declaration"
        )
    declaration_source = locked.availability.get("declaration_source")
    if declaration_source is None:
        raise ValueError("DECLARATION_SOURCE_UNAVAILABLE: re-lock the committed plan")
    provenance = declaration_preflight(
        Path(declaration_source), "experiment", args.allow_uncommitted_declaration
    )
    actual_hashes = {
        declaration["path"]: declaration["sha256"]
        for declaration in provenance["declarations"]
    }
    if actual_hashes != locked.availability.get("declaration_hashes"):
        raise ValueError(
            "DECLARATION_IDENTITY_CHANGED: authored inputs differ from the frozen lock"
        )
    ensure_work_dir(args.work_dir)
    workspace = _workspace(locked.id)
    controller = Controller(workspace / "controller")
    dependent = {phase.id for phase in locked.phases if phase.parent is not None}
    selected = [
        cell
        for cell in locked.cells
        if (args.cell is None or cell.id == args.cell)
        and (
            cell.phase == args.phase
            if args.phase is not None
            else cell.phase not in dependent
        )
    ]
    if not selected:
        raise ValueError("no locked cell matches --cell/--phase")
    worker = args.worker or locked.execution.get("worker")
    if worker is None:
        backend = selected[0].config.runtime.backend
        engine = selected[0].config.runtime.engine
        worker = f"plan-{locked.id}-{backend}"
        capability = controller.register(
            WorkerDefinition(
                worker_id=worker,
                name=worker,
                transport="local",
                python=Path(sys.executable).absolute(),
                root=workspace / "workers" / worker,
                engine=engine,
                backend=backend,
                device_index=selected[0].config.runtime.device_index,
            )
        )
    else:
        available = controller.workers(worker, refresh=True)
        if len(available) != 1:
            raise ValueError(f"selected worker {worker} is unavailable or ambiguous")
        capability = available[0]
    for cell in selected:
        runtime = cell.config.runtime
        if (
            capability.engine != runtime.engine
            or capability.backend != runtime.backend
            or capability.device_index != runtime.device_index
        ):
            raise ValueError(f"worker runtime differs from locked cell {cell.id}")
    requests = [
        locked_cell_request(locked, cell, worker, workspace, controller)
        for cell in selected
    ]
    for request in requests:
        request["declaration_provenance"] = provenance
        request["storage_checks"] = getattr(args, "storage_checks", [])
    submissions = controller.submit_many(requests)
    _emit(
        args,
        {
            "plan_sha256": locked.plan_sha256,
            "submissions": [item.model_dump(mode="json") for item in submissions],
        },
    )


def _collect(args: argparse.Namespace) -> None:
    from sparselab.experiments.evidence import collect_evidence
    from sparselab.experiments.lock import open_lock

    locked = open_lock(Path(args.lock))
    workspace = _workspace(locked.id)
    _emit(args, collect_evidence(locked, workspace))


def _explain(args: argparse.Namespace) -> None:
    from sparselab.experiments.lock import open_lock

    locked = open_lock(Path(args.lock))
    _emit(
        args,
        {
            "plan_sha256": locked.plan_sha256,
            "scientific_sha256": locked.scientific_sha256,
            "cells": [
                {
                    "id": cell.id,
                    "coordinate": cell.coordinate,
                    "phase": cell.phase,
                    "config_sha256": cell.config_sha256,
                    "artifacts": cell.artifacts,
                }
                for cell in locked.cells
            ],
        },
    )


def _reconstruct(args: argparse.Namespace) -> None:
    from sparselab.experiments.lock import open_lock
    from sparselab.experiments.retrospective import retrospective_views

    locked = open_lock(Path(args.lock))
    options = (
        {}
        if args.index is None
        else {"workspace": _workspace(locked.id), "index_path": Path(args.index)}
    )
    _emit(args, retrospective_views(locked, **options))


def _handle(args: argparse.Namespace) -> None:
    try:
        if args.experiment_command in {"validate", "inspect", "diff"}:
            _inspect_declaration(args)
        elif args.experiment_command == "prepare":
            _prepare(args)
        elif args.experiment_command == "lock":
            _lock(args)
        elif args.experiment_command == "run":
            _run(args)
        elif args.experiment_command == "collect":
            _collect(args)
        elif args.experiment_command == "explain":
            _explain(args)
        elif args.experiment_command == "reconstruct":
            _reconstruct(args)
        else:
            raise ValueError(
                f"unsupported experiment command {args.experiment_command}"
            )
    except (ValueError, OSError, TypeError, KeyError) as error:
        _emit(args, {"status": "error", "error": str(error)})
        raise SystemExit(2) from None


def add_commands(commands: argparse._SubParsersAction) -> None:
    for name in (
        "validate",
        "inspect",
        "diff",
        "prepare",
        "lock",
        "run",
        "collect",
        "explain",
        "reconstruct",
    ):
        command = commands.add_parser(name)
        command.add_argument(
            "source"
            if name in {"validate", "inspect", "diff", "prepare", "lock"}
            else "lock"
        )
        command.add_argument("--json", action="store_true")
        if name in {"prepare", "run"}:
            command.add_argument("--allow-uncommitted-declaration", action="store_true")
        if name in {"validate", "inspect", "diff", "lock"}:
            command.add_argument("--max-runs", type=int, default=1000)
        if name == "prepare":
            command.add_argument("--resource-envelope", type=Path)
            from sparselab.cli.main import _tokenizer_batch_arguments

            _tokenizer_batch_arguments(command)
        if name == "run":
            command.add_argument("--cell")
            command.add_argument("--phase")
            command.add_argument("--worker")
        if name == "reconstruct":
            command.add_argument("--index")
        command.set_defaults(handler=_handle)
