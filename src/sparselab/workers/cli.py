"""Command handlers for independent local and SSH workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from sparselab.config.loading import load_config
from sparselab.experiments.matrix import expand
from sparselab.training.manifest import config_sha256

_BACKENDS = ("cpu", "mps", "cuda", "rocm", "xpu", "metal")
_ENGINES = ("pytorch", "mlx")
_TERMINAL_QUEUE_STATES = frozenset({"CANCELLED", "COMPLETE", "FAILED", "INTERRUPTED"})


def _json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str))


def _controller(store: Path, *, transfer_timeout: float = 1800):
    from sparselab.workers.controller import Controller

    return Controller(store, transfer_timeout=transfer_timeout)


def _absolute(value: str, *, argument: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{argument} must be an absolute path")
    return path


def _worker_definition(args: argparse.Namespace):
    from sparselab.workers.models import WorkerDefinition

    store = Path(args.store).resolve()
    ssh_host = args.ssh
    if ssh_host is not None:
        python = _absolute(args.python, argument="--python") if args.python else None
        root = _absolute(args.root, argument="--root") if args.root else None
        if python is None or root is None:
            raise ValueError("SSH worker registration requires --python and --root")
        transport = "ssh"
    else:
        transport = "local"
        python = (
            _absolute(args.python, argument="--python")
            if args.python
            else Path(sys.executable).absolute()
        )
        root = (
            _absolute(args.root, argument="--root")
            if args.root
            else (store.parent / "workers" / args.name).resolve()
        )
    return WorkerDefinition(
        schema_version=1,
        worker_id=args.name,
        name=args.name,
        transport=transport,
        host=ssh_host,
        python=python,
        root=root,
        engine=args.engine,
        backend=args.backend,
        device_index=args.device_index,
    )


def _worker_register(args: argparse.Namespace) -> None:
    capability = _controller(Path(args.store)).register(_worker_definition(args))
    _json(capability.model_dump(mode="json"))


def _worker_status(args: argparse.Namespace) -> None:
    capabilities = _controller(Path(args.store)).workers(args.name, refresh=True)
    payload = [capability.model_dump(mode="json") for capability in capabilities]
    _json(payload if args.name is None else (payload[0] if payload else None))


def _serve_stdio(args: argparse.Namespace) -> None:
    """Run the finite stdio endpoint used by both local and SSH transport."""
    from sparselab.workers.agent import serve_stdio
    from sparselab.workers.models import WorkerDefinition

    definition = WorkerDefinition(
        schema_version=1,
        worker_id=args.worker_id,
        name=args.name,
        transport="local",
        host=None,
        python=Path(sys.executable).absolute(),
        root=_absolute(args.root, argument="--root"),
        engine=args.engine,
        backend=args.backend,
        device_index=args.device_index,
    )
    serve_stdio(definition)


def _execute(args: argparse.Namespace) -> None:
    from sparselab.workers.agent import execute_attempt
    from sparselab.workers.models import WorkerDefinition

    definition = WorkerDefinition(
        schema_version=1,
        worker_id=args.worker_id,
        name=args.name,
        transport="local",
        host=None,
        python=Path(sys.executable).absolute(),
        root=_absolute(args.root, argument="--root"),
        engine=args.engine,
        backend=args.backend,
        device_index=args.device_index,
    )
    execute_attempt(definition, args.attempt_id)


def _matrix_requests(args: argparse.Namespace) -> list[dict[str, object]]:
    expanded = expand(Path(args.matrix), max_runs=args.max_runs)
    if args.dry_run:
        _json(
            [
                {
                    "coordinate": item.coordinate,
                    "config_sha256": config_sha256(item.config.model_dump(mode="json")),
                    "matrix_sha256": item.matrix_sha256,
                    "preferred_worker": item.preferred_worker,
                    "requirements": item.requirements.model_dump(mode="json"),
                }
                for item in expanded
            ]
        )
        return []
    return [
        {
            "config": item.config,
            "worker": args.worker,
            "stage_bundle": Path(args.stage_bundle) if args.stage_bundle else None,
            "promote": Path(args.promote) if args.promote else None,
            "requirements": item.requirements,
            "preferred": item.preferred_worker,
            "matrix": {
                "matrix_sha256": item.matrix_sha256,
                "coordinate": item.coordinate,
            },
        }
        for item in expanded
    ]


def _experiment_submit(args: argparse.Namespace) -> None:
    if bool(args.config) == bool(args.matrix):
        raise ValueError("provide exactly one of CONFIG or --matrix")
    if args.matrix:
        requests = _matrix_requests(args)
        if args.dry_run:
            return
        submissions = _controller(Path(args.store)).submit_many(requests)
        _json([submission.model_dump(mode="json") for submission in submissions])
        return
    if args.dry_run:
        raise ValueError("--dry-run is only available with --matrix")
    config = load_config(Path(args.config))
    submission = _controller(Path(args.store)).submit(
        config,
        worker=args.worker,
        stage_bundle=Path(args.stage_bundle) if args.stage_bundle else None,
        promote=Path(args.promote) if args.promote else None,
    )
    _json(submission.model_dump(mode="json"))


def _experiment_list(args: argparse.Namespace) -> None:
    _json(_controller(Path(args.store)).list_experiments())


def _experiment_cancel(args: argparse.Namespace) -> None:
    _json(_controller(Path(args.store)).cancel(args.run_id))


def _experiment_resume(args: argparse.Namespace) -> None:
    submission = _controller(Path(args.store)).resume(
        args.run_id,
        worker=args.worker,
        allow_runtime_drift=args.allow_runtime_drift,
    )
    _json(submission.model_dump(mode="json"))


def _controller_run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _controller(Path(args.store), transfer_timeout=args.transfer_timeout).run()


def _wait_for_run(controller: Any, run_id: str) -> dict[str, object]:
    """Observe an existing controller, or exclusively drive the submitted run."""
    previous: tuple[object, ...] | None = None
    with controller._process_lock(optional=True) as acquired:
        while True:
            if acquired:
                controller.tick()
            row = next(
                (
                    item
                    for item in controller.list_experiments()
                    if item.get("run_id") == run_id
                ),
                None,
            )
            if row is None:
                raise RuntimeError(
                    f"submitted run {run_id} disappeared from controller"
                )
            state = (
                row.get("status"),
                row.get("queued_reason"),
                row.get("ingestion_status"),
                row.get("ingestion_error"),
            )
            if state != previous:
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": state[0],
                            "queued_reason": state[1],
                            "ingestion_status": state[2],
                            "ingestion_error": state[3],
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                previous = state
            if state[0] in _TERMINAL_QUEUE_STATES and (
                state[2] in {"COMPLETE", "NOT_REQUIRED"}
            ):
                return row
            time.sleep(controller.poll_seconds)


def _run(args: argparse.Namespace) -> None:
    controller = _controller(Path(args.store), transfer_timeout=args.transfer_timeout)
    config = load_config(Path(args.config))
    worker_name = args.worker
    if worker_name is None:
        from sparselab.staging import inspect_runtime
        from sparselab.workers.models import WorkerDefinition

        runtime = inspect_runtime(config)
        root = controller.root.resolve()
        namespace = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
        worker_name = f"local-{runtime.engine}-{runtime.backend}-{config.runtime.device_index}-{namespace}"
        controller.register(
            WorkerDefinition(
                worker_id=worker_name,
                name=worker_name,
                transport="local",
                python=Path(sys.executable).absolute(),
                root=root / ".workers" / worker_name,
                engine=runtime.engine,
                backend=runtime.backend,
                device_index=config.runtime.device_index,
            )
        )
    submission = controller.submit(
        config,
        worker=worker_name,
        stage_bundle=Path(args.stage_bundle) if args.stage_bundle else None,
        promote=Path(args.promote) if args.promote else None,
    )
    result = _wait_for_run(controller, submission.run_id)
    _json({"submission": submission.model_dump(mode="json"), "result": result})
    if result["status"] != "COMPLETE":
        raise SystemExit(1)


def _store_argument(parser: argparse.ArgumentParser, default_store: Path) -> None:
    parser.add_argument(
        "--store", default=str(default_store), help="Controller state root"
    )


def _endpoint_arguments(parser: argparse.ArgumentParser, *, attempt: bool) -> None:
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--engine", choices=_ENGINES, required=True)
    parser.add_argument("--backend", choices=_BACKENDS, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    if attempt:
        parser.add_argument("--attempt-id", required=True)


def add_commands(
    subparsers: argparse._SubParsersAction, *, default_store: Path
) -> None:
    """Add the worker/controller/experiment command families to the main parser."""
    worker = subparsers.add_parser(
        "worker", help="Register and inspect independent workers"
    )
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    register = worker_commands.add_parser("register")
    register.add_argument("name")
    register.add_argument("--backend", choices=_BACKENDS, required=True)
    register.add_argument("--engine", choices=_ENGINES, default="pytorch")
    register.add_argument("--device-index", type=int, default=0)
    register.add_argument("--ssh")
    register.add_argument("--python")
    register.add_argument("--root")
    _store_argument(register, default_store)
    register.set_defaults(handler=_worker_register)
    status = worker_commands.add_parser("status")
    status.add_argument("name", nargs="?")
    status.add_argument("--json", action="store_true")
    _store_argument(status, default_store)
    status.set_defaults(handler=_worker_status)
    serve = worker_commands.add_parser("serve-stdio", help=argparse.SUPPRESS)
    _endpoint_arguments(serve, attempt=False)
    serve.set_defaults(handler=_serve_stdio)
    execute = worker_commands.add_parser("execute", help=argparse.SUPPRESS)
    _endpoint_arguments(execute, attempt=True)
    execute.set_defaults(handler=_execute)

    experiment = subparsers.add_parser(
        "experiment", help="Submit and manage independent experiments"
    )
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    submit = experiment_commands.add_parser("submit")
    submit.add_argument("config", nargs="?")
    submit.add_argument("--matrix")
    submit.add_argument("--worker")
    submit.add_argument("--stage-bundle")
    submit.add_argument("--promote")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--max-runs", type=int, default=1000)
    _store_argument(submit, default_store)
    submit.set_defaults(handler=_experiment_submit)
    listing = experiment_commands.add_parser("list")
    listing.add_argument("--json", action="store_true")
    _store_argument(listing, default_store)
    listing.set_defaults(handler=_experiment_list)
    cancel = experiment_commands.add_parser("cancel")
    cancel.add_argument("run_id")
    _store_argument(cancel, default_store)
    cancel.set_defaults(handler=_experiment_cancel)
    resume = experiment_commands.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--worker")
    resume.add_argument("--allow-runtime-drift", action="store_true")
    _store_argument(resume, default_store)
    resume.set_defaults(handler=_experiment_resume)

    controller = subparsers.add_parser(
        "controller", help="Run the local experiment controller"
    )
    controller_commands = controller.add_subparsers(
        dest="controller_command", required=True
    )
    run_controller = controller_commands.add_parser("run")
    run_controller.add_argument(
        "--transfer-timeout",
        type=float,
        default=1800,
        help="Finite deadline in seconds for each bundle or artifact transfer RPC",
    )
    _store_argument(run_controller, default_store)
    run_controller.set_defaults(handler=_controller_run)

    run = subparsers.add_parser(
        "run", help="Submit one independent run and wait for its outcome"
    )
    run.add_argument("config")
    run.add_argument("--worker")
    run.add_argument("--stage-bundle")
    run.add_argument("--promote")
    run.add_argument(
        "--transfer-timeout",
        type=float,
        default=1800,
        help="Finite deadline in seconds for each bundle or artifact transfer RPC",
    )
    _store_argument(run, default_store)
    run.set_defaults(handler=_run)
