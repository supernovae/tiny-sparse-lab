"""Lightweight recipe routing with explicit inspection and publication operations."""

import argparse
import json
from pathlib import Path


def add_commands(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="recipe_command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Preview an inventoried native recipe; never execute"
    )
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--max-runs", type=int, default=1000)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=handle)
    preparation = commands.add_parser(
        "preparation",
        help="Inspect an inventoried native preparation graph; never execute",
    )
    preparation_actions = preparation.add_subparsers(
        dest="preparation_command", required=True
    )
    preview = preparation_actions.add_parser("inspect")
    preview.add_argument("source", type=Path)
    preview.add_argument("--bindings", type=Path, required=True)
    preview.add_argument("--json", action="store_true")
    preview.set_defaults(handler=handle_preparation)
    workflow = commands.add_parser(
        "workflow", help="Inspect, resolve or publish prepared-input workflow metadata"
    )
    actions = workflow.add_subparsers(dest="workflow_command", required=True)
    for name in ("inspect", "resolve", "publish"):
        action = actions.add_parser(name)
        action.add_argument("source", type=Path)
        action.add_argument("--bindings", type=Path, required=True)
        action.add_argument("--max-runs", type=int, default=1000)
        action.add_argument("--json", action="store_true")
        if name == "publish":
            action.add_argument("--workspace", type=Path, required=True)
        action.set_defaults(handler=handle_workflow)


def handle(args: argparse.Namespace) -> None:
    from sparselab.experiments.recipe import inspect_recipe

    try:
        result = inspect_recipe(args.source, max_runs=args.max_runs)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"sparselab: {error}") from None
    if args.json:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        print(
            f"Recipe {result['id']}: {len(result['cells'])} native config cells; inspect-only, not executable"
        )
        print(f"Recipe SHA-256: {result['recipe_sha256']}")
        print(f"Installed lab SHA-256: {result['installed_lab']['sha256']}")
        for requirement in result["requirements"]:
            print(
                f"{requirement['kind']} {requirement.get('field', '')}: {requirement['status']}"
            )
        for limitation in result["limitations"]:
            print(limitation)


def handle_preparation(args: argparse.Namespace) -> None:
    from sparselab.experiments.preparation_recipe import inspect_preparation_recipe

    try:
        result = inspect_preparation_recipe(args.source, args.bindings)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"sparselab: {error}") from None
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def handle_workflow(args: argparse.Namespace) -> None:
    from sparselab.experiments.workflow_recipe import (
        inspect_workflow,
        publish_workflow,
        resolve_workflow,
    )

    try:
        if args.workflow_command == "inspect":
            result = inspect_workflow(
                args.source, args.bindings, max_runs=args.max_runs
            )
        elif args.workflow_command == "resolve":
            lock = resolve_workflow(args.source, args.bindings, max_runs=args.max_runs)
            result = {
                "format": "sparselab-workflow-resolution-v1",
                "scientific_sha256": lock.scientific_sha256,
                "plan_sha256": lock.plan_sha256,
                "published": False,
                "executable": False,
                "runtime_capability": "unknown",
                "cells": len(lock.cells),
            }
        else:
            path = publish_workflow(
                args.source, args.bindings, args.workspace, max_runs=args.max_runs
            )
            result = {
                "format": "sparselab-workflow-publication-v1",
                "lock": str(path),
                "executable": False,
                "runtime_capability": "unknown",
            }
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"sparselab: {error}") from None
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="sparselab experiment recipe", allow_abbrev=False
    )
    add_commands(parser)
    args = parser.parse_args(argv)
    args.handler(args)
