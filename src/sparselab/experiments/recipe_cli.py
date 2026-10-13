"""Lightweight read-only portable recipe inspection."""

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


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="sparselab experiment recipe", allow_abbrev=False
    )
    add_commands(parser)
    args = parser.parse_args(argv)
    handle(args)
