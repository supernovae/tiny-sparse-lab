"""Command-line entry point for SparseLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparselab.config.loading import load_config
from sparselab.model.inspection import inspect_model
from sparselab.model.transformer import DenseLM


def _inspect(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    values = inspect_model(DenseLM(config.model, config.attention))
    if args.json:
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for name, value in values.items():
            print(f"{name}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparselab", description="Tiny Sparse Lab educational transformer tools."
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Instantiate and account for a dense model."
    )
    inspect.add_argument("config")
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_inspect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
