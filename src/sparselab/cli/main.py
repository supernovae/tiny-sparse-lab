"""Command-line entry point for SparseLab."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparselab", description="Tiny Sparse Lab educational transformer tools."
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    return parser


def main() -> None:
    build_parser().parse_args()
