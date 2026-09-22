"""Command-line entry point for SparseLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.model.inspection import inspect_model
from sparselab.model.transformer import DenseLM
from sparselab.training.trainer import train


def _inspect(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    values = inspect_model(DenseLM(config.model, config.attention))
    print(
        json.dumps(values, indent=2, sort_keys=True)
        if args.json
        else "\n".join(f"{k}: {v}" for k, v in values.items())
    )


def _tokenizer_train(args: argparse.Namespace) -> None:
    print(train_tokenizer(load_tokenizer_config(Path(args.config))))


def _data_prepare(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    print(prepare_data(config, load_tokenizer(config.tokenizer.path)).root)


def _train(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.device:
        config = config.model_copy(update={"device": args.device})
    print(
        train(
            config,
            resume=Path(args.resume) if args.resume else None,
            run_id=args.run_id,
            stop_after_step=args.stop_after_step,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparselab", description="Tiny Sparse Lab educational transformer tools."
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("config")
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_inspect)
    tokenizer = commands.add_parser("tokenizer")
    tokenizer_commands = tokenizer.add_subparsers(
        dest="tokenizer_command", required=True
    )
    tokenizer_train = tokenizer_commands.add_parser("train")
    tokenizer_train.add_argument("config")
    tokenizer_train.set_defaults(handler=_tokenizer_train)
    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_prepare = data_commands.add_parser("prepare")
    data_prepare.add_argument("config")
    data_prepare.set_defaults(handler=_data_prepare)
    training = commands.add_parser("train")
    training.add_argument("config")
    training.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"))
    training.add_argument("--run-id")
    training.add_argument("--resume")
    training.add_argument("--stop-after-step", type=int)
    training.set_defaults(handler=_train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
