"""Command-line entry point for SparseLab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.config.models import RunConfig
from sparselab.data.packing import TokenBlockDataset, prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.data.withheld_facts import verify_manifest, write_manifest
from sparselab.evaluation.generation import generate
from sparselab.evaluation.language_model import evaluate
from sparselab.model.inspection import inspect_model
from sparselab.model.transformer import DenseLM
from sparselab.runtime import select_device
from sparselab.training.checkpoints import load_checkpoint
from sparselab.training.trainer import train


def _dashboard(args: argparse.Namespace) -> None:
    from sparselab.dashboard import app

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(app.__file__)),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
            "--",
            "--runs-dir",
            args.runs_dir,
        ],
        check=True,
    )


def _facts_manifest(args: argparse.Namespace) -> None:
    output = Path(args.output)
    write_manifest(output, args.seed)
    print(output)


def _facts_verify(args: argparse.Namespace) -> None:
    manifest = verify_manifest(Path(args.path))
    print(json.dumps({"sha256": manifest["sha256"], "valid": True}, sort_keys=True))


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


def _run_model(
    run_id: str, runs_dir: Path, checkpoint: str | None, device_name: str | None
) -> tuple[RunConfig, DenseLM, object]:
    run = runs_dir / run_id
    config = RunConfig.model_validate(
        json.loads((run / "resolved_config.yaml").read_text())
    )
    device = select_device(device_name or config.device)
    path = Path(checkpoint) if checkpoint else run / "checkpoints" / "latest.json"
    if path.suffix == ".json":
        path = path.parent / json.loads(path.read_text())["filename"]
    state = load_checkpoint(path)
    model = DenseLM(config.model, config.attention).to(device)
    model.load_state_dict(state["model"])
    return config, model, device


def _eval(args: argparse.Namespace) -> None:
    config, model, device = _run_model(
        args.run_id, Path(args.runs_dir), args.checkpoint, args.device
    )
    data = prepare_data(config, load_tokenizer(config.tokenizer.path))
    print(
        json.dumps(
            evaluate(
                model,
                TokenBlockDataset(
                    data.validation,
                    config.training.seq_len,
                    data.validation_byte_addresses,
                ),
                batch_size=config.training.batch_size,
                max_batches=config.evaluation.max_batches,
                device=device,
            ),
            indent=2,
        )
    )


def _generate(args: argparse.Namespace) -> None:
    config, model, device = _run_model(
        args.run_id, Path(args.runs_dir), None, args.device
    )
    print(
        generate(
            model,
            load_tokenizer(config.tokenizer.path),
            args.prompt,
            config.model.max_seq_len,
            args.max_new_tokens,
            device,
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
    facts = commands.add_parser("facts")
    fact_commands = facts.add_subparsers(dest="fact_command", required=True)
    manifest = fact_commands.add_parser("manifest")
    manifest.add_argument("--seed", type=int, default=0)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(handler=_facts_manifest)
    verify = fact_commands.add_parser("verify")
    verify.add_argument("path")
    verify.set_defaults(handler=_facts_verify)
    data_prepare.set_defaults(handler=_data_prepare)
    training = commands.add_parser("train")
    training.add_argument("config")
    training.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"))
    training.add_argument("--run-id")
    training.add_argument("--resume")
    training.add_argument("--stop-after-step", type=int)
    training.set_defaults(handler=_train)
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("run_id")
    evaluation.add_argument("--runs-dir", default="runs")
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"))
    evaluation.set_defaults(handler=_eval)
    generation = commands.add_parser("generate")
    generation.add_argument("run_id")
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--max-new-tokens", type=int, default=64)
    generation.add_argument("--runs-dir", default="runs")
    generation.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"))
    generation.set_defaults(handler=_generate)
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--runs-dir", default="runs")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(handler=_dashboard)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
