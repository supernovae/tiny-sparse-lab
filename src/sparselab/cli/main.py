"""Command-line entry point for SparseLab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.config.migrate import migrate_file, migrate_v1
from sparselab.config.models import RunConfig
from sparselab.data.packing import TokenBlockDataset, prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.data.withheld_facts import (
    audit_manifest,
    verify_manifest,
    write_manifest,
)
from sparselab.evaluation.byte_memory_transfer import transfer_byte_memory
from sparselab.evaluation.chat import ChatMessage, chat_turn
from sparselab.evaluation.generation import generate
from sparselab.evaluation.language_model import evaluate
from sparselab.evaluation.withheld_facts import (
    evaluate_withheld_facts,
    write_withheld_evaluation,
)
from sparselab.memory import estimate_memory, parameter_inventory
from sparselab.model.inspection import inspect_model
from sparselab.model.memory import ByteAddressMemory
from sparselab.model.portable_engram import export_portable_engram, load_portable_engram
from sparselab.model.transformer import DenseLM
from sparselab.runtime import discover_runtimes, select_device
from sparselab.staging import stage
from sparselab.training.checkpoints import CheckpointManager, load_checkpoint
from sparselab.training.mlx_checkpoints import inspect as inspect_mlx_checkpoint
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


def _facts_audit(args: argparse.Namespace) -> None:
    print(json.dumps(audit_manifest(Path(args.path)), indent=2, sort_keys=True))


def _facts_evaluate(args: argparse.Namespace) -> None:
    config, model, device = _run_model(
        args.run_id, Path(args.runs_dir), args.checkpoint, args.backend
    )
    result = evaluate_withheld_facts(
        model,
        load_tokenizer(config.tokenizer.path),
        Path(args.manifest),
        max_seq_len=config.model.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.runs_dir)
        / args.run_id
        / "evaluations"
        / f"withheld-v2-{result['manifest_sha256']}.json"
    )
    write_withheld_evaluation(output, result)
    print(output)


def _facts_transfer_evaluate(args: argparse.Namespace) -> None:
    _, source, _ = _run_model(
        args.source_run_id, Path(args.runs_dir), args.source_checkpoint, args.backend
    )
    config, target, device = _run_model(
        args.target_run_id, Path(args.runs_dir), args.target_checkpoint, args.backend
    )
    transfer_byte_memory(source.state_dict(), target)
    result = evaluate_withheld_facts(
        target,
        load_tokenizer(config.tokenizer.path),
        Path(args.manifest),
        max_seq_len=config.model.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    result["source_run_id"] = args.source_run_id
    result["target_run_id"] = args.target_run_id
    output = (
        Path(args.output)
        if args.output
        else Path(args.runs_dir)
        / args.target_run_id
        / "evaluations"
        / f"transferred-v2-{args.source_run_id}-{result['manifest_sha256']}.json"
    )
    write_withheld_evaluation(output, result)
    print(output)


def _engram_export(args: argparse.Namespace) -> None:
    config, model, _ = _run_model(
        args.run_id, Path(args.runs_dir), args.checkpoint, args.backend
    )
    if not isinstance(model.memory, ByteAddressMemory):
        raise TypeError("Engram export requires a byte-memory run")
    manifest = export_portable_engram(
        model.memory.table.weight,
        Path(args.output),
        ngram_size=config.model.memory_ngram_size,
    )
    print(json.dumps(manifest.as_dict(), sort_keys=True))


def _engram_inspect(args: argparse.Namespace) -> None:
    package = load_portable_engram(Path(args.path))
    print(json.dumps(package.manifest.as_dict(), sort_keys=True))


def _config_migrate(args: argparse.Namespace) -> None:
    changes = migrate_file(Path(args.input), Path(args.output))
    print("\n".join(changes))


def _checkpoint_inspect(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if (path / "state.json").is_file() or path.name == "state.json":
        report = inspect_mlx_checkpoint(path)
        payload = {
            "valid": report.valid,
            "metadata": report.metadata,
            "files": list(report.files),
            "errors": list(report.errors),
        }
        print(json.dumps(payload, sort_keys=True) if args.json else payload)
        return
    directory = (
        path.parent / json.loads(path.read_text())["relative_path"]
        if path.name in {"latest.json", "best.json"}
        else path
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    print(
        json.dumps(manifest, sort_keys=True)
        if args.json
        else "\n".join(f"{key}: {value}" for key, value in manifest.items())
    )


def _checkpoint_verify(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if (path / "state.json").is_file() or path.name == "state.json":
        report = inspect_mlx_checkpoint(path)
        payload = {
            "valid": report.valid,
            "metadata": report.metadata,
            "files": list(report.files),
            "errors": list(report.errors),
        }
        print(json.dumps(payload, sort_keys=True) if args.json else payload)
        if not report.valid:
            raise SystemExit(1)
        return
    root = path.parent.parent
    report = CheckpointManager(root).verify(
        path, require_training_state=not args.weights_only
    )
    payload = {
        "valid": report.valid,
        "errors": list(report.errors),
        "files": list(report.verified_files),
        "resume_level": report.resume_level,
    }
    print(json.dumps(payload, sort_keys=True) if args.json else payload)
    if not report.valid:
        raise SystemExit(1)


def _inspect(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    inventory = parameter_inventory(config)
    runtime = next(
        info
        for info in discover_runtimes()
        if info.engine == config.runtime.engine
        and info.backend
        == (config.runtime.backend if config.runtime.backend != "auto" else "cpu")
    )
    estimate = estimate_memory(config, runtime, inventory)
    values = {
        **inspect_model(DenseLM(config.model, config.attention)),
        "parameter_inventory": inventory.__dict__,
        "memory_estimate": estimate.__dict__,
        "runtime": runtime.as_dict(),
    }
    print(
        json.dumps(values, indent=2, sort_keys=True)
        if args.json
        else "\n".join(f"{k}: {v}" for k, v in values.items())
    )


def _data_prepare(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    print(prepare_data(config, load_tokenizer(config.tokenizer.path)).root)


def _tokenizer_train(args: argparse.Namespace) -> None:
    print(train_tokenizer(load_tokenizer_config(Path(args.config))))


def _stage(args: argparse.Namespace) -> None:
    print(stage(load_config(Path(args.config)), Path(args.output), args.through))




def _train(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.backend:
        config = config.model_copy(
            update={
                "runtime": config.runtime.model_copy(update={"backend": args.backend})
            }
        )
    print(
        train(
            config,
            resume=Path(args.resume) if args.resume else None,
            promote=Path(args.promote) if args.promote else None,
            recover=Path(args.recover) if args.recover else None,
            run_id=args.run_id,
            stop_after_step=args.stop_after_step,
            allow_runtime_drift=args.allow_runtime_drift,
        )
    )


def _run_model(
    run_id: str, runs_dir: Path, checkpoint: str | None, backend: str | None
) -> tuple[RunConfig, DenseLM, object]:
    run = runs_dir / run_id
    raw_config = json.loads((run / "resolved_config.yaml").read_text())
    config = RunConfig.model_validate(
        migrate_v1(raw_config)
        if raw_config.get("schema_version") == 1
        else raw_config
    )
    chosen_backend = backend or config.runtime.backend
    device = select_device(chosen_backend)
    path = Path(checkpoint) if checkpoint else run / "checkpoints" / "latest.json"
    if path.name in {"latest.json", "best.json"}:
        pointer = json.loads(path.read_text())
        if "filename" in pointer:
            path = path.parent / str(pointer["filename"])
        else:
            weights = CheckpointManager(run).load(path, "promote").model
            model = DenseLM(config.model, config.attention).to(device)
            model.load_state_dict(weights)
            return config, model, device
    if path.is_dir():
        weights = CheckpointManager(run).load(path, "promote").model
    else:
        weights = load_checkpoint(path)["model"]
    model = DenseLM(config.model, config.attention).to(device)
    model.load_state_dict(weights)
    return config, model, device


def _eval(args: argparse.Namespace) -> None:
    config, model, device = _run_model(
        args.run_id, Path(args.runs_dir), args.checkpoint, args.backend
    )
    data = prepare_data(config, load_tokenizer(config.tokenizer.path))
    result = evaluate(
        model,
        TokenBlockDataset(
            data.validation,
            config.training.seq_len,
            data.validation_byte_addresses,
        ),
        batch_size=config.training.micro_batch_size,
        max_batches=config.evaluation.max_batches,
        device=device,
    )
    result.update({"source": "standalone_eval", "device": str(device)})
    evaluations = Path(args.runs_dir) / args.run_id / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    output = evaluations / f"eval_{uuid.uuid4().hex}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**result, "output": str(output)}, indent=2))


def _generate(args: argparse.Namespace) -> None:
    config, model, device = _run_model(
        args.run_id, Path(args.runs_dir), None, args.backend
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



def _chat(args: argparse.Namespace) -> None:
    config, model, device = _run_model(
        args.run_id, Path(args.runs_dir), None, args.backend
    )
    tokenizer = load_tokenizer(config.tokenizer.path)
    history: list[ChatMessage] = []

    def respond(message: str) -> None:
        _, reply = chat_turn(
            model,
            tokenizer,
            history,
            message,
            config.model.max_seq_len,
            args.max_new_tokens,
            device,
            system=args.system,
        )
        print(f"Assistant: {reply}")
        history.extend((ChatMessage("user", message), ChatMessage("assistant", reply)))

    if args.message is not None:
        respond(args.message)
        return
    print("Local greedy chat. Type /exit to end the conversation.")
    while True:
        try:
            message = input("You: ")
        except EOFError:
            print()
            return
        if message.strip().lower() in {"/exit", "/quit"}:
            return
        if not message.strip():
            continue
        respond(message)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparselab", description="Tiny Sparse Lab educational transformer tools."
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_migrate = config_commands.add_parser("migrate")
    config_migrate.add_argument("input")
    config_migrate.add_argument("--output", required=True)
    config_migrate.set_defaults(handler=_config_migrate)
    checkpoint = commands.add_parser("checkpoint")
    checkpoint_commands = checkpoint.add_subparsers(
        dest="checkpoint_command", required=True
    )
    checkpoint_inspect = checkpoint_commands.add_parser("inspect")
    checkpoint_inspect.add_argument("path")
    checkpoint_inspect.add_argument("--json", action="store_true")
    checkpoint_inspect.set_defaults(handler=_checkpoint_inspect)
    checkpoint_verify = checkpoint_commands.add_parser("verify")
    checkpoint_verify.add_argument("path")
    checkpoint_verify.add_argument("--weights-only", action="store_true")
    checkpoint_verify.add_argument("--json", action="store_true")
    checkpoint_verify.set_defaults(handler=_checkpoint_verify)
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
    audit = fact_commands.add_parser("audit")
    audit.add_argument("path")
    audit.set_defaults(handler=_facts_audit)
    facts_evaluate = fact_commands.add_parser("evaluate")
    facts_evaluate.add_argument("run_id")
    facts_evaluate.add_argument("manifest")
    facts_evaluate.add_argument("--runs-dir", default="runs")
    facts_evaluate.add_argument("--checkpoint")
    facts_evaluate.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    facts_evaluate.add_argument("--max-new-tokens", type=int, default=16)
    facts_evaluate.add_argument("--output")
    facts_evaluate.set_defaults(handler=_facts_evaluate)
    transfer_evaluate = fact_commands.add_parser("transfer-evaluate")
    transfer_evaluate.add_argument("source_run_id")
    transfer_evaluate.add_argument("target_run_id")
    transfer_evaluate.add_argument("manifest")
    transfer_evaluate.add_argument("--runs-dir", default="runs")
    transfer_evaluate.add_argument("--source-checkpoint")
    transfer_evaluate.add_argument("--target-checkpoint")
    transfer_evaluate.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    transfer_evaluate.add_argument("--max-new-tokens", type=int, default=16)
    transfer_evaluate.add_argument("--output")
    transfer_evaluate.set_defaults(handler=_facts_transfer_evaluate)
    data_prepare.set_defaults(handler=_data_prepare)
    engram = commands.add_parser("engram")
    engram_commands = engram.add_subparsers(dest="engram_command", required=True)
    engram_export = engram_commands.add_parser("export")
    engram_export.add_argument("run_id")
    engram_export.add_argument("--output", required=True)
    engram_export.add_argument("--runs-dir", default="runs")
    engram_export.add_argument("--checkpoint")
    engram_export.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    engram_export.set_defaults(handler=_engram_export)
    engram_inspect = engram_commands.add_parser("inspect")
    engram_inspect.add_argument("path")
    engram_inspect.set_defaults(handler=_engram_inspect)
    staging = commands.add_parser("stage")
    staging.add_argument("config")
    staging.add_argument(
        "--through", default="smoke", choices=("inspect", "validate", "smoke", "warmup")
    )
    staging.add_argument("--output", required=True)
    staging.set_defaults(handler=_stage)
    training = commands.add_parser("train")
    training.add_argument("config")
    training.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    training.add_argument("--run-id")
    training.add_argument("--resume")
    training.add_argument("--promote")
    training.add_argument("--recover")
    training.add_argument("--allow-runtime-drift", action="store_true")
    training.add_argument("--stop-after-step", type=int)
    training.set_defaults(handler=_train)
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("run_id")
    evaluation.add_argument("--runs-dir", default="runs")
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    evaluation.set_defaults(handler=_eval)
    generation = commands.add_parser("generate")
    generation.add_argument("run_id")
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--max-new-tokens", type=int, default=64)
    generation.add_argument("--runs-dir", default="runs")
    generation.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    generation.set_defaults(handler=_generate)
    chat = commands.add_parser(
        "chat", help="Chat locally with a saved PyTorch run using greedy decoding."
    )
    chat.add_argument("run_id")
    chat.add_argument("--message")
    chat.add_argument("--system")
    chat.add_argument("--max-new-tokens", type=int, default=64)
    chat.add_argument("--runs-dir", default="runs")
    chat.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    chat.set_defaults(handler=_chat)
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--runs-dir", default="runs")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(handler=_dashboard)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
