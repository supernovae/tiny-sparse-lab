"""Command-line entry point for SparseLab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.config.migrate import migrate_file
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.data.withheld_facts import (
    audit_manifest,
    verify_manifest,
    write_manifest,
)
from sparselab.evaluation.byte_memory_transfer import transfer_byte_memory
from sparselab.evaluation.capabilities import (
    capability_card,
    compare_results,
    describe_capability_card,
    evaluate_capability,
    list_capability_cards,
    write_capability_result,
)
from sparselab.evaluation.chat import ChatMessage, assistant_reply, prepare_chat_prompt
from sparselab.evaluation.evidence import experiment_evidence
from sparselab.evaluation.generation import generate
from sparselab.evaluation.inference import load_run, write_inference_result
from sparselab.evaluation.language_model import evaluate
from sparselab.evaluation.withheld_facts import (
    evaluate_withheld_facts,
    write_withheld_evaluation,
)
from sparselab.memory import estimate_memory
from sparselab.model.inspection import inspection_report, parameter_inventory
from sparselab.model.memory import ByteAddressMemory
from sparselab.model.portable_engram import export_portable_engram, load_portable_engram
from sparselab.runtime import discover_runtimes, select_device
from sparselab.staging import stage
from sparselab.training.checkpoints import CheckpointManager
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
    loaded = load_run(args.run_id, Path(args.runs_dir), args.checkpoint, args.backend)
    result = evaluate_withheld_facts(
        loaded.model,
        loaded.tokenizer,
        Path(args.manifest),
        max_seq_len=loaded.config.model.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        device=loaded.device,
    )
    result["identity"] = loaded.identity
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
    source = load_run(
        args.source_run_id, Path(args.runs_dir), args.source_checkpoint, args.backend
    )
    target = load_run(
        args.target_run_id, Path(args.runs_dir), args.target_checkpoint, args.backend
    )
    transfer_byte_memory(source.model.state_dict(), target.model)
    result = evaluate_withheld_facts(
        target.model,
        target.tokenizer,
        Path(args.manifest),
        max_seq_len=target.config.model.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        device=target.device,
    )
    result["source_run_id"] = args.source_run_id
    result["target_run_id"] = args.target_run_id
    result["source_identity"] = source.identity
    result["target_identity"] = target.identity
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
    loaded = load_run(args.run_id, Path(args.runs_dir), args.checkpoint, args.backend)
    if not isinstance(loaded.model.memory, ByteAddressMemory):
        raise TypeError("Engram export requires a byte-memory run")
    manifest = export_portable_engram(
        loaded.model.memory.table.weight,
        Path(args.output),
        ngram_size=loaded.config.model.memory_ngram_size,
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
    runtimes = discover_runtimes()
    backend = config.runtime.backend
    if backend == "auto":
        device = select_device("auto")
        backend = next(
            info.backend
            for info in runtimes
            if info.engine == "pytorch" and info.torch_device == str(device)
        )
    runtime = next(
        info
        for info in runtimes
        if info.engine == config.runtime.engine and info.backend == backend
    )
    if config.runtime.device_index != runtime.device_index:
        runtime = replace(
            runtime,
            device_index=config.runtime.device_index,
            torch_device=None,
            device_name=None,
            device_total_bytes=None,
            device_free_bytes=None,
            device_recommended_bytes=None,
            device_driver_allocated_bytes=None,
            limitations=(
                *runtime.limitations,
                "passive memory readings for this device index are unavailable",
            ),
        )
    estimate = estimate_memory(config, runtime, inventory)
    values = {
        **inspection_report(config),
        "parameter_inventory": inventory.__dict__,
        "memory_estimate": estimate.__dict__,
        "runtime": runtime.as_dict(),
    }
    if args.json:
        print(json.dumps(values, indent=2, sort_keys=True))
        return
    print("Model")
    for field in ("total", "trainable", "active_per_token"):
        print(f"  {field}: {values[field]:,}")
    print("Estimated training memory")
    for field, value in asdict(estimate).items():
        if field.endswith("_bytes") and value is not None:
            print(f"  {field}: {value:,} bytes ({value / 1024**3:.3f} GiB)")
    print("Device")
    print(f"  {runtime.engine}/{runtime.backend}, index {runtime.device_index}")
    for limitation in runtime.limitations:
        print(f"  {limitation}")
    print(f"Result: {estimate.result}")
    for assumption in estimate.assumptions:
        print(f"  {assumption}")


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


def _eval(args: argparse.Namespace) -> None:
    loaded = load_run(args.run_id, Path(args.runs_dir), args.checkpoint, args.backend)
    result = evaluate(
        loaded.model,
        loaded.validation_dataset(),
        batch_size=loaded.config.training.micro_batch_size,
        max_batches=loaded.config.evaluation.max_batches,
        device=loaded.device,
    )
    result.update({"source": "standalone_eval", "identity": loaded.identity})
    path = write_inference_result(loaded.run, "eval", result)
    print(json.dumps({**result, "output": str(path)}, indent=2))


def _generate(args: argparse.Namespace) -> None:
    loaded = load_run(args.run_id, Path(args.runs_dir), args.checkpoint, args.backend)
    print(
        generate(
            loaded.model,
            loaded.tokenizer,
            args.prompt,
            loaded.config.model.max_seq_len,
            args.max_new_tokens,
            loaded.device,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
    )


def _evidence(args: argparse.Namespace) -> None:
    result = experiment_evidence(Path(args.runs_dir) / args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)


def _capability_list(args: argparse.Namespace) -> None:
    print(json.dumps(list_capability_cards(), indent=2, sort_keys=True))


def _capability_describe(args: argparse.Namespace) -> None:
    print(json.dumps(describe_capability_card(args.card), indent=2, sort_keys=True))


def _capability_result(args: argparse.Namespace, run_id: str, checkpoint: str | None):
    loaded = load_run(run_id, Path(args.runs_dir), checkpoint, args.backend)
    result = evaluate_capability(
        capability_card(args.card),
        loaded.model,
        loaded.tokenizer,
        loaded.config.model.max_seq_len,
        loaded.device,
    )
    result["identity"] = loaded.identity
    return result, write_capability_result(loaded.run, result)


def _capability_evaluate(args: argparse.Namespace) -> None:
    result, path = _capability_result(args, args.run_id, args.checkpoint)
    print(json.dumps({**result, "output": str(path)}, indent=2, sort_keys=True))


def _capability_compare(args: argparse.Namespace) -> None:
    # One model resident at a time, including at user-selected larger scales.
    base, base_path = _capability_result(args, args.base_run_id, args.base_checkpoint)
    variant, variant_path = _capability_result(
        args, args.variant_run_id, args.variant_checkpoint
    )
    result = compare_results(base, variant, vary=args.vary)
    result.update({"base_output": str(base_path), "variant_output": str(variant_path)})
    path = write_inference_result(
        Path(args.runs_dir) / args.variant_run_id, "comparison", result
    )
    print(json.dumps({**result, "output": str(path)}, indent=2, sort_keys=True))


def _chat(args: argparse.Namespace) -> None:
    loaded = load_run(args.run_id, Path(args.runs_dir), args.checkpoint, args.backend)
    history: list[ChatMessage] = []
    turns = []
    settings = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    if args.transcript and Path(args.transcript).exists():
        raise FileExistsError(f"transcript already exists: {args.transcript}")

    def respond(message: str) -> None:
        prompt, dropped = prepare_chat_prompt(
            history,
            message,
            loaded.tokenizer,
            loaded.config.model.max_seq_len,
            args.max_new_tokens,
            system=args.system,
        )
        completion = generate(
            loaded.model,
            loaded.tokenizer,
            prompt,
            loaded.config.model.max_seq_len,
            args.max_new_tokens,
            loaded.device,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            strict_context=True,
            stop_sequences=("\nUser:", "\nSystem:", "\nAssistant:"),
        )
        reply = assistant_reply(completion, prompt)
        turn = {
            "user": message,
            "prompt": prompt,
            "response": reply,
            "dropped_history_turns": dropped,
            "prompt_tokens": len(
                loaded.tokenizer.encode(prompt, add_special_tokens=False).ids
            ),
        }
        turns.append(turn)
        if args.json:
            print(
                json.dumps(
                    {"identity": loaded.identity, "generation": settings, **turn},
                    sort_keys=True,
                )
            )
        else:
            if dropped:
                print(
                    f"[Context: dropped {dropped} oldest complete turn(s)]",
                    file=sys.stderr,
                )
            print(f"Assistant: {reply}")
        history.extend((ChatMessage("user", message), ChatMessage("assistant", reply)))

    try:
        if args.message is not None:
            respond(args.message)
            return
        if not args.json:
            print(
                "Local chat. /exit ends; /reset clears context. Greedy unless --temperature is set."
            )
        while True:
            try:
                message = input("" if args.json else "You: ")
            except EOFError:
                return
            if message.strip().lower() in {"/exit", "/quit"}:
                return
            if message.strip().lower() == "/reset":
                history.clear()
                continue
            if not message.strip():
                continue
            try:
                respond(message)
            except ValueError as error:
                print(f"Chat input rejected: {error}", file=sys.stderr)
    finally:
        if args.transcript:
            path = Path(args.transcript)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                json.dump(
                    {
                        "format": "chat_transcript_v1",
                        "identity": loaded.identity,
                        "system": args.system,
                        "generation": settings,
                        "turns": turns,
                        "history": [asdict(item) for item in history],
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    cwd = Path.cwd()
    project_root = next(
        (path for path in (cwd, *cwd.parents) if (path / "pyproject.toml").is_file()),
        cwd,
    )
    runs_dir_default = str(project_root / "runs")
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
    facts_evaluate.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
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
    transfer_evaluate.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
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
    engram_export.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
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
    evaluation.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    evaluation.set_defaults(handler=_eval)
    evidence = commands.add_parser(
        "evidence",
        help="Summarize verified checkpoints and held-out observations for a local run.",
    )
    evidence.add_argument("run_id")
    evidence.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
    evidence.add_argument("--json", action="store_true")
    evidence.set_defaults(handler=_evidence)
    capability = commands.add_parser(
        "capability",
        help="Evaluate or compare a versioned narrow capability card.",
    )
    capability_commands = capability.add_subparsers(
        dest="capability_command", required=True
    )
    capability_list = capability_commands.add_parser("list")
    capability_list.set_defaults(handler=_capability_list)
    capability_describe = capability_commands.add_parser("describe")
    capability_describe.add_argument(
        "card", help="Built-in card name or declarative JSON file"
    )
    capability_describe.set_defaults(handler=_capability_describe)
    capability_evaluate = capability_commands.add_parser("evaluate")
    capability_evaluate.add_argument("run_id")
    capability_evaluate.add_argument("card")
    capability_evaluate.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
    capability_evaluate.add_argument(
        "--checkpoint", help="latest.json, best.json, or a generation path"
    )
    capability_evaluate.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    capability_evaluate.set_defaults(handler=_capability_evaluate)
    capability_compare = capability_commands.add_parser("compare")
    capability_compare.add_argument("base_run_id")
    capability_compare.add_argument("variant_run_id")
    capability_compare.add_argument("card")
    capability_compare.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
    capability_compare.add_argument("--base-checkpoint")
    capability_compare.add_argument("--variant-checkpoint")
    capability_compare.add_argument(
        "--vary",
        choices=("memory", "attention", "ffn", "scale", "none"),
        default="memory",
    )
    capability_compare.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    capability_compare.set_defaults(handler=_capability_compare)
    generation = commands.add_parser("generate")
    generation.add_argument("run_id")
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--max-new-tokens", type=int, default=64)
    generation.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
    generation.add_argument("--checkpoint")
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--top-k", type=int, default=0)
    generation.add_argument("--seed", type=int, default=0)
    generation.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    generation.set_defaults(handler=_generate)
    chat = commands.add_parser(
        "chat", help="Chat with a verified local PyTorch checkpoint."
    )
    chat.add_argument("run_id")
    chat.add_argument("--message")
    chat.add_argument("--system")
    chat.add_argument("--max-new-tokens", type=int, default=16)
    chat.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
    chat.add_argument(
        "--checkpoint", help="latest.json, best.json, or a generation path"
    )
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--top-k", type=int, default=0)
    chat.add_argument("--seed", type=int, default=0)
    chat.add_argument(
        "--transcript",
        help="Save a checkpoint-bound JSON transcript without overwriting",
    )
    chat.add_argument(
        "--json", action="store_true", help="Emit one JSON object per response"
    )
    chat.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    chat.set_defaults(handler=_chat)
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(handler=_dashboard)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
