"""Command-line entry point for SparseLab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
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
from sparselab.engram.packs import compile_pack, inspect_pack, verify_pack
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
from sparselab.evaluation.withheld_facts import (
    evaluate_withheld_facts,
    write_withheld_evaluation,
)
from sparselab.memory import (
    calibrated_estimate,
    calibration_key,
    plan_memory,
    write_resource_proposal,
)
from sparselab.model.inspection import inspection_report, parameter_inventory
from sparselab.model.memory import ByteAddressMemory
from sparselab.model.portable_engram import export_portable_engram, load_portable_engram
from sparselab.staging import inspect_runtime, stage
from sparselab.training.checkpoints import CheckpointManager, _safe_member
from sparselab.training.manifest import source_identity
from sparselab.training.metrics import ExperimentStore
from sparselab.training.trainer import train


def _dashboard(args: argparse.Namespace) -> None:
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
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
    if loaded.engine is not None:
        raise ValueError(
            "withheld-facts evaluation is unsupported for native MLX runs because "
            "candidate likelihood scoring currently requires the PyTorch evaluator"
        )
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
    if source.engine is not None or target.engine is not None:
        raise ValueError(
            "withheld-facts transfer evaluation is unsupported for native MLX runs; "
            "it transfers PyTorch byte-memory tables"
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
    if loaded.engine is not None:
        raise ValueError(
            "Engram export is unsupported for native MLX runs; MLX supports dense "
            "and block-sparse attention only"
        )
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


def _engram_pack_compile(args: argparse.Namespace) -> None:
    from pickle import UnpicklingError

    from pyarrow import ArrowException
    from safetensors import SafetensorError

    try:
        manifest = compile_pack(
            Path(args.source),
            Path(args.output),
            name=args.name,
            namespace=args.namespace,
            default_license=args.license,
            source_name=args.source_name,
            source_revision=args.source_revision,
            created_at=args.created_at,
            lexical_package=Path(args.lexical_package)
            if args.lexical_package
            else None,
            semantic_keys=Path(args.semantic_keys) if args.semantic_keys else None,
            semantic_values=Path(args.semantic_values)
            if args.semantic_values
            else None,
            semantic_metadata=Path(args.semantic_metadata)
            if args.semantic_metadata
            else None,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        EOFError,
        UnpicklingError,
        ArrowException,
        SafetensorError,
    ) as error:
        payload = {
            "valid": False,
            "errors": [{"field": "compile", "reason": str(error)}],
        }
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "pack_id": manifest.pack_id,
                "output": str(args.output),
                "record_count": manifest.record_count,
                "lexical_count": manifest.lexical.table_size
                if manifest.lexical is not None
                else 0,
                "semantic_count": manifest.semantic.entry_count
                if manifest.semantic is not None
                else 0,
            },
            sort_keys=True,
        )
    )


def _engram_pack_inspect(args: argparse.Namespace) -> None:
    try:
        payload = inspect_pack(Path(args.path))
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(
            json.dumps(
                {
                    "valid": False,
                    "errors": [{"field": "inspect", "reason": str(error)}],
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(payload, sort_keys=True))


def _engram_pack_verify(args: argparse.Namespace) -> None:
    try:
        report = verify_pack(Path(args.path), expected_pack_id=args.expected_pack_id)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        payload = {
            "valid": False,
            "pack_id": None,
            "errors": [{"field": "verify", "reason": str(error)}],
            "files": [],
        }
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(1) from None
    payload = {
        "valid": report.valid,
        "pack_id": report.pack_id,
        "errors": [
            {"field": item.field, "reason": item.reason} for item in report.errors
        ],
        "files": [
            {
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in report.files
        ],
    }
    print(json.dumps(payload, sort_keys=True))
    if not report.valid:
        raise SystemExit(1)


def _config_migrate(args: argparse.Namespace) -> None:
    changes = migrate_file(Path(args.input), Path(args.output))
    print("\n".join(changes))


def _legacy_checkpoint_inventory(path: Path, expected_config=None) -> dict[str, object]:
    from sparselab.training.weight_import import load_legacy_weights

    legacy = load_legacy_weights(path, expected_config)
    return {
        **legacy.metadata,
        "engine": "pytorch",
        "requested_backend": legacy.config.runtime.backend,
        "resume_level": "weights_only",
        "source_config_sha256": legacy.source_config_sha256,
        "tensor_inventory": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in legacy.tensors.items()
        },
        "full_resume_unavailable_reason": (
            "Legacy v1 lacks resumable RNG and immutable provenance; "
            "use weights import, then train --promote."
        ),
    }


def _checkpoint_inspect(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if path.suffix == ".pt":
        manifest = _legacy_checkpoint_inventory(path)
    else:
        directory = CheckpointManager(path.parent.parent)._resolve(path)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("unsafe checkpoint directory")
        manifest_path = _safe_member(directory, "manifest.json")
        if manifest_path is None:
            raise ValueError("unsafe checkpoint manifest path")
        manifest = json.loads(manifest_path.read_text())
    print(
        json.dumps(manifest, sort_keys=True)
        if args.json
        else "\n".join(f"{key}: {value}" for key, value in manifest.items())
    )


def _checkpoint_verify(args: argparse.Namespace) -> None:
    path = Path(args.path)
    try:
        expected_config = load_config(Path(args.config)) if args.config else None
    except (ValueError, TypeError) as error:
        payload = {
            "valid": False,
            "errors": [{"field": "config", "reason": str(error)}],
            "files": [],
            "resume_level": None,
        }
        print(json.dumps(payload, sort_keys=True) if args.json else payload)
        raise SystemExit(1) from None
    if path.suffix == ".pt":
        try:
            metadata = _legacy_checkpoint_inventory(path, expected_config)
            payload = {
                "valid": True,
                "errors": [],
                "files": [{"name": path.name, "sha256": metadata["sha256"]}],
                "resume_level": "weights_only",
                "full_resume_unavailable_reason": metadata[
                    "full_resume_unavailable_reason"
                ],
            }
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            payload = {
                "valid": False,
                "errors": [{"field": "legacy_checkpoint", "reason": str(error)}],
                "files": [],
                "resume_level": "weights_only",
            }
    else:
        root = path.parent.parent
        report = CheckpointManager(root).verify(
            path,
            require_training_state=not args.weights_only,
            expected_config=expected_config,
        )
        payload = {
            "valid": report.valid,
            "errors": list(report.errors),
            "files": list(report.verified_files),
            "resume_level": report.resume_level,
        }
    print(json.dumps(payload, sort_keys=True) if args.json else payload)
    if not payload["valid"]:
        raise SystemExit(1)


def _inspect(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    inventory = parameter_inventory(config)
    runtime = inspect_runtime(config)
    observations = ExperimentStore.get_calibration(
        config.logging.root_dir,
        calibration_key(
            config, runtime, source_digest=str(source_identity()["sha256"])
        ),
    )
    estimate = calibrated_estimate(config, runtime, inventory, observations)
    values = {
        **inspection_report(config),
        "parameter_inventory": inventory.__dict__,
        "memory_estimate": estimate.__dict__,
        "runtime": runtime.as_dict(),
    }
    proposal = plan_memory(config, runtime, estimate)
    values["resource_proposal"] = asdict(proposal)
    if args.write_proposal:
        paths = write_resource_proposal(proposal, Path(args.write_proposal))
        values["proposal_paths"] = [str(path) for path in paths]
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
            stage_bundle=Path(args.stage_bundle) if args.stage_bundle else None,
        )
    )


def _eval(args: argparse.Namespace) -> None:
    loaded = load_run(args.run_id, Path(args.runs_dir), args.checkpoint, args.backend)
    result = loaded.evaluate()
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
            engine=loaded.engine,
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
        engine=loaded.engine,
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


def _study_plan(args: argparse.Namespace) -> None:
    from sparselab.experiments.study import plan_study, study_plan_payload

    plan = plan_study(Path(args.path), max_runs=args.max_runs)
    print(json.dumps(study_plan_payload(plan), indent=2, sort_keys=True))


def _study_submit(args: argparse.Namespace) -> None:
    from sparselab.experiments.study import plan_study, submit_study

    plan = plan_study(Path(args.path), max_runs=args.max_runs)
    receipt = submit_study(
        plan,
        Path(args.receipt),
        store=Path(args.store),
        worker=args.worker,
        stage_bundle=Path(args.stage_bundle) if args.stage_bundle else None,
    )
    print(
        json.dumps(
            {
                "study_sha256": plan.study_sha256,
                "receipt": str(Path(args.receipt)),
                "runs": receipt["runs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _study_collect(args: argparse.Namespace) -> None:
    from sparselab.experiments.study import collect_study, plan_study

    plan = plan_study(Path(args.path), max_runs=args.max_runs)
    report, report_path = collect_study(
        plan,
        Path(args.receipt),
        runs_dir=Path(args.runs_dir),
        checkpoint=args.checkpoint,
        backend=args.backend,
    )
    print(json.dumps({**report, "output": str(report_path)}, indent=2, sort_keys=True))


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
            engine=loaded.engine,
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


def _weights_import(args: argparse.Namespace) -> None:
    from sparselab.training.weight_import import import_weights

    result = import_weights(
        Path(args.source),
        Path(args.destination),
        load_config(Path(args.config)),
        source_format=args.format,
        source_tokenizer=Path(args.source_tokenizer),
        provenance=Path(args.provenance),
    )
    print(json.dumps(asdict(result), sort_keys=True, default=str))


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
    weights = commands.add_parser("weights")
    weight_commands = weights.add_subparsers(dest="weights_command", required=True)
    weight_import = weight_commands.add_parser("import")
    weight_import.add_argument("source")
    weight_import.add_argument("destination")
    weight_import.add_argument("--config", required=True)
    weight_import.add_argument(
        "--format",
        required=True,
        choices=("sparselab_legacy_v1", "hf_llama_safetensors"),
    )
    weight_import.add_argument("--source-tokenizer", required=True)
    weight_import.add_argument("--provenance", required=True)
    weight_import.set_defaults(handler=_weights_import)
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
    checkpoint_verify.add_argument("--config")
    checkpoint_verify.add_argument("--weights-only", action="store_true")
    checkpoint_verify.add_argument("--json", action="store_true")
    checkpoint_verify.set_defaults(handler=_checkpoint_verify)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("config")
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("--write-proposal")
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
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
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
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
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
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    engram_export.set_defaults(handler=_engram_export)
    engram_inspect = engram_commands.add_parser("inspect")
    engram_inspect.add_argument("path")
    engram_inspect.set_defaults(handler=_engram_inspect)
    pack = engram_commands.add_parser("pack")
    pack_commands = pack.add_subparsers(dest="engram_pack_command", required=True)
    pack_compile = pack_commands.add_parser("compile")
    pack_compile.add_argument("source")
    pack_compile.add_argument("--output", required=True)
    pack_compile.add_argument("--name", required=True)
    pack_compile.add_argument("--namespace", required=True)
    pack_compile.add_argument("--license")
    pack_compile.add_argument("--source-name")
    pack_compile.add_argument("--source-revision")
    pack_compile.add_argument("--created-at")
    pack_compile.add_argument("--lexical-package")
    pack_compile.add_argument("--semantic-keys")
    pack_compile.add_argument("--semantic-values")
    pack_compile.add_argument("--semantic-metadata")
    pack_compile.set_defaults(handler=_engram_pack_compile)
    pack_inspect = pack_commands.add_parser("inspect")
    pack_inspect.add_argument("path")
    pack_inspect.set_defaults(handler=_engram_pack_inspect)
    pack_verify = pack_commands.add_parser("verify")
    pack_verify.add_argument("path")
    pack_verify.add_argument("--expected-pack-id")
    pack_verify.set_defaults(handler=_engram_pack_verify)
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
    training.add_argument("--stage-bundle")
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
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
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
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
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
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    capability_compare.set_defaults(handler=_capability_compare)

    study = commands.add_parser(
        "study", help="Plan and compare matched architecture experiments."
    )
    study_commands = study.add_subparsers(dest="study_command", required=True)
    study_plan = study_commands.add_parser("plan")
    study_plan.add_argument("path", help="Architecture study YAML")
    study_plan.add_argument("--max-runs", type=int, default=1000)
    study_plan.set_defaults(handler=_study_plan)
    study_submit = study_commands.add_parser("submit")
    study_submit.add_argument("path", help="Architecture study YAML")
    study_submit.add_argument("--receipt", required=True)
    study_submit.add_argument("--worker")
    study_submit.add_argument("--stage-bundle")
    study_submit.add_argument("--max-runs", type=int, default=1000)
    study_submit.add_argument("--store", default=runs_dir_default)
    study_submit.set_defaults(handler=_study_submit)
    study_collect = study_commands.add_parser("collect")
    study_collect.add_argument("path", help="Architecture study YAML")
    study_collect.add_argument("receipt")
    study_collect.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: runs/ at the nearest pyproject.toml, otherwise ./runs)",
    )
    study_collect.add_argument("--checkpoint")
    study_collect.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    study_collect.add_argument("--max-runs", type=int, default=1000)
    study_collect.set_defaults(handler=_study_collect)
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
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    generation.set_defaults(handler=_generate)
    chat = commands.add_parser(
        "chat", help="Chat with a verified local PyTorch or native MLX checkpoint."
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
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
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
    from sparselab.workers.cli import add_commands

    add_commands(commands, default_store=Path(runs_dir_default))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
