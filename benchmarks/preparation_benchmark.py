"""Offline, fresh-process preparation measurements; never a CI timing assertion.

uv run --locked --offline python benchmarks/preparation_benchmark.py \
    --workspace sparselab-work/runtime-prep-v1/benchmarks
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

MIB = 1024 * 1024
BYTE_CAP = MIB


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def corpus(path: Path, target_bytes: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    records = source_bytes = 0
    with path.open("xb") as handle:
        while handle.tell() < target_bytes:
            number = rng.randrange(1_000_000)
            paragraph = (
                f"{path.stem} record-{records}: The café fox 🦊 considers sample {number}, a sparse memory and a blue stone. "
                "Keep document boundaries, Unicode and EOS unchanged.\n"
                f"```python\ndef sparse_{number}(values):\n"
                f"    return [x + {number % 101} for x in values if x % 3]\n```\n"
            )
            text = paragraph * rng.choice((2, 4, 8, 16))
            handle.write(
                (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
            )
            records += 1
            source_bytes += len(text.encode("utf-8"))
    return {
        "records": records,
        "source_utf8_bytes": source_bytes,
        "jsonl_bytes": path.stat().st_size,
    }


def worker(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    import psutil

    process = psutil.Process()
    stop = threading.Event()
    sampled = {"parent_peak_rss_bytes": 0, "process_tree_peak_rss_bytes": 0}

    def sample() -> None:
        while not stop.is_set():
            try:
                parent = process.memory_info().rss
                tree = parent
                for child in process.children(recursive=True):
                    try:
                        tree += child.memory_info().rss
                    except psutil.Error, OSError:
                        pass
                sampled["parent_peak_rss_bytes"] = max(
                    sampled["parent_peak_rss_bytes"], parent
                )
                sampled["process_tree_peak_rss_bytes"] = max(
                    sampled["process_tree_peak_rss_bytes"], tree
                )
            except psutil.Error, OSError:
                pass
            stop.wait(0.025)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        from sparselab.config.models import RunConfig
        from sparselab.data.encoding import PreparationEncoder
        from sparselab.data.packing import prepare_data
        from sparselab.data.tokenizer import load_tokenizer
        from sparselab.resource_envelope import ResourceEnvelope
        from sparselab.training.manifest import sha256_file

        run = RunConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        run = run.model_copy(
            update={"dataset": run.dataset.model_copy(update={"cache_dir": args.cache})}
        )
        observed = {
            "maximum_batch_documents": 0,
            "maximum_batch_source_bytes": 0,
            "encode_calls": 0,
        }
        original = PreparationEncoder.encode

        def bounded_encode(self: PreparationEncoder, texts: list[str]):
            byte_count = sum(len(text.encode("utf-8")) for text in texts)
            if len(texts) > args.documents or byte_count > BYTE_CAP:
                raise ValueError("observed tokenizer batch exceeds benchmark caps")
            observed["maximum_batch_documents"] = max(
                observed["maximum_batch_documents"], len(texts)
            )
            observed["maximum_batch_source_bytes"] = max(
                observed["maximum_batch_source_bytes"], byte_count
            )
            observed["encode_calls"] += 1
            return original(self, texts)

        PreparationEncoder.encode = bounded_encode
        preparation_started = time.perf_counter()
        prepared = prepare_data(
            run,
            load_tokenizer(run.tokenizer.path),
            tokenizer_batch_documents=args.documents,
            tokenizer_batch_source_bytes=BYTE_CAP,
            resource_envelope=ResourceEnvelope(
                resource_envelope_version=1, max_workers=args.threads
            ),
        )
        preparation_seconds = time.perf_counter() - preparation_started
        elapsed = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_SELF)
        children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        current_rss = process.memory_info().rss
        stop.set()
        sampler.join()
        receipt_path = prepared.root.with_name(prepared.root.name + ".preparation.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        arrays = {
            path.name: sha256_file(path) for path in sorted(prepared.root.glob("*.npy"))
        }
        source_bytes = int(receipt["source_bytes"])
        output_tokens = int(receipt["output_tokens"])
        result = {
            "status": "ok",
            "elapsed_wall_seconds": elapsed,
            "preparation_seconds": preparation_seconds,
            "cpu_user_seconds": after.ru_utime
            - before.ru_utime
            + children_after.ru_utime
            - children_before.ru_utime,
            "cpu_system_seconds": after.ru_stime
            - before.ru_stime
            + children_after.ru_stime
            - children_before.ru_stime,
            "source_mb_per_second": source_bytes / 1_000_000 / elapsed,
            "output_tokens_per_second": output_tokens / elapsed,
            "source_utf8_bytes": source_bytes,
            "output_tokens": output_tokens,
            "current_rss_bytes": current_rss,
            "peak_rss_bytes": int(after.ru_maxrss * 1024),
            **sampled,
            "backend": "cpu",
            "runtime": {
                "python": str(Path(sys.executable).resolve()),
                "sys_prefix": sys.prefix,
                "torch_version": str(sys.modules["torch"].__version__),
                "torch_file": sys.modules["torch"].__file__,
                "host": platform.platform(),
            },
            "cpu_count": os.cpu_count(),
            "affinity_cpu_count": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "rayon_threads": receipt["tokenizer_rayon_threads"],
            "batch_documents": args.documents,
            "batch_source_bytes": BYTE_CAP,
            "arrays_sha256": arrays,
            "manifest_sha256": prepared.manifest["manifest_sha256"],
            "cache_identity": prepared.manifest["cache_identity"],
            "observed_batches": observed,
            "operational_receipt": receipt,
        }
        if result["rayon_threads"] != args.threads:
            raise ValueError(
                "actual preparation Rayon cap differs from benchmark request"
            )
        write_json(args.result, result)
    finally:
        stop.set()
        sampler.join()


def benchmark(args: argparse.Namespace) -> int:
    from sparselab.config.models import RunConfig, TokenizerTrainConfig
    from sparselab.data.tokenizer import train_tokenizer
    from sparselab.training.manifest import source_identity

    args.workspace.mkdir(parents=True, exist_ok=True)
    root = args.workspace / f"run-{time.time_ns()}"
    root.mkdir()
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or min(sizes) < 1:
        raise ValueError("sizes must be positive MiB values")
    # Source, eight independent outputs and chunk spools: conservative 32x source budget.
    expected = sum(sizes) * MIB * 32 + 512 * MIB
    storage = os.statvfs(root)
    if storage.f_bavail * storage.f_frsize < expected or storage.f_favail < 2048:
        raise ValueError(f"benchmark requires {expected} free bytes and 2048 inodes")
    fixtures = {}
    for size in sizes:
        folder = root / f"{size}mib"
        folder.mkdir()
        train = corpus(folder / "train.jsonl", size * MIB, 1701)
        validation = corpus(
            folder / "validation.jsonl", min(size * MIB // 32, MIB), 2903
        )
        fixtures[size] = (folder, train, validation)
    first, train, validation = fixtures[sizes[0]]
    dataset = {
        "source": "local_text",
        "license": "generated-offline-fixture",
        "cache_dir": str(first / "unused-cache"),
        "train_path": str(first / "train.jsonl"),
        "validation_path": str(first / "validation.jsonl"),
        "train_max_documents": train["records"],
        "validation_max_documents": validation["records"],
        "train_max_tokens": train["source_utf8_bytes"] + train["records"],
        "validation_max_tokens": validation["source_utf8_bytes"]
        + validation["records"],
    }
    tokenizer = train_tokenizer(
        TokenizerTrainConfig.model_validate(
            {
                "schema_version": 1,
                "vocab_size": 512,
                "min_frequency": 1,
                "max_documents": min(256, train["records"]),
                "output_dir": root / "tokenizer",
                "dataset": dataset,
            }
        )
    )
    threads = sorted({1, min(4, os.cpu_count() or 1)})
    cases = []
    failures = []
    for size, (folder, train, validation) in fixtures.items():
        dataset.update(
            train_path=str(folder / "train.jsonl"),
            validation_path=str(folder / "validation.jsonl"),
            train_max_documents=train["records"],
            validation_max_documents=validation["records"],
            train_max_tokens=train["source_utf8_bytes"] + train["records"],
            validation_max_tokens=validation["source_utf8_bytes"]
            + validation["records"],
        )
        run = RunConfig.model_validate(
            {
                "schema_version": 2,
                "name": "runtime-prep-v1-benchmark",
                "seed": 1701,
                "runtime": {"engine": "pytorch", "backend": "cpu", "precision": "fp32"},
                "model": {
                    "vocab_size": 512,
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "max_seq_len": 32,
                },
                "tokenizer": {"path": tokenizer},
                "dataset": dataset,
                "training": {
                    "micro_batch_size": 1,
                    "seq_len": 16,
                    "max_steps": 2,
                    "max_tokens": 32,
                },
                "optimizer": {"name": "adamw", "warmup_steps": 1},
                "logging": {"root_dir": root / "runs"},
            }
        )
        config = folder / "run.json"
        config.write_text(run.model_dump_json(), encoding="utf-8")
        reference = None
        for thread_count in threads:
            for documents in (1, 16, 64, 256):
                case = folder / f"threads-{thread_count}-batch-{documents}"
                case.mkdir()
                env = os.environ.copy()
                env.update(
                    HF_HUB_OFFLINE="1",
                    HF_DATASETS_OFFLINE="1",
                    UV_OFFLINE="1",
                    SPARSELAB_WORK_DIR=str(root),
                    RAYON_NUM_THREADS=str(thread_count),
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--config",
                    str(config),
                    "--cache",
                    str(case / "cache"),
                    "--documents",
                    str(documents),
                    "--threads",
                    str(thread_count),
                    "--result",
                    str(case / "result.json"),
                ]
                with (
                    (case / "stdout.log").open("wb") as stdout,
                    (case / "stderr.log").open("wb") as stderr,
                ):
                    result = subprocess.run(
                        command, env=env, stdout=stdout, stderr=stderr, check=False
                    )
                if result.returncode:
                    failure = {
                        "size_mib": size,
                        "batch_documents": documents,
                        "rayon_threads": thread_count,
                        "status": "failed",
                        "returncode": result.returncode,
                        "raw_case": str(case),
                    }
                    cases.append(failure)
                    failures.append(failure)
                    continue
                measured = json.loads(
                    (case / "result.json").read_text(encoding="utf-8")
                )
                measured.update(size_mib=size, raw_case=str(case))
                scientific = (
                    measured["arrays_sha256"],
                    measured["manifest_sha256"],
                    measured["cache_identity"],
                )
                if reference is None:
                    reference = scientific
                elif scientific != reference:
                    raise ValueError(f"scientific identity differs for {case}")
                cases.append(measured)
                print(
                    f"{size}MiB threads={thread_count} batch={documents}: "
                    f"{measured['source_mb_per_second']:.3f}MB/s "
                    f"{measured['process_tree_peak_rss_bytes'] / MIB:.1f}MiB tree peak",
                    flush=True,
                )
    selection_threads = max(threads)
    scalar = next(
        (
            case
            for case in cases
            if case.get("status") == "ok"
            and case["size_mib"] == 32
            and case["rayon_threads"] == selection_threads
            and case["batch_documents"] == 1
        ),
        None,
    )
    eligible = (
        []
        if scalar is None
        else [
            case
            for case in cases
            if case.get("status") == "ok"
            and case["size_mib"] == 32
            and case["rayon_threads"] == selection_threads
            and case["batch_documents"] in (16, 64, 256)
            and case["observed_batches"]["maximum_batch_source_bytes"] <= BYTE_CAP
            and case["process_tree_peak_rss_bytes"]
            < 1.5 * scalar["process_tree_peak_rss_bytes"]
        ]
    )
    chosen = (
        min(
            eligible,
            key=lambda case: (case["elapsed_wall_seconds"], case["batch_documents"]),
        )
        if eligible
        else None
    )
    summary = {
        "schema_version": 1,
        "fixture_seed": 1701,
        "raw_workspace": str(root),
        "source_identity": source_identity(),
        "measurement": "one fresh CPU process per case; wall includes imports and encoder startup; bounded-batch instrumentation; tree RSS sampled every25ms",
        "exact_scientific_parity_per_size": not failures,
        "selection": {
            "batch_documents": chosen["batch_documents"] if chosen else 16,
            "rayon_threads": selection_threads,
            "basis": "fastest successful32MiB case below1.5x scalar sampled process-tree peak and1MiB input cap",
            "qualified": chosen is not None,
        },
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key not in {"cache_identity", "operational_receipt"}
            }
            for case in cases
        ],
        "hardware_gate": "CPU only; no accelerator results or general speedup claim",
    }
    write_json(args.summary, summary)
    print(
        f"summary: {args.summary}; selected batch={summary['selection']['batch_documents']}"
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("sparselab-work/runtime-prep-v1/benchmarks"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/benchmarks/runtime-prep-v1.json"),
    )
    parser.add_argument("--sizes", default="1,8,32")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--documents", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    return benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
