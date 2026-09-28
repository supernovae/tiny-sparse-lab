"""Exclusive-write seed-42 evidence for the frozen data-rich reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import time
from collections import Counter
from itertools import islice
from pathlib import Path

import torch

from sparselab.config.loading import load_config
from sparselab.data.datasets import iter_documents
from sparselab.data.local_stories import verify_snapshot
from sparselab.evaluation.generation import generate_with_token_ids
from sparselab.evaluation.inference import load_run
from sparselab.evaluation.text_likelihood import score_story

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
WORK = ROOT / "sparselab-work/experiments/tinystories-dense-30m-data-rich-v1"
OLD_ROOT = ROOT.parent / "tiny-sparse-lab"
RUN_ID = "tinystories-dense-30m-data-rich-v1-seed42"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def inputs() -> tuple[dict, dict]:
    prereg = json.loads((STUDY / "preregistration.json").read_text())
    for relative, expected in prereg["frozen_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"frozen input differs: {relative}")
    source = json.loads((STUDY / "source-receipt.json").read_text())
    config = load_config(ROOT / "configs/tinystories_dense_30m_data_rich_v1.yaml")
    verify_snapshot(config.dataset)
    if digest(ROOT / source["source_manifest"]) != source["source_manifest_sha256"]:
        raise ValueError("source receipt differs from verified snapshot")
    selection = json.loads((STUDY / "tokenizer-selection.json").read_text())
    if (
        digest(ROOT / selection["raw_receipt"]) != selection["raw_receipt_sha256"]
        or digest(config.tokenizer.path) != selection["selected_tokenizer_sha256"]
    ):
        raise ValueError("tokenizer selection receipt differs from selected artifact")
    runtime = json.loads((STUDY / "runtime-selection.json").read_text())
    if (
        digest(ROOT / "configs/tinystories_dense_30m_data_rich_v1.yaml")
        != runtime["final_config_sha256"]
    ):
        raise ValueError("final training config differs from runtime receipt")
    panel = json.loads(
        (
            ROOT / "experiments/research/dense-lm-decoding-v1/evidence/selection.json"
        ).read_text()
    )
    if panel["selected"] != "temperature_0_8":
        raise ValueError("prior selected decoder changed")
    if panel["preregistration_sha256"] != digest(
        ROOT / "experiments/research/dense-lm-decoding-v1/preregistration.json"
    ):
        raise ValueError("prior decoder selection provenance changed")
    return prereg, {
        "config": config,
        "source": source,
        "selection": selection,
        "runtime": runtime,
    }


def checkpoint(model: str, prereg: dict, *, endpoint: dict | None = None):
    if model == "new":
        if endpoint is None:
            endpoint = json.loads((STUDY / "endpoint-receipt.json").read_text())
        if endpoint["config_sha256"] != digest(
            ROOT / "configs/tinystories_dense_30m_data_rich_v1.yaml"
        ):
            raise ValueError("endpoint config identity changed")
        run_id, runs_dir = RUN_ID, WORK / "runs"
        generation = runs_dir / run_id / "checkpoints" / endpoint["generation"]
        expected = endpoint["checkpoint_digest"]
        manifest_sha256 = endpoint["checkpoint_manifest_sha256"]
    else:
        matches = [
            row
            for row in json.loads(
                (ROOT / prereg["reference_checkpoint_receipt"]).read_text()
            )["checkpoints"]
            if row["model"] == model and row["seed"] == 42
        ]
        if len(matches) != 1:
            raise ValueError("missing unique old seed-42 reference")
        row = matches[0]
        run_id = row["run_id"]
        generation = OLD_ROOT / row["generation"]
        runs_dir = generation.parents[2]
        expected = row["digest"]
        manifest_sha256 = row["manifest_sha256"]
    if digest(generation / "manifest.json") != manifest_sha256:
        raise ValueError(f"checkpoint manifest bytes differ: {model}")
    loaded = load_run(run_id, runs_dir, str(generation), backend="rocm")
    if loaded.identity["checkpoint_sha256"] != expected:
        raise ValueError(f"verified checkpoint differs: {model}")
    if not isinstance(loaded.device, torch.device) or loaded.device.type != "cuda":
        raise ValueError("ROCm device is required for comparable execution")
    return loaded


def source_stories(config, *, sample: bool):
    start, stop = (2000, 2256) if sample else (2000, 10000)
    return list(islice(iter_documents(config.dataset, "validation"), start, stop))


def score(model: str, kind: str, prereg: dict, registered: dict) -> None:
    if kind == "heldout" and model != "new":
        raise ValueError("native full-512 primary validation belongs to the new model")
    loaded = checkpoint(model, prereg)
    source = registered["config"]
    sample = kind == "control"
    stories = source_stories(source, sample=sample)
    if len(stories) != (256 if sample else 8000):
        raise ValueError("held-out source-split story count changed")
    receipt_hash = None
    if sample:
        receipt = json.loads((STUDY / "likelihood-sample.json").read_text())
        h = hashlib.sha256()
        for story in stories:
            encoded = story.encode("utf-8")
            h.update(len(encoded).to_bytes(8, "big"))
            h.update(encoded)
        if h.hexdigest() != receipt["ordered_text_sha256"]:
            raise ValueError("sealed raw-story likelihood sample changed")
        receipt_hash = digest(STUDY / "likelihood-sample.json")
    path = WORK / "evaluation" / kind / f"{model}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    window, stride = (128, 64) if sample else (512, 256)
    header = {
        "type": "header",
        "format": "tinystories_data_rich_likelihood_v1",
        "kind": kind,
        "model": model,
        "source_manifest_sha256": registered["source"]["source_manifest_sha256"],
        "sample_receipt_sha256": receipt_hash,
        "checkpoint": loaded.identity,
        "input_length": window,
        "stride": stride,
        "start_context": "eos",
        "end_target": "eos",
        "first_snapshot_validation_index": 2000,
        "story_count": len(stories),
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical(header).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        totals = Counter()
        errors = 0
        for index, story in enumerate(stories, start=2000):
            started = time.monotonic()
            try:
                observation = score_story(
                    loaded.model,
                    loaded.tokenizer,
                    story,
                    device=loaded.device,
                    window_size=window,
                    stride=stride,
                )
                torch.cuda.synchronize(loaded.device)
                result = {"status": "ok", **observation}
                for field in ("nll_nats", "utf8_bytes", "target_count", "token_count"):
                    totals[field] += observation[field]
            except Exception as error:  # noqa: BLE001 - retain every failed story.
                errors += 1
                result = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            row = {
                "type": "story",
                "snapshot_validation_index": index,
                "text_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
                "elapsed_seconds": time.monotonic() - started,
                **result,
            }
            handle.write(canonical(row).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        summary = {
            "type": "summary",
            "successful_stories": len(stories) - errors,
            "failed_stories": errors,
            "sum_nll_nats": totals["nll_nats"],
            "sum_utf8_bytes": totals["utf8_bytes"],
            "sum_targets": totals["target_count"],
            "sum_document_tokens": totals["token_count"],
            "nll_per_target": totals["nll_nats"] / totals["target_count"]
            if totals["target_count"]
            else None,
            "perplexity": math.exp(totals["nll_nats"] / totals["target_count"])
            if totals["target_count"]
            else None,
            "nll_per_byte": totals["nll_nats"] / totals["utf8_bytes"]
            if totals["utf8_bytes"]
            else None,
            "bits_per_byte": totals["nll_nats"] / (totals["utf8_bytes"] * math.log(2))
            if totals["utf8_bytes"]
            else None,
        }
        handle.write(canonical(summary).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(path, summary, flush=True)


def prompts(kind: str, prereg: dict, model: str) -> list[dict]:
    if kind == "regression":
        rows = json.loads((ROOT / "data/dense_lm_v1_prompts.json").read_text())[
            "prompts"
        ]
        if len(rows) != 6:
            raise ValueError("regression panel changed")
        return rows
    if kind in {"development", "test"}:
        rows = json.loads(
            (
                ROOT / f"experiments/research/dense-lm-decoding-v1/{kind}.json"
            ).read_text()
        )["prompts"]
        if len(rows) != (22 if kind == "development" else 55):
            raise ValueError("sealed decoding panel changed")
        return rows
    if model != "new":
        raise ValueError("old models cannot run native long context")
    payload = json.loads((STUDY / "long-prompts.json").read_text())
    subset = "development" if kind == "long-development" else "test"
    return [
        {
            "id": row["id"],
            "text": row["opening"]
            + " "
            + payload["common_middle"]
            + " "
            + row["ending"],
            "anchors": row["anchors"],
            "category": "long_context",
        }
        for row in payload[subset]
    ]


def repeat_excess(ids: list[int], order: int) -> int:
    return sum(
        count - 1
        for count in Counter(zip(*(ids[i:] for i in range(order)))).values()
        if count > 1
    )


def generate(model: str, kind: str, prereg: dict, registered: dict) -> None:
    loaded = checkpoint(model, prereg)
    prior_script = ROOT / "experiments/research/dense-lm-decoding-v1/run.py"
    prior_receipt = json.loads(
        (ROOT / prereg["reference_checkpoint_receipt"]).read_text()
    )
    if (
        digest(prior_script)
        != prior_receipt["frozen_sha256"][
            "experiments/research/dense-lm-decoding-v1/run.py"
        ]
    ):
        raise ValueError("prior mechanical diagnostics source changed")
    diagnose = runpy.run_path(str(prior_script))["diagnostics"]
    rows = prompts(kind, prereg, model)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate frozen prompt ID")
    is_regression = kind == "regression"
    policies = (
        [("greedy", 0.0, 0, None)]
        if is_regression
        else [
            ("greedy", 0.0, 0, None),
            ("temperature_0_8", 0.8, 0, 11),
            ("temperature_0_8", 0.8, 0, 29),
        ]
    )
    path = WORK / "evaluation" / "generation" / f"{kind}-{model}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_source = (
        ROOT / f"experiments/research/dense-lm-decoding-v1/{kind}.json"
        if kind in {"development", "test"}
        else STUDY / "long-prompts.json"
        if kind.startswith("long-")
        else ROOT / "data/dense_lm_v1_prompts.json"
    )
    header = {
        "type": "header",
        "format": "tinystories_data_rich_generation_v1",
        "model": model,
        "kind": kind,
        "checkpoint": loaded.identity,
        "prompt_sha256": digest(prompt_source),
        "preregistration_sha256": digest(STUDY / "preregistration.json"),
        "selected_policy_sha256": digest(
            ROOT / "experiments/research/dense-lm-decoding-v1/evidence/selection.json"
        ),
        "max_new_tokens": 32 if is_regression else 48,
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical(header).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        for prompt in rows:
            length = len(
                loaded.tokenizer.encode(prompt["text"], add_special_tokens=False).ids
            )
            if kind.startswith("long-") and not 257 <= length <= 512 - 48:
                raise ValueError(
                    f"long prompt has invalid new-model length: {prompt['id']} {length}"
                )
            for policy, temperature, top_k, seed in policies:
                started = time.monotonic()
                try:
                    text, ids = generate_with_token_ids(
                        loaded.model,
                        loaded.tokenizer,
                        prompt["text"],
                        loaded.config.model.max_seq_len,
                        32 if is_regression else 48,
                        loaded.device,
                        temperature=temperature,
                        top_k=top_k,
                        seed=0 if seed is None else seed,
                        strict_context=True,
                    )
                    torch.cuda.synchronize(loaded.device)
                    if not text.startswith(prompt["text"]):
                        raise ValueError("prompt prefix changed")
                    result = {
                        "status": "ok",
                        "text": text,
                        "completion": text[len(prompt["text"]) :],
                        "generated_token_ids": ids,
                        "early_eos": len(ids) < (32 if is_regression else 48),
                        "repeated_bigram_excess": repeat_excess(ids, 2),
                        "repeated_trigram_excess": repeat_excess(ids, 3),
                    }
                    if not is_regression:
                        result["diagnostics"] = diagnose(
                            result["completion"],
                            ids,
                            loaded.tokenizer,
                            prompt.get("anchors", {}),
                        )
                except Exception as error:  # noqa: BLE001 - retain every failed cell.
                    result = {
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                cell = {
                    "type": "cell",
                    "prompt_id": prompt["id"],
                    "prompt": prompt["text"],
                    "category": prompt.get("category"),
                    "anchors": prompt.get("anchors", {}),
                    "prompt_tokens": length,
                    "short_subset": length <= 256,
                    "policy": policy,
                    "temperature": temperature,
                    "top_k": top_k,
                    "rng_seed": seed,
                    "checkpoint_digest": loaded.identity["checkpoint_sha256"],
                    "elapsed_seconds": time.monotonic() - started,
                    **result,
                }
                handle.write(canonical(cell).decode("utf-8") + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    print(path, len(rows) * len(policies), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind",
        choices=(
            "control",
            "heldout",
            "regression",
            "development",
            "test",
            "long-development",
            "long-test",
        ),
    )
    parser.add_argument("--model", required=True, choices=("new", "30m", "50m"))
    args = parser.parse_args()
    prereg, registered = inputs()
    if args.kind in {"control", "heldout"}:
        score(args.model, args.kind, prereg, registered)
    else:
        generate(args.model, args.kind, prereg, registered)


if __name__ == "__main__":
    main()
