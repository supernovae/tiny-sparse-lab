"""One-shot, zero-update matched-prompt inference on Card 05's selected v2 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sparselab.evaluation.generation import generate_with_token_ids
from sparselab.evaluation.inference import evaluation_config, load_run
from sparselab.runtime_environments import profile_for_id
from sparselab.runtime_profile import authorize_profile
from sparselab.training.manifest import sha256_file

RUN_ID = "kml-card05-full-tranche-v2"
CHECKPOINT = "step_00004883_gen_000011"
CHECKPOINT_SHA256 = "98934c011671b5f9583e7a70e24fc1f9bc14ae83495a7282f0ac324415964389"
PAIR_STRATA = (
    ("general_prose",) * 3
    + ("explanatory_prose",) * 3
    + ("incident_response_docs",) * 2
)
PARITY_CASES = ((0, "prose"), (0, "question"))


def validate_declaration(
    declaration: dict, tokenizer
) -> list[tuple[int, str, str, bool]]:
    if declaration.get("format") != "kml-card05-matched-diagnostic-v1":
        raise ValueError("unexpected declaration format")
    pairs = declaration.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise ValueError("exactly eight pairs required")
    jobs = []
    record_ids = set()
    for index, (pair, stratum) in enumerate(zip(pairs, PAIR_STRATA, strict=True)):
        if pair.get("stratum") != stratum or pair.get("split") != "test":
            raise ValueError("pair split or stratum changed")
        record_id = pair.get("source_record_id")
        if (
            not isinstance(record_id, str)
            or len(record_id) != 64
            or record_id in record_ids
        ):
            raise ValueError("source record IDs must be distinct and complete")
        record_ids.add(record_id)
        prose, question = pair.get("prose_prompt"), pair.get("question_prompt")
        if not isinstance(prose, str) or not isinstance(question, str):
            raise TypeError("missing prompt")
        if not prose or not question.startswith("Context:\n" + prose + "\nQuestion: "):
            raise ValueError("question must use exactly the same prose excerpt")
        if not question.endswith("\nAnswer:") or not pair.get("gold_answer"):
            raise ValueError("missing question or gold answer")
        for kind, prompt in (("prose", prose), ("question", question)):
            if len(tokenizer.encode(prompt, add_special_tokens=False).ids) + 64 > 1024:
                raise ValueError("prompt exceeds context with generation reserve")
            jobs.append((index, kind, prompt, True))
    for index, kind in PARITY_CASES:
        prompt = pairs[index]["prose_prompt" if kind == "prose" else "question_prompt"]
        jobs.append((index, kind, prompt, False))
    return jobs


def run(
    declaration_path: Path,
    output: Path,
    expected_declaration_sha256: str,
    selected_path: Path,
    runs: Path,
) -> None:
    if output.exists():
        raise FileExistsError("one-shot diagnostic output already exists")
    if sha256_file(declaration_path) != expected_declaration_sha256:
        raise ValueError("frozen declaration identity changed")
    selected = json.loads(selected_path.read_text())
    if (
        selected.get("run_id"),
        selected.get("checkpoint"),
        selected.get("checkpoint_sha256"),
    ) != (RUN_ID, CHECKPOINT, CHECKPOINT_SHA256):
        raise ValueError("selected v2 checkpoint identity changed")
    config = evaluation_config(RUN_ID, runs, CHECKPOINT, "rocm")
    authorization = authorize_profile(profile_for_id("rocm-7900xtx"), config)
    loaded = load_run(RUN_ID, runs, CHECKPOINT, "rocm", authorization=authorization)
    if loaded.identity["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("loaded checkpoint identity changed")
    declaration = json.loads(declaration_path.read_text())
    jobs = validate_declaration(declaration, loaded.tokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for ordinal, (pair_index, kind, prompt, cached) in enumerate(jobs):
            # A started marker is durable before GPU inference. A failed or
            # interrupted item is never silently rerun within this attempt.
            marker = output.parent / f"{ordinal:02d}.started"
            with marker.open("x") as started:
                started.write(f"{pair_index} {kind} cached={cached}\n")
                started.flush()
                os.fsync(started.fileno())
            text, token_ids = generate_with_token_ids(
                loaded.model,
                loaded.tokenizer,
                prompt,
                1024,
                64,
                loaded.device,
                temperature=0,
                top_k=0,
                seed=17,
                stop_sequences=(),
                strict_context=True,
                use_cache=cached,
                engine=loaded.engine,
            )
            if len(token_ids) > 64 or not text.startswith(prompt):
                raise ValueError("generation exceeded cap or changed prompt")
            row = {
                "ordinal": ordinal,
                "pair_index": pair_index,
                "kind": kind,
                "cached": cached,
                "prompt": prompt,
                "completion": text[len(prompt) :],
                "token_ids": token_ids,
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    if sha256_file(declaration_path) != expected_declaration_sha256:
        raise ValueError("declaration changed during inference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--declaration-sha256", required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.declaration, args.output, args.declaration_sha256, args.selected, args.runs
    )


if __name__ == "__main__":
    main()
