"""Explore the verified seed-42 50M and data-rich 30M checkpoints on identical text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run import STUDY, checkpoint, inputs

from sparselab.evaluation.generation import generate_with_token_ids


def compare(prompt: str, models: dict, *, tokens: int, decoder: str, seed: int) -> dict:
    """Reject truncation on either tokenizer before generating any completion."""
    encoded = {
        name: len(loaded.tokenizer.encode(prompt, add_special_tokens=False).ids) or 1
        for name, loaded in models.items()
    }
    for name, loaded in models.items():
        context = loaded.config.model.max_seq_len
        if encoded[name] + tokens > context:
            raise ValueError(
                f"{name} needs {encoded[name]} prompt tokens + {tokens} output tokens "
                f"but its context is {context}; shorten the prompt or output budget"
            )
    results = {}
    with torch.inference_mode():
        for name, loaded in models.items():
            text, ids = generate_with_token_ids(
                loaded.model,
                loaded.tokenizer,
                prompt,
                loaded.config.model.max_seq_len,
                tokens,
                loaded.device,
                temperature=0.0 if decoder == "greedy" else 0.8,
                top_k=0,
                seed=seed,
                strict_context=True,
            )
            results[name] = {
                "checkpoint_sha256": loaded.identity["checkpoint_sha256"],
                "context_limit": loaded.config.model.max_seq_len,
                "prompt_tokens": encoded[name],
                "completion": text[len(prompt) :],
                "text": text,
                "generated_token_ids": ids,
                "early_eos": len(ids) < tokens,
            }
    return {
        "prompt": prompt,
        "decoder": decoder,
        "temperature": 0.0 if decoder == "greedy" else 0.8,
        "top_k": 0,
        "seed": seed,
        "max_new_tokens": tokens,
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="One prompt; omit to enter interactive mode")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 multiline prompt file")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--decoder", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--seed", type=int, default=11, help="Sampled-policy RNG seed")
    parser.add_argument(
        "--json", action="store_true", help="Print one JSON object per prompt"
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or not 0 <= args.seed < 2**64:
        parser.error("max-new-tokens must be positive and seed must be uint64")
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    else:
        prompt = args.prompt
    if prompt is not None and not prompt.strip():
        parser.error("prompt must contain text")
    if args.json and prompt is None:
        parser.error("--json requires --prompt or --prompt-file")

    prereg, _ = inputs()
    models = {name: checkpoint(name, prereg) for name in ("50m", "new")}
    evidence = json.loads((STUDY / "evidence.json").read_text(encoding="utf-8"))
    control = evidence["outputs"]
    if not args.json:
        old = control["control/50m"]["summary"]["bits_per_byte"]
        new = control["control/new"]["summary"]["bits_per_byte"]
        print(
            f"Frozen 256-story bits/UTF-8 byte: 50M={old:.6f}; data-rich 30M={new:.6f}"
        )
        print(
            "Different training data, tokenizer, context, budget, precision and architecture; no causal attribution or human-quality score."
        )
        if prompt is None:
            print("Enter a prompt per line; empty line or Ctrl-D quits.")

    while True:
        if prompt is None:
            try:
                prompt = input("prompt> ")
            except EOFError:
                break
            if not prompt.strip():
                break
        try:
            result = compare(
                prompt,
                models,
                tokens=args.max_new_tokens,
                decoder=args.decoder,
                seed=args.seed,
            )
        except ValueError as error:
            if args.prompt is not None or args.prompt_file is not None:
                parser.error(str(error))
            print(f"Cannot compare: {error}")
        else:
            result["reference_control_bits_per_byte"] = {
                name: control[f"control/{name}"]["summary"]["bits_per_byte"]
                for name in models
            }
            result["interpretation"] = (
                "Single-prompt output is exploratory; the frozen 256-story likelihood "
                "is not a controlled attribution to data variety or model depth."
            )
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                for name, label in (("50m", "Original 50M"), ("new", "Data-rich 30M")):
                    row = result["models"][name]
                    print(
                        f"\n--- {label} ({row['prompt_tokens']}/{row['context_limit']} prompt/context tokens, {len(row['generated_token_ids'])} output tokens) ---"
                    )
                    print(row["text"])
                print()
        if args.prompt is not None or args.prompt_file is not None:
            break
        prompt = None


if __name__ == "__main__":
    main()
