"""Execute the preregistered decoding-only study; no training or weight writes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch

from sparselab.evaluation.generation import generate_with_token_ids
from sparselab.evaluation.inference import load_run

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
WORK = ROOT / "sparselab-work/experiments/dense-lm-decoding-v1"
PRE = STUDY / "preregistration.json"
COLORS = (
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "black",
    "white",
    "brown",
    "gold",
    "gray",
)
DIMENSIONS = (
    "prompt_adherence",
    "entities",
    "stated_attributes",
    "temporal_causal_consistency",
    "repetition",
    "local_readability",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_protocol() -> dict:
    pre = json.loads(PRE.read_text())
    if pre["format"] != "dense_lm_decoding_preregistration_v1":
        raise ValueError("unrecognized preregistration")
    for name, expected in pre["frozen_sha256"].items():
        path = ROOT / name
        if digest(path) != expected:
            raise ValueError(f"frozen input changed: {name}")
    if len(pre["checkpoints"]) != 6 or pre["sample_seeds"] != [11, 29]:
        raise ValueError("incomplete frozen matrix")
    return pre


def prompt_rows(split: str, pre: dict) -> list[dict]:
    payload = json.loads((STUDY / f"{split}.json").read_text())
    rows = payload["prompts"]
    if payload["format"] != "dense_lm_decoding_prompts_v1" or payload["split"] != split:
        raise ValueError("wrong prompt set")
    if len(rows) != (22 if split == "development" else 55):
        raise ValueError("wrong prompt count")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate prompt IDs")
    return rows


def repeated_excess(tokens: list[int], order: int) -> int:
    return sum(
        n - 1
        for n in Counter(zip(*(tokens[i:] for i in range(order)))).values()
        if n > 1
    )


def explicit_checks(completion: str, anchors: dict) -> dict:
    """Only overt relabeling and local color-noun assertions, not semantic parsing."""
    text = completion.lower()
    colors = []
    for entry in anchors.get("attributes", []):
        obj, expected = entry["object"].lower(), entry["value"].lower()
        if expected not in COLORS:
            continue
        noun = re.escape(obj)
        alternatives = "|".join(COLORS)
        phrases = re.findall(
            rf"\b({alternatives})\s+{noun}\b|\b{noun}\s+(?:was|is|looked|became)\s+({alternatives})\b",
            text,
        )
        asserted = [left or right for left, right in phrases]
        colors.append(
            {
                "object": obj,
                "expected": expected,
                "observed": asserted,
                "contradiction": any(color != expected for color in asserted),
            }
        )
    names = []
    for name in anchors.get("names", []):
        matches = re.findall(
            rf"\b{re.escape(name)}\s+(?:was|is)\s+(?:called|named)\s+(\w+)",
            completion,
            flags=re.IGNORECASE,
        )
        names.append(
            {
                "name": name,
                "overt_relabels": [
                    other for other in matches if other.lower() != name.lower()
                ],
            }
        )
    objects = []
    for obj in anchors.get("objects", []):
        matches = re.findall(
            rf"\b{re.escape(obj)}\s+(?:was|is)\s+(?:actually|really)\s+(?:an?\s+)?(\w+)",
            text,
        )
        objects.append(
            {
                "object": obj,
                "overt_relabels": [other for other in matches if other != obj.lower()],
            }
        )
    return {
        "colors": colors,
        "names": names,
        "objects": objects,
        "explicit_contradiction": any(e["contradiction"] for e in colors)
        or any(e["overt_relabels"] for e in (*names, *objects)),
    }


def diagnostics(completion: str, tokens: list[int], tokenizer, anchors: dict) -> dict:
    counts = Counter(tokens)
    sentences = [
        re.sub(r"\s+", " ", s.strip().lower())
        for s in re.split(r"(?<=[.!?])\s+", completion)
        if len(s.strip()) >= 7 and s.rstrip().endswith((".", "!", "?"))
    ]
    special = {tokenizer.token_to_id(x) for x in ("<pad>", "<bos>", "<eos>", "<unk>")}
    anchor = explicit_checks(completion, anchors)
    return {
        "generated_token_count": len(tokens),
        "empty": not completion.strip(),
        "early_stop": len(tokens) < 48,
        "repeated_token_excess": sum(n - 1 for n in counts.values() if n > 1),
        "repeated_bigram_excess": repeated_excess(tokens, 2),
        "repeated_trigram_excess": repeated_excess(tokens, 3),
        "exact_sentence_repetition_excess": sum(
            n - 1 for n in Counter(sentences).values() if n > 1
        ),
        "special_token_count": sum(t in special for t in tokens),
        "explicit_anchors": anchor,
    }


def cells(split: str, pre: dict, selected: str | None = None):
    policies = (
        pre["policies"]
        if split == "development"
        else [
            p for p in pre["policies"] if p["name"] in ("regression_decoder", selected)
        ]
    )
    for row in prompt_rows(split, pre):
        for policy in policies:
            for rng in [None] if policy["temperature"] == 0 else pre["sample_seeds"]:
                yield row, policy, rng


def output_path(split: str, checkpoint: dict) -> Path:
    return WORK / split / f"{checkpoint['model']}-seed{checkpoint['seed']}.jsonl"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def development_digests(pre: dict) -> dict[str, str]:
    paths = [output_path("development", cp) for cp in pre["checkpoints"]]
    for path in paths:
        rows = read_rows(path)
        if len(rows) != 1 + 22 * 5 or any(
            row.get("status") != "ok" for row in rows[1:]
        ):
            raise ValueError(f"incomplete development file: {path}")
    return {path.name: digest(path) for path in paths}


def selection(pre: dict) -> dict:
    receipt = json.loads((WORK / "selection.json").read_text())
    if receipt["development_sha256"] != development_digests(pre) or receipt[
        "preregistration_sha256"
    ] != digest(PRE):
        raise ValueError("selection receipt differs from frozen development evidence")
    if receipt["selected"] not in {"temperature_0_8", "top_k_40"}:
        raise ValueError("invalid selected policy")
    return receipt


def run_checkpoint(
    split: str, pre: dict, checkpoint: dict, selected: str | None
) -> None:
    path = output_path(split, checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: an interrupted or completed evidence file is never silently rerun.
    with path.open("x", encoding="utf-8") as handle:
        header = {
            "type": "header",
            "format": "dense_lm_decoding_cells_v1",
            "split": split,
            "checkpoint": checkpoint,
            "preregistration_sha256": digest(PRE),
            "prompt_sha256": pre["frozen_sha256"][
                f"experiments/research/dense-lm-decoding-v1/{split}.json"
            ],
        }
        handle.write(canonical(header) + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())
        ck = ROOT / checkpoint["generation"]
        if digest(ck / "manifest.json") != checkpoint["manifest_sha256"]:
            raise ValueError("checkpoint manifest bytes changed")
        loaded = load_run(
            checkpoint["run_id"], ck.parent.parent.parent, str(ck), "rocm"
        )
        if (
            loaded.identity["checkpoint_sha256"] != checkpoint["digest"]
            or loaded.identity["step"] != 16384
            or loaded.identity["tokens_seen"] != 16777216
        ):
            raise ValueError("incorrect mature checkpoint")
        if (
            loaded.identity["tokenizer_sha256"] != pre["tokenizer_sha256"]
            or loaded.identity["data_sha256"] != pre["data_sha256"]
        ):
            raise ValueError("checkpoint data/tokenizer drift")
        if not isinstance(loaded.device, torch.device) or loaded.device.type != "cuda":
            raise ValueError("study needs the live ROCm backend")
        torch.cuda.reset_peak_memory_stats(loaded.device)
        for prompt, policy, rng in cells(split, pre, selected):
            started = time.perf_counter()
            try:
                text, ids = generate_with_token_ids(
                    loaded.model,
                    loaded.tokenizer,
                    prompt["text"],
                    loaded.config.model.max_seq_len,
                    48,
                    loaded.device,
                    temperature=policy["temperature"],
                    top_k=policy["top_k"],
                    seed=0 if rng is None else rng,
                    strict_context=True,
                )
                torch.cuda.synchronize(loaded.device)
                if not text.startswith(prompt["text"]):
                    raise ValueError("generated text does not preserve prompt prefix")
                completion = text[len(prompt["text"]) :]
                result = {
                    "status": "ok",
                    "text": text,
                    "completion": completion,
                    "generated_token_ids": ids,
                    "diagnostics": diagnostics(
                        completion, ids, loaded.tokenizer, prompt.get("anchors", {})
                    ),
                }
            except Exception as error:  # noqa: BLE001 - retain failed cell as evidence.
                result = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            row = {
                "type": "cell",
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt["text"],
                "anchors": prompt.get("anchors", {}),
                "model": checkpoint["model"],
                "training_seed": checkpoint["seed"],
                "run_id": checkpoint["run_id"],
                "checkpoint_digest": checkpoint["digest"],
                "checkpoint_step": 16384,
                "decoder": policy,
                "rng_seed": rng,
                "elapsed_seconds": time.perf_counter() - started,
                **result,
            }
            handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if result["status"] != "ok":
                raise RuntimeError(
                    f"generation failed; retained {path} at "
                    f"{prompt['id']} / {policy['name']} / {rng}: {result['error']}"
                )
        print(
            f"{path}: {len(read_rows(path)) - 1} cells, "
            f"peak_allocated={torch.cuda.max_memory_allocated(loaded.device)} "
            f"peak_reserved={torch.cuda.max_memory_reserved(loaded.device)}",
            flush=True,
        )


def choose(pre: dict) -> None:
    development_digests(pre)
    path = WORK / "selection.json"
    scores = {}
    for policy in ("temperature_0_8", "top_k_40"):
        rows = (
            row
            for cp in pre["checkpoints"]
            for row in read_rows(output_path("development", cp))[1:]
            if row["decoder"]["name"] == policy
        )
        data = list(rows)
        assert len(data) == 22 * 2 * 6
        count = sum(row["diagnostics"]["generated_token_count"] for row in data)
        failures = sum(
            row["diagnostics"]["empty"] or row["diagnostics"]["special_token_count"] > 0
            for row in data
        )
        contradictions = sum(
            row["diagnostics"]["explicit_anchors"]["explicit_contradiction"]
            for row in data
        )
        tri = sum(row["diagnostics"]["repeated_trigram_excess"] for row in data)
        bi = sum(row["diagnostics"]["repeated_bigram_excess"] for row in data)
        scores[policy] = {
            "empty_or_pathological": failures,
            "explicit_contradictions": contradictions,
            "emitted_tokens": count,
            "repeated_trigram_excess": tri,
            "repeated_bigram_excess": bi,
            "repeated_trigram_rate": tri / count if count else None,
            "repeated_bigram_rate": bi / count if count else None,
        }

    def rank(name):
        s = scores[name]
        return (
            s["empty_or_pathological"],
            s["explicit_contradictions"],
            s["repeated_trigram_rate"],
            s["repeated_bigram_rate"],
            0 if name == "top_k_40" else 1,
        )

    selected = min(scores, key=rank)
    receipt = {
        "format": "dense_lm_decoder_selection_v1",
        "preregistration_sha256": digest(PRE),
        "development_sha256": development_digests(pre),
        "scores": scores,
        "selected": selected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"selected {selected}; receipt {path}; sha256={digest(path)}", flush=True)


def complete_test(pre: dict) -> dict:
    chosen = selection(pre)["selected"]
    outputs = {}
    expected = {
        (row["id"], policy["name"], rng)
        for row, policy, rng in cells("test", pre, chosen)
    }
    for cp in pre["checkpoints"]:
        path = output_path("test", cp)
        rows = read_rows(path)
        if len(rows) != 1 + len(expected) or any(
            row.get("status") != "ok" for row in rows[1:]
        ):
            raise ValueError(f"incomplete test file: {path}")
        actual = {
            (row["prompt_id"], row["decoder"]["name"], row["rng_seed"])
            for row in rows[1:]
        }
        if actual != expected:
            raise ValueError(f"wrong test cells: {path}")
        outputs[path.name] = digest(path)
    return outputs


def review_bundle(pre: dict) -> None:
    sealed = complete_test(pre)
    selected = selection(pre)["selected"]
    rows = {
        (cp["model"], cp["seed"]): {
            (r["prompt_id"], r["decoder"]["name"], r["rng_seed"]): r
            for r in read_rows(output_path("test", cp))[1:]
        }
        for cp in pre["checkpoints"]
    }
    prompts = {}
    for item in prompt_rows("test", pre):
        prompts.setdefault(item["category"], []).append(item)
    chosen = [
        r
        for category in sorted(prompts)
        for r in sorted(prompts[category], key=lambda x: x["id"])[:2]
    ]
    bundle, key = [], []
    for item, training_seed in itertools.product(chosen, (42, 17, 73)):
        pairs = [
            (
                "model",
                rows[("30m", training_seed)][(item["id"], selected, 11)],
                rows[("50m", training_seed)][(item["id"], selected, 11)],
            )
        ]
        for model in ("30m", "50m"):
            pairs.append(
                (
                    "decoder",
                    rows[(model, training_seed)][
                        (item["id"], "regression_decoder", None)
                    ],
                    rows[(model, training_seed)][(item["id"], selected, 11)],
                )
            )
        for kind, left, right in pairs:
            code = hashlib.sha256(
                f"{digest(PRE)}:{item['id']}:{training_seed}:{kind}:{left['model']}:{right['model']}".encode()
            ).hexdigest()
            if int(code[-1], 16) % 2:
                left, right = right, left
            pair_id = code[:20]
            bundle.append(
                {
                    "pair_id": pair_id,
                    "prompt": item["text"],
                    "category": item["category"],
                    "a": left["text"],
                    "b": right["text"],
                    "dimensions": DIMENSIONS,
                    "allowed_votes": ("A", "B", "tie", "uncertain"),
                    "judgments": [],
                }
            )
            key.append(
                {
                    "pair_id": pair_id,
                    "comparison": kind,
                    "a": {
                        k: left[k]
                        for k in (
                            "model",
                            "training_seed",
                            "decoder",
                            "rng_seed",
                            "checkpoint_digest",
                        )
                    },
                    "b": {
                        k: right[k]
                        for k in (
                            "model",
                            "training_seed",
                            "decoder",
                            "rng_seed",
                            "checkpoint_digest",
                        )
                    },
                }
            )
    folder = WORK / "review"
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("blind.json", bundle),
        ("key.json", key),
        ("test_sha256.json", sealed),
    ):
        with (folder / name).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    print(
        f"blinded pairs={len(bundle)}, key held separately; test files={sealed}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "select", "test", "review"))
    parser.add_argument("--model", choices=("30m", "50m"))
    parser.add_argument("--seed", type=int, choices=(42, 17, 73))
    args = parser.parse_args()
    pre = load_protocol()
    if args.phase in ("development", "test"):
        if args.model is None or args.seed is None:
            parser.error("development/test require --model and --seed")
        (cp,) = [
            x
            for x in pre["checkpoints"]
            if x["model"] == args.model and x["seed"] == args.seed
        ]
        selected = selection(pre)["selected"] if args.phase == "test" else None
        run_checkpoint(args.phase, pre, cp, selected)
    elif args.phase == "select":
        choose(pre)
    else:
        review_bundle(pre)


if __name__ == "__main__":
    main()
