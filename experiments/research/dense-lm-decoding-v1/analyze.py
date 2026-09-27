"""Verify, archive, and describe the sealed decoding study without generating text."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
WORK = ROOT / "sparselab-work/experiments/dense-lm-decoding-v1"
OUT = STUDY / "evidence"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def expected_cells(
    prompts: list[dict], policies: list[dict], seeds: list[int]
) -> set[tuple]:
    return {
        (row["id"], policy["name"], rng)
        for row in prompts
        for policy in policies
        for rng in ([None] if policy["temperature"] == 0 else seeds)
    }


def metric(rows: list[dict]) -> dict:
    metrics = [row["diagnostics"] for row in rows]
    total = sum(m["generated_token_count"] for m in metrics)
    return {
        "cells": len(rows),
        "emitted_tokens": total,
        "elapsed_generation_seconds": sum(row["elapsed_seconds"] for row in rows),
        "premature_empty": sum(m["empty"] for m in metrics),
        "early_eos": sum(m["early_stop"] for m in metrics),
        "special_token_cells": sum(m["special_token_count"] > 0 for m in metrics),
        "exact_sentence_repetition_excess": sum(
            m["exact_sentence_repetition_excess"] for m in metrics
        ),
        "explicit_anchor_contradictions": sum(
            m["explicit_anchors"]["explicit_contradiction"] for m in metrics
        ),
        **{
            key: {
                "excess": sum(m[key] for m in metrics),
                "per_100_emitted_tokens": 100 * sum(m[key] for m in metrics) / total
                if total
                else None,
            }
            for key in (
                "repeated_token_excess",
                "repeated_bigram_excess",
                "repeated_trigram_excess",
            )
        },
    }


def comparisons(test: dict[tuple, dict], selected: str) -> dict:
    pairs = defaultdict(list)
    for seed in (42, 17, 73):
        for model in ("30m", "50m"):
            for prompt_id in sorted(
                {p for m, s, p, _d, _r in test if m == model and s == seed}
            ):
                greedy = test[model, seed, prompt_id, "regression_decoder", None]
                for rng in (11, 29):
                    sampled = test[model, seed, prompt_id, selected, rng]
                    pairs[f"decoder_{model}"].append((greedy, sampled))
        for prompt_id in sorted(
            {p for m, s, p, _d, _r in test if m == "30m" and s == seed}
        ):
            for rng in (11, 29):
                left = test["30m", seed, prompt_id, selected, rng]
                right = test["50m", seed, prompt_id, selected, rng]
                pairs["model_under_selected"].append((left, right))
            left = test["30m", seed, prompt_id, "regression_decoder", None]
            right = test["50m", seed, prompt_id, "regression_decoder", None]
            pairs["model_under_greedy"].append((left, right))
    results = {}
    for name, entries in pairs.items():
        compared = {}
        for key in (
            "repeated_token_excess",
            "repeated_bigram_excess",
            "repeated_trigram_excess",
            "exact_sentence_repetition_excess",
            "generated_token_count",
        ):
            deltas = [
                right["diagnostics"][key] - left["diagnostics"][key]
                for left, right in entries
            ]
            compared[key] = {
                "right_lower": sum(n < 0 for n in deltas),
                "right_higher": sum(n > 0 for n in deltas),
                "ties": sum(n == 0 for n in deltas),
                "sum_delta": sum(deltas),
            }
        for key, field in (
            ("explicit_anchor_contradiction", "explicit_contradiction"),
        ):
            compared[key] = {
                "left_only": sum(
                    a["diagnostics"]["explicit_anchors"][field]
                    and not b["diagnostics"]["explicit_anchors"][field]
                    for a, b in entries
                ),
                "right_only": sum(
                    b["diagnostics"]["explicit_anchors"][field]
                    and not a["diagnostics"]["explicit_anchors"][field]
                    for a, b in entries
                ),
                "both": sum(
                    a["diagnostics"]["explicit_anchors"][field]
                    and b["diagnostics"]["explicit_anchors"][field]
                    for a, b in entries
                ),
            }
        results[name] = {"pairs": len(entries), "metrics": compared}
    return results


def main() -> None:
    pre = json.loads((STUDY / "preregistration.json").read_text())
    for name, expected in pre["frozen_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen input changed: {name}")
    receipt = json.loads((WORK / "selection.json").read_text())
    if receipt["preregistration_sha256"] != digest(STUDY / "preregistration.json"):
        raise ValueError("selection registration mismatch")
    selected = receipt["selected"]
    if selected not in ("top_k_40", "temperature_0_8"):
        raise ValueError("unrecognized selection")
    evidence = {
        "format": "dense_lm_decoding_evidence_v1",
        "preregistration_sha256": digest(STUDY / "preregistration.json"),
        "regression_panel_sha256": pre["frozen_sha256"][
            "data/dense_lm_v1_prompts.json"
        ],
        "regression_checkpoint_observations_sha256": pre["frozen_sha256"][
            "experiments/research/dense-lm-scale-v1/evidence.json"
        ],
        "selection_sha256": digest(WORK / "selection.json"),
        "selected_non_greedy": selected,
        "human_review": "UNAVAILABLE",
        "source_test_files": {},
        "source_development_files": {},
        "split": {},
    }
    selected_policies = [
        p for p in pre["policies"] if p["name"] in ("regression_decoder", selected)
    ]
    test_index = {}
    for split in ("development", "test"):
        prompts = json.loads((STUDY / f"{split}.json").read_text())["prompts"]
        policies = pre["policies"] if split == "development" else selected_policies
        expected = expected_cells(prompts, policies, pre["sample_seeds"])
        by_arm, by_category = defaultdict(list), defaultdict(list)
        for cp in pre["checkpoints"]:
            filename = f"{cp['model']}-seed{cp['seed']}.jsonl"
            source = WORK / split / filename
            rows = read_jsonl(source)
            if (
                rows[0]["checkpoint"] != cp
                or rows[0]["preregistration_sha256"]
                != evidence["preregistration_sha256"]
            ):
                raise ValueError(f"header checkpoint identity mismatch: {source}")
            observed = [
                (row["prompt_id"], row["decoder"]["name"], row["rng_seed"])
                for row in rows[1:]
            ]
            if len(observed) != len(expected) or set(observed) != expected:
                raise ValueError(f"missing or duplicate cells: {source}")
            if any(
                row["status"] != "ok" or row["checkpoint_digest"] != cp["digest"]
                for row in rows[1:]
            ):
                raise ValueError(f"invalid cell: {source}")
            if any(
                row["diagnostics"]["generated_token_count"]
                != len(row["generated_token_ids"])
                or row["text"] != row["prompt"] + row["completion"]
                for row in rows[1:]
            ):
                raise ValueError(f"invalid text/IDs: {source}")
            mapping = evidence[f"source_{split}_files"]
            mapping[filename] = digest(source)
            if (
                split == "development"
                and receipt["development_sha256"][filename] != mapping[filename]
            ):
                raise ValueError(f"selection input changed: {source}")
            for row in rows[1:]:
                key = (
                    cp["model"],
                    cp["seed"],
                    row["prompt_id"],
                    row["decoder"]["name"],
                    row["rng_seed"],
                )
                if split == "test":
                    test_index[key] = row
                by_arm[(cp["model"], cp["seed"], row["decoder"]["name"])].append(row)
                by_category[
                    (cp["model"], row["decoder"]["name"], row["category"])
                ].append(row)
        evidence["split"][split] = {
            "expected_cells": len(expected) * len(pre["checkpoints"]),
            "by_arm": {
                "/".join(map(str, key)): metric(rows)
                for key, rows in sorted(by_arm.items())
            },
            "by_category": {
                "/".join(key): metric(rows) for key, rows in sorted(by_category.items())
            },
        }
    evidence["test_matched_pairs"] = comparisons(test_index, selected)
    test_digests = json.loads((WORK / "review/test_sha256.json").read_text())
    if test_digests != evidence["source_test_files"]:
        raise ValueError("blinded review references different test bytes")
    blind = json.loads((WORK / "review/blind.json").read_text())
    key = json.loads((WORK / "review/key.json").read_text())
    if (
        len(blind) != 198
        or len(key) != 198
        or {row["pair_id"] for row in blind} != {row["pair_id"] for row in key}
    ):
        raise ValueError("incomplete blinded pairs")
    if any(row["judgments"] for row in blind):
        raise ValueError("expected no subjective review votes")
    evidence["blind_review"] = {
        "pair_count": len(blind),
        "blind_sha256": digest(WORK / "review/blind.json"),
        "key_sha256": digest(WORK / "review/key.json"),
        "votes": 0,
        "status": "UNAVAILABLE",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("development", "test"):
        dest_dir = OUT / split
        dest_dir.mkdir(exist_ok=True)
        for filename, expected in evidence[f"source_{split}_files"].items():
            dest = dest_dir / filename
            if dest.exists():
                raise FileExistsError(dest)
            shutil.copyfile(WORK / split / filename, dest)
            if digest(dest) != expected:
                raise ValueError(f"archive changed: {dest}")
    for source, dest in (
        (WORK / "selection.json", OUT / "selection.json"),
        (WORK / "review/blind.json", OUT / "review-blind.json"),
    ):
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copyfile(source, dest)
    summary = OUT / "summary.json"
    if summary.exists():
        raise FileExistsError(summary)
    summary.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(
        f"archived {evidence['split']['development']['expected_cells']} development and {evidence['split']['test']['expected_cells']} test cells; summary_sha256={digest(summary)}; blind_sha256={evidence['blind_review']['blind_sha256']}"
    )


if __name__ == "__main__":
    main()
