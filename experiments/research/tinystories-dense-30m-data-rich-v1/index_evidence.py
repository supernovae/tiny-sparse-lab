"""Freeze small pointers and measured summaries for the ignored evaluation outputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
WORK = ROOT / "sparselab-work/experiments/tinystories-dense-30m-data-rich-v1"
RUN = WORK / "runs/tinystories-dense-30m-data-rich-v1-seed42"


def artifact(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "relative_path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    outputs: dict[str, dict] = {}
    for kind in ("control", "heldout"):
        for path in sorted((WORK / "evaluation" / kind).glob("*.jsonl")):
            data = rows(path)
            expected = 256 if kind == "control" else 8000
            if len(data) != expected + 2 or data[0]["type"] != "header":
                raise ValueError(f"likelihood record incomplete: {path}")
            if any(r["status"] != "ok" for r in data[1:-1]):
                raise ValueError(f"likelihood failures: {path}")
            if data[-1]["successful_stories"] != expected:
                raise ValueError(f"likelihood summary incomplete: {path}")
            outputs[f"{kind}/{path.stem}"] = {
                "artifact": artifact(path),
                "summary": data[-1],
            }
    for path in sorted((WORK / "evaluation/generation").glob("*.jsonl")):
        data = rows(path)
        if data[0]["type"] != "header" or any(r["status"] != "ok" for r in data[1:]):
            raise ValueError(f"generation record incomplete: {path}")
        kind = path.stem.rsplit("-", 1)[0]
        prompt_count = {
            "regression": 6,
            "development": 22,
            "test": 55,
            "long-development": 4,
            "long-test": 4,
        }[kind]
        policies = Counter(r["policy"] for r in data[1:])
        expected_policies = (
            {"greedy": prompt_count}
            if kind == "regression"
            else {"greedy": prompt_count, "temperature_0_8": 2 * prompt_count}
        )
        cells = {(r["prompt_id"], r["policy"], r["rng_seed"]) for r in data[1:]}
        if policies != expected_policies or len(cells) != len(data) - 1:
            raise ValueError(f"generation cells incomplete or duplicated: {path}")
        counts = Counter((r["policy"], r["short_subset"]) for r in data[1:])
        outputs[f"generation/{path.stem}"] = {
            "artifact": artifact(path),
            "cells": len(data) - 1,
            "policy_and_short_counts": [
                {"policy": key[0], "short_subset": key[1], "count": value}
                for key, value in sorted(counts.items())
            ],
            "early_eos": sum(r["early_eos"] for r in data[1:]),
            "explicit_contradictions": sum(
                r.get("diagnostics", {})
                .get("explicit_anchors", {})
                .get("explicit_contradiction", False)
                for r in data[1:]
            ),
        }
    expected = {
        "control/30m",
        "control/50m",
        "control/new",
        "heldout/new",
        *(
            f"generation/{kind}-{model}"
            for kind in ("regression", "development", "test")
            for model in ("30m", "50m", "new")
        ),
        "generation/long-development-new",
        "generation/long-test-new",
    }
    if set(outputs) != expected:
        raise ValueError(f"missing panels: {sorted(expected - set(outputs))}")
    triage = list((RUN / "post-train-triage").glob("*.json"))
    if len(triage) != 1:
        raise ValueError("expected one frozen endpoint triage report")
    report = json.loads(triage[0].read_text())
    if report["core"]["integrity"]["status"] != "PASS":
        raise ValueError("endpoint triage integrity failed")
    evidence = {
        "format": "tinystories_dense_30m_data_rich_evidence_v1",
        "launch": artifact(WORK / "staging/launch-receipt.json"),
        "run_manifest": artifact(RUN / "manifest.json"),
        "endpoint": json.loads((STUDY / "endpoint-receipt.json").read_text()),
        "triage": {
            "artifact": artifact(triage[0]),
            "integrity": report["core"]["integrity"],
            "recommendations": report["recommendations"],
        },
        "outputs": outputs,
    }
    with (STUDY / "evidence.json").open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"indexed {len(outputs)} complete outputs", flush=True)


if __name__ == "__main__":
    main()
