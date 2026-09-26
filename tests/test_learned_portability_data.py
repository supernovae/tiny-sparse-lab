"""Contracts for the learned Engram portability data publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sparselab.data.byte_hash import table_address
from sparselab.data.learned_portability import (
    _query_prompt,
    materialize_learned_portability_data,
)
from sparselab.data.tokenizer import load_tokenizer


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def test_learned_data_replays_and_separates_labels(tmp_path: Path) -> None:
    root = tmp_path / "learned"
    manifest_path = materialize_learned_portability_data(root, fact_count=128)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format"] == "sparselab-learned-portability-data"
    assert manifest["version"] == 1
    assert manifest["ownership"]["calibration"]["count"] == 32
    assert manifest["ownership"]["source_monitor"]["count"] == 32
    assert manifest["ownership"]["held_out"]["count"] == 64
    assert manifest["addressing"]["table_size"] == 65_521
    assert manifest["addressing"]["ngram_size"] == 32
    assert manifest["tokenizer"]["vocab_size"] == 260
    assert manifest["tokenizer"]["merges"] == []
    assert materialize_learned_portability_data(root, fact_count=128) == manifest_path

    facts = _rows(root / "facts.jsonl")
    queries = _rows(root / "queries.jsonl")
    scorer = _rows(root / "scorer.jsonl")
    for split in ("preparation_train.jsonl", "preparation_validation.jsonl"):
        prepared = 0
        for row in _rows(root / split):
            messages = row["messages"]
            prompt = messages[0]["content"]
            if "Copy the supplied symbol" not in prompt:
                continue
            symbol = messages[1]["content"]
            assert f"Copy the supplied symbol {symbol}." in prompt
            assert "{symbol}" not in prompt
            prepared += 1
        assert prepared == (512 if split == "preparation_train.jsonl" else 128)
    tokenizer = load_tokenizer(root / "tokenizer" / "tokenizer.json")
    prompt = _query_prompt(
        tokenizer, "preparation_copy", "k123456789abc.0000", answer="A"
    )
    assert "Copy the supplied symbol A." in prompt
    with pytest.raises(ValueError, match="requires a supplied symbol"):
        _query_prompt(tokenizer, "preparation_copy", "k123456789abc.0000")
    assert len({row["fact_id"] for row in facts}) == 128
    assert len({row["target_row"] for row in facts}) == 128
    assert all(row["target_row"] != 0 for row in facts)
    assert all("expected_symbol" not in row for row in queries)
    assert {row["case_id"] for row in queries} == {row["case_id"] for row in scorer}
    assert all(set(row) == {"case_id", "fact_id", "expected_symbol"} for row in scorer)
    assert all(
        row["answer_prefix"] == row["prompt"] + " "
        and table_address(row["answer_prefix"].encode("utf-8")[-32:], 65_521)
        == row["address"]
        for row in queries
    )
    assert all(
        row["answer_prefix"].encode("utf-8")[-32:]
        == b"|"
        + next(
            fact["key"] for fact in facts if fact["fact_id"] == row["fact_id"]
        ).encode("ascii")
        + b"\n\nAssistant: "
        for row in queries
    )
    assert (
        manifest["sha256"]
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in manifest.items() if key != "sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert all(
        _descriptor(root / str(entry["path"]), root) == entry
        for entry in manifest["files"]
    )


def test_packing_retains_exact_factual_blocks_and_withholds_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "learned"
    materialize_learned_portability_data(root, fact_count=128)
    facts = _rows(root / "facts.jsonl")
    blocks = _rows(root / "block_map.jsonl")
    target_by_fact = {row["fact_id"]: row["target_row"] for row in facts}

    source = [row for row in blocks if row["split"] == "source_train"]
    calibration = [row for row in blocks if row["split"] == "adapter_calibration"]
    assert len(source) == 256
    assert len(calibration) == 64
    assert all(row["token_count_before_eos"] == 127 for row in blocks)
    assert all(
        row["address"] == target_by_fact[row["fact_id"]] for row in source + calibration
    )
    assert {row["fact_id"] for row in calibration} == {
        row["fact_id"] for row in facts if row["ownership"] == "calibration"
    }
    assert not {row["fact_id"] for row in calibration} & {
        row["fact_id"] for row in facts if row["ownership"] == "held_out"
    }
    protected_rows = {
        int(row["target_row"])
        for row in facts
        if row["ownership"] in {"source_monitor", "held_out"}
    }
    prior_accesses = np.concatenate(
        [
            np.load(root / f"{split}_byte_addresses.npy", allow_pickle=False).reshape(
                -1
            )
            for split in (
                "preparation_train",
                "preparation_validation",
                "adapter_calibration",
            )
        ]
    )
    assert protected_rows.isdisjoint(set(prior_accesses.tolist()))
    for split, expected in (("source_train", 256), ("adapter_calibration", 64)):
        values = np.load(root / f"{split}_tokens.npy", allow_pickle=False)
        supervision = np.load(root / f"{split}_supervision.npy", allow_pickle=False)
        addresses = np.load(root / f"{split}_byte_addresses.npy", allow_pickle=False)
        assert len(values) == (expected + 1) * 128
        assert len(supervision) == len(values) == len(addresses)
        assert (len(values) - 1) // 128 == expected
        assert all(
            supervision[block * 128 + 1 : (block + 1) * 128 + 1].sum() == 2
            for block in range(expected)
        )
