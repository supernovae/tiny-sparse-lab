"""Independent read-only tokenizer probe contract and scoring tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from sparselab.corpus.tokenizer_bakeoff import GROUPS
from sparselab.corpus.tokenizer_probes import load_probe_suite, probe_tokenizer

_SUITE = (
    Path(__file__).resolve().parents[1]
    / "experiments/research/devmind-pretrain-v1/tokenizer-probes.json"
)


def _suite_copy(tmp_path: Path, suite: dict) -> Path:
    path = tmp_path / "probe-suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    return path


def _tokenizer(tmp_path: Path) -> Path:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(
        ["independent fitting phrase", "symbols 123 + :", "another tiny example"],
        trainer=BpeTrainer(vocab_size=280, special_tokens=["<unk>"]),
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def test_probe_scores_real_tokenizer_deterministically_without_writing(tmp_path: Path) -> None:
    suite = load_probe_suite(_SUITE)
    assert {sample["group"] for sample in suite["samples"]} == set(GROUPS)
    assert all(sample["provenance"] == "synthetic_syntax" for sample in suite["samples"] if sample["group"] == "logs")
    tokenizer = _tokenizer(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    first = probe_tokenizer(tokenizer, _SUITE)
    assert first == probe_tokenizer(tokenizer, _SUITE)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert first["tokenizer_sha256"] == hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    assert first["suite_sha256"] == hashlib.sha256(_SUITE.read_bytes()).hexdigest()
    loaded = Tokenizer.from_file(str(tokenizer))
    for group in GROUPS:
        texts = [item["text"] for item in suite["samples"] if item["group"] == group]
        row = first["per_kind"][group]
        assert row["samples"] == len(texts)
        assert row["utf8_bytes"] == sum(len(text.encode("utf-8")) for text in texts)
        assert row["tokens"] == sum(len(loaded.encode(text).ids) for text in texts)
        assert row["bytes_per_token"] == row["utf8_bytes"] / row["tokens"]


def test_probe_rejects_missing_tokenizer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tokenizer artifact missing"):
        probe_tokenizer(tmp_path / "missing.json", _SUITE)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda suite: suite.update(schema_version=True), "schema_version"),
        (lambda suite: suite.update(schema_version=2), "schema_version"),
        (lambda suite: suite.update(suite_id=""), "suite_id"),
        (
            lambda suite: suite.update(unexpected="x"),
            "schema_version, suite_id, samples",
        ),
        (
            lambda suite: suite["samples"].append(dict(suite["samples"][0])),
            "duplicate probe sample id",
        ),
        (lambda suite: suite["samples"][0].update(group="fiction"), "invalid probe group"),
        (lambda suite: suite["samples"][0].update(text=" "), "nonempty text"),
        (
            lambda suite: suite["samples"][0].update(provenance="source_document"),
            "invalid probe provenance",
        ),
        (
            lambda suite: suite["samples"][0].update(provenance=[]),
            "invalid probe provenance",
        ),
        (
            lambda suite: suite["samples"][0].update(train_source_id="train-1"),
            "source IDs are forbidden",
        ),
        (
            lambda suite: suite.update(
                samples=[s for s in suite["samples"] if s["group"] != "logs"]
            ),
            "missing groups",
        ),
        (
            lambda suite: suite["samples"][-1].update(
                provenance="independently_authored"
            ),
            "synthetic_syntax",
        ),
        (
            lambda suite: suite["samples"][0].update(text=chr(0xD800)),
            "valid UTF-8 text",
        ),
    ],
)
def test_probe_suite_validation(tmp_path: Path, mutation, expected: str) -> None:
    suite = json.loads(_SUITE.read_text(encoding="utf-8"))
    mutation(suite)
    with pytest.raises(ValueError, match=expected):
        load_probe_suite(_suite_copy(tmp_path, suite))


def test_probe_rejects_non_object_and_malformed_json(tmp_path: Path) -> None:
    for payload in ('[]', '{broken', '"text"'):
        path = tmp_path / "invalid.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError):
            load_probe_suite(path)
