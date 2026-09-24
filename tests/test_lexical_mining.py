from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from sparselab.data import lexical_mining
from sparselab.data.lexical_mining import analyze_training_corpus
from sparselab.model.memory import TokenNgramMemory


def _write_conversations(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                }
            )
            + "\n"
            for user, assistant in pairs
        ),
        encoding="utf-8",
    )


def _tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "User:": 1,
                "Assistant:": 2,
                "a": 3,
                "b": 4,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(path))


def _ngram_keys(ids: list[int], order: int) -> set[tuple[int, ...]]:
    return {
        tuple(
            ids[position - offset] if position >= offset else 0
            for offset in range(order)
        )
        for position in range(len(ids))
    }


def _address_for_key(key: tuple[int, ...], table_size: int, head: int) -> int:
    address = head + 1
    for token_id in key:
        address = (address * (257 + head * 2) + token_id) % table_size
    return address


def test_analyze_training_corpus_reports_train_only_counts_entropy_and_addresses(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a", "b"), ("b", "a")])
    _tokenizer(tokenizer)

    analysis = analyze_training_corpus(
        train,
        tokenizer,
        table_size=17,
        memory_dim=3,
        ngram_orders=(1, 2),
        hash_heads=2,
    )

    first, second = [1, 3, 2, 4], [1, 4, 2, 3]
    assert analysis["documents"] == 2
    assert analysis["tokens"] == 8
    assert analysis["token_frequency"] == [
        {"token_id": token, "count": 2} for token in range(1, 5)
    ]
    assert analysis["document_frequency"] == [
        {"token_id": token, "count": 2} for token in range(1, 5)
    ]
    assert analysis["empirical_unigram_entropy_bits_per_token"] == pytest.approx(2.0)
    assert analysis["training_corpus"] == {
        "sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "bytes": train.stat().st_size,
    }
    assert analysis["tokenizer"] == {
        "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    }
    assert analysis["addressing"] == {
        "algorithm": "token-ngram-recurrence-v1",
        "recurrence": "address=(address*(257+hash_head*2)+shifted_token_id)%table_size",
        "initial_address": "hash_head+1",
        "document_boundary": "zero_padded",
        "collision_key": "distinct zero-padded token n-gram keys across corpus",
        "collision_definition": (
            "distinct keys minus occupied buckets; repeated lookups are separate"
        ),
        "distinct_key_limits": {
            "max_keys": lexical_mining.MAX_UNIQUE_NGRAMS,
            "max_key_components": lexical_mining.MAX_UNIQUE_NGRAM_COMPONENTS,
        },
        "table_size": 17,
        "ngram_orders": [1, 2],
        "hash_heads": 2,
    }
    for stream in analysis["streams"]:
        assert isinstance(stream, dict)
        order, head = stream["order"], stream["hash_head"]
        ngram_keys = _ngram_keys(first, order) | _ngram_keys(second, order)
        addresses = {_address_for_key(key, 17, head) for key in ngram_keys}
        collisions = len(ngram_keys) - len(addresses)
        repeated_accesses = 8 - len(ngram_keys)
        assert stream == {
            "order": order,
            "hash_head": head,
            "lookup_count": 8,
            "distinct_ngram_count": len(ngram_keys),
            "repeated_accesses": repeated_accesses,
            "unique_occupied_buckets": len(addresses),
            "collisions": collisions,
            "collision_rate": collisions / len(ngram_keys),
            "repeated_access_rate": repeated_accesses / 8,
            "utilization": len(addresses) / 17,
        }
    assert analysis["storage"] == {
        "table_fp32_bytes": 17 * 3 * 4 * 4,
        "table_count": 4,
        "memory_dim": 3,
        "excludes": ["output_projection_weights", "gate_weights"],
    }


def test_analyze_training_corpus_has_deterministic_content_identity_and_reads_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    held_out = tmp_path / "validation.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a", "b")])
    held_out.write_text("not valid JSON\n", encoding="utf-8")
    _tokenizer(tokenizer)
    original = lexical_mining.iter_conversations
    opened: list[Path] = []

    def train_only(path: Path):
        opened.append(path)
        assert path == train
        yield from original(path)

    monkeypatch.setattr(lexical_mining, "iter_conversations", train_only)
    first = analyze_training_corpus(
        train, tokenizer, table_size=19, memory_dim=2, ngram_orders=[2]
    )
    second = analyze_training_corpus(
        train, tokenizer, table_size=19, memory_dim=2, ngram_orders=[2]
    )

    assert opened == [train, train]
    assert held_out.name not in json.dumps(first)
    assert first == second
    body = dict(first)
    identity = body.pop("analysis_sha256")
    assert identity == hashlib.sha256(lexical_mining.canonical_json(body)).hexdigest()


@pytest.mark.parametrize(
    ("records", "kwargs", "message"),
    [
        ([], {}, "is empty"),
        ([("a", "b")], {"table_size": 0}, "table_size"),
        ([("a", "b")], {"ngram_orders": (2, 2)}, "duplicates"),
        ([("a", "b")], {"ngram_orders": ()}, "nonempty"),
    ],
)
def test_analyze_training_corpus_rejects_empty_or_invalid_configuration(
    tmp_path: Path,
    records: list[tuple[str, str]],
    kwargs: dict[str, object],
    message: str,
) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, records)
    _tokenizer(tokenizer)
    arguments: dict[str, object] = {
        "table_size": 17,
        "memory_dim": 2,
        "ngram_orders": (2,),
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        analyze_training_corpus(train, tokenizer, **arguments)  # type: ignore[arg-type]


def test_analyze_training_corpus_entropy_is_empirical_unigram_entropy(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a", "a"), ("a", "b")])
    _tokenizer(tokenizer)

    analysis = analyze_training_corpus(
        train, tokenizer, table_size=31, memory_dim=1, ngram_orders=(2,)
    )

    counts = [2, 2, 3, 1]
    expected = -sum((count / 8) * math.log2(count / 8) for count in counts)
    assert analysis["empirical_unigram_entropy_bits_per_token"] == pytest.approx(
        expected
    )


def test_address_streams_match_token_ngram_memory_exactly(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a b", "a"), ("b", "a b")])
    _tokenizer(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    memory = TokenNgramMemory(
        hidden_dim=4,
        table_size=17,
        ngram_size=2,
        value_dim=3,
        ngram_orders=(1, 2),
        hash_heads=2,
    )

    expected: dict[tuple[int, int], list[int]] = {
        (order, head): [] for order in (1, 2) for head in range(2)
    }
    distinct_keys = {order: set() for order in (1, 2)}
    for document in lexical_mining.iter_conversations(train):
        tokens = tokenizer.encode(document, add_special_tokens=False).ids
        input_ids = torch.tensor([tokens], dtype=torch.long)
        for order, keys in distinct_keys.items():
            keys.update(_ngram_keys(tokens, order))
        for (order, head), addresses in expected.items():
            addresses.extend(
                memory.addresses(input_ids, order=order, seed=head).flatten().tolist()
            )

    analysis = analyze_training_corpus(
        train,
        tokenizer_path,
        table_size=17,
        memory_dim=3,
        ngram_orders=(1, 2),
        hash_heads=2,
    )
    for stream in analysis["streams"]:
        addresses = expected[stream["order"], stream["hash_head"]]
        unique = len(set(addresses))
        distinct_count = len(distinct_keys[stream["order"]])
        collisions = distinct_count - unique
        repeated_accesses = len(addresses) - distinct_count
        assert stream["lookup_count"] == len(addresses)
        assert stream["distinct_ngram_count"] == distinct_count
        assert stream["repeated_accesses"] == repeated_accesses
        assert stream["unique_occupied_buckets"] == unique
        assert stream["collisions"] == collisions
        assert stream["collision_rate"] == collisions / distinct_count
        assert stream["repeated_access_rate"] == repeated_accesses / len(addresses)


def test_address_bitmap_limit_rejects_unbounded_table_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collision-bitmap memory limit"):
        analyze_training_corpus(
            tmp_path / "unused-train.jsonl",
            tmp_path / "unused-tokenizer.json",
            table_size=lexical_mining.MAX_ADDRESS_BITMAP_BITS + 1,
            memory_dim=1,
            ngram_orders=(1,),
        )


def test_exact_unique_ngram_tracking_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a", "b")])
    _tokenizer(tokenizer)
    monkeypatch.setattr(lexical_mining, "MAX_UNIQUE_NGRAMS", 1)

    with pytest.raises(ValueError, match="unique n-gram limit"):
        analyze_training_corpus(
            train, tokenizer, table_size=17, memory_dim=1, ngram_orders=(1,)
        )
