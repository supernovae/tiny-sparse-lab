"""Corpus-only lexical statistics for token n-gram memory planning.

This module intentionally consumes only one explicitly supplied local-chat training
JSONL file.  It does not construct a model or inspect validation, test, or card
inputs; its address arithmetic mirrors :class:`TokenNgramMemory` directly.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Final

from sparselab.data.conversations import iter_conversations
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.manifest import canonical_json, sha256_file

MAX_INPUT_BYTES: Final = 256 * 1024 * 1024
MAX_DOCUMENTS: Final = 1_000_000
MAX_TOKENS: Final = 100_000_000
MAX_TOKENS_PER_DOCUMENT: Final = 1_000_000
MAX_TABLE_SIZE: Final = 1_000_000_000
MAX_MEMORY_DIM: Final = 1_000_000
MAX_NGRAM_ORDER: Final = 64
MAX_HASH_HEADS: Final = 64
MAX_ADDRESS_STREAMS: Final = 64
MAX_ADDRESS_BITMAP_BITS: Final = 512 * 1024 * 1024
MAX_HASH_OPERATIONS: Final = 50_000_000
MAX_UNIQUE_NGRAMS: Final = 1_000_000
MAX_UNIQUE_NGRAM_COMPONENTS: Final = 8_000_000


def _positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _validated_orders(ngram_orders: object) -> tuple[int, ...]:
    if not isinstance(ngram_orders, tuple | list) or not ngram_orders:
        raise ValueError("ngram_orders must be a nonempty tuple or list of integers")
    orders = tuple(
        _positive_int(order, "each ngram order", MAX_NGRAM_ORDER)
        for order in ngram_orders
    )
    if len(set(orders)) != len(orders):
        raise ValueError("ngram_orders must not contain duplicates")
    return orders


def _ngram_key(token_ids: list[int], position: int, order: int) -> tuple[int, ...]:
    return tuple(
        token_ids[position - offset] if position >= offset else 0
        for offset in range(order)
    )


def _address_for_key(ngram_key: tuple[int, ...], *, table_size: int, head: int) -> int:
    address = head + 1
    multiplier = 257 + head * 2
    for shifted in ngram_key:
        address = (address * multiplier + shifted) % table_size
    return address


def _file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{name} must be an existing regular file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{name} is empty: {path}")
    if size > MAX_INPUT_BYTES:
        raise ValueError(f"{name} exceeds {MAX_INPUT_BYTES} byte limit: {path}")
    return path


def analyze_training_corpus(
    train_jsonl: Path | str,
    tokenizer_path: Path | str,
    *,
    table_size: int,
    memory_dim: int,
    ngram_orders: tuple[int, ...] | list[int],
    hash_heads: int = 1,
) -> dict[str, object]:
    """Analyze one caller-designated training corpus without allocating tables.

    Only ``train_jsonl`` is opened. Its split role is asserted by the caller;
    this function never discovers or reads validation/test files.
    """
    table_size = _positive_int(table_size, "table_size", MAX_TABLE_SIZE)
    memory_dim = _positive_int(memory_dim, "memory_dim", MAX_MEMORY_DIM)
    hash_heads = _positive_int(hash_heads, "hash_heads", MAX_HASH_HEADS)
    orders = _validated_orders(ngram_orders)
    stream_count = len(orders) * hash_heads
    hash_steps_per_token = sum(orders) * hash_heads
    if stream_count > MAX_ADDRESS_STREAMS:
        raise ValueError(f"order/head combinations exceed {MAX_ADDRESS_STREAMS}")
    if table_size * stream_count > MAX_ADDRESS_BITMAP_BITS:
        raise ValueError(
            "configured tables exceed the exact collision-bitmap memory limit"
        )

    train_path = _file(Path(train_jsonl), "train_jsonl")
    tokenizer_file = _file(Path(tokenizer_path), "tokenizer_path")
    tokenizer = load_tokenizer(tokenizer_file)
    bitmaps = {
        (order, head): bytearray((table_size + 7) // 8)
        for order in orders
        for head in range(hash_heads)
    }
    unique_address_counts = {key: 0 for key in bitmaps}
    unique_ngram_keys: dict[int, set[tuple[int, ...]]] = {
        order: set() for order in orders
    }
    unique_ngram_counts = {order: 0 for order in orders}
    unique_ngram_total = 0
    unique_ngram_components = 0
    token_frequency: Counter[int] = Counter()
    document_frequency: Counter[int] = Counter()
    document_count = 0
    token_count = 0

    for document in iter_conversations(train_path):
        document_count += 1
        if document_count > MAX_DOCUMENTS:
            raise ValueError(f"train_jsonl exceeds {MAX_DOCUMENTS} document limit")
        token_ids = tokenizer.encode(document, add_special_tokens=False).ids
        if len(token_ids) > MAX_TOKENS_PER_DOCUMENT:
            raise ValueError(
                f"train_jsonl document {document_count} exceeds "
                f"{MAX_TOKENS_PER_DOCUMENT} token limit"
            )
        token_count += len(token_ids)
        if token_count > MAX_TOKENS:
            raise ValueError(f"train_jsonl exceeds {MAX_TOKENS} token limit")
        if token_count * hash_steps_per_token > MAX_HASH_OPERATIONS:
            raise ValueError(
                f"train_jsonl exceeds {MAX_HASH_OPERATIONS} address-step limit"
            )
        token_frequency.update(token_ids)
        document_frequency.update(set(token_ids))
        for position in range(len(token_ids)):
            for order in orders:
                ngram_key = _ngram_key(token_ids, position, order)
                known_keys = unique_ngram_keys[order]
                if ngram_key in known_keys:
                    continue
                if unique_ngram_total >= MAX_UNIQUE_NGRAMS:
                    raise ValueError(
                        f"train_jsonl exceeds {MAX_UNIQUE_NGRAMS} unique n-gram limit"
                    )
                if unique_ngram_components + order > MAX_UNIQUE_NGRAM_COMPONENTS:
                    raise ValueError(
                        "train_jsonl exceeds the unique n-gram component limit"
                    )
                known_keys.add(ngram_key)
                unique_ngram_counts[order] += 1
                unique_ngram_total += 1
                unique_ngram_components += order
                for head in range(hash_heads):
                    address = _address_for_key(
                        ngram_key, table_size=table_size, head=head
                    )
                    bitmap = bitmaps[order, head]
                    byte_index, bit_index = divmod(address, 8)
                    mask = 1 << bit_index
                    if not bitmap[byte_index] & mask:
                        bitmap[byte_index] |= mask
                        unique_address_counts[order, head] += 1

    if document_count == 0:
        raise ValueError("train_jsonl contains no conversations")
    if token_count == 0:
        raise ValueError("train_jsonl tokenized to no tokens")

    entropy_bits = -sum(
        (count / token_count) * math.log2(count / token_count)
        for count in token_frequency.values()
    )
    streams: list[dict[str, object]] = []
    for order in orders:
        distinct_ngrams = unique_ngram_counts[order]
        repeated_accesses = token_count - distinct_ngrams
        for head in range(hash_heads):
            unique = unique_address_counts[order, head]
            collisions = distinct_ngrams - unique
            streams.append(
                {
                    "order": order,
                    "hash_head": head,
                    "lookup_count": token_count,
                    "distinct_ngram_count": distinct_ngrams,
                    "repeated_accesses": repeated_accesses,
                    "unique_occupied_buckets": unique,
                    "collisions": collisions,
                    "collision_rate": collisions / distinct_ngrams,
                    "repeated_access_rate": repeated_accesses / token_count,
                    "utilization": unique / table_size,
                }
            )

    table_bytes = table_size * memory_dim * stream_count * 4
    result: dict[str, object] = {
        "format": "sparselab-lexical-corpus-analysis",
        "version": 1,
        "input_scope": "one_explicit_training_jsonl",
        "training_corpus": {
            "sha256": sha256_file(train_path),
            "bytes": train_path.stat().st_size,
        },
        "tokenizer": {"sha256": sha256_file(tokenizer_file)},
        "documents": document_count,
        "tokens": token_count,
        "token_frequency": [
            {"token_id": token_id, "count": count}
            for token_id, count in sorted(token_frequency.items())
        ],
        "document_frequency": [
            {"token_id": token_id, "count": count}
            for token_id, count in sorted(document_frequency.items())
        ],
        "empirical_unigram_entropy_bits_per_token": entropy_bits,
        "addressing": {
            "algorithm": "token-ngram-recurrence-v1",
            "recurrence": "address=(address*(257+hash_head*2)+shifted_token_id)%table_size",
            "initial_address": "hash_head+1",
            "document_boundary": "zero_padded",
            "collision_key": "distinct zero-padded token n-gram keys across corpus",
            "collision_definition": (
                "distinct keys minus occupied buckets; repeated lookups are separate"
            ),
            "distinct_key_limits": {
                "max_keys": MAX_UNIQUE_NGRAMS,
                "max_key_components": MAX_UNIQUE_NGRAM_COMPONENTS,
            },
            "table_size": table_size,
            "ngram_orders": list(orders),
            "hash_heads": hash_heads,
        },
        "streams": streams,
        "storage": {
            "table_fp32_bytes": table_bytes,
            "table_count": stream_count,
            "memory_dim": memory_dim,
            "excludes": ["output_projection_weights", "gate_weights"],
        },
    }
    result["analysis_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result


__all__ = ["analyze_training_corpus"]
