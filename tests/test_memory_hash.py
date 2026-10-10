"""Regression: memory-table addresses must not degenerate at any table size.

With the legacy base-257 recurrence, ``memory_table_size: 257`` turned hash
head 0 of every token n-gram table into a single-token table (the address was
the oldest token in the window modulo 257) and sent every raw-byte address to
row 0. Table sizes coprime with the multiplier must keep their legacy
addresses bit-for-bit, so existing checkpoints and packs stay valid.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from sparselab.address_hash import (
    BYTE_SCHEME_V1,
    BYTE_SCHEME_V2,
    TOKEN_SCHEME_V1,
    TOKEN_SCHEME_V2,
    byte_scheme,
    token_ngram_address,
    token_scheme,
)
from sparselab.data.byte_hash import hash_bytes, table_address
from sparselab.data.lexical_mining import _address_for_key
from sparselab.model.memory import TokenNgramMemory
from sparselab.model.portable_engram import (
    export_portable_engram,
    load_portable_engram,
)
from sparselab.research.portability_campaign import _token_address

TABLE_SIZES = (257, 514, 251, 1021)
VOCAB = 512


def _legacy_token_address(key: tuple[int, ...], table_size: int, head: int) -> int:
    address = head + 1
    for token in key:
        address = (address * (257 + 2 * head) + token) % table_size
    return address


def _expected_distinct(draws: int, table_size: int) -> float:
    return table_size * (1 - (1 - 1 / table_size) ** draws)


def _ids(seed: int = 0, batch: int = 8, length: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, length), generator=generator)


@pytest.mark.parametrize("table_size", TABLE_SIZES)
@pytest.mark.parametrize("order", (2, 3))
@pytest.mark.parametrize("head", (0, 1))
def test_token_ngram_addresses_use_many_rows_and_whole_window(
    table_size: int, order: int, head: int
) -> None:
    memory = TokenNgramMemory(16, table_size, 3, 8, (2, 3), 2)
    ids = _ids()
    addresses = memory.addresses(ids, order, head)[:, order - 1 :]
    assert int(addresses.min()) >= 0 and int(addresses.max()) < table_size
    draws = addresses.numel()
    assert addresses.unique().numel() >= 0.8 * _expected_distinct(draws, table_size)

    # Not a single-token table: the address is not the oldest token mod size...
    oldest = ids[:, : ids.shape[1] - (order - 1)] % table_size
    assert (addresses == oldest).float().mean() < 0.05
    # ...and changing any single token in the window moves the address.
    for position in range(order - 1, ids.shape[1], 7):
        window = ids[0, position - order + 1 : position + 1].clone()
        for slot in range(order):
            altered = ids[:1].clone()
            moved_rows = 0
            for delta in range(1, 6):
                altered[0, position - order + 1 + slot] = (window[slot] + delta) % VOCAB
                row = memory.addresses(altered, order, head)[0, position]
                moved_rows += int(row != addresses[0, position - order + 1])
            assert moved_rows >= 4, (position, slot)


@pytest.mark.parametrize("table_size", (*TABLE_SIZES, 17, 127, 259, 8192))
def test_tensor_and_python_token_hashes_agree(table_size: int) -> None:
    memory = TokenNgramMemory(16, table_size, 3, 8, (2, 3), 3)
    ids = _ids(seed=1, batch=2, length=24)
    for order in (2, 3):
        for head in range(3):
            addresses = memory.addresses(ids, order, head)
            for batch in range(ids.shape[0]):
                for position in range(order - 1, ids.shape[1]):
                    key = tuple(int(ids[batch, position - k]) for k in range(order))
                    expected = token_ngram_address(key, table_size, head)
                    assert int(addresses[batch, position]) == expected
                    assert (
                        _address_for_key(key, table_size=table_size, head=head)
                        == expected
                    )
    for position in range(2, ids.shape[1]):
        history = ids[0, : position + 1].tolist()
        assert _token_address(history, 3, table_size) == int(
            memory.addresses(ids, 3, 0)[0, position]
        )


@pytest.mark.parametrize("table_size", (17, 127, 251, 1021, 8192, 65521))
def test_coprime_table_sizes_keep_legacy_token_addresses(table_size: int) -> None:
    rng = random.Random(table_size)
    for head in range(3):
        assert token_scheme(table_size, head) == TOKEN_SCHEME_V1
        for _ in range(200):
            key = tuple(rng.randrange(VOCAB) for _ in range(rng.choice((2, 3))))
            assert token_ngram_address(key, table_size, head) == (
                _legacy_token_address(key, table_size, head)
            )


def test_degenerate_sizes_switch_scheme_per_head() -> None:
    assert token_scheme(257, 0) == TOKEN_SCHEME_V2
    assert token_scheme(257, 1) == TOKEN_SCHEME_V1
    assert token_scheme(514, 0) == TOKEN_SCHEME_V2
    assert token_scheme(259, 1) == TOKEN_SCHEME_V2  # 259 = 257 + 2
    assert token_scheme(7, 1) == TOKEN_SCHEME_V2  # 7 divides 259


@pytest.mark.parametrize("table_size", TABLE_SIZES)
def test_byte_addresses_are_not_constant(table_size: int) -> None:
    rng = random.Random(table_size)
    values = [bytes(rng.randrange(256) for _ in range(3)) for _ in range(2000)]
    rows = [table_address(value, table_size) for value in values]
    assert all(0 <= row < table_size for row in rows)
    assert len(set(rows)) >= 0.8 * _expected_distinct(len(set(values)), table_size)
    # Every byte of the suffix matters, not only the first or last.
    for value, row in zip(values[:50], rows[:50], strict=True):
        for slot in range(3):
            altered = bytearray(value)
            moved = 0
            for delta in range(1, 6):
                altered[slot] = (value[slot] + delta) % 256
                moved += int(table_address(bytes(altered), table_size) != row)
            assert moved >= 4


@pytest.mark.parametrize("table_size", (127, 251, 1021, 8192, 65521))
def test_coprime_table_sizes_keep_legacy_byte_addresses(table_size: int) -> None:
    assert byte_scheme(table_size) == BYTE_SCHEME_V1
    rng = random.Random(table_size)
    for _ in range(200):
        value = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 33)))
        assert table_address(value, table_size) == hash_bytes(value) % table_size


def test_byte_scheme_marks_multiples_of_257() -> None:
    assert byte_scheme(257) == BYTE_SCHEME_V2
    assert byte_scheme(514) == BYTE_SCHEME_V2
    assert byte_scheme(256) == BYTE_SCHEME_V1


def test_portable_pack_records_and_requires_the_table_size_scheme(
    tmp_path: Path,
) -> None:
    table = torch.zeros((257, 4), dtype=torch.float32)
    manifest = export_portable_engram(table, tmp_path / "new.enbyte", ngram_size=3)
    assert manifest.hashing == BYTE_SCHEME_V2
    load_portable_engram(
        tmp_path / "new.enbyte", expected_shape=(257, 4), expected_ngram_size=3
    )

    export_portable_engram(
        table, tmp_path / "old.enbyte", ngram_size=3, hashing=BYTE_SCHEME_V1
    )
    with pytest.raises(ValueError, match="addressing algorithm is unsupported"):
        load_portable_engram(
            tmp_path / "old.enbyte", expected_shape=(257, 4), expected_ngram_size=3
        )

    legacy = torch.zeros((256, 4), dtype=torch.float32)
    assert (
        export_portable_engram(legacy, tmp_path / "legacy.enbyte", ngram_size=3).hashing
        == BYTE_SCHEME_V1
    )
