"""Shared memory-table address hashing for token n-gram and raw-byte tables.

Two schemes exist per table size:

* **v1 (legacy)** reduces a base-257 polynomial modulo ``table_size`` at every
  step. When ``gcd(multiplier, table_size) == 1`` the step is a bijection on the
  previous address, so v1 is a real hash and every address it produces is kept
  bit-for-bit: existing checkpoints, prepared byte addresses and portable packs
  at those sizes are unaffected.
* **v2 (mixed)** is used only when the multiplier shares a factor with
  ``table_size`` (for example ``table_size == 257`` or ``514``). There v1
  degenerates: with ``gcd == table_size`` every step discards the previous
  address, so an "n-gram" address is just the oldest token modulo the table
  size, and the byte hash's trailing ``* 257`` sends every input to row 0. v2
  mixes in the prime field ``2**31 - 1`` with a large odd multiplier and reduces
  modulo ``table_size`` only once at the end, so no table size collapses it.

All arithmetic is plain ``*``, ``+`` and ``%`` on non-negative values below
``2**62``, so the same helpers work on Python ints and int64 torch tensors.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import gcd
from typing import Final

LEGACY_BASE: Final = 257
MIX_MODULUS: Final = 2**31 - 1
MIX_MULTIPLIER: Final = 1_103_515_245

TOKEN_SCHEME_V1: Final = "token-ngram-recurrence-v1"
TOKEN_SCHEME_V2: Final = "token-ngram-recurrence-v2"
BYTE_SCHEME_V1: Final = "poly257-terminal-v1"
BYTE_SCHEME_V2: Final = "mix31-terminal-v2"


def _positive(table_size: int) -> int:
    if type(table_size) is not int or table_size <= 0:
        raise ValueError("table_size must be positive")
    return table_size


def token_multiplier(head: int) -> int:
    """The legacy per-hash-head multiplier ``257 + 2 * head``."""
    if type(head) is not int or head < 0:
        raise ValueError("hash head must be a non-negative integer")
    return LEGACY_BASE + 2 * head


def token_uses_legacy(table_size: int, head: int) -> bool:
    """True when v1 is a proper hash for this table size and hash head."""
    return gcd(token_multiplier(head), _positive(table_size)) == 1


def token_scheme(table_size: int, head: int) -> str:
    return TOKEN_SCHEME_V1 if token_uses_legacy(table_size, head) else TOKEN_SCHEME_V2


def token_address_init(head: int) -> int:
    token_multiplier(head)
    return head + 1


def token_address_step[Value](
    address: Value, token: Value, table_size: int, head: int
) -> Value:
    """Fold one token (newest first) into a running address."""
    if token_uses_legacy(table_size, head):
        return (address * token_multiplier(head) + token) % table_size  # type: ignore[operator]
    multiplier = MIX_MULTIPLIER + 2 * head
    return (address * multiplier + token + 1) % MIX_MODULUS  # type: ignore[operator]


def token_address_finish[Value](address: Value, table_size: int, head: int) -> Value:
    """Reduce a running address to a table row."""
    if token_uses_legacy(table_size, head):
        return address
    return address % table_size  # type: ignore[operator]


def token_ngram_address(key: Iterable[int], table_size: int, head: int) -> int:
    """Address of one zero-padded token n-gram key, newest token first."""
    address = token_address_init(head)
    for token in key:
        address = token_address_step(address, token, table_size, head)
    return token_address_finish(address, table_size, head)


def byte_uses_legacy(table_size: int) -> bool:
    return gcd(LEGACY_BASE, _positive(table_size)) == 1


def byte_scheme(table_size: int) -> str:
    """The raw-byte hashing identifier recorded for a table of this size."""
    return BYTE_SCHEME_V1 if byte_uses_legacy(table_size) else BYTE_SCHEME_V2


def mixed_byte_address(value: bytes, table_size: int) -> int:
    """v2 raw-byte address with a terminal separator step."""
    _positive(table_size)
    address = 1
    for byte in value:
        address = (address * MIX_MULTIPLIER + byte + 1) % MIX_MODULUS
    address = (address * MIX_MULTIPLIER) % MIX_MODULUS
    return address % table_size
