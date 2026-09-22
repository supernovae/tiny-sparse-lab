"""Tokenizer-independent raw UTF-8 byte address primitives."""

BASE = 257


def hash_bytes(value: bytes) -> int:
    """Hash raw bytes with a terminal separator; no normalization is performed."""
    result = 0
    for byte in value:
        result = result * BASE + byte + 1
    return result * BASE


def hash_text(value: str) -> int:
    """Hash the exact UTF-8 representation of text."""
    return hash_bytes(value.encode("utf-8"))


def table_address(value: bytes, table_size: int) -> int:
    if table_size <= 0:
        raise ValueError("table_size must be positive")
    return hash_bytes(value) % table_size
