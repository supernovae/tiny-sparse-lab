"""Small source-backed validation example (no execution by corpus derivation)."""

from pathlib import PurePosixPath


def suffix_of(name: str) -> str:
    """Return lexical POSIX suffix without consulting the filesystem."""
    return PurePosixPath(name).suffix


def has_valid_name(name: str) -> bool:
    if not name:
        raise ValueError("name must not be empty")
    return suffix_of(name) in {".md", ".txt"}
