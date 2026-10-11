"""tests/slow_tests.txt must keep naming real tests (the fast/full split)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_slow_list_entries_name_existing_tests() -> None:
    lines = (ROOT / "tests/slow_tests.txt").read_text().splitlines()
    entries = [line for line in lines if line.strip() and not line.startswith("#")]
    assert entries == sorted(set(entries))
    for entry in entries:
        path, _, name = entry.partition("::")
        source = (ROOT / path).read_text()
        if name:
            function = name.rsplit("::", 1)[-1]
            assert re.search(
                rf"^\s*(async )?def {re.escape(function)}\(", source, re.MULTILINE
            ), entry


def test_known_failures_name_existing_tests_with_an_issue() -> None:
    lines = (ROOT / "tests/known_failures.txt").read_text().splitlines()
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        node, _, reason = (part.strip() for part in line.partition("|"))
        assert re.search(r"\(#\d+\)$", reason), line
        path, _, name = node.partition("::")
        source = (ROOT / path).read_text()
        if name:
            assert re.search(rf"^\s*def {re.escape(name)}\(", source, re.MULTILINE), (
                line
            )
