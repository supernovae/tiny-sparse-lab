"""Deterministic fact-identity splits for leakage diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    value: str

    @property
    def key(self) -> tuple[str, str]:
        return self.subject, self.relation

    def statement(self) -> str:
        return f"{self.subject}'s {self.relation} is {self.value}."

    def prompt(self) -> str:
        return f"{self.subject}'s {self.relation} is"


_FACTS = (
    Fact("Ava", "favorite color", "red"),
    Fact("Ben", "favorite color", "blue"),
    Fact("Cora", "favorite color", "green"),
    Fact("Dax", "favorite color", "gold"),
    Fact("Eli", "favorite animal", "otter"),
    Fact("Fae", "favorite animal", "panda"),
    Fact("Gus", "favorite animal", "falcon"),
    Fact("Hana", "favorite animal", "turtle"),
)


def split_facts(seed: int = 0) -> tuple[tuple[Fact, ...], tuple[Fact, ...]]:
    """Return a stable 75/25 split by whole fact identity."""
    rotated = _FACTS[seed % len(_FACTS) :] + _FACTS[: seed % len(_FACTS)]
    return rotated[:6], rotated[6:]


def training_documents(seed: int = 0) -> tuple[str, ...]:
    train, held_out = split_facts(seed)
    assert not {fact.key for fact in train} & {fact.key for fact in held_out}
    return tuple(fact.statement() for fact in train)


def evaluation_cases(seed: int = 0) -> tuple[tuple[str, str], ...]:
    _, held_out = split_facts(seed)
    return tuple((fact.prompt(), fact.value) for fact in held_out)


def diagnostic_manifest(seed: int = 0) -> dict[str, object]:
    """Return canonical split evidence without training or model artifacts."""
    train, held_out = split_facts(seed)
    payload = {
        "format_version": 1,
        "seed": seed,
        "training_statements": [fact.statement() for fact in train],
        "held_out_cases": [
            {"prompt": fact.prompt(), "expected_value": fact.value} for fact in held_out
        ],
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def write_manifest(path: Path, seed: int = 0) -> None:
    """Atomically write immutable diagnostic evidence."""
    target = path
    manifest = diagnostic_manifest(seed)
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"conflicting diagnostic manifest exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
