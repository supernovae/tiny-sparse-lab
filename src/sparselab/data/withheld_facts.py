"""Deterministic fact-identity splits for leakage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


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
