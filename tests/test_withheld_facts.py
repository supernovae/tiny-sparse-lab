from __future__ import annotations

from sparselab.data.withheld_facts import (
    evaluation_cases,
    split_facts,
    training_documents,
)


def test_withheld_facts_are_disjoint_by_identity_and_value() -> None:
    train, held_out = split_facts(3)
    assert len(train) == 6
    assert len(held_out) == 2
    assert not {fact.key for fact in train} & {fact.key for fact in held_out}
    assert not {fact.value for fact in held_out} & set(
        " ".join(training_documents(3)).split()
    )


def test_evaluation_prompts_exclude_held_out_values() -> None:
    for prompt, value in evaluation_cases(5):
        assert value not in prompt
        assert prompt.endswith(" is")
