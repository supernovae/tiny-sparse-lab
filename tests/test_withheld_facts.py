from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.data.withheld_facts import (
    evaluation_cases,
    split_facts,
    training_documents,
    write_manifest,
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


def test_manifest_is_canonical_and_conflict_safe(tmp_path: Path) -> None:
    path = tmp_path / "facts.json"
    write_manifest(path, seed=2)
    manifest = json.loads(path.read_text())
    digest = manifest.pop("sha256")
    canonical = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    assert digest == hashlib.sha256(canonical).hexdigest()
    assert manifest["training_statements"] == list(training_documents(2))
    assert [
        (case["prompt"], case["expected_value"]) for case in manifest["held_out_cases"]
    ] == list(evaluation_cases(2))
    write_manifest(path, seed=2)
    with pytest.raises(FileExistsError, match="conflicting"):
        write_manifest(path, seed=3)
