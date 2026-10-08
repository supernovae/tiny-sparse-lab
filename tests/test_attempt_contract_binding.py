"""Offline declaration and lock compatibility for optional attempt contracts."""

import pytest

from sparselab.experiments.lock import _identities
from sparselab.experiments.plan import AttemptContractReference, Execution


def test_absent_contract_keeps_legacy_execution_and_lock_identity() -> None:
    legacy_execution = {
        "worker": None,
        "backend": None,
        "workspace": None,
        "min_free_bytes": 0,
        "min_free_inodes": 0,
    }
    assert Execution().model_dump(mode="json") == legacy_execution
    assert Execution(attempt_contract=None).model_dump(mode="json") == legacy_execution
    legacy_lock = {"execution": legacy_execution, "cells": [], "id": "legacy"}
    current_lock = {
        "execution": Execution().model_dump(mode="json"),
        "cells": [],
        "id": "legacy",
    }
    assert _identities(current_lock) == _identities(legacy_lock)


@pytest.mark.parametrize(
    "reference",
    [
        {"path": "../escape.json", "sha256": "a" * 64},
        {"path": "/absolute.json", "sha256": "a" * 64},
        {"path": "contract.json", "sha256": "not-a-digest"},
    ],
)
def test_contract_reference_rejects_unsafe_or_unpinned_values(reference: dict) -> None:
    with pytest.raises(ValueError):
        AttemptContractReference.model_validate(reference)


def test_declared_contract_changes_only_operational_lock_identity() -> None:
    reference = AttemptContractReference(path="contract.json", sha256="a" * 64)
    old = {"execution": Execution().model_dump(mode="json"), "cells": []}
    new = {
        "execution": Execution(attempt_contract=reference).model_dump(mode="json"),
        "cells": [],
    }
    old_science, old_full = _identities(old)
    new_science, new_full = _identities(new)
    assert old_science == new_science
    assert old_full != new_full
    assert new["execution"]["attempt_contract"] == reference.model_dump()
