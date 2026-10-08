"""The optional BPE frequency choice retains historical export identities."""

from __future__ import annotations

import hashlib

import pytest

from sparselab.corpus.export import _export_request
from sparselab.training.manifest import canonical_json


def test_default_export_request_keeps_historical_hash_and_two_is_distinct() -> None:
    historical = {
        "release_id": "a" * 64,
        "view": "lm",
        "base_config_sha256": "b" * 64,
        "vocab_size": 32768,
    }
    default = _export_request("a" * 64, "lm", "b" * 64, 32768, 1)
    assert default == historical
    assert hashlib.sha256(canonical_json(default)).hexdigest() == hashlib.sha256(
        canonical_json(historical)
    ).hexdigest()
    twice = _export_request("a" * 64, "lm", "b" * 64, 32768, 2)
    assert twice == {**historical, "min_frequency": 2}
    assert hashlib.sha256(canonical_json(twice)).digest() != hashlib.sha256(
        canonical_json(default)
    ).digest()


def test_export_request_rejects_invalid_minimum_frequency() -> None:
    for invalid in (0, -1, True, "2"):
        with pytest.raises(ValueError, match="min_frequency"):
            _export_request("a" * 64, "lm", "b" * 64, 32768, invalid)
