"""Small real-file checks for relocated tokenizer evidence."""

import hashlib

import pytest

from sparselab.corpus.tokenizer_observation import _read, _relocation_mapper


def test_tokenizer_relocation_maps_both_roots_and_detects_changed_bytes(tmp_path):
    source_work = tmp_path / "source" / "work"
    source_receipts = tmp_path / "source" / "receipts"
    target_work = tmp_path / "target" / "work"
    target_receipts = tmp_path / "target" / "receipts"
    target_work.mkdir(parents=True)
    target_receipts.mkdir(parents=True)
    token = target_work / "tokenizer.json"
    report = target_receipts / "report.json"
    token.write_bytes(b"tokenizer")
    report.write_bytes(b"completed")
    map_path, map_values = _relocation_mapper(
        source_work, target_work, source_receipts, target_receipts
    )
    for old, new in (
        (source_work / "tokenizer.json", token),
        (source_receipts / "report.json", report),
    ):
        assert map_path(old) == new
        digest = hashlib.sha256(new.read_bytes()).hexdigest()
        assert _read(map_path(old), digest) == new.read_bytes()
    assert map_values(
        {"dataset": {"train_path": str(source_work / "train")}, "vocab_size": 32768}
    ) == {"dataset": {"train_path": str(target_work / "train")}, "vocab_size": 32768}
    with pytest.raises(ValueError, match="outside relocation roots"):
        map_path(tmp_path / "other" / "report.json")
    with pytest.raises(ValueError, match="invalid tokenizer relocation path"):
        map_path(source_work / ".." / "outside")
    report.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="input changed"):
        _read(
            map_path(source_receipts / "report.json"),
            hashlib.sha256(b"completed").hexdigest(),
        )
