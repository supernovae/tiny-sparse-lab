import hashlib
import json

import pytest

from sparselab.corpus.bounded_metadata import metadata_operation
from sparselab.corpus.project import SplitDeclaration
from sparselab.corpus.split_assignments import (
    Assignments,
    assignment_path,
    verify_document_assignments,
)
from sparselab.training.manifest import canonical_json


def test_compact_lookup_reuses_parsing_and_closes_owned_maps(tmp_path, monkeypatch):
    from sparselab.corpus import split_lookup

    path, ref = _bound(
        tmp_path,
        [
            {"document_id": "alpha", "split": "train"},
            {"document_id": "\u03b2eta", "split": "validation"},
        ],
    )
    original, calls = split_lookup.records_from_path, []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(split_lookup, "records_from_path", counted)
    with metadata_operation(tmp_path):
        first = Assignments(path, ref)
        second = Assignments(path, ref)
        assert (
            dict(first) == dict(second) == {"alpha": "train", "\u03b2eta": "validation"}
        )
        assert len(calls) == 1
        owned = first.index
    assert owned.maps == []


def _bound(tmp_path, rows, count=None):
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / ("assignments-" + digest + ".jsonl")
    path.write_bytes(raw)
    return path, {"sha256": digest, "rows": len(rows) if count is None else count}


def test_bound_assignments_lookup_and_cold_disagreement(tmp_path):
    path, ref = _bound(
        tmp_path,
        [{"document_id": "a", "split": "test"}, {"document_id": "b", "split": "train"}],
    )
    identity = {"split": {"assignment_file": ref}}
    with metadata_operation(tmp_path):
        mapping = Assignments(assignment_path(tmp_path, ref), ref)
        assert dict(mapping) == {"a": "test", "b": "train"}
        assert mapping.get("absent") is None
        verify_document_assignments(
            tmp_path, identity, [{"document_id": "a", "split": "test"}]
        )
        for doc in [
            {"document_id": "a", "split": "train"},
            {"document_id": "absent", "split": "test"},
        ]:
            with pytest.raises(ValueError, match="differs from frozen"):
                verify_document_assignments(tmp_path, identity, [doc])
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        assignment_path(tmp_path, ref)
    assert not list(tmp_path.glob(".metadata-*"))


@pytest.mark.parametrize(
    "rows,count,match",
    [
        ([{"document_id": "a", "split": "train"}], 2, "count mismatch"),
        (
            [
                {"document_id": "b", "split": "train"},
                {"document_id": "a", "split": "test"},
            ],
            2,
            "unordered",
        ),
        (
            [
                {"document_id": "a", "split": "train"},
                {"document_id": "a", "split": "train"},
            ],
            2,
            "unordered",
        ),
        ([{"document_id": "a", "split": "invalid"}], 1, "invalid"),
        ([{"document_id": "a", "split": "train", "extra": 1}], 1, "invalid"),
    ],
)
def test_assignment_refusals(tmp_path, rows, count, match):
    _path, ref = _bound(tmp_path, rows, count)
    with pytest.raises(ValueError, match=match), metadata_operation(tmp_path):
        Assignments(assignment_path(tmp_path, ref), ref)
    assert not list(tmp_path.glob(".metadata-*"))


def test_assignment_schema_and_path_refuse_ambiguous_unsafe_and_symlink(tmp_path):
    base = {"schema_version": 1, "unit": "document", "assignments": {}}
    ref = {"sha256": "a" * 64, "rows": 1}
    SplitDeclaration.model_validate({**base, "assignment_file": ref})
    for invalid in [
        base,
        {**base, "assignments": {"a": "train"}, "assignment_file": ref},
        {**base, "assignment_file": {**ref, "rows": True}},
        {**base, "unit": "source_repository", "assignment_file": ref},
    ]:
        with pytest.raises(ValueError):
            SplitDeclaration.model_validate(invalid)
    with pytest.raises(ValueError):
        assignment_path(tmp_path, {**ref, "sha256": "../outside"})
    target = tmp_path / "target"
    target.write_text(json.dumps({"document_id": "a", "split": "train"}) + "\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (tmp_path / ("assignments-" + digest + ".jsonl")).symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        assignment_path(tmp_path, {"sha256": digest, "rows": 1})


@pytest.mark.parametrize(
    "options",
    [
        {"exclude_none": True},
        {"exclude": {"assignment_file"}},
        {"include": {"unit", "assignments"}},
    ],
)
def test_inline_assignment_serializer_respects_optional_field_selection(options):
    declaration = SplitDeclaration.model_validate(
        {"schema_version": 1, "unit": "document", "assignments": {"doc": "train"}}
    )
    output = declaration.model_dump(mode="json", **options)
    assert output["assignments"] == {"doc": "train"}
    assert "assignment_file" not in output
