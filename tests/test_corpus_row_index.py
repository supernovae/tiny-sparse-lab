import pytest

from sparselab.corpus.bounded_metadata import metadata_operation
from sparselab.corpus.row_index import RowIndex


def test_index_streams_values_and_joins_without_iteration(tmp_path, monkeypatch):
    with metadata_operation(tmp_path) as store:
        docs = RowIndex(
            store.connection,
            [{"id": "b", "source_id": "s"}, {"id": "a", "source_id": "s"}],
            lambda r: r["id"],
        )
        spans = RowIndex(
            store.connection,
            [
                {"id": "a", "raw_path": "x", "line_start": 1},
                {"id": "b", "raw_path": "x", "line_start": 2},
            ],
            lambda r: r["id"],
        )
        assert len(docs) == 2
        assert docs["a"]["id"] == "a"
        assert docs.get("missing") is None
        assert [r["id"] for r in docs.values()] == ["b", "a"]
        monkeypatch.setattr(
            RowIndex, "__iter__", lambda _: pytest.fail("join materialized keys")
        )
        assert docs.same_keys(spans)
        assert [r["id"] for r in docs.source_order(spans)] == ["a", "b"]
    assert not list(tmp_path.glob(".metadata-*"))


def test_index_duplicate_and_storage_cap_cleanup(tmp_path):
    with (
        pytest.raises(ValueError, match="duplicate evidence"),
        metadata_operation(tmp_path) as store,
    ):
        RowIndex(store.connection, [{"id": "a"}, {"id": "a"}], lambda r: r["id"])
    with (
        pytest.raises(ValueError, match="database byte cap"),
        metadata_operation(tmp_path, max_database_bytes=65536) as store,
    ):
        RowIndex(
            store.connection,
            ({"id": i, "text": "x" * 32000} for i in range(10)),
            lambda r: r["id"],
        )
    assert not list(tmp_path.glob(".metadata-*"))


def test_index_refuses_record_cap_and_detects_key_difference(tmp_path):
    with metadata_operation(tmp_path) as store:
        left = RowIndex(store.connection, [{"id": "a"}], lambda r: r["id"])
        right = RowIndex(store.connection, [{"id": "b"}], lambda r: r["id"])
        assert not left.same_keys(right)
        with pytest.raises(ValueError, match="record-byte cap"):
            RowIndex(
                store.connection,
                [{"id": "a", "text": "x" * (8 * 1024**2)}],
                lambda r: r["id"],
                max_record_bytes=8 * 1024**2,
            )
