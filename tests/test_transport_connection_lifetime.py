"""Tiny local ledgers: transaction lifetime must not depend on garbage collection."""

import sqlite3

import pytest
from test_bounded_acquisition import admission, pinned_files
from test_transport_initial_gate import fixture

from sparselab.data.bounded_acquisition import DatasetTransport


@pytest.mark.parametrize("kind", ["corpus", "dataset"])
def test_connections_close_after_success_and_failure(tmp_path, monkeypatch, kind):
    # Strong references deliberately prevent GC from hiding unclosed connections.
    connections = []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    if kind == "corpus":
        ledger, _ = fixture(tmp_path)
    else:
        _, lock, _ = pinned_files(tmp_path / "input", monkeypatch)
        ledger = DatasetTransport(
            lock,
            admission(lock),
            tmp_path / "cache",
            tmp_path / "output",
            tmp_path / "work",
            {},
        )
        ledger.initialize()

    def assert_closed():
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        connections.clear()

    assert_closed()
    ledger.reserve_metadata(10)
    assert_closed()
    for _ in range(300):
        ledger.receipt()
        ledger._mutate(lambda state: None)
        assert_closed()
    before = ledger.receipt()
    assert before["metadata_charged"] == 10

    def fail(state):
        state["metadata_charged"] = 999
        raise RuntimeError("fixture interruption")

    with pytest.raises(RuntimeError, match="fixture interruption"):
        ledger._mutate(fail)
    assert_closed()
    assert ledger.receipt() == before
    assert_closed()
    # SQL errors must close too, including a failed read before row decoding.
    with connect(ledger.path) as db:
        db.execute("DROP TABLE state")
    db.close()
    with pytest.raises(ValueError, match="ledger is invalid"):
        ledger._read()
    assert_closed()
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        ledger._mutate(lambda state: None)
    assert_closed()
