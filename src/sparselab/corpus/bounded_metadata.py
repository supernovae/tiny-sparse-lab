"""Bounded readers for immutable JSON manifests with disk-backed row arrays.

SQLite is disposable operation-local storage, never authoritative evidence.
The JSON bytes and existing native verifiers remain the source of truth.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from sparselab.training.manifest import canonical_json

_MAX_INPUT_BYTES = 256 * 1024**2


class DiskRows(list):
    """Read-only list interface for ordered JSON rows, valid inside its store."""

    def __init__(self, store, identifier):
        self.store = store
        self.identifier = identifier
        self._length = 0

    def __len__(self):
        return self._length

    def __iter__(self):
        cursor = self.store.connection.execute(
            "SELECT payload FROM rows WHERE collection=? ORDER BY ordinal",
            (self.identifier,),
        )
        try:
            for (payload,) in cursor:
                yield json.loads(payload)
        finally:
            cursor.close()

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("disk-backed rows require an integer index")
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        row = self.store.connection.execute(
            "SELECT payload FROM rows WHERE collection=? AND ordinal=?",
            (self.identifier, index),
        ).fetchone()
        return json.loads(row[0])

    def __eq__(self, other):
        if not isinstance(other, (list, tuple)) or len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    def __ne__(self, other):
        return not self == other

    def __contains__(self, value):
        return any(row == value for row in self)

    def __reversed__(self):
        for index in range(len(self) - 1, -1, -1):
            yield self[index]

    def count(self, value):
        return sum(row == value for row in self)

    def index(self, value, start=0, stop=None):
        start, stop, _ = slice(start, stop).indices(len(self))
        for index in range(start, stop):
            if self[index] == value:
                return index
        raise ValueError("value is not in disk-backed rows")

    def _materialization(self, *args, **kwargs):
        raise TypeError(
            "explicitly iterate disk-backed rows; materialization is unsupported"
        )

    copy = __add__ = __radd__ = __mul__ = __rmul__ = _materialization

    def _readonly(self, *args, **kwargs):
        raise TypeError("disk-backed rows are read-only; use the owning store")

    append = extend = insert = clear = pop = remove = reverse = sort = _readonly
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _readonly


class MetadataStore:
    """Explicit finite database cap; SQLite cache size is not an RSS guarantee."""

    def __init__(self, directory: Path, *, max_database_bytes: int):
        if type(max_database_bytes) is not int or max_database_bytes < 16384:
            raise ValueError("metadata database cap must be at least 16384 bytes")
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError("metadata scratch directory must not be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix=".metadata-", dir=directory)
        self.connection = sqlite3.connect(Path(self.temporary.name) / "rows.sqlite")
        self.next_identifier = 0
        try:
            self.connection.execute("PRAGMA page_size=4096")
            self.connection.execute(
                f"PRAGMA max_page_count={max_database_bytes // 4096}"
            )
            self.connection.execute("PRAGMA cache_size=-4096")
            self.connection.execute("PRAGMA temp_store=FILE")
            # No rollback log is needed for disposable, unpublished derived rows.
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute(
                "CREATE TABLE rows(collection INTEGER, ordinal INTEGER, payload BLOB, "
                "PRIMARY KEY(collection, ordinal)) WITHOUT ROWID"
            )
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.connection.close()
        self.temporary.cleanup()

    def new_rows(self):
        self.next_identifier += 1
        return DiskRows(self, self.next_identifier)

    def append(self, rows, value):
        if rows.store is not self:
            raise ValueError("row collection belongs to another metadata store")
        try:
            self.connection.execute(
                "INSERT INTO rows VALUES(?,?,?)",
                (
                    rows.identifier,
                    rows._length,
                    canonical_json(value),
                ),
            )
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise ValueError("metadata database byte cap exceeded") from error
            raise
        rows._length += 1

    def load(
        self,
        path: Path,
        *,
        max_input_bytes: int,
        max_header_bytes=4 * 1024**2,
        max_record_bytes=8 * 1024**2,
    ):
        """Parse JSON strictly; spill records/selected_rows arrays without rewriting input."""
        for value in (max_input_bytes, max_header_bytes, max_record_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("metadata limits must be positive integers")
        if (
            any(part.is_symlink() for part in (path, *path.parents))
            or not path.is_file()
        ):
            raise ValueError("unsafe metadata input")
        if path.stat().st_size > max_input_bytes:
            raise ValueError("metadata input-byte cap exceeded")
        with path.open("rb") as stream:
            return _Reader(
                stream, self, max_input_bytes, max_header_bytes, max_record_bytes
            ).read()


class _Reader:
    def __init__(self, stream, store, input_limit, header_limit, row_limit):
        self.stream, self.store = stream, store
        self.input_limit, self.header_limit, self.row_limit = (
            input_limit,
            header_limit,
            row_limit,
        )
        self.buffer = b""
        self.offset = self.total = self.header = 0
        self.row_bytes = None

    def peek(self):
        if self.offset == len(self.buffer):
            self.buffer = self.stream.read(min(4096, self.input_limit - self.total + 1))
            self.offset = 0
        return self.buffer[self.offset : self.offset + 1]

    def take(self):
        value = self.peek()
        if not value:
            raise ValueError("truncated metadata JSON")
        self.offset += 1
        self.total += 1
        if self.total > self.input_limit:
            raise ValueError("metadata input-byte cap exceeded")
        if self.row_bytes is None:
            self.header += 1
            if self.header > self.header_limit:
                raise ValueError("metadata header-byte cap exceeded")
        else:
            self.row_bytes += 1
            if self.row_bytes > self.row_limit:
                raise ValueError("metadata record-byte cap exceeded")
        return value

    def whitespace(self):
        while self.peek() and self.peek() in b" \t\r\n":
            self.take()

    def expect(self, value):
        self.whitespace()
        if self.take() != value:
            raise ValueError("malformed metadata JSON")

    def string(self):
        self.expect(b'"')
        raw = bytearray(b'"')
        escaped = False
        while True:
            char = self.take()
            raw.extend(char)
            if char == b'"' and not escaped:
                return json.loads(raw)
            escaped = char == b"\\" and not escaped

    @contextmanager
    def record(self):
        self.row_bytes = 0
        try:
            yield
        finally:
            self.row_bytes = None

    def value(self, depth=0, *, spill=False):
        if depth > 64:
            raise ValueError("metadata nesting cap exceeded")
        self.whitespace()
        char = self.peek()
        if char == b'"':
            return self.string()
        if char == b"{":
            self.take()
            result = {}
            self.whitespace()
            if self.peek() == b"}":
                self.take()
                return result
            while True:
                key = self.string()
                if key in result:
                    raise ValueError("duplicate metadata object key")
                self.expect(b":")
                result[key] = self.value(
                    depth + 1,
                    spill=self.row_bytes is None
                    and key in {"records", "selected_rows", "files"},
                )
                self.whitespace()
                end = self.take()
                if end == b"}":
                    return result
                if end != b",":
                    raise ValueError("malformed metadata object")
        if char == b"[":
            self.take()
            result = self.store.new_rows() if spill else []
            self.whitespace()
            if self.peek() == b"]":
                self.take()
                return result
            while True:
                if spill:
                    with self.record():
                        item = self.value(depth + 1)
                        self.whitespace()
                        end = self.take()
                    self.store.append(result, item)
                else:
                    result.append(self.value(depth + 1))
                    self.whitespace()
                    end = self.take()
                if end == b"]":
                    return result
                if end != b",":
                    raise ValueError("malformed metadata array")
        raw = bytearray()
        while self.peek() and self.peek() not in b" \t\r\n,]}":
            raw.extend(self.take())
        if not raw:
            raise ValueError("missing metadata value")
        result = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonfinite metadata number")
            ),
        )

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("nonfinite metadata number")
        return result

    def read(self):
        value = self.value()
        self.whitespace()
        if self.peek():
            raise ValueError("trailing metadata JSON")
        return value


_ACTIVE_STORE: ContextVar[MetadataStore | None] = ContextVar(
    "corpus_metadata_store", default=None
)


@contextmanager
def metadata_operation(directory: Path, *, max_database_bytes=1024**3):
    """Share disposable storage through nested native verification calls."""
    existing = _ACTIVE_STORE.get()
    if existing is not None:
        yield existing
        return
    with MetadataStore(directory, max_database_bytes=max_database_bytes) as store:
        token = _ACTIVE_STORE.set(store)
        try:
            yield store
        finally:
            _ACTIVE_STORE.reset(token)


def read_metadata(path: Path):
    store = _ACTIVE_STORE.get()
    if store is None:
        return json.loads(path.read_text(encoding="utf-8"))
    return store.load(path, max_input_bytes=_MAX_INPUT_BYTES)


class DiskDecisions(dict):
    """Integer-keyed decision lookup sharing the operation's bounded database."""

    def __init__(self, store):
        self.store = store
        self.rows = store.new_rows()

    def __len__(self):
        return self.rows._length

    def __iter__(self):
        cursor = self.store.connection.execute(
            "SELECT ordinal FROM rows WHERE collection=? ORDER BY ordinal",
            (self.rows.identifier,),
        )
        try:
            for (key,) in cursor:
                yield key
        finally:
            cursor.close()

    def __contains__(self, key):
        if type(key) is not int or not 0 <= key <= 2**63 - 1:
            return False
        return (
            self.store.connection.execute(
                "SELECT 1 FROM rows WHERE collection=? AND ordinal=?",
                (self.rows.identifier, key),
            ).fetchone()
            is not None
        )

    def __getitem__(self, key):
        if type(key) is not int or not 0 <= key <= 2**63 - 1:
            raise KeyError(key)
        row = self.store.connection.execute(
            "SELECT payload FROM rows WHERE collection=? AND ordinal=?",
            (self.rows.identifier, key),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        if type(key) is not int or not 0 <= key <= 2**63 - 1:
            raise ValueError(
                "metadata decision index must be a nonnegative signed 64-bit integer"
            )
        if key in self:
            raise ValueError("duplicate metadata decision index")
        try:
            self.store.connection.execute(
                "INSERT INTO rows VALUES(?,?,?)",
                (
                    self.rows.identifier,
                    key,
                    canonical_json(value),
                ),
            )
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise ValueError("metadata database byte cap exceeded") from error
            raise
        self.rows._length += 1

    def __eq__(self, other):
        from collections.abc import Mapping

        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        return all(key in other and value == other[key] for key, value in self.items())

    def __ne__(self, other):
        return not self == other

    def copy(self):
        raise TypeError(
            "explicitly iterate disk-backed decisions; materialization is unsupported"
        )

    def keys(self):
        return iter(self)

    def items(self):
        for key in self:
            yield key, self[key]

    def values(self):
        yield from self.rows


def decision_index():
    store = _ACTIVE_STORE.get()
    return DiskDecisions(store) if store is not None else {}
