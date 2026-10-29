"""Durable content tokenization, independent of mixture quotas and ordering.

Only encoding is reused. Current release rights, split isolation, deduplication
and scheduling are still checked by the mixture producer on every invocation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from sparselab.canonical import canonical_json
from sparselab.corpus.mixture_index import MAX_ROW_BYTES


class TokenCache:
    def __init__(self, root: Path, tokenizer_sha: str):
        root.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(
            canonical_json(["encode-no-special-v1", tokenizer_sha])
        ).hexdigest()
        path = root / f"{name}.sqlite"
        if path.is_symlink():
            raise ValueError("token cache must not be a symlink")
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA cache_size=-4096;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS tokens (
                content TEXT PRIMARY KEY, payload BLOB NOT NULL, digest TEXT NOT NULL
            );
        """)
        self.pending = 0

    def encode(self, tokenizer, text):
        content = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self.db.execute(
            "SELECT payload, digest FROM tokens WHERE content=?", (content,)
        ).fetchone()
        if row is not None:
            payload, digest = row
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("cached token payload digest mismatch")
            ids = json.loads(payload)
            if not isinstance(ids, list) or any(
                type(i) is not int or i < 0 for i in ids
            ):
                raise ValueError("invalid cached token IDs")
            return ids
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        payload = canonical_json(ids)
        if len(payload) > MAX_ROW_BYTES:
            raise ValueError("mixture encoded record-byte cap exceeded")
        self.db.execute(
            "INSERT INTO tokens VALUES(?,?,?)",
            (content, payload, hashlib.sha256(payload).hexdigest()),
        )
        self.pending += 1
        if self.pending >= 256:
            self.db.commit()
            self.pending = 0
        return ids

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Valid encodings survive quota failure or cooperative interruption.
        try:
            self.db.commit()
        finally:
            self.db.close()
