"""Owned, authenticated inspection work; unsigned or stale state is never reused.

The existing host-local ProofStore trust boundary applies: immutable files owned
by the service UID, not protection against a compromised same-UID owner. Only a
completed native inspection publishes a result. SQLite rows are disposable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from sparselab.corpus.bounded_metadata import current_metadata_store
from sparselab.training.manifest import canonical_json, sha256_file
from sparselab.verification_proofs import (
    _fingerprint,
    _sync,
    _trusted,
    file_binding,
    verification_options,
)
from sparselab.verifier_authority import verifier_authority


def metadata_sha256(value):
    """Stream DiskRows through Python's iterator, never the list C encoder."""
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def capture(paths):
    result = {}
    for path in sorted({Path(p).absolute() for p in paths}):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("unsafe inspection dependency")
        members = [path, *sorted(path.rglob("*"))] if path.is_dir() else [path]
        result[str(path)] = [
            [p.relative_to(path).as_posix() if p != path else ".", *_fingerprint(p)]
            for p in members
        ]
    return result


def unchanged(dependencies):
    if capture(dependencies) != dependencies:
        raise ValueError("inspection inputs changed during execution")


class InspectionCache:
    """Reuse successful native results, including completed sources after a stop."""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.proofs = verification_options(self.root)["proof_store"]
        self.directory = self.root / "cache" / "admission-inspection"
        self.writable = self.proofs is not None
        self.unavailable_reason = None if self.writable else "untrusted_work_root"
        self.authority = {}
        if self.proofs is None:
            return
        try:
            self.authority = {
                "corpus": verifier_authority("source_snapshot", 1),
                "inspection": {
                    name: sha256_file(Path(__file__).with_name(name))
                    for name in (
                        "admission_inspection.py",
                        "admission_inspection_cache.py",
                        "rights.py",
                        "jsonl_records.py",
                        "structure_cleaning.py",
                    )
                },
            }
        except OSError, ValueError, TypeError, KeyError, SyntaxError:
            self.proofs = None
            self._unavailable("verifier_authority_unavailable")

    def _unavailable(self, reason):
        self.writable = False
        self.unavailable_reason = reason

    def _cleanup(self, temporary):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                self._unavailable("cache_cleanup_unavailable")

    def _path(self, kind, request):
        key = hashlib.sha256(canonical_json([kind, request])).hexdigest()
        return self.directory / f"{kind}-{key}.json"

    def _binding(self, path, raw, request, dependencies):
        binding = file_binding(
            path,
            hashlib.sha256(raw).hexdigest(),
            closure={"request": request, "authority": self.authority},
        )
        binding["dependencies"] = [
            [path, rows] for path, rows in sorted(dependencies.items())
        ]
        return binding

    def load(self, kind, request, *, limit):
        if self.proofs is None:
            return None
        path = self._path(kind, request)
        try:
            if not self.proofs.trusted(path) or path.stat().st_size > limit:
                return None
            raw = path.read_bytes()
            value = json.loads(raw)
            if set(value) != {"dependencies", "payload"}:
                return None
            paths = value["dependencies"]
            if not isinstance(paths, list) or any(
                not isinstance(p, str)
                or not Path(p).is_absolute()
                or ".." in Path(p).parts
                or not Path(p).is_relative_to(self.root)
                for p in paths
            ):
                return None
            dependencies = capture(paths)
            binding = self._binding(path, raw, request, dependencies)
            if not self.proofs.lookup(binding):
                return None
            return value["payload"], dependencies
        except OSError, ValueError, TypeError, KeyError, json.JSONDecodeError:
            return None

    def publish(self, kind, request, payload, dependencies, *, limit):
        unchanged(dependencies)
        if self.proofs is None or not self.writable:
            return
        # A proof cannot cover a dependency outside its registered trust root.
        dependencies = {
            path: rows
            for path, rows in dependencies.items()
            if Path(path).is_relative_to(self.root)
        }
        raw = (
            canonical_json({"dependencies": sorted(dependencies), "payload": payload})
            + b"\n"
        )
        if len(raw) > limit:
            return  # Cache size is optional; the independently checked result stands.
        if any(p.is_symlink() for p in (self.directory, *self.directory.parents)):
            self._unavailable("untrusted_cache_directory")
            return
        path = self._path(kind, request)
        from sparselab.data.verification import verify_file

        temporary = None
        try:
            self.directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self.proofs.trusted(self.directory.parent):
                self._unavailable("untrusted_cache_directory")
                return
            self.directory.mkdir(exist_ok=True, mode=0o700)
            if not _trusted(self.root, self.directory, private=True):
                self._unavailable("untrusted_cache_directory")
                return
            with tempfile.NamedTemporaryFile(
                prefix=".inspection-", dir=self.directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._unavailable("cache_write_unavailable")
            self._cleanup(temporary)
            return
        except BaseException:
            self._cleanup(temporary)
            raise
        try:
            unchanged(dependencies)
            try:
                os.replace(temporary, path)
                _sync(self.directory)
                binding = self._binding(path, raw, request, dependencies)
                verify_file(
                    path,
                    expected_sha256=binding["sha256"],
                    proof_store=self.proofs,
                    verification_mode="verified_reuse",
                    binding=binding,
                )
            except OSError, ValueError:
                # An optional cache cannot veto the live native-validated result.
                # Keep reads enabled so other authenticated sources can still hit.
                self._unavailable("cache_write_unavailable")
        finally:
            self._cleanup(temporary)


class SourceIndex:
    """One source's offsets and lengths, with bounded SQLite cache and no text."""

    def __init__(self):
        self.connection = current_metadata_store().connection
        self.connection.execute("DROP TABLE IF EXISTS temp.inspection_candidates")
        self.connection.execute(
            "CREATE TEMP TABLE inspection_candidates(ordinal INTEGER PRIMARY KEY, "
            "path TEXT,digest TEXT,row_index INTEGER,physical_line INTEGER,"
            "byte_offset INTEGER,decision TEXT,issues INTEGER,length INTEGER)"
        )
        self.rows = 0

    def add(self, row, length):
        self.connection.execute(
            "INSERT INTO inspection_candidates VALUES(?,?,?,?,?,?,?,?,?)",
            (
                row["_ordinal"],
                row["path"],
                row["digest"],
                row["row_index"],
                row["physical_line"],
                row["_offset"],
                row["decision"],
                bool(row["issues"]),
                length,
            ),
        )
        self.rows += 1

    def median(self):
        count = self.connection.execute(
            "SELECT count(*) FROM inspection_candidates WHERE decision='qualify'"
        ).fetchone()[0]
        if not count:
            return 0, 0
        self.connection.execute(
            "CREATE INDEX inspection_lengths ON inspection_candidates(decision,length)"
        )
        values = self.connection.execute(
            "SELECT length FROM inspection_candidates WHERE decision='qualify' "
            "ORDER BY length LIMIT ? OFFSET ?",
            (2 if count % 2 == 0 else 1, (count - 1) // 2),
        ).fetchall()
        return sum(value[0] for value in values) * (2 if count % 2 else 1), count

    def __iter__(self):
        cursor = self.connection.execute(
            "SELECT * FROM inspection_candidates ORDER BY ordinal"
        )
        try:
            for row in cursor:
                yield dict(
                    zip(
                        (
                            "_ordinal",
                            "path",
                            "digest",
                            "row_index",
                            "physical_line",
                            "_offset",
                            "decision",
                            "issues",
                            "retained_length_bytes",
                        ),
                        row,
                        strict=True,
                    )
                )
        finally:
            cursor.close()
