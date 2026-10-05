"""Immutable publication, verification and read-only corpus inspection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from sparselab.training.manifest import canonical_json, sha256_file

if TYPE_CHECKING:
    from sparselab.verification_proofs import ProofStore, VerificationMode


@dataclass(frozen=True)
class _VerifiedRelease:
    """Full-verifier proof valid only while its typed closure remains unchanged."""

    manifest: dict[str, Any]
    closure_key: tuple[object, ...]


_verified_releases: ContextVar[dict[Path, _VerifiedRelease] | None] = ContextVar(
    "verified_releases", default=None
)


@contextmanager
def _verification_operation():
    """Share full authentication with nested consumers, never across operations."""
    if _verified_releases.get() is not None:
        yield
        return
    token = _verified_releases.set({})
    try:
        yield
    finally:
        _verified_releases.reset(token)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _safe(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise ValueError("unsafe release artifact path")
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("release artifact escapes directory")
    return target


def _files(root: Path, inventory: dict[str, Any]) -> None:
    if not isinstance(inventory, dict):
        raise TypeError("missing artifact inventory")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != set(inventory):
        raise ValueError("release artifact inventory mismatch")
    for name, entry in inventory.items():
        path = _safe(root, name)
        if (
            not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"tampered artifact: {name}")


def _verify_rights_files(
    root: Path,
    sources: dict[str, dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Bind prospective file decisions to the pinned bytes and metadata."""
    from sparselab.corpus.rights import RightsPolicy, resolve_file_rights

    report = _load(root / "license-report.json")
    if report.get("schema_version") not in (2, 3) or report.get("sources") != list(
        sources.values()
    ):
        raise ValueError("prospective rights report/source mismatch")
    if report["schema_version"] == 3 and report.get("training_use_policy") != (
        "allowed_unless_explicitly_prohibited"
    ):
        raise ValueError("unrecognized prospective training-use policy")
    rows = report["files"]
    indexed = {(row["source_id"], row["path"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate rights file entry")
    expected = {
        (source_id, item["path"])
        for source_id, snapshot in snapshots.items()
        for item in snapshot["files"]
    }
    if set(indexed) != expected:
        raise ValueError("rights file inventory differs from pinned snapshots")
    for source_id, snapshot in snapshots.items():
        source = sources[source_id]
        policy = RightsPolicy.model_validate(source["rights_policy"])
        nested_path = policy.nested_metadata_path
        folder = (
            root.parent.parent
            / "snapshots"
            / source_id
            / source["snapshot_sha256"]
            / "files"
        )
        metadata = (
            {nested_path: _load(_safe(folder, nested_path))} if nested_path else {}
        )
        for file in snapshot["files"]:
            path = file["path"]
            recorded = indexed[source_id, path]
            if any(
                recorded.get(key) != value
                for key, value in (
                    ("sha256", file["sha256"]),
                    ("size", file["size"]),
                    ("canonical_uri", source["canonical_uri"]),
                    ("revision", source["revision"]),
                )
            ):
                raise ValueError("rights source file attribution mismatch")
            if path == nested_path:
                if recorded["role"] != "license_metadata" or "rights" in recorded:
                    raise ValueError("invalid rights metadata role")
                continue
            if recorded["role"] not in {"document", "transform_input"}:
                raise ValueError("unexpected rights file role")
            # The resolver examines only the first 30 lines (and no lines for
            # dataset shards). Never materialize a whole source shard here.
            with _safe(folder, path).open("rb") as stream:
                raw = (
                    b""
                    if path.endswith((".jsonl", ".json", ".parquet"))
                    else b"".join(stream.readline() for _ in range(30))
                )
            resolved = resolve_file_rights(
                policy,
                path,
                raw,
                nested_metadata=metadata,
                prospective_private_research=source.get("explicit_training_restriction")
                == "none_found",
            )
            if recorded["rights"] != resolved.model_dump(mode="json") or (
                recorded.get("license_url") != source["license_url"]
            ):
                raise ValueError("rights decision differs from pinned file evidence")
    return indexed


def _iter_rows(path: Path):
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _streaming_v3(identity: dict[str, Any]) -> bool:
    return (
        identity["release"].get("schema_version") == 3
        and not identity["release"]["chat"]["selected"]
        and all(spec["kind"] == "lm_text" for spec in identity["transforms"])
        and identity["release"].get("fraction") is None
    )


def _validate_rows_v3(
    root: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> None:
    """Validate LM-only source evidence with disk-backed identity joins."""
    from sparselab.corpus.acquisition import verify_snapshot
    from sparselab.corpus.pipeline import _normalized, _origin_keys
    from sparselab.corpus.provenance import shape_for_record, validate_lineage

    source_rows = _load(root / "sources.json")
    sources = {row["id"]: row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("duplicate source identity")
    receipt = _load(
        root / ("build.json" if (root / "build.json").exists() else "manifest.json")
    )
    pinned_snapshots = {row["source_id"]: row["sha256"] for row in receipt["snapshots"]}
    if len(pinned_snapshots) != len(receipt["snapshots"]) or pinned_snapshots != {
        source["id"]: source["snapshot_sha256"]
        for source in source_rows
        if source["snapshot_sha256"]
    }:
        raise ValueError("source snapshot inventory mismatch")
    snapshots = {}
    for source in source_rows:
        snapshot_id = source["snapshot_sha256"]
        if not snapshot_id:
            continue
        snapshot = verify_snapshot(
            root.parent.parent / "snapshots" / source["id"] / snapshot_id,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        snapshots[source["id"]] = snapshot
        declaration = snapshot["declaration"]
        if any(
            source[key] != declaration[key]
            for key in (
                "id",
                "kind",
                "canonical_uri",
                "revision",
                "license",
                "source_family",
            )
        ) or source.get("origin", "primary_source") != declaration.get(
            "origin", "primary_source"
        ):
            raise ValueError("source declaration attribution mismatch")
        if (
            declaration.get("schema_version") != 3
            or source["rights_policy"] != declaration["rights"]
            or source["license_url"] != declaration["license_url"]
            or source["redistribution"] != declaration["rights"]["redistribution_mode"]
            or source.get("explicit_training_restriction")
            != declaration.get("explicit_training_restriction")
        ):
            raise ValueError("prospective source rights declaration mismatch")
    rights = _verify_rights_files(root, sources, snapshots)

    # Keep the keyed index on the corpus filesystem, not a RAM-backed /tmp.
    with tempfile.TemporaryDirectory(
        prefix=".verify-corpus-", dir=root.parent
    ) as temporary:
        db = sqlite3.connect(Path(temporary) / "evidence.sqlite")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-8192")
        try:
            db.execute(
                "CREATE TABLE docs (id TEXT PRIMARY KEY, source TEXT, split TEXT, "
                "representative TEXT, dropped TEXT, raw_path TEXT, line_start INTEGER, "
                "text TEXT, data TEXT)"
            )
            db.execute("CREATE TABLE spans (id TEXT PRIMARY KEY, data TEXT)")
            db.execute("CREATE TABLE lineage (id TEXT PRIMARY KEY)")
            db.execute(
                "CREATE TABLE stage_docs (stage TEXT, id TEXT, PRIMARY KEY (stage, id))"
            )
            db.execute("CREATE TABLE view_docs (id TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE duplicate_keys (key TEXT, id TEXT)")
            for span in _iter_rows(root / "spans.jsonl"):
                try:
                    db.execute(
                        "INSERT INTO spans VALUES (?, ?)",
                        (span["record_id"], json.dumps(span)),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError("duplicate document evidence span") from error
            for doc in _iter_rows(root / "documents.jsonl"):
                text = doc["text"]
                if (
                    doc["content_sha256"]
                    != hashlib.sha256(text.encode("utf-8")).hexdigest()
                ):
                    raise ValueError("document content digest mismatch")
                span_row = db.execute(
                    "SELECT data FROM spans WHERE id=?", (doc["document_id"],)
                ).fetchone()
                if span_row is None:
                    raise ValueError("document evidence span inventory mismatch")
                span = json.loads(span_row[0])
                try:
                    db.execute(
                        "INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            doc["document_id"],
                            doc["source_id"],
                            doc["split"],
                            doc["representative_id"],
                            doc["drop_reason"],
                            span["raw_path"],
                            span["line_start"],
                            text,
                            json.dumps(
                                {
                                    key: value
                                    for key, value in doc.items()
                                    if key != "text"
                                }
                            ),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError("duplicate document identity") from error
                for key in (
                    "raw:" + doc["raw_content_sha256"],
                    "normalized:" + doc["content_sha256"],
                    *("origin:" + value for value in _origin_keys(doc)),
                ):
                    db.execute(
                        "INSERT INTO duplicate_keys VALUES (?, ?)",
                        (key, doc["document_id"]),
                    )
            db.commit()
            db.execute(
                "CREATE INDEX docs_raw_order_idx "
                "ON docs(source, raw_path, line_start, id)"
            )
            if (
                db.execute(
                    "SELECT COUNT(*) FROM docs LEFT JOIN spans ON docs.id=spans.id "
                    "WHERE spans.id IS NULL"
                ).fetchone()[0]
                or db.execute(
                    "SELECT COUNT(*) FROM spans LEFT JOIN docs ON docs.id=spans.id "
                    "WHERE docs.id IS NULL"
                ).fetchone()[0]
            ):
                raise ValueError("document evidence span inventory mismatch")
            if db.execute(
                "SELECT COUNT(*) FROM docs AS doc LEFT JOIN docs AS representative "
                "ON doc.representative=representative.id WHERE representative.id IS NULL "
                "OR representative.representative!=representative.id OR "
                "(doc.id=doc.representative AND doc.dropped IS NOT NULL) OR "
                "(doc.id!=doc.representative AND doc.dropped IS NULL) OR "
                "(doc.dropped NOT IN ('duplicate','contaminated_heldout')) OR "
                "(doc.split!='train' AND representative.split!=doc.split) OR "
                "(doc.dropped='contaminated_heldout' AND "
                "(doc.split!='train' OR representative.split='train')) OR "
                "(doc.dropped='duplicate' AND doc.split='train' "
                "AND representative.split!='train')"
            ).fetchone()[0]:
                raise ValueError("invalid duplicate representative identity")
            db.execute("CREATE INDEX duplicate_key_idx ON duplicate_keys (key)")
            if db.execute(
                "SELECT COUNT(*) FROM (SELECT duplicate_keys.key FROM duplicate_keys "
                "JOIN docs ON docs.id=duplicate_keys.id GROUP BY duplicate_keys.key "
                "HAVING COUNT(DISTINCT docs.representative)!=1)"
            ).fetchone()[0]:
                raise ValueError("duplicate group representative mismatch")

            previous_file = None
            raw_sha = ""
            with ExitStack() as stack:
                raw_stream = None
                current_line = 0
                current_offset = 0
                for doc_json, text, span_json in db.execute(
                    "SELECT docs.data, docs.text, spans.data "
                    "FROM docs INDEXED BY docs_raw_order_idx "
                    "JOIN spans ON docs.id=spans.id "
                    "ORDER BY docs.source, docs.raw_path, docs.line_start, docs.id"
                ):
                    doc, span = json.loads(doc_json), json.loads(span_json)
                    doc["text"] = text
                    source = sources[doc["source_id"]]
                    file = rights.get((doc["source_id"], span["raw_path"]))
                    if file is None or file["role"] != "document":
                        raise ValueError("document lacks pinned file rights")
                    decision = file["rights"]
                    if (
                        source.get("explicit_training_restriction") != "none_found"
                        or doc["source_revision"] != source["revision"]
                        or doc["source_family"] != source["source_family"]
                        or decision["training_eligibility"]
                        not in ("eligible", "eligible_with_obligations")
                        or doc.get("schema_version") != 3
                        or doc.get("rights") != decision
                        or doc.get("file_sha256") != file["sha256"]
                        or doc["license"]
                        != (
                            decision["detected_spdx_expression"]
                            or source["rights_policy"]["spdx_expression"]
                            or source["license"]
                        )
                        or doc["redistribution"] != decision["redistribution_mode"]
                    ):
                        raise ValueError("document rights attribution mismatch")
                    raw_path = _safe(
                        root.parent.parent
                        / "snapshots"
                        / doc["source_id"]
                        / source["snapshot_sha256"]
                        / "files",
                        span["raw_path"],
                    )
                    if raw_path != previous_file:
                        stack.close()
                        raw_stream = stack.enter_context(raw_path.open("rb"))
                        raw_sha = sha256_file(raw_path)
                        current_line, current_offset = 0, 0
                        previous_file = raw_path
                    if (
                        span["source_id"] != doc["source_id"]
                        or span["snapshot_sha256"] != source["snapshot_sha256"]
                        or span["raw_sha256"] != raw_sha
                        or span["raw_content_sha256"] != doc["raw_content_sha256"]
                    ):
                        raise ValueError("document snapshot span mismatch")
                    normalizer = (
                        "cnxml-text-v1"
                        if raw_path.suffix.lower() == ".cnxml"
                        else "normalizer-nfc-markdown-v1"
                    )
                    if (
                        span.get("normalizer") != normalizer
                        or doc["document_id"]
                        != _digest(
                            [
                                source["snapshot_sha256"],
                                doc["source_location"],
                                raw_sha,
                                normalizer,
                            ]
                        )
                        or span.get("section_path") != doc["section_path"]
                    ):
                        raise ValueError("document snapshot identity mismatch")
                    if "#lines=" in doc["source_location"]:
                        start, end = span["line_start"], span["line_end"]
                        if start < 1 or end < start:
                            raise ValueError("document raw line range mismatch")
                        if start <= current_line:
                            raw_stream.seek(0)
                            current_line, current_offset = 0, 0
                        while current_line < start - 1:
                            line = raw_stream.readline()
                            if not line:
                                raise ValueError("document raw line range mismatch")
                            current_line += 1
                            current_offset += len(line)
                        byte_start = current_offset
                        digest = hashlib.sha256()
                        while current_line < end:
                            line = raw_stream.readline()
                            if not line:
                                raise ValueError("document raw line range mismatch")
                            current_line += 1
                            current_offset += len(line)
                            digest.update(line)
                        if (
                            span["byte_start"] != byte_start
                            or span["byte_end"] != current_offset
                            or digest.hexdigest() != doc["raw_content_sha256"]
                        ):
                            raise ValueError("document raw byte range mismatch")
                        if doc["source_location"] != (
                            f"{span['raw_path']}#lines={start}-{end}"
                        ):
                            raise ValueError("document source location mismatch")
                    elif (
                        "#row=" in doc["source_location"]
                        and raw_path.suffix == ".jsonl"
                    ):
                        index = span["line_start"]
                        if (
                            index < 1
                            or span["line_end"] != index
                            or doc["source_location"]
                            != f"{span['raw_path']}#row={index}"
                            or span["byte_start"] is not None
                            or span["byte_end"] is not None
                        ):
                            raise ValueError("document raw row range mismatch")
                        if index <= current_line:
                            raw_stream.seek(0)
                            current_line = 0
                        while current_line < index:
                            line = raw_stream.readline()
                            if not line:
                                raise ValueError("document raw row range mismatch")
                            if line.strip():
                                current_line += 1
                        item = json.loads(line)
                        if (
                            not isinstance(item, dict)
                            or not isinstance(
                                item.get(
                                    snapshots[doc["source_id"]]["declaration"][
                                        "acquisition"
                                    ]["text_field"]
                                ),
                                str,
                            )
                            or _normalized(
                                item[
                                    snapshots[doc["source_id"]]["declaration"][
                                        "acquisition"
                                    ]["text_field"]
                                ]
                            )
                            != doc["text"]
                            or doc["raw_content_sha256"] != doc["content_sha256"]
                        ):
                            raise ValueError("document raw row content mismatch")
                    else:
                        raise ValueError("unsupported document snapshot span")

            for row in _iter_rows(root / "lineage.jsonl"):
                record_id = row["record_id"]
                doc_record = db.execute(
                    "SELECT data, text FROM docs WHERE id=?", (record_id,)
                ).fetchone()
                if doc_record is None or row["record_kind"] != "document":
                    raise ValueError("unexpected LM-only lineage record")
                doc = json.loads(doc_record[0])
                doc["text"] = doc_record[1]
                validate_lineage(row)
                if (
                    row["parent_document_ids"] != [record_id]
                    or row["original_parent_document_ids"] != [record_id]
                    or row["representative_parent_document_ids"]
                    != [doc["representative_id"]]
                    or row["representative_id"] != doc["representative_id"]
                    or row["drop_reason"] != doc["drop_reason"]
                    or row["source_family_ids"] != [doc["source_family"]]
                    or row["split"] != doc["split"]
                    or row["origin"]
                    != sources[doc["source_id"]].get("origin", "primary_source")
                    or row["modalities"] != [doc["modality"]]
                    or row["domains"] != doc["domains"]
                    or row["shape"]
                    != shape_for_record(
                        "document",
                        "raw_document",
                        domains=doc["domains"],
                        parent_document_ids=[record_id],
                    )
                    or row["verification"]["status"] != "schema_validated"
                    or row["verification"]["evidence"].get("schema_id")
                    != "normalized_document_v1"
                    or row["rendered_sha256"] != doc["content_sha256"]
                ):
                    raise ValueError("document lineage mismatch")
                try:
                    db.execute(
                        "INSERT INTO lineage VALUES (?)",
                        (record_id,),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError("duplicate lineage ID") from error
            db.commit()
            if db.execute(
                "SELECT COUNT(*) FROM docs LEFT JOIN lineage ON docs.id=lineage.id "
                "WHERE lineage.id IS NULL"
            ).fetchone()[0]:
                raise ValueError("document lineage mismatch")
            receipt = _load(
                root
                / ("build.json" if (root / "build.json").exists() else "manifest.json")
            )
            identity = receipt.get("identity", receipt.get("build_identity"))
            transforms = identity["transforms"]
            for stage in receipt["stages"]:
                transform = next(
                    spec for spec in transforms if spec["id"] == stage["id"]
                )
                selected_sources = set(transform["inputs"]) & set(sources)
                for row in _iter_rows(root / "stages" / f"{stage['id']}.jsonl"):
                    doc = db.execute(
                        "SELECT text, split, source, dropped FROM docs WHERE id=?",
                        (row["record_id"],),
                    ).fetchone()
                    if (
                        doc is None
                        or set(row) != {"record_id", "text", "split"}
                        or row["text"] != doc[0]
                        or row["split"] != doc[1]
                        or (selected_sources and doc[2] not in selected_sources)
                        or doc[3]
                    ):
                        raise ValueError(
                            "LM stage record differs from selected document"
                        )
                    try:
                        db.execute(
                            "INSERT INTO stage_docs VALUES (?, ?)",
                            (stage["id"], row["record_id"]),
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError("duplicate LM stage identity") from error
                count = db.execute(
                    "SELECT COUNT(*) FROM docs WHERE dropped IS NULL AND "
                    "(? = 1 OR source IN (SELECT value FROM json_each(?))) "
                    "AND id NOT IN (SELECT id FROM stage_docs WHERE stage=?)",
                    (
                        not bool(selected_sources),
                        json.dumps(sorted(selected_sources)),
                        stage["id"],
                    ),
                ).fetchone()[0]
                if count:
                    raise ValueError("LM stage omitted selected documents")
            db.commit()
            release = identity["release"]
            allowed_sources = set().union(
                *(
                    set(spec["inputs"]) & set(sources)
                    if set(spec["inputs"]) & set(sources)
                    else set(sources)
                    for spec in transforms
                )
            )
            shapes = release.get("include_shapes")
            origins = release.get("include_origins")
            eligible = (shapes is None or "raw_document" in shapes) and (
                origins is None
                or any(
                    source.get("origin", "primary_source") in origins
                    for source in sources.values()
                )
            )
            for split in ("train", "validation", "test"):
                with (
                    (root / "lm" / f"{split}.jsonl").open("rb") as view,
                    (root / "lm" / f"{split}.lineage.jsonl").open("rb") as links,
                ):
                    for payload, link in zip_longest(view, links):
                        if payload is None or link is None:
                            raise ValueError("LM lineage count mismatch")
                        row, ref = json.loads(payload), json.loads(link)
                        if (
                            set(row) != {"text"}
                            or not isinstance(row["text"], str)
                            or not row["text"].strip()
                            or ref.get("split") != split
                        ):
                            raise ValueError("invalid LM record or lineage")
                        found = db.execute(
                            "SELECT text, split, dropped FROM docs WHERE id=?",
                            (ref["record_id"],),
                        ).fetchone()
                        if found is None or found != (row["text"], split, None):
                            raise ValueError("invalid LM record or lineage")
                        source = db.execute(
                            "SELECT source FROM docs WHERE id=?", (ref["record_id"],)
                        ).fetchone()[0]
                        if (
                            not eligible
                            or source not in allowed_sources
                            or origins is not None
                            and sources[source].get("origin", "primary_source")
                            not in origins
                        ):
                            raise ValueError("LM view includes an unselected document")
                        try:
                            db.execute(
                                "INSERT INTO view_docs VALUES (?)", (ref["record_id"],)
                            )
                        except sqlite3.IntegrityError as error:
                            raise ValueError("duplicate LM view identity") from error
                for name in (f"chat/{split}.jsonl", f"chat/{split}.lineage.jsonl"):
                    if (root / name).stat().st_size:
                        raise ValueError("unexpected chat records in LM-only release")
            if eligible:
                for source_id in allowed_sources:
                    if (
                        origins is not None
                        and sources[source_id].get("origin", "primary_source")
                        not in origins
                    ):
                        continue
                    if db.execute(
                        "SELECT COUNT(*) FROM docs WHERE source=? AND dropped IS NULL "
                        "AND id NOT IN (SELECT id FROM view_docs)",
                        (source_id,),
                    ).fetchone()[0]:
                        raise ValueError("LM view omitted selected documents")
            for name in (
                "chat/records.jsonl",
                "scenarios.jsonl",
                "generations.jsonl",
                "lexical/candidates.jsonl",
                "semantic/candidates.jsonl",
                "tool_episodes.jsonl",
            ):
                if (root / name).stat().st_size:
                    raise ValueError("unexpected generated records in LM-only release")
        finally:
            db.close()


def _validate_rows(
    root: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> None:
    documents = _rows(root / "documents.jsonl")
    document_map = {doc["document_id"]: doc for doc in documents}
    if len(document_map) != len(documents):
        raise ValueError("duplicate document identity")
    for doc in documents:
        if doc["content_sha256"] != hashlib.sha256(doc["text"].encode()).hexdigest():
            raise ValueError("document content digest mismatch")
    spans = {item["record_id"]: item for item in _rows(root / "spans.jsonl")}
    sources = {item["id"]: item for item in _load(root / "sources.json")}
    from sparselab.corpus.acquisition import verify_snapshot

    source_origins = {
        key: value.get("origin", "primary_source") for key, value in sources.items()
    }

    snapshots: dict[str, dict[str, Any]] = {}
    for source in sources.values():
        snapshot_id = source["snapshot_sha256"]
        if not snapshot_id:
            continue
        snapshot = verify_snapshot(
            root.parent.parent / "snapshots" / source["id"] / snapshot_id,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        snapshots[source["id"]] = snapshot
        declaration = snapshot["declaration"]
        if any(
            source[key] != declaration[key]
            for key in (
                "id",
                "kind",
                "canonical_uri",
                "revision",
                "license",
                "source_family",
            )
        ) or source.get("origin", "primary_source") != declaration.get(
            "origin", "primary_source"
        ):
            raise ValueError("source declaration attribution mismatch")
        if "rights_policy" in source:
            if (
                declaration.get("schema_version") not in (2, 3)
                or source["rights_policy"] != declaration["rights"]
                or source["license_url"] != declaration["license_url"]
                or source["redistribution"]
                != declaration["rights"]["redistribution_mode"]
            ):
                raise ValueError("prospective source rights declaration mismatch")
            if declaration["schema_version"] == 3 and (
                source.get("explicit_training_restriction")
                != declaration.get("explicit_training_restriction")
            ):
                raise ValueError(
                    "source training restriction state differs from declaration"
                )
        elif source["redistribution"] != declaration["redistribution"]:
            raise ValueError("source declaration attribution mismatch")
    prospective = any("rights_policy" in source for source in sources.values())
    rights_files = _verify_rights_files(root, sources, snapshots) if prospective else {}
    if set(spans) != set(document_map):
        raise ValueError("document evidence span inventory mismatch")
    previous_file: Path | None = None
    raw = b""
    raw_sha = ""
    offsets: list[int] = []
    for doc in sorted(
        documents,
        key=lambda item: (item["source_id"], spans[item["document_id"]]["raw_path"]),
    ):
        span = spans[doc["document_id"]]
        source = sources[doc["source_id"]]
        if (
            doc["source_revision"] != source["revision"]
            or doc["source_family"] != source["source_family"]
        ):
            raise ValueError("document attribution mismatch")
        if prospective:
            file = rights_files.get((doc["source_id"], span["raw_path"]))
            if file is None or file["role"] != "document":
                raise ValueError("document lacks pinned file rights")
            decision = file["rights"]
            if (
                source.get("explicit_training_restriction", "none_found")
                != "none_found"
            ):
                raise ValueError("document source has unresolved training restriction")
            if (
                decision["training_eligibility"]
                not in ("eligible", "eligible_with_obligations")
                or doc.get("schema_version")
                != (3 if "explicit_training_restriction" in source else 2)
                or doc.get("rights") != decision
                or doc.get("file_sha256") != file["sha256"]
                or doc["license"]
                != (
                    decision["detected_spdx_expression"]
                    or source["rights_policy"]["spdx_expression"]
                    or source["license"]
                )
                or doc["redistribution"] != decision["redistribution_mode"]
            ):
                raise ValueError("document rights attribution mismatch")
        elif (
            doc["license"] != source["license"]
            or doc["redistribution"] != source["redistribution"]
        ):
            raise ValueError("document attribution mismatch")
        snapshot = (
            root.parent.parent
            / "snapshots"
            / doc["source_id"]
            / source["snapshot_sha256"]
        )
        raw_path = _safe(snapshot / "files", span["raw_path"])
        if raw_path != previous_file:
            raw = raw_path.read_bytes()
            raw_sha = hashlib.sha256(raw).hexdigest()
            offsets = [0]
            if "#lines=" in doc["source_location"]:
                for line in raw.decode("utf-8").splitlines(keepends=True):
                    offsets.append(offsets[-1] + len(line.encode("utf-8")))
            previous_file = raw_path
        if (
            span["source_id"] != doc["source_id"]
            or span["snapshot_sha256"] != source["snapshot_sha256"]
            or raw_sha != span["raw_sha256"]
            or span["raw_content_sha256"] != doc["raw_content_sha256"]
        ):
            raise ValueError("document snapshot span mismatch")
        if "#lines=" in doc["source_location"]:
            start, end = span["line_start"], span["line_end"]
            if start < 1 or end < start or end >= len(offsets):
                raise ValueError("document raw line range mismatch")
            byte_start, byte_end = span["byte_start"], span["byte_end"]
            if (
                byte_start != offsets[start - 1]
                or byte_end != offsets[end]
                or hashlib.sha256(raw[byte_start:byte_end]).hexdigest()
                != doc["raw_content_sha256"]
            ):
                raise ValueError("document raw byte range mismatch")
    lineages = _rows(root / "lineage.jsonl")
    lineage_map = {row["record_id"]: row for row in lineages}
    if len(lineage_map) != len(lineages):
        raise ValueError("duplicate lineage ID")
    classified = any("origin" in row for row in lineages)
    if classified:
        from sparselab.corpus.pipeline import _path_scenario, _scenario, _scenario_shape
        from sparselab.corpus.provenance import (
            rendered_digest,
            shape_for_record,
            validate_lineage,
        )
        from sparselab.data.conversations import _v2_document

        receipt = _load(
            root / ("build.json" if (root / "build.json").exists() else "manifest.json")
        )
        stage_rows = {
            (stage["id"], record.get("record_id", record.get("scenario_id"))): record
            for stage in receipt["stages"]
            for record in _rows(root / "stages" / f"{stage['id']}.jsonl")
            if record.get("record_id") or record.get("scenario_id")
        }
        chats = {
            record["record_id"]: record for record in _rows(root / "chat/records.jsonl")
        }
        scenarios = {
            record["scenario_id"]: record for record in _rows(root / "scenarios.jsonl")
        }
        generations = {
            record["record_id"]: record for record in _rows(root / "generations.jsonl")
        }
        semantic = {
            record["record_id"]: record
            for record in _rows(root / "semantic/candidates.jsonl")
        }
        for row in lineages:
            validate_lineage(row, documents=document_map)
            expected_origin = (
                source_origins[document_map[row["record_id"]]["source_id"]]
                if row["record_kind"] == "document"
                else "deterministic_synthetic"
                if row["record_kind"] in {"scenario", "tool_episode"}
                or row["record_kind"] == "chat_sft"
                and row.get("scenario_id")
                else "multi_source_synthetic"
                if len(
                    {
                        document_map[parent]["source_id"]
                        for parent in row["parent_document_ids"]
                    }
                )
                > 1
                else "free_generated_synthetic"
                if row["record_kind"] == "generation"
                and not row["parent_document_ids"]
                or row["record_kind"] == "chat_sft"
                and row.get("generation_id")
                and not row["parent_document_ids"]
                else "source_transformed_synthetic"
            )
            if row["origin"] != expected_origin:
                raise ValueError("lineage origin disagrees with source and transform")
            expected_domains = (
                sorted(document_map[row["record_id"]]["domains"])
                if row["record_kind"] == "document"
                else ["systems_scenarios"]
                if row.get("generator_world_id")
                else sorted(
                    {
                        domain
                        for parent in row["parent_document_ids"]
                        for domain in document_map[parent]["domains"]
                    }
                )
            )
            if (
                sorted(row.get("domains", [])) != expected_domains
                or row["shape"]["attributes"]["source_domains"] != expected_domains
            ):
                raise ValueError("shape source domain attribution mismatch")
            evidence = row["verification"]["evidence"]
            if row["verification"]["status"] == "oracle_verified":
                scenario = scenarios.get(row.get("scenario_id", row["record_id"]))
                if scenario is None:
                    raise ValueError("oracle scenario reference mismatch")
                world = scenario["world_state"]
                generator_id = scenario["generator_id"]
                if generator_id == "pathlib_path_suffix_v1":
                    expected_scenario = _path_scenario(
                        scenario["world_seed"],
                        scenario["scenario_family_id"],
                        scenario["transform_id"],
                    )
                    expected_result = PurePosixPath(world["path"]).suffix
                else:
                    expected_scenario = _scenario(
                        generator_id,
                        scenario["world_seed"],
                        scenario["scenario_family_id"],
                        scenario["template_family_id"],
                        scenario["transform_id"],
                    )
                    expected_result = expected_scenario["oracle_answer"]
                implementation = receipt.get("identity", receipt.get("build_identity"))[
                    "implementation_sha256"
                ]
                if (
                    evidence["oracle_answer"] != scenario["oracle_answer"]
                    or evidence["generator_world_id"] != scenario["generator_world_id"]
                    or evidence["oracle_identity"]
                    != _digest([world, scenario["oracle_answer"]])
                    or evidence["world_state"] != world
                    or evidence["oracle_version"] != scenario["generator_version"]
                    or evidence["oracle_implementation"] != implementation
                    or evidence["interpreter"] != scenario["interpreter"]
                    or evidence["actual_result"] != expected_result
                    or evidence["comparison_status"] != "match"
                    or expected_result != scenario["oracle_answer"]
                    or expected_scenario
                    != {key: value for key, value in scenario.items() if key != "split"}
                    or (
                        evidence.get("receipt") != scenario.get("oracle_receipt")
                        if generator_id != "pathlib_path_suffix_v1"
                        else "receipt" in evidence
                    )
                ):
                    raise ValueError("oracle verification evidence mismatch")
            kind = row["record_kind"]
            shape_id = row["shape"]["id"]
            attributes = row["shape"]["attributes"]
            if kind == "document" and (
                shape_id != "raw_document"
                or row["shape"]
                != shape_for_record(
                    kind,
                    shape_id,
                    domains=row["domains"],
                    parent_document_ids=row["parent_document_ids"],
                )
            ):
                raise ValueError("document shape mismatch")
            if kind == "document":
                if (
                    row["verification"]["status"] != "schema_validated"
                    or evidence.get("schema_id") != "normalized_document_v1"
                ):
                    raise ValueError("document verification classification mismatch")
                continue
            if kind in {"chat_sft", "tool_episode"}:
                record = chats.get(row["record_id"])
                payload = (
                    {
                        key: record[key]
                        for key in ("format_version", "loss_mode", "messages")
                    }
                    if record is not None
                    else None
                )
            else:
                payload = stage_rows.get((row["transform_id"], row["record_id"]))
            source_payload = record if kind in {"chat_sft", "tool_episode"} else payload
            expected_shape = {
                "lexical_candidate": "lexical_inventory",
                "semantic_candidate": "definition",
                "scenario": _scenario_shape(source_payload, "scenario")
                if source_payload is not None
                else None,
                "generation": (source_payload.get("parsed_output") or {}).get(
                    "shape", "direct_qa"
                )
                if source_payload is not None
                else None,
                "tool_episode": "tool_trace",
                "chat_sft": (
                    source_payload.get("semantic_shape")
                    or (
                        "troubleshooting_scenario"
                        if source_payload.get("scenario_id")
                        else "direct_qa"
                    )
                )
                if source_payload is not None
                else None,
            }.get(kind)
            if (
                shape_id != expected_shape
                or attributes["task_family"] != expected_shape
            ):
                raise ValueError("lineage shape classification mismatch")
            if row["shape"] != shape_for_record(
                kind,
                shape_id,
                domains=row["domains"],
                parent_document_ids=row["parent_document_ids"],
                scenario_id=row.get("scenario_id"),
            ):
                raise ValueError("lineage shape attributes mismatch")
            if source_payload is not None:
                evidence_payload = (
                    generations.get(source_payload["generation_id"])
                    if source_payload.get("generation_id")
                    else semantic.get(source_payload["semantic_id"])
                    if source_payload.get("semantic_id")
                    else source_payload
                )
                if (
                    kind == "generation"
                    or kind == "chat_sft"
                    and source_payload.get("generation_id")
                ) and row.get("generator") != evidence_payload.get("generator"):
                    raise ValueError("lineage generator metadata mismatch")
                if (
                    kind != "generation"
                    and not (kind == "chat_sft" and source_payload.get("generation_id"))
                    and "generator" in row
                ):
                    raise ValueError("unexpected lineage generator metadata")
                expected_status = (
                    "oracle_verified"
                    if kind == "scenario" or source_payload.get("scenario_id")
                    else evidence_payload.get("validation_status", "schema_validated")
                )
                if (
                    row["verification"]["status"] != expected_status
                    or row.get("validation_status") != expected_status
                ):
                    raise ValueError(
                        "lineage verification differs from recorded status"
                    )
                if (
                    expected_status == "schema_validated"
                    and evidence.get("schema_id") != f"{kind}_v1"
                ):
                    raise ValueError("lineage schema evidence mismatch")
            if row["verification"]["status"] == "source_entailed":
                expected_answer = (
                    source_payload["messages"][-1]["content"]
                    if kind == "chat_sft"
                    else source_payload["parsed_output"]["answer"]
                    if kind == "generation"
                    else source_payload["value"]
                )
                if evidence.get("answer") != expected_answer:
                    raise ValueError(
                        "source-entailed answer differs from rendered record"
                    )
            if payload is None:
                raise ValueError("lineage payload missing")
            if kind in {"chat_sft", "tool_episode"}:
                actual_rendered = rendered_digest(
                    _v2_document(payload, Path("<corpus-record>"), 1).text, text=True
                )
            else:
                actual_rendered = rendered_digest(payload)
            if actual_rendered != row["rendered_sha256"]:
                raise ValueError("lineage rendered digest mismatch")
            if kind == "generation":
                generation = generations.get(row["record_id"])
                if row["verification"]["status"] == "source_entailed" and (
                    generation["parsed_output"] is None
                    or generation["parsed_output"]["answer"] not in evidence["passage"]
                    or generation["parsed_output"]["citation_id"]
                    != evidence["document_id"]
                ):
                    raise ValueError("generation answer not entailed by citation")
                if generation != payload or _digest(generation["generator"]) != row.get(
                    "generator_identity"
                ):
                    raise ValueError("generation identity mismatch")
            elif kind == "scenario":
                scenario = scenarios.get(row["record_id"])
                if scenario != payload or _digest(
                    [scenario["generator_id"], scenario["generator_version"]]
                ) != row.get("generator_identity"):
                    raise ValueError("scenario generator identity mismatch")
            elif kind in {"chat_sft", "tool_episode"}:
                if generation_id := row.get("generation_id"):
                    source = generations.get(generation_id)
                    if source is None or source["generator_identity"] != row.get(
                        "generator_identity"
                    ):
                        raise ValueError("chat generation identity mismatch")
                    if source.get("parsed_output") is None or row["verification"][
                        "status"
                    ] != source["validation_status"].replace(
                        "source_grounded", "source_entailed"
                    ):
                        raise ValueError("chat verification differs from generation")
                elif scenario_id := row.get("scenario_id"):
                    source = scenarios.get(scenario_id)
                    if source is None or _digest(
                        [source["generator_id"], source["generator_version"]]
                    ) != row.get("generator_identity"):
                        raise ValueError("chat scenario identity mismatch")
                elif semantic_id := row.get("semantic_id"):
                    fact = semantic.get(semantic_id)
                    if (
                        fact is None
                        or record.get("semantic_id") != semantic_id
                        or row.get("generator_identity")
                        != _digest(
                            [
                                "semantic_chat_v1",
                                row["transform_id"],
                                record["semantic_shape"],
                            ]
                        )
                        or evidence["document_id"] != fact["evidence_document_id"]
                        or evidence["passage"] != fact["evidence_passage"]
                        or evidence["span"] != fact["evidence_span"]
                        or record["messages"][-1]["content"]
                        not in fact["evidence_passage"]
                    ):
                        raise ValueError("semantic chat generator/evidence mismatch")
                else:
                    raise ValueError("chat missing generator reference")
    for row in lineages:
        parents = row["parent_document_ids"]
        if not set(parents).issubset(document_map) or (
            row["split"] not in {"train", "validation", "test"}
            and not (
                row["record_kind"] == "lexical_candidate"
                and row["split"] == "inventory"
            )
        ):
            raise ValueError("dangling lineage or invalid split")
        if not set(row["representative_parent_document_ids"]).issubset(document_map):
            raise ValueError("invalid representative lineage")
        if row["split"] == "inventory":
            if len({document_map[parent]["split"] for parent in parents}) < 2:
                raise ValueError("unnecessary cross-split lexical inventory")
        elif any(document_map[parent]["split"] != row["split"] for parent in parents):
            raise ValueError("cross-split lineage")
        if row["representative_parent_document_ids"] != sorted(
            {document_map[parent]["representative_id"] for parent in parents}
        ):
            raise ValueError("incorrect duplicate representative lineage")
    for doc in documents:
        row = lineage_map.get(doc["document_id"])
        if (
            row is None
            or row["record_kind"] != "document"
            or row["split"] != doc["split"]
        ):
            raise ValueError("document lineage mismatch")
    chats = {row["record_id"]: row for row in _rows(root / "chat/records.jsonl")}
    for split in ("train", "validation", "test"):
        lm = _rows(root / "lm" / f"{split}.jsonl")
        lm_lineage = _rows(root / "lm" / f"{split}.lineage.jsonl")
        if len(lm) != len(lm_lineage):
            raise ValueError("LM lineage count mismatch")
        for row, link in zip(lm, lm_lineage, strict=True):
            doc = document_map[link["record_id"]]
            if classified and (
                link["split"] != split
                or lineage_map[link["record_id"]]["record_kind"] != "document"
                or lineage_map[link["record_id"]]["rendered_sha256"]
                != hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            ):
                raise ValueError("LM selected-view lineage mismatch")
            if (
                set(row) != {"text"}
                or not isinstance(row["text"], str)
                or not row["text"].strip()
                or doc["text"] != row["text"]
                or doc["split"] != split
                or doc["drop_reason"]
            ):
                raise ValueError("invalid LM record or lineage")
        chat = _rows(root / "chat" / f"{split}.jsonl")
        chat_lineage = _rows(root / "chat" / f"{split}.lineage.jsonl")
        if len(chat) != len(chat_lineage):
            raise ValueError("chat lineage count mismatch")
        for row, link in zip(chat, chat_lineage, strict=True):
            record = chats[link["record_id"]]
            if classified and (
                link["split"] != split
                or lineage_map[link["record_id"]]["record_kind"]
                not in {"chat_sft", "tool_episode"}
                or lineage_map[link["record_id"]]["rendered_sha256"]
                != hashlib.sha256(
                    _v2_document(row, Path("<corpus-record>"), 1).text.encode("utf-8")
                ).hexdigest()
            ):
                raise ValueError("chat selected-view lineage mismatch")
            if (
                record["split"] != split
                or lineage_map[link["record_id"]]["split"] != split
                or row
                != {
                    key: record[key]
                    for key in ("format_version", "loss_mode", "messages")
                }
            ):
                raise ValueError("invalid chat record lineage")
        from sparselab.data.conversations import iter_rendered_conversations

        list(iter_rendered_conversations(root / "chat" / f"{split}.jsonl"))


def _verify_stages(
    root: Path, stages: list[dict[str, Any]], identity: dict[str, Any]
) -> None:
    declarations = {spec["id"]: spec for spec in identity["transforms"]}
    if len(stages) != len(declarations) or {stage["id"] for stage in stages} != set(
        declarations
    ):
        raise ValueError("stage receipt inventory mismatch")
    for stage in stages:
        declaration = declarations[stage["id"]]
        if (
            stage["kind"] != declaration["kind"]
            or stage["version"] != declaration["version"]
            or stage["parameters_sha256"] != _digest(declaration["parameters"])
            or stage["inputs"] != declaration["inputs"]
            or stage["implementation_sha256"] != identity["implementation_sha256"]
        ):
            raise ValueError("stage declaration identity mismatch")
        output = _safe(root, f"stages/{stage['id']}.jsonl")
        if _streaming_v3(identity):
            digest = hashlib.sha256()
            digest.update(canonical_json([stage["id"]])[:-1])
            digest.update(b",[")
            count = 0
            for index, row in enumerate(_iter_rows(output)):
                if index:
                    digest.update(b",")
                digest.update(canonical_json(row))
                count = index + 1
            digest.update(b"]]")
            output_id = digest.hexdigest()
        else:
            rows = _rows(output)
            count = len(rows)
            output_id = _digest([stage["id"], rows])
        if (
            sha256_file(output) != stage["output_sha256"]
            or count != stage["output_count"]
            or output_id != stage["output_id"]
        ):
            raise ValueError("stage receipt does not match output")


def verify_build(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    _domain_cold: bool = False,
) -> dict[str, Any]:
    """Verify staged build and its pinned snapshots."""
    path = Path(path).resolve()
    with _verification_operation():
        if (
            not _domain_cold
            and proof_store is not None
            and verification_mode == "verified_reuse"
        ):
            from sparselab.experiments.artifacts import verify_artifact
            from sparselab.experiments.plan import Artifact

            manifest = _load(path / "build.json")
            verify_artifact(
                Artifact(
                    kind="corpus_build",
                    version=manifest["schema_version"],
                    producer="sparselab",
                    identifier=manifest["build_id"],
                    sha256=manifest["build_id"],
                    path=str(path),
                ),
                path / "build.json",
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            return manifest
        return _verify_build_cold(
            path,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )


def _verify_build_cold(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Verify all staged outputs and their claimed immutable input snapshots."""
    path = Path(path).resolve()
    manifest = _load(path / "build.json")
    if (
        manifest["schema_version"] != 1
        or manifest["build_id"] != path.name
        or _digest(manifest["identity"]) != path.name
    ):
        raise ValueError("invalid build identity")
    _files(
        path,
        {
            **manifest["files"],
            "build.json": {
                "sha256": sha256_file(path / "build.json"),
                "size": (path / "build.json").stat().st_size,
            },
        },
    )
    _verify_stages(path, manifest["stages"], manifest["identity"])
    from sparselab.corpus.acquisition import verify_snapshot

    for snapshot in manifest["snapshots"]:
        receipt = verify_snapshot(
            path.parent.parent
            / "snapshots"
            / snapshot["source_id"]
            / snapshot["sha256"],
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if receipt["snapshot_sha256"] != snapshot["sha256"]:
            raise ValueError("build snapshot identity mismatch")
    (_validate_rows_v3 if _streaming_v3(manifest["identity"]) else _validate_rows)(
        path,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    return manifest


def _manifest_payload(build: dict[str, Any]) -> dict[str, Any]:
    files = {key: value for key, value in build["files"].items() if key != "build.json"}
    return {
        "schema_version": 1,
        "corpus_id": build["identity"]["project_id"],
        "build_id": build["build_id"],
        "build_identity": build["identity"],
        "stages": build["stages"],
        "snapshots": build["snapshots"],
        "files": files,
    }


def freeze(
    build_dir: Path,
    work_root: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> Path:
    """Publish a verified build at its content-derived full SHA-256, never replacing it."""
    build_dir = Path(build_dir).resolve()
    build = verify_build(
        build_dir, proof_store=proof_store, verification_mode=verification_mode
    )
    release_config = build["identity"]["release"]
    for view in ("lm", "chat"):
        selected = release_config[view]
        if not selected.get("selected", False):
            continue
        for split in selected.get("training_splits", ("train", "validation")):
            if split not in {"train", "validation"}:
                raise ValueError("test split cannot be a training view")
            with (build_dir / view / f"{split}.jsonl").open("rb") as stream:
                if not any(
                    chunk.strip() for chunk in iter(lambda: stream.read(64 * 1024), b"")
                ):
                    raise ValueError(f"selected {view}/{split} training view is empty")
    payload = _manifest_payload(build)
    release_id = _digest(payload)
    destination = (
        Path(work_root) / "corpora" / payload["corpus_id"] / "releases" / release_id
    )
    if destination.exists():
        verify_release(
            destination, proof_store=proof_store, verification_mode=verification_mode
        )
        if _load(destination / "manifest.json") != {
            **payload,
            "release_id": release_id,
        }:
            raise ValueError("existing release has different identity")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".release-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        for name in payload["files"]:
            source = _safe(build_dir, name)
            target = _safe(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (staging / "manifest.json").write_bytes(
            canonical_json({**payload, "release_id": release_id}) + b"\n"
        )
        verify_release(
            staging,
            expected_id=release_id,
            proof_store=proof_store,
            verification_mode=verification_mode,
            _domain_cold=True,
        )
        from sparselab.corpus.acquisition import _sync_dir
        from sparselab.engram.packs import _rename_noreplace

        for entry in sorted(staging.rglob("*"), reverse=True):
            if entry.is_file():
                with entry.open("rb") as handle:
                    os.fsync(handle.fileno())
            elif entry.is_dir():
                _sync_dir(entry)
        _sync_dir(staging)
        try:
            _rename_noreplace(staging, destination)
            _sync_dir(destination.parent)
        except FileExistsError:
            verify_release(
                destination,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
    return destination


def verify_release(
    path: Path,
    *,
    expected_id: str | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    _domain_cold: bool = False,
) -> dict[str, Any]:
    """Verify a release, optionally reusing a signed typed artifact receipt."""
    path = Path(path).resolve()
    with _verification_operation():
        if (
            not _domain_cold
            and proof_store is not None
            and verification_mode == "verified_reuse"
        ):
            from sparselab.experiments.artifacts import verify_artifact
            from sparselab.experiments.plan import Artifact

            manifest = _load(path / "manifest.json")
            if manifest["release_id"] != (expected_id or path.name):
                raise ValueError("release directory identity mismatch")
            verify_artifact(
                Artifact(
                    kind="corpus_release",
                    version=manifest["schema_version"],
                    producer="sparselab",
                    identifier=manifest["release_id"],
                    sha256=manifest["release_id"],
                    path=str(path),
                ),
                path / "manifest.json",
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            return manifest
        return _verify_release_cold(
            path,
            expected_id=expected_id,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )


def _release_closure_key(path: Path, manifest: dict[str, Any]) -> tuple[object, ...]:
    """Use the artifact verifier's typed inventory and external snapshot binding."""
    from sparselab.experiments.artifacts import _artifact_key
    from sparselab.experiments.plan import Artifact

    release_id = manifest["release_id"]
    return _artifact_key(
        Artifact(
            kind="corpus_release",
            version=manifest["schema_version"],
            producer="sparselab",
            identifier=release_id,
            sha256=release_id,
            path=str(path),
        ),
        path,
    )


def _verify_release_cold(
    path: Path,
    *,
    expected_id: str | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Verify the complete release including snapshotted source and lineage evidence."""
    path = Path(path).resolve()
    proofs = _verified_releases.get()
    if proofs is not None and path in proofs:
        proof = proofs[path]
        if proof.closure_key != _release_closure_key(
            path, proof.manifest
        ) or proof.manifest["release_id"] != (expected_id or path.name):
            raise ValueError("release changed within verification operation")
        return proof.manifest
    manifest = _load(path / "manifest.json")
    release_id = manifest.get("release_id")
    if release_id != (expected_id or path.name) or len(release_id) != 64:
        raise ValueError("release directory identity mismatch")
    if (
        _digest({key: value for key, value in manifest.items() if key != "release_id"})
        != release_id
    ):
        raise ValueError("release manifest identity mismatch")
    closure_key = _release_closure_key(path, manifest) if proofs is not None else None
    _files(
        path,
        {
            **manifest["files"],
            "manifest.json": {
                "sha256": sha256_file(path / "manifest.json"),
                "size": (path / "manifest.json").stat().st_size,
            },
        },
    )
    _verify_stages(path, manifest["stages"], manifest["build_identity"])
    from sparselab.corpus.acquisition import verify_snapshot

    for snapshot in manifest["snapshots"]:
        receipt = verify_snapshot(
            path.parent.parent
            / "snapshots"
            / snapshot["source_id"]
            / snapshot["sha256"],
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if receipt["snapshot_sha256"] != snapshot["sha256"]:
            raise ValueError("release snapshot identity mismatch")
    (
        _validate_rows_v3
        if _streaming_v3(manifest["build_identity"])
        else _validate_rows
    )(path, proof_store=proof_store, verification_mode=verification_mode)
    if proofs is not None:
        if closure_key != _release_closure_key(path, manifest):
            raise ValueError("release changed during verification operation")
        proofs[path] = _VerifiedRelease(manifest, closure_key)
    return manifest


def describe(path: Path, *, tokenizer: Path | None = None) -> dict[str, Any]:
    manifest = verify_release(path)
    report = _load(Path(path) / "report.json")
    from sparselab.corpus.measurement import measure_views

    measured = measure_views(
        Path(path), tokenizer, release_spec=manifest["build_identity"]["release"]
    )
    if report.get("schema_version") in (2, 3) and tokenizer is not None:
        from sparselab.corpus.measurement import measure_source_rights

        counted = measure_source_rights(Path(path), tokenizer)
        rights = report["rights"]
        report = {
            **report,
            "rights": {
                **rights,
                "training_eligibility": {
                    state: {
                        **info,
                        "training_documents": counted[state]["documents"],
                        "source_tokens": counted[state]["source_tokens"],
                        "token_count_reason": (
                            None
                            if counted[state]["source_tokens"] is not None
                            else "not_eligible_for_training"
                        ),
                    }
                    for state, info in rights["training_eligibility"].items()
                },
            },
        }
    result = {
        "release_id": manifest["release_id"],
        "report": report,
        "measurement": measured,
    }
    if tokenizer is not None:
        result["tokenizer_sha256"] = measured["tokenizer_sha256"]
        result["token_totals"] = {
            f"{view}/{split}": item["actual_tokens"]
            for view, splits in measured["views"].items()
            for split, item in splits.items()
        }
    return result


def sources(path: Path) -> dict[str, Any]:
    verify_release(path)
    return {"sources": _load(Path(path) / "sources.json")}


def audit(path: Path) -> dict[str, Any]:
    verify_release(path)
    return _load(Path(path) / "audit.json")


def sample(
    path: Path,
    *,
    domain: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    verify_release(path)
    if limit < 0 or limit > 1000:
        raise ValueError("sample limit must be between 0 and 1000")
    if limit == 0:
        return {"records": []}
    root = Path(path)
    documents = {row["document_id"]: row for row in _rows(root / "documents.jsonl")}
    result = []
    for row in _rows(root / "lineage.jsonl"):
        if kind and row["record_kind"] not in (
            {"chat_sft", "tool_episode"} if kind == "chat" else {kind}
        ):
            continue
        if status and row.get("validation_status") != status:
            continue
        domains = sorted(
            set(row.get("domains", []))
            | {
                value
                for parent in row["parent_document_ids"]
                for value in documents[parent]["domains"]
            }
        )
        if domain and domain not in domains:
            continue
        result.append({**row, "domains": domains})
        if len(result) >= limit:
            break
    return {"records": result}


def review(path: Path, **filters: Any) -> dict[str, Any]:
    return sample(path, **filters)


def lineage(path: Path, record_id: str) -> dict[str, Any]:
    verify_release(path)
    root = Path(path)
    document_map = {row["document_id"]: row for row in _rows(root / "documents.jsonl")}
    spans = {row["record_id"]: row for row in _rows(root / "spans.jsonl")}
    sources_map = {row["id"]: row for row in _load(root / "sources.json")}
    for row in _rows(root / "lineage.jsonl"):
        if row["record_id"] == record_id:
            parents = [
                {
                    "document": document_map[key],
                    "span": spans[key],
                    "source": sources_map[document_map[key]["source_id"]],
                }
                for key in row["parent_document_ids"]
            ]
            result = {"lineage": row, "parents": parents}
            if scenario_id := row.get("scenario_id"):
                result["scenario"] = next(
                    item
                    for item in _rows(root / "scenarios.jsonl")
                    if item["scenario_id"] == scenario_id
                )
            if generation_id := row.get("generation_id"):
                result["generation"] = next(
                    item
                    for item in _rows(root / "generations.jsonl")
                    if item["record_id"] == generation_id
                )
            return result
    raise ValueError(f"unknown record ID: {record_id}")


def consumers(path: Path, runs_dir: Path) -> dict[str, Any]:
    manifest = verify_release(path)
    release = Path(path)
    result = []
    from sparselab.training.manifest import read_manifest

    evidence = ("manifest.json", "report.json", "license-report.json", "audit.json")
    for run_manifest in sorted(Path(runs_dir).rglob("manifest.json")):
        if run_manifest.parent.name == "corpus":
            continue
        run = run_manifest.parent
        try:
            record = read_manifest(run_manifest)
            dataset = record["effective_config"]["dataset"]
            if dataset.get("revision") != manifest["release_id"]:
                continue
            inventory = {item["relative_path"]: item for item in record["artifacts"]}
            for name in (*evidence, "export.json"):
                relative = f"corpus/{name}"
                artifact = inventory[relative]
                owned = _safe(run, relative)
                if (
                    owned.stat().st_size != artifact["size_bytes"]
                    or sha256_file(owned) != artifact["sha256"]
                ):
                    raise ValueError("tampered run evidence artifact")
                if name != "export.json" and sha256_file(owned) != sha256_file(
                    release / name
                ):
                    raise ValueError("run corpus evidence does not match release")
            export = _load(run / "corpus/export.json")
            if export.get("release_id") != manifest["release_id"]:
                raise ValueError("run export release mismatch")
            checkpoints = [
                {"path": str(file.relative_to(run)), "sha256": sha256_file(file)}
                for file in sorted((run / "checkpoints").rglob("manifest.json"))
            ]
            result.append(
                {
                    "run_id": record["run_id"],
                    "run_path": str(run),
                    "checkpoints": checkpoints,
                }
            )
        except OSError, ValueError, KeyError, TypeError:
            continue
    return {"release_id": manifest["release_id"], "consumers": result}
