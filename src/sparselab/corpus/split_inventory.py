"""Verified, pre-build document IDs for frozen family split assignments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from sparselab.corpus.acquisition import verify_acquisition
from sparselab.corpus.jsonl_records import records_from_bytes
from sparselab.corpus.pipeline import _records_for_file
from sparselab.corpus.project import Project
from sparselab.corpus.rights import resolve_file_rights
from sparselab.training.manifest import canonical_json, sha256_file


def _verified_hf_metadata(
    raw: bytes, file: dict[str, Any], retrieval: dict[str, Any], revision: str
) -> dict[int, dict[str, Any]]:
    """Bind nested source metadata to every selected row in the shard receipt."""
    matching = [
        shard
        for shard in retrieval.get("shards", [])
        if shard.get("output_path") == file["path"]
    ]
    if len(matching) != 1:
        raise ValueError("split inventory lacks one pinned HF shard receipt")
    shard = matching[0]
    selected = shard["selected_rows"]
    rows = list(records_from_bytes(raw, source=file["path"]))
    if len(rows) != len(selected):
        raise ValueError("split inventory HF row count differs from receipt")
    result: dict[int, dict[str, Any]] = {}
    for parsed, receipt in zip(rows, selected, strict=True):
        row = parsed.value
        envelope = row.get("_sparselab_source")
        if not isinstance(envelope, dict):
            raise TypeError("split inventory HF row lacks provenance")
        original = {
            key: value for key, value in row.items() if key != "_sparselab_source"
        }
        digest = hashlib.sha256(canonical_json(original)).hexdigest()
        index = receipt["source_row_index"]
        if (
            type(index) is not int
            or index in result
            or envelope.get("source_row_index") != index
            or envelope.get("source_row_sha256") != digest
            or receipt["source_row_sha256"] != digest
            or envelope.get("dataset_revision") != revision
            or envelope.get("source_shard_path") != shard["source_shard_path"]
            or envelope.get("source_shard_sha256") != shard["source_shard_sha256"]
        ):
            raise ValueError("split inventory HF row provenance mismatch")
        metadata = original.get("metadata")
        result[index] = metadata if isinstance(metadata, dict) else {}
    return result


def _wiki_locator_hint(locator: str) -> str | None:
    try:
        parsed = urlsplit(locator)
        host = (parsed.hostname or "").lower().removeprefix("www.")
    except ValueError:
        return None
    if parsed.scheme != "https" or not (
        host in {"wikipedia.com", "wikipedia.org"}
        or host.endswith((".wikipedia.org", ".wikibooks.org", ".wikimedia.org"))
    ):
        return None
    if parsed.path.startswith("/wiki/"):
        title = unquote(parsed.path[6:]).replace("_", " ").casefold()
    elif parsed.path == "/w/index.php":
        title = (
            parse_qs(parsed.query).get("title", [""])[0].replace("_", " ").casefold()
        )
    else:
        return None
    return f"wikimedia-page:{host}:{title}" if title else None


def _family_hint(
    source: Any, document: dict[str, Any], source_metadata: dict[str, Any]
) -> str:
    """Conservative grouping hint; a reviewer must merge mirrored/topic siblings."""
    metadata = document.get("metadata") or {}
    if source.kind == "huggingface_dataset":
        if "project_gutenberg" in source.canonical_uri:
            identifier = source_metadata.get("id") or metadata.get("id")
            if identifier:
                return f"gutenberg-work:{identifier}"
            locator = source_metadata.get("url") or metadata.get("url")
            if isinstance(locator, str):
                path = urlsplit(locator).path
                if path.startswith("/ebooks/"):
                    return f"gutenberg-work:{path.split('/')[2].split('.')[0]}"
        else:
            page = source_metadata.get("page_id") or metadata.get("page_id")
            if page:
                return f"wikimedia-page:{page}"
            locator = (
                source_metadata.get("page_uri")
                or source_metadata.get("url")
                or metadata.get("page_uri")
                or metadata.get("url")
            )
            if isinstance(locator, str) and (hint := _wiki_locator_hint(locator)):
                return hint
    if source.kind == "git":
        return f"git-file:{source.id}:{document['source_location'].split('#', 1)[0]}"
    return f"unresolved-family:{document['document_id']}"


def write_split_inventory(
    project: Project, work_root: Path, output: Path
) -> dict[str, Any]:
    """Write metadata-only IDs from cold-verified immutable acquisition bytes.

    Candidate family hints are not final assignments. The reviewed mapping can
    group multiple hints together before it is frozen into `splits.yaml`.
    """
    root = work_root.resolve()
    destination = output.resolve()
    if not destination.is_relative_to(root) or destination.exists():
        raise ValueError("split inventory requires a new path within the work root")
    lock = verify_acquisition(project, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".split-inventory-", dir=destination.parent, delete=False
        ) as stream:
            temporary = stream.name
            for source in project.sources:
                entry = lock["sources"][source.id]
                if entry["snapshot_path"] is None:
                    continue
                snapshot_path = Path(entry["snapshot_path"])
                manifest = json.loads((snapshot_path / "manifest.json").read_text())
                for file in sorted(manifest["files"], key=lambda item: item["path"]):
                    path = file["path"]
                    if source.rights and path == source.rights.nested_metadata_path:
                        continue
                    raw = (snapshot_path / "files" / path).read_bytes()
                    source_metadata = (
                        _verified_hf_metadata(
                            raw, file, manifest["retrieval"], source.revision
                        )
                        if source.kind == "huggingface_dataset"
                        else {}
                    )
                    file_rights = (
                        resolve_file_rights(source.rights, path, raw)
                        if source.rights is not None
                        else None
                    )
                    for document, _ in _records_for_file(
                        raw,
                        path,
                        source,
                        entry["snapshot_sha256"],
                        file_rights=file_rights,
                    ):
                        index = document.get("metadata", {}).get("source_row_index")
                        if (
                            source.kind == "huggingface_dataset"
                            and index not in source_metadata
                        ):
                            raise ValueError(
                                "split inventory document lacks pinned HF row"
                            )
                        row = {
                            "document_id": document["document_id"],
                            "source_id": source.id,
                            "snapshot_sha256": entry["snapshot_sha256"],
                            "source_location": document["source_location"],
                            "content_sha256": document["content_sha256"],
                            "raw_content_sha256": document["raw_content_sha256"],
                            "family_hint": _family_hint(
                                source, document, source_metadata.get(index, {})
                            ),
                            "source_row_index": index,
                        }
                        stream.write(canonical_json(row) + b"\n")
                        counts[source.id] += 1
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError("split inventory output already exists") from error
        os.unlink(temporary)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "project_id": project.config.id,
        "path": str(destination),
        "sha256": sha256_file(destination),
        "counts": dict(sorted(counts.items())),
        "documents": sum(counts.values()),
    }
