"""Bounded, pre-release inspection of verified admission draft rows and files.

This is an inspection receipt, not a rights decision. It reuses the acquired
snapshot, physical JSONL reader, record-admission verifier and versioned cleaner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sparselab.corpus.acquisition import verify_acquisition
from sparselab.corpus.jsonl_records import records_from_path
from sparselab.corpus.project import Project, load_project, source_declaration_payload
from sparselab.corpus.rights import verify_record_admission
from sparselab.corpus.structure_cleaning import (
    NORMALIZER_V2,
    NORMALIZER_V3,
    clean_structure,
)
from sparselab.training.manifest import canonical_json, sha256_file


def _inside(path: Path, root: Path | None = None) -> Path:
    if (
        any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
        or (root is not None and not path.resolve().is_relative_to(root))
    ):
        raise ValueError("unsafe or missing admission inspection input")
    return path.resolve()


def _selection(path: Path, root: Path, source_ids: set[str]) -> dict[str, Any]:
    selected = _inside(path)
    if selected.stat().st_size > 65536:
        raise ValueError("admission inspection selection is too large")
    value = json.loads(selected.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "seed", "normalizer", "sources"}
        or value["format"] != "sparselab-admission-inspection-selection-v1"
        or not isinstance(value["seed"], str)
        or not 1 <= len(value["seed"]) <= 128
        or value["normalizer"] not in {NORMALIZER_V2, NORMALIZER_V3}
        or not isinstance(value["sources"], list)
        or len(value["sources"]) != len(source_ids)
    ):
        raise ValueError("invalid admission inspection selection")
    seen: set[str] = set()
    for source in value["sources"]:
        if (
            not isinstance(source, dict)
            or set(source) != {"source_id", "strata", "exception_count"}
            or source["source_id"] not in source_ids
            or source["source_id"] in seen
            or type(source["exception_count"]) is not int
            or not 0 <= source["exception_count"] <= 32
            or not isinstance(source["strata"], list)
            or not 1 <= len(source["strata"]) <= 16
        ):
            raise ValueError("invalid admission inspection source selection")
        seen.add(source["source_id"])
        names: set[str] = set()
        for stratum in source["strata"]:
            if (
                not isinstance(stratum, dict)
                or set(stratum)
                != {"id", "count", "path_prefix", "length_band", "issues_nonempty"}
                or not isinstance(stratum["id"], str)
                or not 1 <= len(stratum["id"]) <= 64
                or stratum["id"] in names
                or type(stratum["count"]) is not int
                or not 0 <= stratum["count"] <= 128
                or stratum["length_band"] not in {None, "short", "long"}
                or stratum["issues_nonempty"] not in {None, True, False}
                or (
                    stratum["path_prefix"] is not None
                    and (
                        not isinstance(stratum["path_prefix"], str)
                        or not stratum["path_prefix"]
                    )
                )
            ):
                raise ValueError("invalid admission inspection stratum")
            names.add(stratum["id"])
        if not 1 <= sum(group["count"] for group in source["strata"]) <= 128:
            raise ValueError("admission inspection sample count is out of bounds")
    return value


def _candidate(
    source: Any,
    entry: dict[str, Any],
    decision: dict[str, Any],
    *,
    path: str,
    raw: str,
    digest: str,
    row_index: int | None = None,
    physical_line: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "source_id": source.id,
        "path": path,
        "digest": digest,
        "row_index": row_index,
    }
    return {
        "item_id": hashlib.sha256(canonical_json(identity)).hexdigest(),
        "source_id": source.id,
        "path": path,
        "digest": digest,
        "row_index": row_index,
        "physical_line": physical_line,
        "decision": decision["decision"],
        "reason": decision["reason"],
        "issues": decision["issues"],
        "rights": entry["rights"],
        "license_label": entry["license_label"],
        "source_revision": entry["source_revision"],
        "canonical_uri": source.canonical_uri,
        "license_url": source.license_url,
        "snapshot_sha256": entry["snapshot_sha256"],
        "metadata": metadata or {},
        "raw": raw,
        "raw_text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "length_bytes": len(raw.encode("utf-8")),
    }


def _candidates(
    project: Project,
    root: Path,
    draft: dict[str, Any],
    lock: dict[str, Any],
    max_input_bytes: int,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    result: dict[str, list[dict[str, Any]]] = {}
    consumed = 0
    entries = {entry["source_id"]: entry for entry in draft["sources"]}
    for source in project.sources:
        entry = entries[source.id]
        snapshot = Path(lock["sources"][source.id]["snapshot_path"])
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
        rows: list[dict[str, Any]] = []
        if source.kind == "git":
            decisions = {item["path"]: item for item in entry["files"]}
            if len(decisions) != len(entry["files"]):
                raise ValueError("duplicate draft file decision")
            for file in files:
                path = file["path"]
                size = (snapshot / "files" / path).stat().st_size
                if consumed + size > max_input_bytes:
                    raise ValueError("admission inspection input-byte cap exceeded")
                raw = (snapshot / "files" / path).read_bytes()
                consumed += len(raw)
                if consumed > max_input_bytes:
                    raise ValueError("admission inspection input-byte cap exceeded")
                if hashlib.sha256(raw).hexdigest() != file["sha256"]:
                    raise ValueError("snapshot file changed during inspection")
                decision = decisions.pop(path)
                if decision["sha256"] != file["sha256"]:
                    raise ValueError("draft file identity differs from snapshot")
                rows.append(
                    _candidate(
                        source,
                        entry,
                        decision,
                        path=path,
                        raw=raw.decode("utf-8"),
                        digest=file["sha256"],
                    )
                )
            if decisions:
                raise ValueError("draft contains unobserved file decisions")
        elif source.kind == "huggingface_dataset":
            if len(files) != 1 or len(manifest["retrieval"]["shards"]) != 1:
                raise ValueError("inspection requires one declared JSONL shard")
            file = files[0]
            selected = manifest["retrieval"]["shards"][0]["selected_rows"]
            decisions = entry["records"]
            shard = snapshot / "files" / file["path"]
            consumed += shard.stat().st_size
            if consumed > max_input_bytes:
                raise ValueError("admission inspection input-byte cap exceeded")
            parsed = records_from_path(shard)
            for receipt, decision, record in zip(
                selected, decisions, parsed, strict=True
            ):
                value = record.value
                if not isinstance(value, dict) or not isinstance(
                    value.get("text"), str
                ):
                    raise TypeError("inspection row lacks text")
                envelope = value.get("_sparselab_source")
                original = {
                    key: item
                    for key, item in value.items()
                    if key != "_sparselab_source"
                }
                digest = hashlib.sha256(canonical_json(original)).hexdigest()
                index = receipt["source_row_index"]
                if (
                    not isinstance(envelope, dict)
                    or envelope.get("source_row_index") != index
                    or envelope.get("source_row_sha256") != digest
                    or decision["source_row_index"] != index
                    or decision["source_row_sha256"] != digest
                    or receipt["source_row_sha256"] != digest
                ):
                    raise ValueError(
                        "inspection row identity differs from draft or snapshot"
                    )
                rows.append(
                    _candidate(
                        source,
                        entry,
                        decision,
                        path=file["path"],
                        raw=value["text"],
                        digest=digest,
                        row_index=index,
                        physical_line=record.physical_line,
                        metadata=original.get("metadata")
                        if isinstance(original.get("metadata"), dict)
                        else {},
                    )
                )
            if sha256_file(shard) != file["sha256"]:
                raise ValueError("snapshot shard changed during inspection")
        else:
            raise ValueError("unsupported admission inspection source kind")
        result[source.id] = rows
    return result, consumed


def _rank(
    seed: str, source: str, stratum: str, row: dict[str, Any]
) -> tuple[str, str, int]:
    value = f"{seed} | {source} | {stratum} | {row['digest']}".encode()
    return hashlib.sha256(value).hexdigest(), row["path"], row["row_index"] or 0


def _clip(value: str, limit: int) -> tuple[str, bool]:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value, False
    return raw[:limit].decode("utf-8", errors="ignore"), True


def _display(
    row: dict[str, Any], *, normalizer: str, max_excerpt_bytes: int
) -> dict[str, Any]:
    cleaned = clean_structure(
        row["raw"], source_id=row["source_id"], normalizer=normalizer
    )
    raw_excerpt, raw_truncated = _clip(row["raw"], max_excerpt_bytes)
    clean_excerpt, clean_truncated = _clip(cleaned["text"], max_excerpt_bytes)
    structure = cleaned["structure"]
    return {
        **{key: value for key, value in row.items() if key != "raw"},
        "source_span": {
            "path": row["path"],
            "row_index": row["row_index"],
            "physical_line": row["physical_line"],
        },
        "raw_excerpt": raw_excerpt,
        "raw_truncated": raw_truncated,
        "cleaned_excerpt": clean_excerpt,
        "cleaned_truncated": clean_truncated,
        "cleaned_bytes": len(cleaned["text"].encode("utf-8")),
        "cleaned_text_sha256": hashlib.sha256(cleaned["text"].encode()).hexdigest(),
        "cleaning_decision": structure["decision"],
        "cleaning_flags": structure["flags"],
        "removed_spans": structure["removed_spans"][:32],
        "removed_spans_complete": len(structure["removed_spans"]) <= 32,
        "retained_spans": structure["retained_spans"][:32],
        "retained_spans_complete": len(structure["retained_spans"]) <= 32,
    }


def _inspect(
    project: Project,
    project_path: Path,
    work_root: Path,
    draft_path: Path,
    policy_document: Path,
    selection_path: Path,
    *,
    max_input_bytes: int,
    max_excerpt_bytes: int,
    max_output_bytes: int,
) -> dict[str, Any]:
    root = work_root.resolve(strict=True)
    for value in (max_input_bytes, max_excerpt_bytes, max_output_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("inspection caps must be positive integers")
    if (
        max_input_bytes > 2 * 1024**3
        or max_excerpt_bytes > 4096
        or max_output_bytes > 4 * 1024**2
    ):
        raise ValueError("inspection cap exceeds native maximum")
    if (
        sum(source.acquisition.max_bytes for source in project.sources)
        > max_input_bytes
    ):
        raise ValueError(
            "admission inspection input-byte cap is below declared source maxima"
        )
    draft_file = _inside(draft_path, root)
    if draft_file.stat().st_size > 128 * 1024**2:
        raise ValueError("admission inspection draft is too large")
    draft = json.loads(draft_file.read_text(encoding="utf-8"))
    policy = _inside(policy_document)
    if policy.stat().st_size > 1024**2:
        raise ValueError("admission inspection policy is too large")
    if draft.get("schema_version") != 2 or draft.get("policy_sha256") != sha256_file(
        policy
    ):
        raise ValueError("inspection draft or policy identity mismatch")
    selection = _selection(
        selection_path, root, {source.id for source in project.sources}
    )
    lock = verify_acquisition(project, root)
    snapshots = {
        source.id: json.loads(
            (
                Path(lock["sources"][source.id]["snapshot_path"]) / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        for source in project.sources
    }
    verify_record_admission(
        draft,
        {source.id: source_declaration_payload(source) for source in project.sources},
        snapshots,
        root / "corpora" / project.config.id / "snapshots",
    )
    rows_by_source, consumed = _candidates(project, root, draft, lock, max_input_bytes)
    items: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for source_spec in selection["sources"]:
        source_id = source_spec["source_id"]
        source_rows = rows_by_source[source_id]
        eligible = [row for row in source_rows if row["decision"] == "qualify"]
        for row in eligible:
            row["retained_length_bytes"] = len(
                clean_structure(
                    row["raw"],
                    source_id=source_id,
                    normalizer=selection["normalizer"],
                )["text"].encode("utf-8")
            )
        lengths = sorted(row["retained_length_bytes"] for row in eligible)
        middle = len(lengths) // 2
        median2 = (
            (lengths[middle - 1] + lengths[middle])
            if len(lengths) % 2 == 0 and lengths
            else (2 * lengths[middle] if lengths else 0)
        )
        buckets: dict[str, list[dict[str, Any]]] = {
            group["id"]: [] for group in source_spec["strata"]
        }
        outside = 0
        for row in eligible:
            for group in source_spec["strata"]:
                prefix = group["path_prefix"]
                band = group["length_band"]
                issue = group["issues_nonempty"]
                if (
                    (prefix is None or row["path"].startswith(prefix))
                    and (
                        band is None
                        or (
                            row["retained_length_bytes"] * 2 <= median2
                            if band == "short"
                            else row["retained_length_bytes"] * 2 > median2
                        )
                    )
                    and (issue is None or bool(row["issues"]) == issue)
                ):
                    buckets[group["id"]].append(row)
                    break
            else:
                outside += 1
        selected: list[tuple[str, dict[str, Any]]] = []
        remaining: list[tuple[str, dict[str, Any]]] = []
        sample_population = 0
        for group in source_spec["strata"]:
            name = group["id"]
            ordered = sorted(
                buckets[name],
                key=lambda row: _rank(selection["seed"], source_id, name, row),
            )
            count = group["count"]
            selected.extend((name, row) for row in ordered[:count])
            if count:
                sample_population += len(ordered)
                remaining.extend((name, row) for row in ordered[count:])
            else:
                outside += len(ordered)
            coverage.append(
                {
                    "source_id": source_id,
                    "stratum": name,
                    "population": len(ordered),
                    "requested": count,
                    "initial_selected": min(count, len(ordered)),
                }
            )
        requested = sum(group["count"] for group in source_spec["strata"])
        if len(selected) < requested:
            remaining.sort(
                key=lambda pair: _rank(selection["seed"], source_id, pair[0], pair[1])
            )
            selected.extend(remaining[: requested - len(selected)])
        if len(selected) != min(requested, sample_population):
            raise ValueError("inspection selection is incomplete")
        for name, row in selected:
            item = _display(
                row,
                normalizer=selection["normalizer"],
                max_excerpt_bytes=max_excerpt_bytes,
            )
            item["stratum"] = name
            items.append(item)
        held = [row for row in source_rows if row["decision"] == "quarantine"]
        held.sort(
            key=lambda row: _rank(selection["seed"], source_id, "quarantine", row)
        )
        exceptions.extend(
            _display(
                row,
                normalizer=selection["normalizer"],
                max_excerpt_bytes=max_excerpt_bytes,
            )
            for row in held[: source_spec["exception_count"]]
        )
        coverage.append(
            {
                "source_id": source_id,
                "stratum": "__summary__",
                "qualify_population": len(eligible),
                "outside_protocol_population": outside,
                "requested": requested,
                "selected": len(selected),
                "sample_shortfall": requested - len(selected),
                "quarantine_population": len(held),
                "exceptions_shown": min(len(held), source_spec["exception_count"]),
                "exceptions_incomplete": len(held) > source_spec["exception_count"],
            }
        )
    payload = {
        "format": "sparselab-admission-inspection-v1",
        "project_id": project.config.id,
        "project_path": str(_inside(project_path)),
        "project_sha256": lock["project_sha256"],
        "acquisition_lock_sha256": sha256_file(
            root / "corpora" / project.config.id / "acquisition.json"
        ),
        "draft_path": str(draft_path.resolve()),
        "draft_sha256": sha256_file(draft_path),
        "policy_path": str(policy),
        "policy_sha256": sha256_file(policy),
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "normalizer": selection["normalizer"],
        "normalizer_source_sha256": sha256_file(
            Path(__file__).with_name("structure_cleaning.py")
        ),
        "snapshot_sha256": {
            source.id: lock["sources"][source.id]["snapshot_sha256"]
            for source in project.sources
        },
        "max_input_bytes": max_input_bytes,
        "max_excerpt_bytes": max_excerpt_bytes,
        "max_output_bytes": max_output_bytes,
        "input_bytes": consumed,
        "items": items,
        "quarantine_exceptions": exceptions,
        "coverage": coverage,
    }
    if len(canonical_json(payload)) + 1 > max_output_bytes:
        raise ValueError("admission inspection output-byte cap exceeded")
    return payload


def inspect_admission(
    project: Project,
    project_path: Path,
    work_root: Path,
    draft: Path,
    policy_document: Path,
    selection: Path,
    output: Path,
    *,
    max_input_bytes: int,
    max_excerpt_bytes: int,
    max_output_bytes: int,
) -> dict[str, Any]:
    root = work_root.resolve(strict=True)
    if (
        any(part.is_symlink() for part in (output, *output.parents))
        or output.exists()
        or not output.resolve().is_relative_to(root)
    ):
        raise ValueError("inspection requires unused output within work root")
    payload = _inspect(
        project,
        project_path,
        root,
        draft,
        policy_document,
        selection,
        max_input_bytes=max_input_bytes,
        max_excerpt_bytes=max_excerpt_bytes,
        max_output_bytes=max_output_bytes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json(payload) + b"\n")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "items": len(payload["items"]),
        "exceptions": len(payload["quarantine_exceptions"]),
        "input_bytes": payload["input_bytes"],
    }


def verify_admission_inspection(path: Path, work_root: Path) -> dict[str, Any]:
    root = work_root.resolve(strict=True)
    verified_path = _inside(path, root)
    if verified_path.stat().st_size > 4 * 1024**2:
        raise ValueError("admission inspection receipt is too large")
    observed = json.loads(verified_path.read_text(encoding="utf-8"))
    if (
        not isinstance(observed, dict)
        or observed.get("format") != "sparselab-admission-inspection-v1"
    ):
        raise ValueError("invalid admission inspection receipt")
    project_path = Path(observed["project_path"])
    project = load_project(project_path)
    expected = _inspect(
        project,
        project_path,
        root,
        Path(observed["draft_path"]),
        Path(observed["policy_path"]),
        Path(observed["selection_path"]),
        max_input_bytes=observed["max_input_bytes"],
        max_excerpt_bytes=observed["max_excerpt_bytes"],
        max_output_bytes=observed["max_output_bytes"],
    )
    if observed != expected:
        raise ValueError("admission inspection receipt differs from verified inputs")
    return observed
