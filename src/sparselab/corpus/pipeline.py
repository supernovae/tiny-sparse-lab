"""Deterministic, evidence-preserving corpus normalization and derivation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from sparselab.config.models import StrictModel
from sparselab.corpus.project import ReleaseDeclaration, release_declaration_payload
from sparselab.corpus.provenance import (
    DERIVED_ORIGIN,
    DETERMINISTIC_ORIGIN,
    HUMAN_ORIGIN,
    INFERENCE_ORIGIN,
    MULTI_SOURCE_ORIGIN,
    SOURCE_ORIGIN,
    rendered_digest,
    shape_for_record,
    validate_lineage,
    verification,
)
from sparselab.corpus.rights import FileRights, resolve_file_rights
from sparselab.training.manifest import canonical_json, sha256_file


class NormalizedDocument(StrictModel):
    """Versioned document retaining source attribution and raw evidence identity."""

    schema_version: Literal[1, 2]
    document_id: str
    source_id: str
    modality: Literal["text"]
    source_revision: str
    source_location: str
    license: str
    redistribution: str
    domains: list[str]
    document_kind: str
    title: str
    section_path: list[str]
    language: str
    text: str
    content_sha256: str
    raw_content_sha256: str
    source_family: str
    file_sha256: str | None = None
    rights: dict[str, Any] | None = None
    metadata: dict[str, str | int | float | bool | None] | None = None
    split: str | None = None
    representative_id: str | None = None
    drop_reason: str | None = None


class GenerationRecord(StrictModel):
    """Captured response with explicit, non-self-attested validation status."""

    schema_version: Literal[1]
    request_id: str
    generator: dict[str, Any]
    generator_identity: str
    source_document_ids: list[str]
    raw_output: str
    parsed_output: dict[str, Any] | None
    validation_status: Literal[
        "unverified",
        "schema_validated",
        "source_entailed",
        "oracle_verified",
        "cross_source_verified",
        "human_reviewed",
        "rejected",
    ]
    rejection_reason: str | None
    evidence: dict[str, Any] | None
    seed: int
    transform_id: str
    record_id: str
    split: Literal["train", "validation", "test"]


class ScenarioRecord(StrictModel):
    """Deterministic, filesystem-free oracle world."""

    schema_version: Literal[1]
    generator_id: str
    generator_version: str
    world_seed: int
    scenario_family_id: str
    generator_world_id: str
    world_state: dict[str, str]
    oracle_answer: str
    rendered_example: dict[str, str]
    interpreter: str
    transform_id: str
    scenario_id: str
    split: Literal["train", "validation", "test"] | None = None


class ToolEpisode(StrictModel):
    """Static inert tool transcript bound to an oracle or source."""

    schema_version: Literal[1]
    goal: str
    tool_call: dict[str, Any]
    tool_result: str
    interpretation: str
    next_bounded_action: str
    verification: str
    final_response: str
    source_document_ids: list[str]
    scenario_id: str
    transform_id: str
    record_id: str


class Generator(Protocol):
    """Provider-neutral response interface; receipts, not providers, define identity."""

    def generate(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]: ...


class RecordedResponses:
    """Replay a pinned ledger; never infer a missing or changed response."""

    adapter_id = "recorded_responses"
    adapter_version = "1"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = {row["request_id"]: row for row in responses}
        if len(self._responses) != len(responses):
            raise ValueError("duplicate inference request ID")

    def generate(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        response = self._responses.get(request["request_id"])
        if response is None or any(
            response.get(name) != request[name]
            for name in (
                "prompt_sha256",
                "source_document_ids",
                "provider",
                "model",
                "model_revision",
            )
        ):
            raise ValueError("recorded response request/prompt mismatch")
        raw = response.get("raw_output")
        if not isinstance(raw, str) or not isinstance(response.get("seed"), int):
            raise TypeError("recorded response requires raw output and seed")
        return raw, response


SPLITS = ("train", "validation", "test")
KINDS = frozenset(
    {
        "lm_text",
        "lexical_candidates",
        "semantic_candidates",
        "deterministic_scenarios",
        "inference_qa",
        "manual_semantic",
        "chat_sft",
        "tool_episode",
    }
)
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
LEXEME = re.compile(
    r"(?<![\w-])--[\w-]+|\b[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+\b|\b[A-Z][A-Z_0-9]{2,}\b|\b[a-zA-Z_][\w]*(?:_[\w]+)+\b"
)
KEY_VALUE = re.compile(
    r"^[ \t]*([A-Za-z_][\w.-]*)[ \t]*[:=][ \t]*(\S.*?)[ \t]*$", re.MULTILINE
)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(canonical_json(row) + b"\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _model(value: Any) -> dict[str, Any]:
    if isinstance(value, ReleaseDeclaration):
        return release_declaration_payload(value)
    return (
        value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
    )


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _sections(text: str, markdown: bool) -> list[tuple[str, list[str], int, int]]:
    lines = text.splitlines(keepends=True)
    if not markdown:
        return [(text, [], 1, max(1, len(lines)))]
    starts = [(0, [])]
    ancestry: list[str] = []
    fenced = False
    fence_char = ""
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        fence = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence:
            if not fenced:
                fenced, fence_char = True, fence.group(1)[0]
            elif fence.group(1)[0] == fence_char:
                fenced = False
        if fenced or fence:
            continue
        match = HEADING.match(line.rstrip("\r\n"))
        if match:
            level = len(match.group(1))
            ancestry = ancestry[: level - 1] + [match.group(2).strip()]
            if index == 0:
                starts[0] = (0, ancestry.copy())
            else:
                starts.append((index, ancestry.copy()))
    result = []
    for position, (start, path) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        passage = "".join(lines[start:end])
        if passage.strip():
            result.append((passage, path, start + 1, end))
    return result


def _records_for_file(
    raw: bytes,
    name: str,
    source: Any,
    snapshot_id: str,
    *,
    file_rights: FileRights | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    suffix = Path(name).suffix.lower()
    if suffix != ".parquet":
        if b"\x00" in raw:
            raise ValueError("binary input")
        text = _normalized(raw.decode("utf-8", errors="strict"))
        if not text.strip():
            raise ValueError("empty input")
    else:
        text = ""
    suffix = Path(name).suffix.lower()
    if (
        suffix in {".jsonl", ".json", ".parquet"}
        and source.kind == "huggingface_dataset"
    ):
        if suffix == ".parquet":
            import io

            import pyarrow.parquet as pq

            objects = pq.read_table(io.BytesIO(raw)).to_pylist()
        elif suffix == ".jsonl":
            objects = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            loaded = json.loads(text)
            objects = loaded if isinstance(loaded, list) else [loaded]
        field = source.acquisition.text_field
        max_rows = source.acquisition.max_rows
        passages = [
            (item[field], [], n, n) for n, item in enumerate(objects[:max_rows], 1)
        ]
    elif suffix in {".md", ".markdown"}:
        passages = _sections(text, True)
    else:
        passages = _sections(text, False)
    rows = []
    raw_sha = hashlib.sha256(raw).hexdigest()
    rights_payload = file_rights.model_dump(mode="json") if file_rights else None
    file_license = (
        (
            file_rights.detected_spdx_expression
            or source.rights.spdx_expression
            or source.license
        )
        if file_rights
        else source.license
    )
    byte_offsets = [] if source.kind == "huggingface_dataset" else [0]
    if source.kind != "huggingface_dataset":
        for line in raw.decode("utf-8").splitlines(keepends=True):
            byte_offsets.append(byte_offsets[-1] + len(line.encode("utf-8")))
    for content, ancestry, start, end in passages:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty/nontext document")
        location = (
            f"{name}#lines={start}-{end}"
            if source.kind != "huggingface_dataset"
            else f"{name}#row={start}"
        )
        identity = digest(
            [snapshot_id, location, raw_sha, "normalizer-nfc-markdown-v1"]
        )
        byte_start = byte_offsets[start - 1] if byte_offsets else None
        byte_end = byte_offsets[end] if byte_offsets else None
        raw_passage = (
            raw[byte_start:byte_end]
            if byte_start is not None and byte_end is not None
            else content.encode("utf-8")
        )
        raw_content_sha = hashlib.sha256(raw_passage).hexdigest()
        document = {
            "schema_version": source.schema_version,
            "document_id": identity,
            "source_id": source.id,
            "modality": source.modality,
            "source_revision": source.revision,
            "source_location": location,
            "license": file_license,
            "redistribution": (
                file_rights.redistribution_mode
                if file_rights
                else source.redistribution
            ),
            "domains": list(source.domains),
            "document_kind": next(iter(source.document_kinds)),
            "title": ancestry[-1] if ancestry else Path(name).name,
            "section_path": ancestry,
            "language": "en",
            "text": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "raw_content_sha256": raw_content_sha,
            "source_family": source.source_family,
            **(
                {"file_sha256": raw_sha, "rights": rights_payload}
                if file_rights
                else {}
            ),
        }
        document = NormalizedDocument.model_validate(document).model_dump(
            mode="json", exclude_none=True
        )
        evidence = {
            "record_id": identity,
            "snapshot_sha256": snapshot_id,
            "source_id": source.id,
            "raw_path": name,
            "raw_sha256": raw_sha,
            "raw_content_sha256": raw_content_sha,
            "line_start": start,
            "line_end": end,
            "byte_start": byte_start,
            "byte_end": byte_end,
            "section_path": ancestry,
        }
        rows.append((document, evidence))
    return rows


def _split(record: dict[str, Any], policy: dict[str, Any]) -> str:
    unit = policy["unit"]
    if unit == "document":
        keys = record.get("parent_document_ids") or [
            record.get("document_id") or record.get("scenario_family_id")
        ]
    elif unit in {"source_repository", "source_document_family"}:
        keys = record.get("source_family_ids") or [
            record.get("source_family") or record.get("scenario_family_id")
        ]
    else:
        field = {
            "scenario_family": "scenario_family_id",
            "generator_world": "generator_world_id",
            "template_family": "template_family_id",
        }[unit]
        keys = [record.get(field)]
    keys = [k for k in keys if k]
    if not keys:
        raise ValueError(f"missing {unit} split lineage")
    assignments = policy["assignments"]
    choices = {assignments[k] for k in keys if k in assignments}
    if len(choices) != 1 or any(k not in assignments for k in keys):
        raise ValueError(f"missing/conflicting split assignment: {keys}")
    choice = choices.pop()
    if choice not in SPLITS:
        raise ValueError("invalid split")
    return choice


def _lineage(
    record: dict[str, Any],
    parents: list[dict[str, Any]],
    stage: str,
    *,
    family: str | None = None,
    world: str | None = None,
    template: str | None = None,
) -> dict[str, Any]:
    ids = sorted({p["document_id"] for p in parents})
    families = sorted({p["source_family"] for p in parents})
    return {
        "record_id": record["record_id"],
        "record_kind": record["kind"],
        "transform_id": stage,
        "parent_document_ids": ids,
        "original_parent_document_ids": ids,
        "source_family_ids": families,
        "scenario_family_id": family,
        "generator_world_id": world,
        "template_family_id": template,
        "domains": ["systems_scenarios"]
        if world
        else sorted({domain for parent in parents for domain in parent["domains"]}),
        "validation_status": record.get("validation_status", "schema_validated"),
    }


def _path_scenario(seed: int, family: str, stage: str) -> dict[str, Any]:
    from random import Random

    rng = Random(seed)
    parts = [
        "",
        "workspace",
        "project",
        f"module{rng.randrange(1000)}",
        f"item{rng.randrange(1000)}.py",
    ]
    path = "/".join(parts)
    suffix = PurePosixPath(path).suffix
    world = {"path": path, "operation": "suffix"}
    result = {
        "schema_version": 1,
        "generator_id": "pathlib_path_suffix_v1",
        "generator_version": "1",
        "world_seed": seed,
        "scenario_family_id": family,
        "generator_world_id": f"{family}:{seed}",
        "world_state": world,
        "oracle_answer": suffix,
        "rendered_example": {
            "question": f"What is the suffix of {path}?",
            "answer": suffix,
        },
        "interpreter": "pathlib.PurePosixPath",
        "transform_id": stage,
    }
    result["scenario_id"] = digest(result)
    return ScenarioRecord.model_validate(result).model_dump(
        mode="json", exclude_none=True
    )


def _snapshot_file(
    project: Any, lock: dict[str, Any], source_id: str, name: str
) -> bytes:
    entry = lock["sources"][source_id]
    root = Path(entry["snapshot_path"])
    from sparselab.corpus.acquisition import verify_snapshot

    manifest = verify_snapshot(root)
    matches = [item for item in manifest["files"] if item["path"] == name]
    if len(matches) != 1:
        raise ValueError(f"ambiguous/missing snapshotted input: {source_id}/{name}")
    return (root / "files" / matches[0]["path"]).read_bytes()


def _ordered_transforms(project: Any) -> list[Any]:
    sources = {source.id for source in project.sources}
    stages = {stage.id: stage for stage in project.transforms}
    if len(stages) != len(project.transforms) or sources.intersection(stages):
        raise ValueError("ambiguous transform/source ID")
    ordered: list[Any] = []
    pending = dict(stages)
    while pending:
        ready = [
            stage
            for stage in pending.values()
            if all(
                dependency in sources or dependency in {item.id for item in ordered}
                for dependency in stage.inputs
            )
        ]
        if not ready:
            raise ValueError("unknown or cyclic transform dependency")
        stage = min(ready, key=lambda value: value.id)
        ordered.append(stage)
        del pending[stage.id]
    return ordered


@contextmanager
def _staged_build(root: Path, build_id: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as temporary:
        staging = Path(temporary)
        try:
            yield staging
        except Exception as exc:
            diagnostic = root / "diagnostics" / f"{build_id}.json"
            details = (
                json.loads(diagnostic.read_text(encoding="utf-8"))
                if diagnostic.exists()
                else {}
            )
            details.setdefault("error", str(exc))
            details["build_id"] = build_id
            for name in ("rejected.jsonl", "generations.jsonl", "audit.json"):
                source = staging / name
                if source.exists():
                    details[name] = (
                        _rows(source)
                        if name.endswith(".jsonl")
                        else json.loads(source.read_text(encoding="utf-8"))
                    )
            _json(diagnostic, details)
            raise


def build(project: Any, work_root: Path, offline: bool = False) -> Path:
    """Build an immutable derived inventory from the exact acquisition lock."""
    from sparselab.corpus.acquisition import verify_acquisition

    lock = verify_acquisition(project, work_root)
    root = Path(work_root) / "corpora" / project.config.id / "builds"
    root.mkdir(parents=True, exist_ok=True)
    transforms = _ordered_transforms(project)
    stage_specs = [_model(t) for t in transforms]
    for t in stage_specs:
        if t["kind"] not in KINDS:
            raise ValueError(f"unregistered transform: {t['kind']}")
    identity = {
        "project_id": project.config.id,
        "project": _model(project.config),
        "lock": {
            key: {
                "declaration_sha256": value["declaration_sha256"],
                "snapshot_sha256": value.get("snapshot_sha256"),
            }
            for key, value in sorted(lock["sources"].items())
        },
        "transforms": stage_specs,
        "split": _model(project.splits),
        "release": _model(project.release),
        "implementation_sha256": sha256_file(Path(__file__)),
        **(
            {
                "rights_implementation_sha256": sha256_file(
                    Path(__file__).with_name("rights.py")
                )
            }
            if project.release.schema_version == 2
            else {}
        ),
    }
    build_id = digest(identity)
    target = root / build_id
    if target.exists():
        from sparselab.corpus.release import verify_build

        verify_build(target)
        return target
    with _staged_build(root, build_id) as staging:
        documents: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        rights_files: list[dict[str, Any]] = []
        auxiliary_files: set[tuple[str, str]] = set()
        for spec in stage_specs:
            params = spec["parameters"]
            if spec["kind"] == "inference_qa":
                if not all(
                    params.get(name)
                    for name in (
                        "generator_source_id",
                        "prompt_template_path",
                        "responses_path",
                    )
                ):
                    raise ValueError("incomplete recorded inference generator")
                auxiliary_files.update(
                    (params["generator_source_id"], params[name])
                    for name in ("prompt_template_path", "responses_path")
                )
            elif spec["kind"] == "manual_semantic":
                auxiliary_files.add((params["source_id"], params["path"]))
        for source in sorted(project.sources, key=lambda s: s.id):
            entry = lock["sources"].get(source.id)
            if (
                source.redistribution == "rejected"
                or not entry
                or not entry.get("snapshot_path")
            ):
                continue
            from sparselab.corpus.acquisition import verify_snapshot

            snapshot = verify_snapshot(Path(entry["snapshot_path"]))
            nested_path = source.rights.nested_metadata_path if source.rights else None
            nested_metadata = (
                {
                    nested_path: json.loads(
                        (
                            Path(entry["snapshot_path"]) / "files" / nested_path
                        ).read_text(encoding="utf-8")
                    )
                }
                if nested_path
                else {}
            )
            for file in sorted(snapshot["files"], key=lambda f: f["path"]):
                name = file["path"]
                if name == nested_path:
                    rights_files.append(
                        {
                            "source_id": source.id,
                            "path": name,
                            "sha256": file["sha256"],
                            "size": file["size"],
                            "role": "license_metadata",
                            "canonical_uri": source.canonical_uri,
                            "revision": source.revision,
                        }
                    )
                    continue
                raw = (Path(entry["snapshot_path"]) / "files" / name).read_bytes()
                decision = (
                    resolve_file_rights(
                        source.rights, name, raw, nested_metadata=nested_metadata
                    )
                    if source.rights
                    else None
                )
                if decision is not None:
                    rights_files.append(
                        {
                            "source_id": source.id,
                            "path": name,
                            "sha256": file["sha256"],
                            "size": file["size"],
                            "role": "transform_input"
                            if (source.id, name) in auxiliary_files
                            else "document",
                            "canonical_uri": source.canonical_uri,
                            "revision": source.revision,
                            "license_url": source.license_url,
                            "rights": decision.model_dump(mode="json"),
                        }
                    )
                    if decision.training_eligibility not in {
                        "eligible",
                        "eligible_with_obligations",
                    }:
                        rejected.append(
                            {
                                "source_id": source.id,
                                "path": name,
                                "reason": f"rights {decision.training_eligibility}: {decision.reason}",
                            }
                        )
                        if (source.id, name) in auxiliary_files:
                            raise ValueError(
                                f"unresolved rights for transform input: {source.id}/{name}"
                            )
                        continue
                if (source.id, name) in auxiliary_files:
                    continue
                try:
                    pairs = _records_for_file(
                        raw,
                        name,
                        source,
                        entry["snapshot_sha256"],
                        file_rights=decision,
                    )
                    documents.extend(doc for doc, _ in pairs)
                    evidence.extend(span for _, span in pairs)
                except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                    rejected.append(
                        {
                            "source_id": source.id,
                            "path": name,
                            "reason": str(exc),
                        }
                    )
        _jsonl(staging / "rejected.jsonl", rejected)
        documents.sort(key=lambda d: d["document_id"])
        policy = _model(project.splits)
        for doc in documents:
            doc["split"] = _split(doc, policy)
        groups: dict[str, list[str]] = defaultdict(list)
        normalized: dict[str, list[str]] = defaultdict(list)
        by_id = {d["document_id"]: d for d in documents}
        for doc in documents:
            groups[doc["raw_content_sha256"]].append(doc["document_id"])
            normalized[doc["content_sha256"]].append(doc["document_id"])
        duplicate_groups = []
        representative = {d["document_id"]: d["document_id"] for d in documents}
        for method, mapping in (("raw", groups), ("normalized", normalized)):
            for content_id, ids in sorted(mapping.items()):
                if len(ids) < 2:
                    continue
                ids = sorted(ids)
                duplicate_groups.append(
                    {
                        "method": method,
                        "sha256": content_id,
                        "origins": ids,
                        "splits": sorted({by_id[i]["split"] for i in ids}),
                        "representative": ids[0],
                    }
                )
                if len({by_id[i]["split"] for i in ids}) > 1:
                    diagnostic = {
                        "duplicates": duplicate_groups,
                        "rejected": rejected,
                        "error": "cross-split exact text overlap",
                    }
                    _json(staging / "audit.json", diagnostic)
                    _json(root / "diagnostics" / f"{build_id}.json", diagnostic)
                    raise ValueError("cross-split exact text overlap")
                if method == "normalized" and project.release.schema_version == 2:
                    rights_signatures = {
                        canonical_json(
                            {
                                "license": by_id[i]["license"],
                                "redistribution": by_id[i]["redistribution"],
                                "training_eligibility": by_id[i]["rights"][
                                    "training_eligibility"
                                ],
                                "license_references": by_id[i]["rights"][
                                    "license_references"
                                ],
                                "notices": by_id[i]["rights"]["notices"],
                                "training_restriction": by_id[i]["rights"][
                                    "training_restriction"
                                ],
                            }
                        )
                        for i in ids
                    }
                    if len(rights_signatures) > 1:
                        diagnostic = {
                            "duplicates": duplicate_groups,
                            "rejected": rejected,
                            "error": "duplicate source text has incompatible rights evidence",
                        }
                        _json(staging / "audit.json", diagnostic)
                        _json(root / "diagnostics" / f"{build_id}.json", diagnostic)
                        raise ValueError(
                            "duplicate source text has incompatible rights evidence"
                        )
                for i in ids[1:]:
                    representative[i] = min(representative[i], ids[0])

        def chosen(identifier: str) -> str:
            current = identifier
            while representative[current] != current:
                current = representative[current]
            representative[identifier] = current
            return current

        for identifier in representative:
            chosen(identifier)
        for doc in documents:
            doc["representative_id"] = representative[doc["document_id"]]
            doc["drop_reason"] = (
                "duplicate" if doc["representative_id"] != doc["document_id"] else None
            )
        kept = [d for d in documents if not d["drop_reason"]]
        stages: list[dict[str, Any]] = []
        lexical: list[dict[str, Any]] = []
        semantic: list[dict[str, Any]] = []
        scenarios: list[dict[str, Any]] = []
        generations: list[dict[str, Any]] = []
        chats: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []
        for transform in transforms:
            spec = _model(transform)
            kind, stage_id = spec["kind"], spec["id"]
            params = spec.get("parameters", {})
            inputs = spec.get("inputs", [])
            selected_sources = set(inputs) & {source.id for source in project.sources}
            selected = [
                doc
                for doc in kept
                if not selected_sources or doc["source_id"] in selected_sources
            ]
            if kind == "lm_text":
                output = [
                    {
                        "record_id": d["document_id"],
                        "text": d["text"],
                        "split": d["split"],
                    }
                    for d in selected
                ]
            elif kind == "lexical_candidates":
                terms: dict[str, dict[str, Any]] = {}
                for doc in selected:
                    for term in LEXEME.findall(doc["text"]):
                        item = terms.setdefault(
                            term,
                            {
                                "term": term,
                                "occurrences": 0,
                                "document_ids": set(),
                                "domains": Counter(),
                            },
                        )
                        item["occurrences"] += 1
                        item["document_ids"].add(doc["document_id"])
                        item["domains"].update(doc["domains"])
                output = [
                    {
                        "record_id": digest([stage_id, term]),
                        "term": term,
                        "occurrences": item["occurrences"],
                        "document_ids": sorted(item["document_ids"]),
                        "domain_counts": dict(sorted(item["domains"].items())),
                    }
                    for term, item in sorted(terms.items())
                ]
                for candidate in output:
                    parents = [
                        by_id[identifier] for identifier in candidate["document_ids"]
                    ]
                    record = _lineage(
                        {
                            "record_id": candidate["record_id"],
                            "kind": "lexical_candidate",
                        },
                        parents,
                        stage_id,
                    )
                    splits = {parent["split"] for parent in parents}
                    record["split"] = splits.pop() if len(splits) == 1 else "inventory"
                    lineage.append(record)
                lexical.extend(output)
            elif kind == "semantic_candidates":
                output = []
                for doc in selected:
                    for match in KEY_VALUE.finditer(doc["text"]):
                        passage = match.group(0)
                        row = {
                            "subject": match.group(1),
                            "relation": "has_value",
                            "value": match.group(2),
                            "preconditions": [],
                            "consequences": [],
                            "product": None,
                            "version": doc["source_revision"],
                            "validity_interval": None,
                            "method": "deterministic",
                            "evidence_document_id": doc["document_id"],
                            "evidence_passage": passage,
                            "evidence_span": [match.start(), match.end()],
                            "validation_status": "source_entailed",
                        }
                        row["record_id"] = digest([stage_id, row])
                        output.append(row)
                        lineage.append(
                            _lineage(
                                {
                                    "record_id": row["record_id"],
                                    "kind": "semantic_candidate",
                                },
                                [doc],
                                stage_id,
                            )
                        )
                semantic.extend(output)
            elif kind == "manual_semantic":
                raw = _snapshot_file(project, lock, params["source_id"], params["path"])
                output = []
                for entry in (
                    json.loads(line)
                    for line in raw.decode("utf-8").splitlines()
                    if line
                ):
                    doc = by_id[entry["evidence_document_id"]]
                    start, end = entry["evidence_span"]
                    if doc["text"][start:end] != entry["evidence_passage"]:
                        raise ValueError("manual semantic evidence span mismatch")
                    row = {
                        **entry,
                        "method": "manual",
                        "validation_status": "source_entailed",
                    }
                    row["record_id"] = digest([stage_id, row])
                    lineage.append(
                        _lineage(
                            {
                                "record_id": row["record_id"],
                                "kind": "semantic_candidate",
                            },
                            [doc],
                            stage_id,
                        )
                    )
                    output.append(row)
                semantic.extend(output)
            elif kind == "deterministic_scenarios":
                seeds, families = params["world_seeds"], params["scenario_families"]
                if (
                    params["generator"] != "pathlib_path_suffix_v1"
                    or len(seeds) != len(families)
                    or len(set(seeds)) != len(seeds)
                ):
                    raise ValueError("invalid scenario generator configuration")
                output = [
                    _path_scenario(seed, family, stage_id)
                    for seed, family in zip(seeds, families, strict=True)
                ]
                for row in output:
                    row["split"] = _split(row, policy)
                    ScenarioRecord.model_validate(row)
                    lineage.append(
                        {
                            **_lineage(
                                {
                                    "record_id": row["scenario_id"],
                                    "kind": "scenario",
                                    "validation_status": "oracle_verified",
                                },
                                [],
                                stage_id,
                                family=row["scenario_family_id"],
                                world=row["generator_world_id"],
                                template=digest([stage_id, "path_suffix_prompt_v1"]),
                            ),
                            "scenario_id": row["scenario_id"],
                            "generator_identity": digest(
                                [row["generator_id"], row["generator_version"]]
                            ),
                            "oracle_identity": digest(
                                [row["world_state"], row["oracle_answer"]]
                            ),
                        }
                    )
                scenarios.extend(output)
            elif kind == "inference_qa":
                required = (
                    "generator_source_id",
                    "backend",
                    "provider",
                    "model",
                    "model_revision",
                    "prompt_template_path",
                    "responses_path",
                    "max_input_chars",
                    "generation_config",
                )
                if (
                    any(not params.get(key) for key in required)
                    or params["backend"] != "recorded_responses"
                ):
                    raise ValueError("incomplete recorded inference generator")
                source_id = params["generator_source_id"]
                template = _snapshot_file(
                    project, lock, source_id, params["prompt_template_path"]
                )
                response_bytes = _snapshot_file(
                    project, lock, source_id, params["responses_path"]
                )
                output = []
                responses = [
                    json.loads(line)
                    for line in response_bytes.decode("utf-8").splitlines()
                    if line
                ]
                adapter = RecordedResponses(responses)
                replay: Generator = adapter
                for response in responses:
                    request_id = response["request_id"]
                    parents = [by_id[item] for item in response["source_document_ids"]]
                    prompt = template.decode("utf-8").format(
                        passages="\n".join(
                            f"[{d['document_id']}] {d['text'][: params['max_input_chars']]}"
                            for d in parents
                        )
                    )
                    request = {
                        "request_id": request_id,
                        "source_document_ids": response["source_document_ids"],
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        **{
                            name: params[name]
                            for name in ("provider", "model", "model_revision")
                        },
                    }
                    try:
                        raw_output, response = replay.generate(request)
                    except ValueError as exc:
                        _json(
                            root / "diagnostics" / f"{build_id}.json",
                            {
                                "error": str(exc),
                                "request_id": request_id,
                                "source_document_ids": response["source_document_ids"],
                                "raw_output": response.get("raw_output"),
                            },
                        )
                        raise
                    generator = {
                        "adapter_id": adapter.adapter_id,
                        "adapter_version": adapter.adapter_version,
                        "adapter_module_sha256": identity["implementation_sha256"],
                        "provider": params["provider"],
                        "model": params["model"],
                        "model_revision": params["model_revision"],
                        "template_id": params["prompt_template_path"],
                        "template_sha256": hashlib.sha256(template).hexdigest(),
                        "generation_config": params["generation_config"],
                        "seed": response["seed"],
                        "raw_output_sha256": hashlib.sha256(
                            raw_output.encode()
                        ).hexdigest(),
                    }
                    try:
                        parsed = json.loads(raw_output)
                        answer, cited = parsed["answer"], parsed["citation_id"]
                        if (
                            not isinstance(answer, str)
                            or not answer.strip()
                            or cited not in {p["document_id"] for p in parents}
                        ):
                            raise ValueError("invalid answer/citation")
                        parent = by_id[cited]
                        start = parent["text"][: params["max_input_chars"]].find(answer)
                        status = "source_entailed" if start >= 0 else "unverified"
                        grounding_evidence = (
                            {
                                "document_id": cited,
                                "passage": answer,
                                "span": [start, start + len(answer)],
                            }
                            if start >= 0
                            else None
                        )
                        reason = (
                            None
                            if status == "source_entailed"
                            else "answer absent from cited passage"
                        )
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        json.JSONDecodeError,
                    ) as exc:
                        parsed, status, reason, grounding_evidence = (
                            None,
                            "rejected",
                            str(exc),
                            None,
                        )
                    record = {
                        "schema_version": 1,
                        "request_id": request_id,
                        "generator": generator,
                        "generator_identity": digest(generator),
                        "source_document_ids": response["source_document_ids"],
                        "raw_output": raw_output,
                        "parsed_output": parsed,
                        "validation_status": status,
                        "rejection_reason": reason,
                        "evidence": grounding_evidence,
                        "seed": response["seed"],
                        "transform_id": stage_id,
                    }
                    record["record_id"] = digest(record)
                    record["split"] = _split(
                        _lineage(
                            {"record_id": record["record_id"], "kind": "generation"},
                            parents,
                            stage_id,
                        ),
                        policy,
                    )
                    GenerationRecord.model_validate(record)
                    output.append(record)
                    if status == "rejected":
                        rejected.append(
                            {
                                "record_id": record["record_id"],
                                "request_id": request_id,
                                "raw_output": raw_output,
                                "generator_identity": record["generator_identity"],
                                "source_document_ids": record["source_document_ids"],
                                "reason": reason,
                            }
                        )
                    lineage.append(
                        {
                            **_lineage(
                                {
                                    "record_id": record["record_id"],
                                    "kind": "generation",
                                    "validation_status": status,
                                },
                                parents,
                                stage_id,
                                template=hashlib.sha256(template).hexdigest(),
                            ),
                            "generation_id": record["record_id"],
                            "generator_identity": record["generator_identity"],
                        }
                    )
                generations.extend(output)
            elif kind in {"chat_sft", "tool_episode"}:
                output = []
                accepted = set(
                    _model(project.release).get(
                        "accepted_generation_statuses",
                        ["source_entailed", "oracle_verified", "human_reviewed"],
                    )
                )
                for scenario in scenarios:
                    if kind == "chat_sft":
                        messages = [
                            {
                                "role": "user",
                                "content": scenario["rendered_example"]["question"],
                            },
                            {"role": "assistant", "content": scenario["oracle_answer"]},
                        ]
                    else:
                        call_id = scenario["scenario_id"][:16]
                        messages = [
                            {
                                "role": "user",
                                "content": scenario["rendered_example"]["question"],
                            },
                            {
                                "role": "assistant",
                                "content": "Inspect the declared path world.",
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "name": "pathlib_suffix",
                                        "arguments": {
                                            "path": scenario["world_state"]["path"]
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": scenario["oracle_answer"],
                            },
                            {
                                "role": "assistant",
                                "content": f"The tool result is {scenario['oracle_answer']}; checking against the declared path suffix gives {scenario['oracle_answer']}. Final answer: {scenario['oracle_answer']}",
                            },
                        ]
                    row = {
                        "record_id": digest(
                            [stage_id, scenario["scenario_id"], messages]
                        ),
                        "kind": kind,
                        "format_version": 2,
                        "loss_mode": "assistant_only",
                        "messages": messages,
                        "scenario_id": scenario["scenario_id"],
                        "validation_status": "oracle_verified",
                        "split": scenario["split"],
                    }
                    output.append(row)
                    lineage.append(
                        {
                            **_lineage(
                                row,
                                [],
                                stage_id,
                                family=scenario["scenario_family_id"],
                                world=scenario["generator_world_id"],
                                template=digest([stage_id, kind]),
                            ),
                            "scenario_id": scenario["scenario_id"],
                            "generator_identity": digest(
                                [
                                    scenario["generator_id"],
                                    scenario["generator_version"],
                                ]
                            ),
                            "oracle_identity": digest(
                                [scenario["world_state"], scenario["oracle_answer"]]
                            ),
                        }
                    )
                    if kind == "tool_episode":
                        tools.append(
                            ToolEpisode.model_validate(
                                {
                                    "schema_version": 1,
                                    "goal": messages[0]["content"],
                                    "tool_call": messages[1]["tool_calls"][0],
                                    "tool_result": messages[2]["content"],
                                    "interpretation": "Suffix returned by bounded path oracle",
                                    "next_bounded_action": "Compare the path suffix",
                                    "verification": scenario["oracle_answer"],
                                    "final_response": messages[-1]["content"],
                                    "source_document_ids": [],
                                    "scenario_id": scenario["scenario_id"],
                                    "transform_id": stage_id,
                                    "record_id": row["record_id"],
                                }
                            ).model_dump(mode="json")
                        )
                if kind == "chat_sft":
                    for gen in generations:
                        if (
                            gen["validation_status"] not in accepted
                            or gen["validation_status"] == "rejected"
                        ):
                            continue
                        parsed = gen["parsed_output"]
                        row = {
                            "record_id": digest([stage_id, gen["record_id"]]),
                            "kind": kind,
                            "format_version": 2,
                            "loss_mode": "assistant_only",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"Based on source passage {parsed['citation_id']}, answer the question.",
                                },
                                {"role": "assistant", "content": parsed["answer"]},
                            ],
                            "generation_id": gen["record_id"],
                            "validation_status": gen["validation_status"],
                            "split": gen["split"],
                        }
                        output.append(row)
                        source_lineage = next(
                            item
                            for item in lineage
                            if item["record_id"] == gen["record_id"]
                        )
                        lineage.append(
                            {
                                **source_lineage,
                                "record_id": row["record_id"],
                                "record_kind": kind,
                                "transform_id": stage_id,
                            }
                        )
                    requested_shapes = params.get("semantic_shapes", [])
                    if params.get("include_semantic_qa", False):
                        requested_shapes = list(
                            dict.fromkeys([*requested_shapes, "direct_qa"])
                        )
                    allowed_shapes = {
                        "direct_qa",
                        "troubleshooting_scenario",
                        "decision_record",
                    }
                    if not isinstance(requested_shapes, list) or (
                        set(requested_shapes) - allowed_shapes
                    ):
                        raise ValueError("unknown semantic chat shape")
                    for fact in semantic:
                        doc = by_id[fact["evidence_document_id"]]
                        if (
                            selected_sources
                            and doc["source_id"] not in selected_sources
                        ):
                            continue
                        key, value = fact["subject"], fact["value"]
                        if value not in fact["evidence_passage"]:
                            raise ValueError(
                                "semantic answer absent from source passage"
                            )
                        for shape_id in requested_shapes:
                            question = {
                                "direct_qa": f"What value is recorded for {key}?",
                                "troubleshooting_scenario": (
                                    f"Checking the recorded configuration: which setting has value {value}?"
                                ),
                                "decision_record": (
                                    f"For the recorded setting {key}, which recorded value should be used?"
                                ),
                            }[shape_id]
                            answer = {
                                "direct_qa": value,
                                "troubleshooting_scenario": key,
                                "decision_record": value,
                            }[shape_id]
                            messages = [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer},
                            ]
                            row = {
                                "record_id": digest(
                                    [stage_id, fact["record_id"], shape_id, messages]
                                ),
                                "kind": kind,
                                "format_version": 2,
                                "loss_mode": "assistant_only",
                                "messages": messages,
                                "semantic_id": fact["record_id"],
                                "semantic_shape": shape_id,
                                "validation_status": "source_entailed",
                                "split": doc["split"],
                            }
                            output.append(row)
                            lineage.append(
                                {
                                    **_lineage(
                                        row,
                                        [doc],
                                        stage_id,
                                        template=digest(
                                            [stage_id, shape_id, "semantic_v1"]
                                        ),
                                    ),
                                    "semantic_id": fact["record_id"],
                                    "generator_identity": digest(
                                        ["semantic_chat_v1", stage_id, shape_id]
                                    ),
                                }
                            )
                chats.extend(output)
            else:
                raise ValueError(f"unregistered transform: {kind}")
            _jsonl(staging / "stages" / f"{stage_id}.jsonl", output)
            stages.append(
                {
                    "id": stage_id,
                    "kind": kind,
                    "version": spec["version"],
                    "parameters_sha256": digest(params),
                    "inputs": inputs,
                    "implementation_sha256": sha256_file(Path(__file__)),
                    "output_sha256": sha256_file(
                        staging / "stages" / f"{stage_id}.jsonl"
                    ),
                    "output_count": len(output),
                    "output_id": digest([stage_id, output]),
                }
            )
        for doc in documents:
            lineage.append(
                {
                    "record_id": doc["document_id"],
                    "record_kind": "document",
                    "transform_id": "normalizer-nfc-markdown-v1",
                    "parent_document_ids": [doc["document_id"]],
                    "original_parent_document_ids": [doc["document_id"]],
                    "representative_id": doc["representative_id"],
                    "drop_reason": doc["drop_reason"],
                    "source_family_ids": [doc["source_family"]],
                    "split": doc["split"],
                }
            )
        for item in lineage:
            if "split" not in item:
                item["split"] = _split(item, policy)
            for parent in item["parent_document_ids"]:
                if parent not in by_id:
                    raise ValueError("dangling parent document")
            item["representative_parent_document_ids"] = sorted(
                {representative[p] for p in item["parent_document_ids"]}
            )
        payloads = {
            row["record_id"]: row
            for collection in (lexical, semantic, generations, chats)
            for row in collection
        }
        payloads.update({row["scenario_id"]: row for row in scenarios})
        source_origins = {source.id: source.origin for source in project.sources}
        for item in lineage:
            kind = item["record_kind"]
            record_id = item["record_id"]
            payload = by_id[record_id] if kind == "document" else payloads[record_id]
            if kind == "document":
                origin = (
                    HUMAN_ORIGIN
                    if source_origins[payload["source_id"]] == HUMAN_ORIGIN
                    else SOURCE_ORIGIN
                )
                shape_id = "raw_document"
                item["domains"] = payload["domains"]
                status, method, proof = (
                    "schema_validated",
                    "normalizer_v1",
                    {"schema_id": "normalized_document_v1"},
                )
                rendered = rendered_digest(payload["text"], text=True)
            else:
                source_ids = {
                    by_id[doc_id]["source_id"] for doc_id in item["parent_document_ids"]
                }
                origin = (
                    DETERMINISTIC_ORIGIN
                    if kind in {"scenario", "tool_episode"}
                    or (kind == "chat_sft" and "scenario_id" in payload)
                    else MULTI_SOURCE_ORIGIN
                    if len(source_ids) > 1
                    else INFERENCE_ORIGIN
                    if kind == "generation"
                    and not source_ids
                    or kind == "chat_sft"
                    and "generation_id" in payload
                    and not source_ids
                    else DERIVED_ORIGIN
                )
                shape_id = {
                    "lexical_candidate": "lexical_inventory",
                    "semantic_candidate": "definition",
                    "scenario": "troubleshooting_scenario",
                    "generation": "direct_qa",
                    "tool_episode": "tool_trace",
                    "chat_sft": payload.get(
                        "semantic_shape",
                        "troubleshooting_scenario"
                        if "scenario_id" in payload
                        else "direct_qa",
                    ),
                }[kind]
                evidence_row = (
                    payloads[payload["semantic_id"]]
                    if "semantic_id" in payload
                    else payloads[payload["generation_id"]]
                    if "generation_id" in payload
                    else payload
                )
                status = evidence_row.get(
                    "validation_status", item["validation_status"]
                )
                if status == "source_entailed":
                    proof = {
                        "document_id": evidence_row.get("evidence_document_id")
                        or evidence_row["evidence"]["document_id"],
                        "passage": evidence_row.get("evidence_passage")
                        or evidence_row["evidence"]["passage"],
                        "span": evidence_row.get("evidence_span")
                        or evidence_row["evidence"]["span"],
                        "answer": (
                            payload["messages"][-1]["content"]
                            if kind == "chat_sft"
                            else evidence_row["parsed_output"]["answer"]
                            if kind == "generation"
                            else evidence_row["value"]
                        ),
                    }
                    method = "verbatim_source_span"
                elif status == "oracle_verified":
                    scenario = (
                        payloads[payload["scenario_id"]]
                        if "scenario_id" in payload
                        else payload
                    )
                    proof = {
                        "oracle_identity": digest(
                            [scenario["world_state"], scenario["oracle_answer"]]
                        ),
                        "generator_world_id": scenario["generator_world_id"],
                        "oracle_answer": scenario["oracle_answer"],
                        "oracle_implementation": identity["implementation_sha256"],
                        "oracle_version": scenario["generator_version"],
                        "interpreter": scenario["interpreter"],
                        "world_state": scenario["world_state"],
                        "actual_result": scenario["oracle_answer"],
                        "comparison_status": "match",
                    }
                    method = "pathlib_pureposix_oracle_v1"
                elif status in {"rejected", "unverified"}:
                    proof = {
                        "reason": evidence_row.get("rejection_reason")
                        or "response not entailed by cited passage"
                    }
                    method = "recorded_response_validation"
                else:
                    proof = {"schema_id": f"{kind}_v1"}
                    method = "schema_validation"
                if kind in {"chat_sft", "tool_episode"}:
                    from sparselab.data.conversations import _v2_document

                    chat_row = {
                        field: payload[field]
                        for field in ("format_version", "loss_mode", "messages")
                    }
                    rendered = rendered_digest(
                        _v2_document(chat_row, Path("<corpus-record>"), 1).text,
                        text=True,
                    )
                else:
                    rendered = rendered_digest(payload)
            item["origin"] = origin
            item["modalities"] = sorted(
                {by_id[parent]["modality"] for parent in item["parent_document_ids"]}
                or {"text"}
            )
            item["origin_schema_version"] = 1
            item["shape"] = shape_for_record(
                kind,
                shape_id,
                domains=item.get("domains", []),
                parent_document_ids=item["parent_document_ids"],
                scenario_id=item.get("scenario_id"),
            )
            item["verification"] = verification(status, method, proof)
            item["validation_status"] = status
            item["rendered_sha256"] = rendered
            if kind == "chat_sft" and "generation_id" in payload:
                item["generator_identity"] = payloads[payload["generation_id"]][
                    "generator_identity"
                ]
                item["generator"] = payloads[payload["generation_id"]]["generator"]
            elif kind == "generation":
                item["generator"] = payload["generator"]
            validate_lineage(item, documents=by_id)
        _jsonl(staging / "documents.jsonl", documents)
        _jsonl(staging / "spans.jsonl", evidence)
        _jsonl(staging / "lineage.jsonl", sorted(lineage, key=lambda r: r["record_id"]))
        _jsonl(staging / "lexical/candidates.jsonl", lexical)
        _jsonl(staging / "semantic/candidates.jsonl", semantic)
        _jsonl(staging / "scenarios.jsonl", scenarios)
        _jsonl(staging / "generations.jsonl", generations)
        _jsonl(staging / "tool_episodes.jsonl", tools)
        _jsonl(staging / "rejected.jsonl", rejected)
        _jsonl(staging / "chat/records.jsonl", chats)
        ledger_by_id = {item["record_id"]: item for item in lineage}
        release_spec = _model(project.release)
        include_shapes = release_spec.get("include_shapes")
        include_origins = release_spec.get("include_origins")

        def included(record_id: str) -> bool:
            item = ledger_by_id[record_id]
            return (
                include_shapes is None or item["shape"]["id"] in include_shapes
            ) and (include_origins is None or item["origin"] in include_origins)

        for split in SPLITS:
            chosen_lm = [
                d
                for d in kept
                if d["split"] == split
                and any(
                    t.kind == "lm_text" and (not t.inputs or d["source_id"] in t.inputs)
                    for t in transforms
                )
                and included(d["document_id"])
            ]
            chosen_chat = [
                row
                for row in chats
                if row["split"] == split and included(row["record_id"])
            ]
            _jsonl(
                staging / "lm" / f"{split}.jsonl",
                [{"text": d["text"]} for d in chosen_lm],
            )
            _jsonl(
                staging / "lm" / f"{split}.lineage.jsonl",
                [{"record_id": d["document_id"], "split": split} for d in chosen_lm],
            )
            _jsonl(
                staging / "chat" / f"{split}.jsonl",
                [
                    {k: row[k] for k in ("format_version", "loss_mode", "messages")}
                    for row in chosen_chat
                ],
            )
            _jsonl(
                staging / "chat" / f"{split}.lineage.jsonl",
                [
                    {"record_id": row["record_id"], "split": split}
                    for row in chosen_chat
                ],
            )
        fraction = project.release.fraction
        if fraction is not None:
            from tokenizers import Tokenizer

            from sparselab.corpus.project import project_path
            from sparselab.corpus.selection import select_fraction
            from sparselab.data.conversations import iter_rendered_conversations

            tokenizer_path = project_path(project.root, fraction.tokenizer_path)
            if sha256_file(tokenizer_path) != fraction.tokenizer_sha256.lower():
                raise ValueError("fraction tokenizer SHA-256 changed during build")
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            candidates = []
            for view in ("lm", "chat"):
                selected_view = getattr(project.release, view)
                if (
                    not selected_view.selected
                    or "train" not in selected_view.training_splits
                ):
                    continue
                payload_path = staging / view / "train.jsonl"
                links = _rows(staging / view / "train.lineage.jsonl")
                texts = (
                    [row["text"] for row in _rows(payload_path)]
                    if view == "lm"
                    else [row.text for row in iter_rendered_conversations(payload_path)]
                )
                for link, text in zip(links, texts, strict=True):
                    record = ledger_by_id[link["record_id"]]
                    candidates.append(
                        {
                            "record_id": link["record_id"],
                            "tokens": len(tokenizer.encode(text).ids),
                            "origin": record["origin"],
                            "source_family_ids": record["source_family_ids"],
                        }
                    )
            chosen_ids = select_fraction(
                candidates, fraction.generated_share, fraction.train_tokens
            )
            for view in ("lm", "chat"):
                selected_view = getattr(project.release, view)
                if (
                    not selected_view.selected
                    or "train" not in selected_view.training_splits
                ):
                    continue
                path = staging / view / "train.jsonl"
                links_path = staging / view / "train.lineage.jsonl"
                pairs = [
                    (row, link)
                    for row, link in zip(_rows(path), _rows(links_path), strict=True)
                    if link["record_id"] in chosen_ids
                ]
                _jsonl(path, [row for row, _ in pairs])
                _jsonl(links_path, [link for _, link in pairs])
        from sparselab.data.conversations import iter_rendered_conversations

        for split in SPLITS:
            list(iter_rendered_conversations(staging / "chat" / f"{split}.jsonl"))
        warnings = []
        leakage = []
        for field in (
            "parent_document_ids",
            "source_family_ids",
            "scenario_family_id",
            "generator_world_id",
            "template_family_id",
        ):
            families: dict[str, set[str]] = defaultdict(set)
            for item in lineage:
                if item["split"] == "inventory":
                    continue
                value = item.get(field)
                for family in (
                    value if isinstance(value, list) else ([value] if value else [])
                ):
                    families[family].add(item["split"])
            for family, assignments in sorted(families.items()):
                if len(assignments) < 2:
                    continue
                overlap = {
                    "family_type": field,
                    "family_id": family,
                    "splits": sorted(assignments),
                }
                leakage.append(overlap)
                if (
                    (policy["unit"] == "document" and field == "parent_document_ids")
                    or (
                        policy["unit"]
                        in {"source_document_family", "source_repository"}
                        and field == "source_family_ids"
                    )
                    or (
                        policy["unit"] == "scenario_family"
                        and field == "scenario_family_id"
                    )
                    or (
                        policy["unit"] == "generator_world"
                        and field == "generator_world_id"
                    )
                    or (
                        policy["unit"] == "template_family"
                        and field == "template_family_id"
                    )
                ):
                    _json(
                        root / "diagnostics" / f"{build_id}.json",
                        {"error": "shared split policy unit", "leakage": leakage},
                    )
                    raise ValueError(f"split policy unit {family} spans splits")
                warnings.append(
                    f"{field} {family} overlaps {', '.join(sorted(assignments))}"
                )
        audit = {
            "duplicates": duplicate_groups,
            "rejected": rejected,
            "warnings": warnings,
            "drop_count": len(documents) - len(kept),
            "leakage": leakage,
        }
        _json(staging / "audit.json", audit)
        _json(staging / "splits.json", policy)
        sources = [
            {
                "id": s.id,
                "kind": s.kind,
                "canonical_uri": s.canonical_uri,
                "revision": s.revision,
                "license": s.license,
                "redistribution": s.rights.redistribution_mode
                if s.rights
                else s.redistribution,
                **(
                    {
                        "license_url": s.license_url,
                        "rights_policy": s.rights.model_dump(mode="json"),
                    }
                    if s.rights
                    else {}
                ),
                "origin": s.origin,
                "source_family": s.source_family,
                "snapshot_sha256": lock["sources"].get(s.id, {}).get("snapshot_sha256"),
                "reproducibility_class": (
                    "generator_dependent"
                    if s.kind == "inference_generator"
                    else "script_reproducible_not_redistributed"
                    if (s.rights.redistribution_mode if s.rights else s.redistribution)
                    in {
                        "reference_only",
                        "derived_only",
                        "unknown",
                        "rejected",
                        "metadata_reconstruction_only",
                        "not_redistributable",
                        "review_required",
                    }
                    else "fully_reproducible"
                ),
            }
            for s in project.sources
        ]
        _json(staging / "sources.json", sources)
        if project.release.schema_version == 2:
            file_decisions = [
                item for item in rights_files if item["role"] == "document"
            ]
            rights_counts = Counter(
                item["rights"]["training_eligibility"] for item in file_decisions
            )
            bytes_by_state: Counter[str] = Counter()
            spdx_counts: Counter[str] = Counter()
            modes: Counter[str] = Counter()
            source_spdx = {s.id: s.rights.spdx_expression for s in project.sources}
            for item in file_decisions:
                decision = item["rights"]
                bytes_by_state[decision["training_eligibility"]] += item["size"]
                spdx_counts[
                    decision["detected_spdx_expression"]
                    or source_spdx[item["source_id"]]
                    or "unknown"
                ] += 1
                modes[decision["redistribution_mode"]] += 1
            rights_report = {
                "schema_version": 2,
                "publication_mode": project.release.publication_mode,
                "weight_license_status": "separate_analysis_required",
                "sources": sources,
                "files": rights_files,
                "training_eligibility": {
                    state: {
                        "files": rights_counts[state],
                        "source_bytes": bytes_by_state[state],
                        "source_tokens": None,
                        "token_count_reason": "tokenizer_not_declared",
                    }
                    for state in (
                        "eligible",
                        "eligible_with_obligations",
                        "review_required",
                        "ineligible",
                    )
                },
                "spdx_expressions": dict(spdx_counts),
                "redistribution_modes": dict(modes),
                "unresolved_rights_files": rights_counts["review_required"],
                "advisory": "Source/derived-data rights and model-weight licensing are separate decisions; not a legal conclusion.",
            }
        else:
            rights_report = {
                "sources": sources,
                "redistribution_classes": dict(
                    Counter(s.redistribution for s in project.sources)
                ),
                "advisory": "Declared attribution and redistribution classifications are inventory metadata, not legal review.",
            }
        _json(staging / "license-report.json", rights_report)
        mixture = _model(project.release).get("mixture", {})
        actual = {
            domain: {
                "documents": sum(domain in d["domains"] for d in kept),
                "utf8_bytes": sum(
                    len(d["text"].encode()) for d in kept if domain in d["domains"]
                ),
                "actual_tokens": None,
                "token_count_reason": "tokenizer_not_declared",
            }
            for domain in sorted(
                set(mixture) | {domain for d in kept for domain in d["domains"]}
            )
        }
        generation_statuses = Counter(g["validation_status"] for g in generations)
        generation_statuses["oracle_verified"] += len(scenarios)
        generation_total = len(generations) + len(scenarios)
        source_counts = Counter(d["source_id"] for d in kept)
        report = {
            "schema_version": project.release.schema_version,
            "corpus_id": project.config.id,
            "requested_mixture": mixture,
            "actual_mixture": actual,
            "document_count": len(documents),
            "kept_document_count": len(kept),
            "split_counts": {s: sum(d["split"] == s for d in kept) for s in SPLITS},
            "view_counts": {
                view: {
                    split: len(_rows(staging / view / f"{split}.jsonl"))
                    for split in SPLITS
                }
                for view in ("lm", "chat")
            },
            "source_counts": dict(source_counts),
            "source_concentration": {
                source: count / len(kept)
                for source, count in sorted(source_counts.items())
            }
            if kept
            else {},
            "kind_counts": dict(Counter(d["document_kind"] for d in kept)),
            "lexical_count": len(lexical),
            "semantic_count": len(semantic),
            "chat_count": len(chats),
            "generation_statuses": dict(generation_statuses),
            "generation_validation_ratios": {
                status: count / generation_total
                for status, count in sorted(generation_statuses.items())
            }
            if generation_total
            else {},
            "generator_count": len({g["generator_identity"] for g in generations}),
            "scenario_count": len(scenarios),
            "dedupe": {
                "groups": len(duplicate_groups),
                "dropped": len(documents) - len(kept),
            },
            "warnings": warnings,
            "token_count_reason": None
            if fraction is not None
            else "tokenizer_not_declared",
        }
        if project.release.schema_version == 2:
            report["rights"] = {
                key: value
                for key, value in rights_report.items()
                if key
                in {
                    "training_eligibility",
                    "spdx_expressions",
                    "redistribution_modes",
                    "unresolved_rights_files",
                    "weight_license_status",
                    "publication_mode",
                }
            }
        from sparselab.corpus.measurement import summarize_release

        report["measurement"] = summarize_release(
            staging,
            release_spec=release_spec,
            tokenizer=tokenizer_path if fraction is not None else None,
        )
        _json(staging / "report.json", report)
        (staging / "report.md").write_text(
            f"# Corpus {project.config.id}\n\nDocuments: {len(kept)} retained / {len(documents)} total.\n\n"
            + (
                f"Training tokens: {report['measurement']['training_mixture']['actual_tokens']} with pinned tokenizer {fraction.tokenizer_sha256}.\n"
                if fraction is not None
                else "No tokenizer declared; token counts unavailable.\n"
            ),
            encoding="utf-8",
        )
        _json(
            staging / "build.json",
            {
                "schema_version": 1,
                "build_id": build_id,
                "identity": identity,
                "stages": stages,
                "files": {
                    str(p.relative_to(staging)): {
                        "sha256": sha256_file(p),
                        "size": p.stat().st_size,
                    }
                    for p in sorted(staging.rglob("*"))
                    if p.is_file()
                },
                "snapshots": [
                    {"source_id": key, "sha256": v["snapshot_sha256"]}
                    for key, v in sorted(lock["sources"].items())
                    if v.get("snapshot_path")
                ],
            },
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
        _rename_noreplace(staging, target)
        _sync_dir(root)
    return target
