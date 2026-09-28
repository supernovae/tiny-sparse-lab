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
from sparselab.training.manifest import canonical_json, sha256_file


class NormalizedDocument(StrictModel):
    """Versioned document retaining source attribution and raw evidence identity."""

    schema_version: Literal[1]
    document_id: str
    source_id: str
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
        "source_grounded",
        "oracle_verified",
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
    raw: bytes, name: str, source: Any, snapshot_id: str
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
            "schema_version": 1,
            "document_id": identity,
            "source_id": source.id,
            "source_revision": source.revision,
            "source_location": location,
            "license": source.license,
            "redistribution": source.redistribution,
            "domains": list(source.domains),
            "document_kind": next(iter(source.document_kinds)),
            "title": ancestry[-1] if ancestry else Path(name).name,
            "section_path": ancestry,
            "language": "en",
            "text": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "raw_content_sha256": raw_content_sha,
            "source_family": source.source_family,
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
        "validation_status": record.get("validation_status", "source_grounded"),
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
            for file in sorted(snapshot["files"], key=lambda f: f["path"]):
                if (source.id, file["path"]) in auxiliary_files:
                    continue
                raw = (
                    Path(entry["snapshot_path"]) / "files" / file["path"]
                ).read_bytes()
                try:
                    pairs = _records_for_file(
                        raw, file["path"], source, entry["snapshot_sha256"]
                    )
                    documents.extend(doc for doc, _ in pairs)
                    evidence.extend(span for _, span in pairs)
                except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                    rejected.append(
                        {
                            "source_id": source.id,
                            "path": file["path"],
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
                            "validation_status": "source_grounded",
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
                        "validation_status": "source_grounded",
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
                        status = "source_grounded" if start >= 0 else "unverified"
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
                            if status == "source_grounded"
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
                        ["source_grounded", "oracle_verified", "human_reviewed"],
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
        for split in SPLITS:
            chosen_lm = [
                d
                for d in kept
                if d["split"] == split
                and any(
                    t.kind == "lm_text" and (not t.inputs or d["source_id"] in t.inputs)
                    for t in transforms
                )
            ]
            chosen_chat = [row for row in chats if row["split"] == split]
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
        from sparselab.data.conversations import iter_rendered_conversations

        for split in SPLITS:
            list(iter_rendered_conversations(staging / "chat" / f"{split}.jsonl"))
        warnings = []
        leakage = []
        for field in (
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
                    (
                        policy["unit"] == "source_document_family"
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
                "redistribution": s.redistribution,
                "source_family": s.source_family,
                "snapshot_sha256": lock["sources"].get(s.id, {}).get("snapshot_sha256"),
                "reproducibility_class": (
                    "generator_dependent"
                    if s.kind == "inference_generator"
                    else "script_reproducible_not_redistributed"
                    if s.redistribution
                    in {"reference_only", "derived_only", "unknown", "rejected"}
                    else "fully_reproducible"
                ),
            }
            for s in project.sources
        ]
        _json(staging / "sources.json", sources)
        _json(
            staging / "license-report.json",
            {
                "sources": sources,
                "redistribution_classes": dict(
                    Counter(s.redistribution for s in project.sources)
                ),
                "advisory": "Declared attribution and redistribution classifications are inventory metadata, not legal review.",
            },
        )
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
            "schema_version": 1,
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
            "token_count_reason": "tokenizer_not_declared",
        }
        _json(staging / "report.json", report)
        (staging / "report.md").write_text(
            f"# Corpus {project.config.id}\n\nDocuments: {len(kept)} retained / {len(documents)} total.\n\nNo tokenizer declared; token counts unavailable.\n",
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
