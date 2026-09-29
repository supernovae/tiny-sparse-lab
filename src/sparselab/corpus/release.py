"""Immutable publication, verification and read-only corpus inspection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from sparselab.training.manifest import canonical_json, sha256_file


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
    if report.get("schema_version") != 2 or report.get("sources") != list(
        sources.values()
    ):
        raise ValueError("prospective rights report/source mismatch")
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
            raw = _safe(folder, path).read_bytes()
            resolved = resolve_file_rights(policy, path, raw, nested_metadata=metadata)
            if recorded["rights"] != resolved.model_dump(mode="json") or (
                recorded.get("license_url") != source["license_url"]
            ):
                raise ValueError("rights decision differs from pinned file evidence")
    return indexed


def _validate_rows(root: Path) -> None:
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
            root.parent.parent / "snapshots" / source["id"] / snapshot_id
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
                declaration.get("schema_version") != 2
                or source["rights_policy"] != declaration["rights"]
                or source["license_url"] != declaration["license_url"]
                or source["redistribution"]
                != declaration["rights"]["redistribution_mode"]
            ):
                raise ValueError("prospective source rights declaration mismatch")
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
                decision["training_eligibility"]
                not in ("eligible", "eligible_with_obligations")
                or doc.get("schema_version") != 2
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
                document_map[row["record_id"]]["domains"]
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
                row.get("domains") != expected_domains
                or row["shape"]["attributes"]["source_domains"] != expected_domains
            ):
                raise ValueError("shape source domain attribution mismatch")
            evidence = row["verification"]["evidence"]
            if row["verification"]["status"] == "oracle_verified":
                scenario = scenarios.get(row.get("scenario_id", row["record_id"]))
                if scenario is None:
                    raise ValueError("oracle scenario reference mismatch")
                world = scenario["world_state"]
                expected_result = (
                    PurePosixPath(world["path"]).suffix
                    if world.get("operation") == "suffix"
                    else None
                )
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
                "scenario": "troubleshooting_scenario",
                "generation": "direct_qa",
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
                    or generation["parsed_output"]["answer"] != evidence["passage"]
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
        rows = _rows(output)
        if (
            sha256_file(output) != stage["output_sha256"]
            or len(rows) != stage["output_count"]
            or _digest([stage["id"], rows]) != stage["output_id"]
        ):
            raise ValueError("stage receipt does not match output")


def verify_build(path: Path) -> dict[str, Any]:
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
            / snapshot["sha256"]
        )
        if receipt["snapshot_sha256"] != snapshot["sha256"]:
            raise ValueError("build snapshot identity mismatch")
    _validate_rows(path)
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


def freeze(build_dir: Path, work_root: Path) -> Path:
    """Publish a verified build at its content-derived full SHA-256, never replacing it."""
    build_dir = Path(build_dir).resolve()
    build = verify_build(build_dir)
    release_config = build["identity"]["release"]
    for view in ("lm", "chat"):
        selected = release_config[view]
        if not selected.get("selected", False):
            continue
        for split in selected.get("training_splits", ("train", "validation")):
            if split not in {"train", "validation"}:
                raise ValueError("test split cannot be a training view")
            if not (build_dir / view / f"{split}.jsonl").read_bytes().strip():
                raise ValueError(f"selected {view}/{split} training view is empty")
    payload = _manifest_payload(build)
    release_id = _digest(payload)
    destination = (
        Path(work_root) / "corpora" / payload["corpus_id"] / "releases" / release_id
    )
    if destination.exists():
        verify_release(destination)
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
        verify_release(staging, expected_id=release_id)
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
            verify_release(destination)
    return destination


def verify_release(path: Path, *, expected_id: str | None = None) -> dict[str, Any]:
    """Verify the complete release including snapshotted source and lineage evidence."""
    path = Path(path).resolve()
    manifest = _load(path / "manifest.json")
    release_id = manifest.get("release_id")
    if release_id != (expected_id or path.name) or len(release_id) != 64:
        raise ValueError("release directory identity mismatch")
    if (
        _digest({key: value for key, value in manifest.items() if key != "release_id"})
        != release_id
    ):
        raise ValueError("release manifest identity mismatch")
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
            / snapshot["sha256"]
        )
        if receipt["snapshot_sha256"] != snapshot["sha256"]:
            raise ValueError("release snapshot identity mismatch")
    _validate_rows(path)
    return manifest


def describe(path: Path, *, tokenizer: Path | None = None) -> dict[str, Any]:
    manifest = verify_release(path)
    report = _load(Path(path) / "report.json")
    from sparselab.corpus.measurement import measure_views

    measured = measure_views(
        Path(path), tokenizer, release_spec=manifest["build_identity"]["release"]
    )
    if report.get("schema_version") == 2 and tokenizer is not None:
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
