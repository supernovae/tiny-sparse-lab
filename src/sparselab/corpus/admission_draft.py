"""Conservative admission draft from verified bounded acquisition snapshots.

This prepares decisions for policy review. It never turns a flagged or
conflicting record into an eligible one, and the existing admission verifier
remains the final build/release gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from sparselab.corpus.acquisition import verify_acquisition
from sparselab.corpus.project import Project, source_declaration_payload
from sparselab.corpus.rights import (
    RightsPolicy,
    _admission_exceptions,
    _admission_profile,
    _git_file_exceptions,
    resolve_file_rights,
    verify_record_admission,
)
from sparselab.training.manifest import canonical_json, sha256_file


def _decision(value: str, reason: str) -> dict[str, Any]:
    return {"decision": value, "reason": reason, "issues": []}


def _issue(field: str, remedy: str) -> dict[str, str]:
    return {
        "field": field,
        "owner": "KML Card 03 source audit",
        "remedy": remedy,
        "decision_impact": "Optional descriptive gap; source-policy coverage and pinned row identity still govern local use.",
    }


def _optional_issues(record: dict[str, Any], profile: str) -> list[dict[str, str]]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return []
    issues: list[dict[str, str]] = []
    if profile == "gutenberg":
        if not metadata.get("edition"):
            issues.append(
                _issue(
                    "metadata.edition",
                    "Reconcile edition from the retained Gutenberg book ID and source URL before later redistribution.",
                )
            )
    else:
        authors = metadata.get("authors")
        if not isinstance(authors, list) or len(authors) <= 1:
            issues.append(
                _issue(
                    "metadata.authors",
                    "Use the canonical page history for attribution; reconcile the optional author list before later redistribution.",
                )
            )
        locator = metadata.get("url")
        if isinstance(locator, str):
            try:
                has_revision = "oldid" in parse_qs(urlsplit(locator).query)
            except ValueError:
                has_revision = False
            if not has_revision:
                issues.append(
                    _issue(
                        "metadata.url.oldid",
                        "Retain the pinned row digest and canonical page/history route; recover the optional revision query if needed.",
                    )
                )
    return issues


def _additional_draft_flags(record: dict[str, Any], profile: str) -> tuple[str, ...]:
    """Conservatively hold observed scale-audit patterns for manual review.

    These flags only narrow a draft; they do not change the historical rights
    verifier or turn a flagged row into an eligible one.
    """
    if profile != "wikimedia":
        return ()
    content = record.get("text")
    if not isinstance(content, str):
        return ()
    flags: list[str] = []
    if re.search(r'["“”]', content):
        flags.append("quoted_text_context")
    if re.search(
        r"(?i)\b(?:sexual orientation|coming out|came out|partner|"
        r"accused|alleged|allegations?|misconduct|scandal)\b",
        content,
    ):
        flags.append("sensitive_biography_or_allegation_context")
    return tuple(flags)


def draft_admission_manifest(
    project: Project,
    work_root: Path,
    template_path: Path,
    policy_document: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Draft one complete v2 manifest; flagged cases remain quarantined.

    The policy template supplies reviewed source policies, not row decisions:
    ``policy_id``, ``policy_sha256`` and one item per source with ``source_id``,
    ``license_label`` and ``rights``. The result binds immutable snapshots and
    covers every selected HF row and Git file. No rights review is inferred
    from this mechanical draft.
    """
    root = work_root.resolve()
    output = output_path.resolve()
    if not output.is_relative_to(root) or output.exists() or output_path.is_symlink():
        raise ValueError("admission draft requires a new output inside the work root")
    if template_path.is_symlink() or not template_path.is_file():
        raise ValueError("admission policy template is missing or symlinked")
    if policy_document.is_symlink() or not policy_document.is_file():
        raise ValueError("admission policy document is missing or symlinked")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    if (
        not isinstance(template, dict)
        or set(template) != {"policy_id", "policy_sha256", "sources"}
        or not isinstance(template["sources"], list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"source_id", "license_label", "rights"}
            for item in template["sources"]
        )
    ):
        raise ValueError("invalid admission policy template")
    if sha256_file(policy_document) != template["policy_sha256"]:
        raise ValueError("admission policy document SHA-256 mismatch")
    policies = {item["source_id"]: item for item in template["sources"]}
    if len(policies) != len(template["sources"]) or set(policies) != {
        source.id for source in project.sources
    }:
        raise ValueError("admission policy template source mismatch")

    lock = verify_acquisition(project, root)
    snapshots: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    counts = {"qualify": 0, "exclude": 0, "quarantine": 0}
    for source in project.sources:
        source_id = source.id
        row = lock["sources"][source_id]
        if row["snapshot_path"] is None:
            raise ValueError(f"missing acquired snapshot: {source_id}")
        snapshot_dir = Path(row["snapshot_path"])
        snapshot = json.loads(
            (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
        )
        snapshots[source_id] = snapshot
        policy = policies[source_id]
        rights = RightsPolicy.model_validate(policy["rights"])
        entry: dict[str, Any] = {
            "source_id": source_id,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "source_revision": source.revision,
            "license_label": policy["license_label"],
            "rights": rights.model_dump(mode="json"),
        }
        if source.kind == "huggingface_dataset":
            if len(snapshot["files"]) != 1 or len(snapshot["retrieval"]["shards"]) != 1:
                raise ValueError("admission draft requires one selected HF shard")
            file = snapshot["files"][0]
            shard = snapshot["retrieval"]["shards"][0]
            entry.update(
                sample_path=file["path"],
                sample_sha256=file["sha256"],
                source_shard_path=shard["source_shard_path"],
                source_shard_sha256=shard["source_shard_sha256"],
            )
            selected = shard["selected_rows"]
            decisions: list[dict[str, Any]] = []
            profile = _admission_profile(source_declaration_payload(source))
            with (snapshot_dir / "files" / file["path"]).open("rb") as stream:
                for receipt in selected:
                    line = stream.readline()
                    if not line:
                        raise ValueError("selected HF rows exceed snapshot file")
                    record = json.loads(line)
                    envelope = record.pop("_sparselab_source", None)
                    original_sha = hashlib.sha256(canonical_json(record)).hexdigest()
                    if (
                        not isinstance(envelope, dict)
                        or envelope.get("source_row_index")
                        != receipt["source_row_index"]
                        or original_sha != receipt["source_row_sha256"]
                    ):
                        raise ValueError("HF row differs from acquisition receipt")
                    conflict, flags = _admission_exceptions(record, profile)
                    flags = tuple(
                        sorted(
                            set(flags) | set(_additional_draft_flags(record, profile))
                        )
                    )
                    if conflict is not None:
                        state = (
                            "exclude"
                            if "outside content namespace" in conflict
                            else "quarantine"
                        )
                        choice = _decision(state, conflict)
                    elif flags:
                        choice = _decision(
                            "quarantine",
                            "manual exception review required: " + ", ".join(flags),
                        )
                    else:
                        choice = _decision(
                            "qualify",
                            "covered by reviewed source policy; no automated exception",
                        )
                    choice["issues"] = _optional_issues(record, profile)
                    choice.update(
                        source_row_index=receipt["source_row_index"],
                        source_row_sha256=original_sha,
                    )
                    decisions.append(choice)
                    counts[state if conflict is not None else choice["decision"]] += 1
                if stream.readline():
                    raise ValueError("snapshot has undeclared HF rows")
            entry["records"] = decisions
        elif source.kind == "git":
            if rights.spdx_expression is None:
                raise ValueError("Git admission policy requires SPDX expression")
            decisions = []
            for file in snapshot["files"]:
                path = file["path"]
                raw = (snapshot_dir / "files" / path).read_bytes()
                if hashlib.sha256(raw).hexdigest() != file["sha256"]:
                    raise ValueError("Git file differs from verified snapshot")
                material, flagged = _git_file_exceptions(raw, rights.spdx_expression)
                checked = resolve_file_rights(
                    rights,
                    path,
                    b"\n".join(raw.splitlines()[:30]) + b"\n",
                    prospective_private_research=True,
                )
                if path == "LICENSE" or Path(path).name == "README.md":
                    choice = _decision(
                        "exclude", "rights context, not incident training prose"
                    )
                elif material or checked.training_eligibility not in {
                    "eligible",
                    "eligible_with_obligations",
                }:
                    choice = _decision(
                        "quarantine", "material or unknown file-level rights exception"
                    )
                elif flagged:
                    choice = _decision(
                        "quarantine", "manual third-party/notice review required"
                    )
                else:
                    choice = _decision(
                        "qualify",
                        "covered by reviewed source policy; no automated exception",
                    )
                choice.update(path=path, sha256=file["sha256"])
                decisions.append(choice)
                counts[choice["decision"]] += 1
            entry["files"] = decisions
        else:
            raise ValueError("admission draft source kind is unsupported")
        entries.append(entry)

    manifest = {
        "schema_version": 2,
        "policy_id": template["policy_id"],
        "policy_sha256": template["policy_sha256"],
        "sources": entries,
    }
    verify_record_admission(
        manifest,
        {source.id: source_declaration_payload(source) for source in project.sources},
        snapshots,
        root / "corpora" / project.config.id / "snapshots",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json(manifest) + b"\n")
    return {
        "schema_version": 2,
        "path": str(output),
        "sha256": sha256_file(output),
        "counts": counts,
    }
