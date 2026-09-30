"""Conservative, file-level rights decisions for declared corpus sources.

``nested_metadata`` maps repository-relative ``license-metadata.json`` paths
to decoded JSON documents. Supports Rust REUSE's hierarchical ``files`` root
with directory/file/group nodes, nested children, and license ``spdx`` objects,
as well as flat ``files`` lists with path(s), license and directory prefixes.
The most specific matching file/group/directory overrides ancestor metadata.
Contradictions at equal specificity require review. Metadata is never loaded
implicitly from the filesystem.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import field_validator, model_validator

from sparselab.config.models import StrictModel

TrainingEligibility = Literal[
    "eligible", "eligible_with_obligations", "review_required", "ineligible"
]
RedistributionMode = Literal[
    "redistributable_under_source_terms",
    "metadata_reconstruction_only",
    "derived_only",
    "not_redistributable",
    "review_required",
]
WeightLicenseStatus = Literal["separate_analysis_required"]


class TrainingRestriction(StrictModel):
    kind: Literal["prohibited", "conditional"]
    basis: str

    @field_validator("basis")
    @classmethod
    def nonblank_basis(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("training restriction requires a nonblank basis")
        return value


class RightsPolicy(StrictModel):
    training_eligibility: TrainingEligibility
    redistribution_mode: RedistributionMode
    weight_license_status: WeightLicenseStatus = "separate_analysis_required"
    spdx_expression: str | None = None
    license_references: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()
    training_restriction: TrainingRestriction | None = None
    allowed_boundary_paths: tuple[str, ...] = ()
    nested_metadata_path: str | None = None

    @field_validator("nested_metadata_path")
    @classmethod
    def safe_metadata_path(cls, value: str | None) -> str | None:
        if value is not None:
            _safe_path(value)
            if PurePosixPath(value).name != "license-metadata.json":
                raise ValueError("nested_metadata_path must name license-metadata.json")
        return value

    @field_validator("spdx_expression")
    @classmethod
    def nonblank_spdx(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("SPDX expression must not be blank")
        return value

    @field_validator("license_references", "notices")
    @classmethod
    def nonblank_entries(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("rights references and notices must not be blank")
        return values

    @field_validator("allowed_boundary_paths")
    @classmethod
    def safe_allowlist(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _safe_path(value.rstrip("/"))
        return values

    @model_validator(mode="after")
    def consistent_restriction(self) -> RightsPolicy:
        if (
            self.training_restriction is not None
            and self.training_restriction.kind == "prohibited"
            and self.training_eligibility != "ineligible"
        ):
            raise ValueError("prohibited training requires ineligible policy")
        if (
            self.training_restriction is not None
            and self.training_restriction.kind == "conditional"
            and self.training_eligibility in ("eligible", "eligible_with_obligations")
        ):
            raise ValueError("conditional training cannot be declared eligible")
        return self


class FileRights(StrictModel):
    path: str
    detected_spdx_expression: str | None
    training_eligibility: TrainingEligibility
    redistribution_mode: RedistributionMode
    weight_license_status: WeightLicenseStatus
    license_references: tuple[str, ...]
    notices: tuple[str, ...]
    training_restriction: TrainingRestriction | None
    boundary_flags: tuple[str, ...]
    reason: str


# The registry is intentionally finite: absence of recognition is a review, not
# an inference from a project name or from a nearly matching identifier.
_PERMISSIVE = frozenset(
    {
        "MIT",
        "MIT-0",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "ISC",
        "Zlib",
        "Unlicense",
        "PSF-2.0",
        "Python-2.0",
        "CC0-1.0",
        "BSL-1.0",
        "Artistic-2.0",
    }
)
_ATTRIBUTION = frozenset(
    {"CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0", "ODC-By-1.0"}
)
_COPYLEFT = frozenset(
    {
        "MPL-2.0",
        "GPL-2.0-only",
        "GPL-2.0",
        "GPL-2.0+",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0",
        "GPL-3.0+",
        "GPL-3.0-or-later",
        "LGPL-2.1-only",
        "LGPL-2.1",
        "LGPL-2.1+",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0",
        "LGPL-3.0+",
        "LGPL-3.0-or-later",
        "AGPL-3.0-only",
        "AGPL-3.0",
        "AGPL-3.0+",
        "AGPL-3.0-or-later",
    }
)
_RESTRICTED = frozenset(
    {
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-ND-4.0",
        "CC-BY-NC-ND-4.0",
        "CC-BY-NC-3.0",
        "CC-BY-NC-SA-3.0",
        "CC-BY-ND-3.0",
        "CC-BY-NC-ND-3.0",
    }
)
_TOKEN = re.compile(r"\(|\)|AND\b|OR\b|WITH\b|[A-Za-z0-9][A-Za-z0-9.\-+]*", re.ASCII)
_SPDX_LINE = re.compile(r"SPDX-License-Identifier\s*:\s*([^\r\n]+)", re.IGNORECASE)
_COPYRIGHT_LINE = re.compile(r"\b(?:SPDX-FileCopyrightText|Copyright\b)", re.IGNORECASE)
_BOUNDARIES = frozenset(
    {
        "vendor",
        "third_party",
        "third-party",
        "external",
        "generated",
        "bundled",
        "fixtures",
    }
)


def _safe_path(path: str) -> str:
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
        or PurePosixPath(path).as_posix() != path
    ):
        raise ValueError(f"unsafe rights path: {path!r}")
    return path


def _license_class(expression: str) -> str:
    """Parse a bounded SPDX grammar; no best-case treatment of OR branches."""
    tokens = _TOKEN.findall(expression)
    if not tokens or "".join(tokens) != re.sub(r"\s+", "", expression):
        return "unknown"
    position = 0

    def atom() -> set[str]:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("missing license")
        token = tokens[position]
        position += 1
        if token == "(":
            result = disjunction()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("unclosed SPDX expression")
            position += 1
            return result
        if token in ("AND", "OR", "WITH", ")"):
            raise ValueError("missing license identifier")
        if position < len(tokens) and tokens[position] == "WITH":
            position += 1
            if position >= len(tokens):
                raise ValueError("missing SPDX exception")
            exception = tokens[position]
            position += 1
            if exception != "Linux-syscall-note" or token not in {
                "GPL-2.0", "GPL-2.0-only", "GPL-2.0+", "GPL-2.0-or-later"
            }:
                return {"unknown"}
        if token in _PERMISSIVE:
            return {"permissive"}
        if token in _ATTRIBUTION or token in _COPYLEFT:
            return {"obligations"}
        if token in _RESTRICTED:
            return {"restricted"}
        return {"unknown"}

    def conjunction() -> set[str]:
        nonlocal position
        result = atom()
        while position < len(tokens) and tokens[position] == "AND":
            position += 1
            result |= atom()
        return result

    def disjunction() -> set[str]:
        nonlocal position
        result = conjunction()
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            result |= conjunction()
        return result

    try:
        classes = disjunction()
        if position != len(tokens):
            return "unknown"
    except ValueError:
        return "unknown"
    if "restricted" in classes:
        return "restricted"
    if "unknown" in classes:
        return "unknown"
    if "obligations" in classes:
        return "obligations"
    return "permissive"


def _metadata_expression(
    path: str, nested_metadata: Mapping[str, Any]
) -> tuple[str | None, bool]:
    matches: list[tuple[int, int, str]] = []

    def add(
        scope: str, license_id: Any, *, directory: bool, metadata_depth: int
    ) -> None:
        if isinstance(license_id, Mapping):
            license_id = license_id.get("spdx")
        if not isinstance(license_id, str) or not license_id.strip():
            raise ValueError("license metadata requires a nonblank SPDX expression")
        if path == scope or (directory and (not scope or path.startswith(scope + "/"))):
            matches.append((metadata_depth, len(scope), license_id))

    def walk(node: Any, base: str, metadata_depth: int) -> None:
        if not isinstance(node, Mapping):
            raise TypeError("license metadata nodes must be objects")
        kind = node.get("type")
        if kind == "root":
            for child in node.get("children", []):
                walk(child, base, metadata_depth)
            return
        if kind in ("directory", "file"):
            name = node.get("name")
            if not isinstance(name, str):
                raise ValueError("license metadata node requires a name")
            scope = (
                base
                if name == "."
                else "/".join(filter(None, (base, _safe_path(name))))
            )
            if "license" in node:
                add(
                    scope,
                    node["license"],
                    directory=kind == "directory",
                    metadata_depth=metadata_depth,
                )
            for child in node.get("children", []):
                walk(child, scope, metadata_depth)
            return
        if kind == "group":
            for name in node.get("files", []):
                add(
                    "/".join(filter(None, (base, _safe_path(name)))),
                    node.get("license"),
                    directory=False,
                    metadata_depth=metadata_depth,
                )
            for name in node.get("directories", []):
                add(
                    "/".join(filter(None, (base, _safe_path(name.rstrip("/"))))),
                    node.get("license"),
                    directory=True,
                    metadata_depth=metadata_depth,
                )
            return
        raise ValueError(f"unrecognized license metadata node: {kind!r}")

    for metadata_path, document in nested_metadata.items():
        _safe_path(metadata_path)
        if PurePosixPath(metadata_path).name != "license-metadata.json":
            raise ValueError("nested metadata keys must name license-metadata.json")
        if not isinstance(document, Mapping):
            raise TypeError("license metadata must be an object")
        entries = document.get("files")
        parent = PurePosixPath(metadata_path).parent
        base = "" if parent.as_posix() == "." else parent.as_posix()
        depth = len(parent.parts)
        if isinstance(entries, Mapping):
            walk(entries, base, depth)
        elif isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise TypeError("license metadata entries must be objects")
                license_id = entry.get("license", entry.get("spdx_expression"))
                paths = entry.get(
                    "paths", entry.get("files", entry.get("path", entry.get("name")))
                )
                if not isinstance(paths, (str, list, tuple)):
                    raise TypeError("license metadata entry requires path(s)")
                if isinstance(paths, str):
                    paths = [paths]
                for relative in paths:
                    if not isinstance(relative, str):
                        raise TypeError("license metadata path must be text")
                    directory = relative.endswith("/")
                    clean = _safe_path(relative.rstrip("/"))
                    add(
                        "/".join(filter(None, (base, clean))),
                        license_id,
                        directory=directory,
                        metadata_depth=depth,
                    )
        else:
            raise TypeError("license metadata requires a files root or list")
    if not matches:
        return None, False
    best = max((depth, length) for depth, length, _ in matches)
    selected = {value for depth, length, value in matches if (depth, length) == best}
    return (next(iter(selected)), False) if len(selected) == 1 else (None, True)


def resolve_file_rights(
    policy: RightsPolicy,
    path: str,
    raw_bytes: bytes,
    *,
    nested_metadata: Mapping[str, Any] | None = None,
    prospective_private_research: bool = False,
) -> FileRights:
    """Resolve pinned file evidence under the declared release training-use policy."""
    _safe_path(path)
    flags = tuple(
        sorted({part for part in path.split("/") if part.lower() in _BOUNDARIES})
    )
    allowed = any(
        path == item.rstrip("/") or (item.endswith("/") and path.startswith(item))
        for item in policy.allowed_boundary_paths
    )
    # Dataset rows are not file-level license headers. Avoid copying every line
    # of a multi-gigabyte selected shard just to inspect a handful of notices.
    header: list[bytes] = []
    if not path.endswith((".jsonl", ".json", ".parquet")):
        offset = 0
        for _ in range(30):
            end = raw_bytes.find(b"\n", offset)
            if end < 0:
                header.append(raw_bytes[offset:])
                break
            header.append(raw_bytes[offset:end])
            offset = end + 1
    expressions = set()
    file_notices: list[str] = []
    for line in header:
        text = line.decode("utf-8", errors="replace").strip()
        match = _SPDX_LINE.search(text)
        if match:
            expressions.add(
                re.sub(r"(?:\s*\*/|\s*-->)\s*$", "", match.group(1)).strip()
            )
        if _COPYRIGHT_LINE.search(text):
            file_notices.append(text)
    metadata_license, metadata_conflict = _metadata_expression(
        path, nested_metadata or {}
    )
    detected = next(iter(expressions)) if len(expressions) == 1 else None
    if metadata_license is not None:
        if detected is not None and detected != metadata_license:
            metadata_conflict = True
        detected = metadata_license if detected is None else detected
    effective = detected or policy.spdx_expression
    reasons: list[str] = []
    if flags and not allowed:
        reasons.append("external or generated path boundary: " + ", ".join(flags))
    if len(expressions) > 1 or metadata_conflict:
        reasons.append("conflicting file license evidence")
    # A recognized file-level expression supersedes a repository default.
    # Only contradictory file-vs-metadata evidence is a conflict; legitimate
    # per-file license exceptions must remain separately classifiable.
    classification = _license_class(effective) if effective is not None else "unknown"
    if classification == "unknown" and not prospective_private_research:
        reasons.append("unknown or unrecognized SPDX expression")
    elif classification == "restricted" and not prospective_private_research:
        reasons.append("restrictive SPDX license requires review")
    if policy.training_restriction is not None:
        reasons.append(
            f"{policy.training_restriction.kind} ML training: {policy.training_restriction.basis}"
        )
    if policy.training_eligibility == "ineligible" or (
        policy.training_restriction is not None
        and policy.training_restriction.kind == "prohibited"
    ):
        training: TrainingEligibility = "ineligible"
    elif policy.training_eligibility == "review_required" or reasons:
        training = "review_required"
    elif (
        policy.training_eligibility == "eligible_with_obligations"
        or classification == "obligations"
        or (prospective_private_research and classification in {"unknown", "restricted"})
        or (classification == "permissive" and not policy.notices)
    ):
        training = "eligible_with_obligations"
    else:
        training = "eligible"
    redistribution = policy.redistribution_mode
    if reasons and redistribution == "redistributable_under_source_terms":
        redistribution = "review_required"
    if not reasons:
        reasons.append(
            "private research candidate; license obligations and explicit source terms retained"
            if prospective_private_research
            else "recognized SPDX under declared source terms"
        )
    return FileRights(
        path=path,
        detected_spdx_expression=detected,
        training_eligibility=training,
        redistribution_mode=redistribution,
        weight_license_status=policy.weight_license_status,
        license_references=policy.license_references,
        notices=(*policy.notices, *file_notices),
        training_restriction=policy.training_restriction,
        boundary_flags=flags,
        reason="; ".join(reasons),
    )
