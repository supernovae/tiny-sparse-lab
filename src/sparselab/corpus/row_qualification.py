"""Declared HF row screening and family provenance; never grants review authority."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from sparselab.config.models import StrictModel
from sparselab.training.manifest import canonical_json

LEGACY_HF_ROW_URIS = {
    "https://huggingface.co/datasets/common-pile/project_gutenberg_filtered",
    "https://huggingface.co/datasets/common-pile/wikimedia_filtered",
}

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}


class RowQualification(StrictModel):
    profile: Literal["web_text", "synthetic_text", "code"]
    family_fields: tuple[str, ...] = Field(min_length=1, max_length=8)
    language_field: str | None = None
    allowed_languages: tuple[str, ...] = ()
    license_field: str | None = None
    license_labels: dict[str, str] = Field(default_factory=dict)
    repository_field: str | None = None
    path_field: str | None = None
    synthetic_provenance_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def complete(self):
        names = (
            *self.family_fields,
            *self.synthetic_provenance_fields,
            self.language_field,
            self.license_field,
            self.repository_field,
            self.path_field,
        )
        if any(
            name is not None
            and (
                len(name) > 512
                or len(name.split(".")) > 8
                or any(
                    not part
                    or part != part.strip()
                    or len(part) > 128
                    or part.startswith("_sparselab")
                    for part in name.split(".")
                )
            )
            for name in names
        ):
            raise ValueError("qualification requires bounded original row field names")
        if len(set(self.family_fields)) != len(self.family_fields):
            raise ValueError("duplicate qualification family field")
        if bool(self.language_field) != bool(self.allowed_languages) or any(
            not value.strip() for value in self.allowed_languages
        ):
            raise ValueError(
                "language field and explicit allowed values are required together"
            )
        if self.profile != "code" and any(
            (
                self.license_field,
                self.license_labels,
                self.repository_field,
                self.path_field,
            )
        ):
            raise ValueError("code-only qualification fields require the code profile")
        if self.profile != "synthetic_text" and self.synthetic_provenance_fields:
            raise ValueError(
                "synthetic provenance fields require the synthetic_text profile"
            )
        if self.profile == "synthetic_text" and not self.synthetic_provenance_fields:
            raise ValueError("synthetic rows require original provenance fields")
        if self.profile == "code" and (
            not all(
                (
                    self.language_field,
                    self.license_field,
                    self.repository_field,
                    self.path_field,
                )
            )
            or not self.license_labels
            or any(
                not key.strip() or value not in PERMISSIVE
                for key, value in self.license_labels.items()
            )
            or self.family_fields != (self.repository_field,)
        ):
            raise ValueError(
                "code qualification requires language, permissive license mapping, path and repository family"
            )
        return self


def _present(value: object) -> bool:
    return (
        value is not None
        and (not isinstance(value, str) or bool(value.strip()))
        and value != []
        and value != {}
    )


def field_value(row: dict, path: str | None):
    """Resolve an explicit dotted path through dictionaries, never arrays."""
    if path is None:
        return None
    value = row
    for segment in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(segment)
    return value


def family_hint(row: dict, policy: RowQualification) -> str:
    values = [field_value(row, field) for field in policy.family_fields]
    if policy.profile == "code" and isinstance(values[0], str):
        values[0] = values[0].casefold()
    if any(not _present(value) for value in values):
        raise ValueError("missing declared family provenance")
    return (
        policy.profile + "-family:" + hashlib.sha256(canonical_json(values)).hexdigest()
    )


def provenance(row: dict, policy: RowQualification) -> dict:
    result = {"profile": policy.profile, "family_hint": family_hint(row, policy)}
    if policy.profile == "synthetic_text":
        values = {
            key: field_value(row, key) for key in policy.synthetic_provenance_fields
        }
        if any(not _present(value) for value in values.values()):
            raise ValueError("missing synthetic provenance")
        result["synthetic_provenance_sha256"] = hashlib.sha256(
            canonical_json(values)
        ).hexdigest()
        result["synthetic_provenance_fields"] = list(values)
    if policy.profile == "code":
        result["attribution"] = {
            "repository": field_value(row, policy.repository_field),
            "path": field_value(row, policy.path_field),
            "license_label": field_value(row, policy.license_field),
            "language": field_value(row, policy.language_field),
            "license_expression": policy.license_labels.get(
                field_value(row, policy.license_field)
            )
            if isinstance(field_value(row, policy.license_field), str)
            else None,
        }
    return result


def exceptions(
    row: dict, policy: RowQualification, text_field: str
) -> tuple[str | None, tuple[str, ...]]:
    from sparselab.corpus.rights import _ADMISSION_PRIVATE, _git_file_exceptions

    text = row.get(text_field)
    if not isinstance(text, str) or not text.strip():
        return "empty or nontext declared content field", ()
    if _ADMISSION_PRIVATE.search(text):
        return "private identifier or credential in source row", ()
    try:
        provenance(row, policy)
    except TypeError, ValueError:
        return "missing or invalid declared family/synthetic provenance", ()
    if (
        policy.language_field
        and field_value(row, policy.language_field) not in policy.allowed_languages
    ):
        return "record outside declared language policy", ()
    if policy.profile != "code":
        return None, ()
    license_value = field_value(row, policy.license_field)
    expression = (
        policy.license_labels.get(license_value)
        if isinstance(license_value, str)
        else None
    )
    if expression is None:
        return "missing or incompatible repository license label", ()
    repository = field_value(row, policy.repository_field)
    path = field_value(row, policy.path_field)
    if (
        not isinstance(repository, str)
        or not repository.strip()
        or not isinstance(path, str)
    ):
        return "missing code repository or path", ()
    parts = PurePosixPath(path).parts
    if (
        not parts
        or path.startswith("/")
        or "\\" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        return "unsafe code path", ()
    if any(
        part.casefold()
        in {
            "vendor",
            "vendored",
            "node_modules",
            "third_party",
            "third-party",
            "generated",
            "dist",
            "build",
        }
        for part in parts
    ):
        return "vendor or generated code path", ()
    material, flags = _git_file_exceptions(text.encode("utf-8"), expression)
    if material:
        return "material or conflicting file-level code notice", ()
    if "generated" in text[:4096].casefold() and any(
        marker in text[:4096].casefold()
        for marker in ("do not edit", "automatically generated", "auto-generated")
    ):
        return "generated code notice", ()
    return None, ("code_notice_review",) if flags else ()
