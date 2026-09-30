"""Checked-in, strictly validated corpus recipes."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from sparselab.config.models import StrictModel
from sparselab.corpus.rights import RightsPolicy

_ID = re.compile(r"[a-z][a-z0-9_-]*\Z")
_HEX = re.compile(r"[0-9a-fA-F]{64}\Z")
_GIT_REV = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


def safe_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError(f"unsafe logical path: {value!r}")
    return value


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class LocalFile(StrictModel):
    path: str
    name: str

    @field_validator("name")
    @classmethod
    def name_safe(cls, value: str) -> str:
        return safe_name(value)

    @field_validator("path")
    @classmethod
    def path_nonblank(cls, value: str) -> str:
        return _nonblank(value)


class GitAcquisition(StrictModel):
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    max_bytes: int = Field(gt=0)

    @field_validator("include", "exclude")
    @classmethod
    def patterns_safe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            safe_name(value.replace("*", "x").replace("?", "x"))
        return values

    @model_validator(mode="after")
    def selection_required(self) -> GitAcquisition:
        if not self.include:
            raise ValueError("Git include cannot be empty")
        return self


class HttpAcquisition(StrictModel):
    expected_sha256: str
    max_bytes: int = Field(gt=0)

    @field_validator("expected_sha256")
    @classmethod
    def digest_required(cls, value: str) -> str:
        if not _HEX.fullmatch(value):
            raise ValueError("expected_sha256 must be SHA-256 hex")
        return value.lower()


class HuggingFaceAcquisition(StrictModel):
    config: str
    split: str
    include: tuple[str, ...]
    text_field: str
    max_rows: int = Field(gt=0)
    max_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def selection_required(self) -> HuggingFaceAcquisition:
        if not self.include:
            raise ValueError("HF include cannot be empty")
        for pattern in self.include:
            safe_name(pattern.replace("*", "x").replace("?", "x"))
        for value in (self.config, self.split, self.text_field):
            _nonblank(value)
        return self


class LocalAcquisition(StrictModel):
    files: tuple[LocalFile, ...]
    max_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def unique_files(self) -> LocalAcquisition:
        if not self.files or len({entry.name for entry in self.files}) != len(
            self.files
        ):
            raise ValueError("local files require distinct logical names")
        return self


class DeterministicAcquisition(StrictModel):
    generator: Literal["pathlib_path_suffix_v1"]
    generator_version: str

    @field_validator("generator_version")
    @classmethod
    def version_required(cls, value: str) -> str:
        return _nonblank(value)


class InferenceAcquisition(StrictModel):
    backend: Literal["recorded_responses"]
    provider: str
    model: str
    model_revision: str

    @field_validator("provider", "model", "model_revision")
    @classmethod
    def identity_required(cls, value: str) -> str:
        return _nonblank(value)


_ACQUISITION = {
    "git": GitAcquisition,
    "http_document": HttpAcquisition,
    "huggingface_dataset": HuggingFaceAcquisition,
    "local": LocalAcquisition,
    "deterministic_generator": DeterministicAcquisition,
    "inference_generator": InferenceAcquisition,
}


class SourceDeclaration(StrictModel):
    schema_version: Literal[1, 2]
    id: str
    kind: Literal[
        "git",
        "http_document",
        "huggingface_dataset",
        "local",
        "deterministic_generator",
        "inference_generator",
    ]
    origin: Literal["primary_source", "human_authored"] = "primary_source"
    modality: Literal["text"] = "text"
    canonical_uri: str
    revision: str
    license: str
    license_url: str | None = None
    notes: str | None = None
    redistribution: (
        Literal[
            "redistributable", "derived_only", "reference_only", "unknown", "rejected"
        ]
        | None
    ) = None
    rights: RightsPolicy | None = None
    domains: tuple[str, ...]
    document_kinds: tuple[str, ...]
    source_family: str
    acquisition: (
        GitAcquisition
        | HttpAcquisition
        | HuggingFaceAcquisition
        | LocalAcquisition
        | DeterministicAcquisition
        | InferenceAcquisition
    )
    rejection_reason: str | None = None

    @field_validator("id", "source_family")
    @classmethod
    def identifier_safe(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError(f"invalid corpus identifier: {value!r}")
        return value

    @field_validator("canonical_uri", "revision", "license")
    @classmethod
    def identity_nonblank(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def validate_kind(self) -> SourceDeclaration:
        if type(self.acquisition) is not _ACQUISITION[self.kind]:
            raise ValueError("acquisition does not match source kind")
        if (
            not self.domains
            or not self.document_kinds
            or any(
                not _ID.fullmatch(value)
                for value in (*self.domains, *self.document_kinds)
            )
            or len(set(self.domains)) != len(self.domains)
            or len(set(self.document_kinds)) != len(self.document_kinds)
        ):
            raise ValueError(
                "domains and document_kinds must contain distinct safe IDs"
            )
        if self.kind == "git" and not _GIT_REV.fullmatch(self.revision):
            raise ValueError("Git revision must be an exact commit hash")
        if self.kind == "huggingface_dataset" and not _GIT_REV.fullmatch(self.revision):
            raise ValueError("HF revision must be a pinned commit hash")
        if self.schema_version == 1:
            if self.rights is not None or self.redistribution is None:
                raise ValueError(
                    "v1 source needs legacy redistribution and no rights policy"
                )
            if self.redistribution == "rejected" and (
                not self.rejection_reason or not self.rejection_reason.strip()
            ):
                raise ValueError("rejected sources require rejection_reason")
            if self.redistribution != "rejected" and self.rejection_reason is not None:
                raise ValueError("rejection_reason only applies to rejected sources")
        else:
            if (
                self.rights is None
                or self.redistribution is not None
                or self.rejection_reason is not None
            ):
                raise ValueError(
                    "v2 source needs rights, not legacy redistribution/rejection"
                )
            if self.rights.nested_metadata_path and self.kind != "git":
                raise ValueError(
                    "nested repository rights metadata requires Git acquisition"
                )
            if (
                self.rights.spdx_expression
                and self.rights.spdx_expression != self.license
            ):
                raise ValueError("declared license differs from source SPDX expression")
            if self.kind not in {"local", "deterministic_generator"} and (
                self.rights.training_eligibility
                in {"eligible", "eligible_with_obligations"}
                and (not self.license_url or not self.rights.license_references)
            ):
                raise ValueError(
                    "eligible external source requires license URL and references"
                )
        return self


def source_declaration_payload(source: SourceDeclaration) -> dict[str, Any]:
    """Preserve v1 acquisition receipts byte-for-byte when the v2 schema is added."""
    result = source.model_dump(mode="json")
    if source.schema_version == 1:
        result.pop("rights")
    return result


class TransformDeclaration(StrictModel):
    id: str
    version: str
    kind: Literal[
        "lm_text",
        "lexical_candidates",
        "semantic_candidates",
        "deterministic_scenarios",
        "inference_qa",
        "manual_semantic",
        "chat_sft",
        "tool_episode",
    ]
    parameters: dict[str, Any]
    inputs: tuple[str, ...]

    @field_validator("id")
    @classmethod
    def id_safe(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("invalid transform ID")
        return value

    @field_validator("version")
    @classmethod
    def version_required(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def inputs_unique(self) -> TransformDeclaration:
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("duplicate transform input")
        return self


class SplitDeclaration(StrictModel):
    schema_version: Literal[1]
    unit: Literal[
        "document",
        "source_repository",
        "source_document_family",
        "scenario_family",
        "generator_world",
        "template_family",
    ]
    family_key: str | None = None
    assignments: dict[str, Literal["train", "validation", "test"]]

    @model_validator(mode="after")
    def family_required(self) -> SplitDeclaration:
        if self.unit not in ("document", "source_repository") and not self.family_key:
            raise ValueError("family_key required for family split")
        if not self.assignments:
            raise ValueError("explicit split assignments required")
        return self


class ViewDeclaration(StrictModel):
    selected: bool
    training_splits: tuple[Literal["train", "validation", "test"], ...]

    @model_validator(mode="after")
    def splits_valid(self) -> ViewDeclaration:
        if len(set(self.training_splits)) != len(self.training_splits):
            raise ValueError("duplicate training split")
        if "test" in self.training_splits:
            raise ValueError("test is held out of training")
        if not self.selected and self.training_splits:
            raise ValueError("unselected view cannot declare training splits")
        return self


class FractionDeclaration(StrictModel):
    """Exact tokenizer-based train selection; infeasible budgets are errors."""

    generated_share: float
    train_tokens: int
    tokenizer_path: str
    tokenizer_sha256: str

    @model_validator(mode="after")
    def valid_fraction(self) -> FractionDeclaration:
        if not 0 <= self.generated_share <= 1:
            raise ValueError("generated_share must be between zero and one")
        if self.train_tokens <= 0:
            raise ValueError("train_tokens must be positive")
        safe_name(self.tokenizer_path)
        if not _HEX.fullmatch(self.tokenizer_sha256):
            raise ValueError("tokenizer_sha256 must be a full SHA-256")
        return self


class ReleaseDeclaration(StrictModel):
    schema_version: Literal[1, 2]
    mixture: dict[str, float]
    accepted_generation_statuses: tuple[
        Literal[
            "source_entailed",
            "oracle_verified",
            "cross_source_verified",
            "human_reviewed",
            "schema_validated",
            "unverified",
        ],
        ...,
    ] = ("source_entailed", "oracle_verified", "human_reviewed")
    lm: ViewDeclaration
    chat: ViewDeclaration
    include_shapes: tuple[str, ...] | None = None
    include_origins: tuple[str, ...] | None = None
    fraction: FractionDeclaration | None = None
    keep_nonredistributable_local: bool = True
    publication_mode: (
        Literal["metadata_reconstruction_only", "redistributable_under_source_terms"]
        | None
    ) = None

    @model_validator(mode="after")
    def weights_valid(self) -> ReleaseDeclaration:
        if (
            not self.mixture
            or any(not 0 <= v <= 1 for v in self.mixture.values())
            or abs(sum(self.mixture.values()) - 1) > 1e-8
        ):
            raise ValueError("mixture weights must sum to one")
        if (self.schema_version == 1 and self.publication_mode is not None) or (
            self.schema_version == 2 and self.publication_mode is None
        ):
            raise ValueError("v2 releases require an explicit publication_mode")
        from sparselab.corpus.provenance import ORIGINS, SHAPES

        for values, allowed, label in (
            (self.include_shapes, SHAPES, "shape"),
            (self.include_origins, ORIGINS, "origin"),
        ):
            if values is not None and (not values or len(values) != len(set(values))):
                raise ValueError(f"{label} filter must contain unique values")
            if values is not None and set(values) - set(allowed):
                raise ValueError(f"unknown {label} in release filter")
        return self


def release_declaration_payload(release: ReleaseDeclaration) -> dict[str, Any]:
    result = release.model_dump(mode="json")
    if release.schema_version == 1:
        result.pop("publication_mode")
    return result


class ProjectConfig(StrictModel):
    schema_version: Literal[1]
    id: str
    sources: tuple[str, ...]
    transforms: tuple[str, ...]
    splits: str
    release: str

    @field_validator("id")
    @classmethod
    def id_safe(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("invalid project ID")
        return value

    @model_validator(mode="after")
    def references_safe(self) -> ProjectConfig:
        if (
            not self.sources
            or len(set(self.sources)) != len(self.sources)
            or len(set(self.transforms)) != len(self.transforms)
        ):
            raise ValueError("project references must be nonempty and unique")
        if list(self.sources) != sorted(self.sources):
            raise ValueError("source references must be sorted")
        for group, paths in (
            ("sources", self.sources),
            ("transforms", self.transforms),
        ):
            for path in paths:
                safe_name(path)
                if not path.startswith(group + "/") or not path.endswith(".yaml"):
                    raise ValueError(f"invalid {group} declaration path: {path}")
        for path in (self.splits, self.release):
            safe_name(path)
        return self


class Project(StrictModel):
    root: Path
    config: ProjectConfig
    sources: tuple[SourceDeclaration, ...]
    transforms: tuple[TransformDeclaration, ...]
    splits: SplitDeclaration
    release: ReleaseDeclaration

    @model_validator(mode="after")
    def rights_protocol_consistent(self) -> Project:
        if self.release.schema_version == 2:
            if any(source.schema_version != 2 for source in self.sources):
                raise ValueError(
                    "v2 release requires explicit v2 rights for every source"
                )
        elif any(source.schema_version != 1 for source in self.sources):
            raise ValueError("v2 source requires v2 release publication policy")
        return self


def project_path(root: Path, name: str) -> Path:
    safe_name(name)
    candidate = root / name
    if candidate.is_symlink() or any(
        part.is_symlink() for part in candidate.parents if part != root.parent
    ):
        raise ValueError(f"symlink in project path: {name}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"project path escapes root: {name}")
    return candidate


def _yaml(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or symlinked declaration: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def verify_fraction_tokenizer(project: Project) -> None:
    """Require the exact pinned tokenizer before any fractional release build."""
    if project.release.fraction is None:
        return
    from sparselab.training.manifest import sha256_file

    tokenizer = project_path(project.root, project.release.fraction.tokenizer_path)
    if not tokenizer.is_file() or tokenizer.is_symlink():
        raise ValueError("fraction tokenizer must be a regular project file")
    if sha256_file(tokenizer) != project.release.fraction.tokenizer_sha256.lower():
        raise ValueError("fraction tokenizer SHA-256 mismatch")


def load_project(path: Path | str) -> Project:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked corpus project: {path}")
    root = path.parent
    config = ProjectConfig.model_validate(_yaml(path))
    sources = tuple(
        SourceDeclaration.model_validate(_yaml(project_path(root, p)))
        for p in config.sources
    )
    transforms = tuple(
        TransformDeclaration.model_validate(_yaml(project_path(root, p)))
        for p in config.transforms
    )
    source_ids = {source.id for source in sources}
    transform_by_id = {transform.id: transform for transform in transforms}
    if (
        len(source_ids) != len(sources)
        or len(transform_by_id) != len(transforms)
        or source_ids & transform_by_id.keys()
    ):
        raise ValueError("duplicate source or transform ID")
    visiting: set[str] = set()
    visited: set[str] = set()

    def check_inputs(transform_id: str) -> None:
        if transform_id in visiting:
            raise ValueError("cyclic transform inputs")
        if transform_id in visited:
            return
        visiting.add(transform_id)
        for input_id in transform_by_id[transform_id].inputs:
            if input_id in transform_by_id:
                check_inputs(input_id)
            elif input_id not in source_ids:
                raise ValueError(f"unknown transform input: {input_id}")
        visiting.remove(transform_id)
        visited.add(transform_id)

    for transform in transforms:
        check_inputs(transform.id)
    splits = SplitDeclaration.model_validate(_yaml(project_path(root, config.splits)))
    release = ReleaseDeclaration.model_validate(
        _yaml(project_path(root, config.release))
    )
    project = Project(
        root=root,
        config=config,
        sources=sources,
        transforms=transforms,
        splits=splits,
        release=release,
    )
    verify_fraction_tokenizer(project)
    return project
