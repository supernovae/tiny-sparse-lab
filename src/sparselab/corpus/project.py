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


class HFBoundedShard(StrictModel):
    """One exact, pinned input shard; scan only its bounded row prefix."""

    path: str
    expected_sha256: str
    max_shard_bytes: int = Field(gt=0)
    max_scanned_rows: int = Field(gt=0)
    hash_modulus: int = Field(gt=0)
    hash_remainders: tuple[int, ...]

    @model_validator(mode="after")
    def valid_selection(self) -> HFBoundedShard:
        safe_name(self.path)
        if any(char in self.path for char in "*?[]"):
            raise ValueError("bounded HF shard must be an exact path")
        if not self.path.endswith((".parquet", ".jsonl", ".jsonl.gz", ".json.gz")):
            raise ValueError("bounded HF shard must be Parquet or JSONL stream")
        if not _HEX.fullmatch(self.expected_sha256):
            raise ValueError("bounded HF shard needs a SHA-256 checksum")
        if (
            not self.hash_remainders
            or len(set(self.hash_remainders)) != len(self.hash_remainders)
            or any(not 0 <= n < self.hash_modulus for n in self.hash_remainders)
        ):
            raise ValueError("invalid bounded HF hash remainders")
        return self


class HuggingFaceAcquisition(StrictModel):
    config: str
    split: str
    include: tuple[str, ...] = ()
    bounded_shards: tuple[HFBoundedShard, ...] | None = None
    text_field: str
    max_rows: int = Field(gt=0)
    max_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def selection_required(self) -> HuggingFaceAcquisition:
        if bool(self.include) == bool(self.bounded_shards):
            raise ValueError("HF requires either include or bounded_shards, not both")
        for pattern in self.include:
            safe_name(pattern.replace("*", "x").replace("?", "x"))
        if self.bounded_shards:
            paths = [shard.path for shard in self.bounded_shards]
            if len(paths) != len(set(paths)):
                raise ValueError("duplicate bounded HF shard path")
            for path in paths:
                parts = path.split("/")
                if self.config not in parts and not Path(path).name.startswith(
                    self.config + "-"
                ):
                    raise ValueError("bounded HF shard is outside declared config")
                if self.split not in parts and not any(
                    part.startswith((self.split + "-", self.split + "."))
                    for part in parts
                ):
                    raise ValueError("bounded HF shard is outside declared split")
        for value in (self.config, self.split, self.text_field):
            _nonblank(value)
        return self


class WikimediaDumpAcquisition(StrictModel):
    expected_sha1: str
    expected_sha256: str | None = None
    checksum_uri: str
    text_field: Literal["text"] = "text"
    max_compressed_bytes: int = Field(gt=0)
    max_decompressed_bytes: int = Field(gt=0)
    max_scanned_pages: int = Field(gt=0)
    max_selected_pages: int = Field(gt=0)
    max_emitted_bytes: int = Field(gt=0)

    @property
    def max_rows(self) -> int:
        return self.max_selected_pages

    @model_validator(mode="after")
    def pinned_checksum(self) -> WikimediaDumpAcquisition:
        if not re.fullmatch(r"[a-fA-F0-9]{40}", self.expected_sha1):
            raise ValueError("Wikimedia dump needs official SHA-1 checksum")
        if self.expected_sha256 is not None and not _HEX.fullmatch(
            self.expected_sha256
        ):
            raise ValueError("Wikimedia dump SHA-256 must be hex")
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
    generator: Literal[
        "pathlib_path_suffix_v1",
        "filesystem_judgment_v1",
        "platform_fault_v1",
        "deployment_change_v1",
        "code_test_workflow_v1",
        "filesystem_judgment_v2",
        "platform_fault_v2",
        "deployment_change_v2",
        "code_test_workflow_v2",
    ]
    generator_version: Literal["1", "2"]

    @model_validator(mode="after")
    def version_matches_generator(self) -> DeterministicAcquisition:
        expected = self.generator.rsplit("_v", 1)[-1]
        if self.generator_version != expected:
            raise ValueError("generator_version must match registered generator ID")
        return self


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
    "wikimedia_dump": WikimediaDumpAcquisition,
    "local": LocalAcquisition,
    "deterministic_generator": DeterministicAcquisition,
    "inference_generator": InferenceAcquisition,
}


class SourceDeclaration(StrictModel):
    schema_version: Literal[1, 2, 3]
    id: str
    kind: Literal[
        "git",
        "http_document",
        "huggingface_dataset",
        "wikimedia_dump",
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
    explicit_training_restriction: (
        Literal["none_found", "restricted", "incompatible", "unknown"] | None
    ) = None
    domains: tuple[str, ...]
    document_kinds: tuple[str, ...]
    source_family: str
    acquisition: (
        GitAcquisition
        | HttpAcquisition
        | HuggingFaceAcquisition
        | WikimediaDumpAcquisition
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
        if self.kind == "wikimedia_dump":
            spec = self.acquisition
            assert isinstance(spec, WikimediaDumpAcquisition)
            if not re.fullmatch(r"20\d{6}", self.revision):
                raise ValueError("Wikimedia dump needs an exact date")
            base = "https://dumps.wikimedia.org/"
            book = f"enwikibooks/{self.revision}/"
            wiki = f"enwiki/{self.revision}/"
            book_name = (
                f"enwikibooks-{self.revision}-pages-articles-multistream.xml.bz2"
            )
            wikipedia_name = (
                rf"enwiki-{self.revision}-pages-articles-multistream\d+"
                r"\.xml-p\d+p\d+\.bz2"
            )
            is_book = self.canonical_uri == base + book + book_name
            is_wikipedia = (
                self.canonical_uri.startswith(base + wiki)
                and re.fullmatch(
                    wikipedia_name, self.canonical_uri.removeprefix(base + wiki)
                )
                is not None
            )
            if not is_book and not is_wikipedia:
                raise ValueError(
                    "Wikimedia dump requires exact dated HTTPS article file"
                )
            prefix = base + (book if is_book else wiki)
            project_name = "enwikibooks" if is_book else "enwiki"
            if (
                spec.checksum_uri
                != prefix + f"{project_name}-{self.revision}-sha1sums.txt"
            ):
                raise ValueError(
                    "Wikimedia dump requires matching official checksum URI"
                )
        if self.schema_version == 1:
            if (
                self.rights is not None
                or self.redistribution is None
                or self.explicit_training_restriction is not None
            ):
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
                    "v2/v3 source needs rights, not legacy redistribution/rejection"
                )
            if (
                self.schema_version == 2
                and self.explicit_training_restriction is not None
            ):
                raise ValueError("explicit_training_restriction requires v3 source")
            if self.schema_version == 3:
                state = self.explicit_training_restriction
                restriction = self.rights.training_restriction
                if state is None:
                    raise ValueError("v3 source needs explicit_training_restriction")
                if state == "incompatible" and (
                    self.rights.training_eligibility != "ineligible"
                    or restriction is None
                    or restriction.kind != "prohibited"
                ):
                    raise ValueError("incompatible training requires prohibited basis")
                if state in {
                    "restricted",
                    "unknown",
                } and self.rights.training_eligibility not in {
                    "review_required",
                    "ineligible",
                }:
                    raise ValueError(
                        "restricted/unknown training requires rights review"
                    )
                if state == "none_found" and restriction is not None:
                    raise ValueError("none_found conflicts with a training restriction")
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
    """Preserve v1/v2 acquisition receipts byte-for-byte when v3 is added."""
    result = source.model_dump(mode="json")
    if source.schema_version == 1:
        result.pop("rights")
    if source.schema_version in (1, 2):
        result.pop("explicit_training_restriction")
    if source.kind == "huggingface_dataset":
        if source.acquisition.bounded_shards is None:
            result["acquisition"].pop("bounded_shards")
        else:
            result["acquisition"].pop("include")
    return result


class TransformDeclaration(StrictModel):
    id: str
    version: str
    kind: Literal[
        "lm_text",
        "lexical_candidates",
        "semantic_candidates",
        "deterministic_scenarios",
        "source_qa",
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
    schema_version: Literal[1, 2, 3]
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
    training_use_policy: Literal["allowed_unless_explicitly_prohibited"] | None = None

    @model_validator(mode="after")
    def weights_valid(self) -> ReleaseDeclaration:
        if (
            not self.mixture
            or any(not 0 <= v <= 1 for v in self.mixture.values())
            or abs(sum(self.mixture.values()) - 1) > 1e-8
        ):
            raise ValueError("mixture weights must sum to one")
        if (self.schema_version == 1 and self.publication_mode is not None) or (
            self.schema_version in (2, 3) and self.publication_mode is None
        ):
            raise ValueError("v2/v3 releases require an explicit publication_mode")
        if (self.schema_version == 3) != (self.training_use_policy is not None):
            raise ValueError("training_use_policy is required only for v3 releases")
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
    if release.schema_version in (1, 2):
        result.pop("training_use_policy")
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
        if any(
            source.schema_version != self.release.schema_version
            for source in self.sources
        ):
            raise ValueError(
                "source rights schema must match release publication policy"
            )
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
