"""Strict, data-only Campaign v1 declarations and typed dependency validation."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from sparselab.campaign.policy import CorpusReadinessPolicy
from sparselab.config.models import StrictModel
from sparselab.experiments.plan import Artifact, read_document

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
Scope = Literal["corpus", "tokenizer", "model", "runtime", "evaluation", "release"]


def safe_path(base: Path, reference: str) -> Path:
    """Resolve a declaration-relative path without accepting traversal or symlinks.

    The target need not exist: availability is checked by the consumer, not by
    declaration validation. Existing components beneath the base may not be links.
    """
    if (
        not reference
        or "\\" in reference
        or "\x00" in reference
        or Path(reference).is_absolute()
        or PureWindowsPath(reference).is_absolute()
        or bool(PureWindowsPath(reference).drive)
        or any(part == ".." for part in Path(reference).parts)
        or not Path(reference).parts
        or Path(reference) == Path(".")
    ):
        raise ValueError(f"unsafe declaration-relative path: {reference!r}")
    root = base.resolve(strict=False)
    parts = Path(reference).parts
    target = root / reference
    for index in range(len(parts) + 1):
        component = root.joinpath(*parts[:index])
        if component.is_symlink():
            raise ValueError(f"symlinked declaration path: {component}")
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError(f"path escapes declaration directory: {reference!r}")
    return target


def operational_path(base: Path, reference: str) -> Path:
    """Validate an existing or missing artifact/lock location without rebasing it.

    Scientific declaration references use ``safe_path`` instead. Operational
    locations can be absolute in the independent persistent state root.
    """
    declared = Path(reference)
    if (
        not reference
        or "\x00" in reference
        or "\\" in reference
        or PureWindowsPath(reference).drive
        or ".." in declared.parts
        or declared == Path(".")
    ):
        raise ValueError(f"unsafe operational path: {reference!r}")
    target = declared if declared.is_absolute() else base / declared
    target = target.absolute()
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked operational path: {component}")
    return target


class Stage(StrictModel):
    id: str
    scope: Scope
    requires: tuple[str, ...] = ()

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("stage ID must be a safe nonempty identifier")
        return value

    @field_validator("requires")
    @classmethod
    def valid_requirements(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _ID.fullmatch(item) for item in value) or len(set(value)) != len(
            value
        ):
            raise ValueError("requirements must be unique safe stage IDs")
        return value


class ArtifactReference(Stage):
    kind: Literal["artifact_reference"]
    artifact: Artifact

    @model_validator(mode="after")
    def valid_artifact(self) -> ArtifactReference:
        scopes = {
            "source_snapshot": "corpus",
            "corpus_build": "corpus",
            "corpus_release": "corpus",
            "corpus_export": "corpus",
            "tokenizer": "tokenizer",
            "prepared_data": "model",
            "stage_bundle": "model",
            "checkpoint": "model",
            "capability_card": "evaluation",
            "prompt_set": "evaluation",
        }
        if self.artifact.kind not in scopes or self.artifact.from_phase is not None:
            raise ValueError(
                "artifact_reference requires a verifiable external artifact"
            )
        if self.scope != scopes[self.artifact.kind]:
            raise ValueError("artifact_reference scope must match artifact kind")
        return self


class CorpusRelease(Stage):
    kind: Literal["corpus_release"]
    scope: Literal["corpus"]
    project: str


class CorpusReadiness(Stage):
    kind: Literal["corpus_readiness"]
    scope: Literal["corpus"]
    corpus: str
    policy: CorpusReadinessPolicy
    tokenizer: str | None = None


class TokenizerReference(Stage):
    kind: Literal["tokenizer_reference"]
    scope: Literal["tokenizer"]
    artifact: Artifact

    @model_validator(mode="after")
    def valid_artifact(self) -> TokenizerReference:
        if self.artifact.kind != "tokenizer" or self.artifact.from_phase is not None:
            raise ValueError(
                "tokenizer_reference requires an external tokenizer artifact"
            )
        return self


class TokenMeasurement(Stage):
    kind: Literal["token_measurement"]
    scope: Literal["tokenizer"]
    corpus: str
    tokenizer: str


class ExperimentPlanStage(Stage):
    kind: Literal["experiment_plan"]
    scope: Literal["model"]
    source: str
    mode: Literal["lock", "reference"]
    lock: str | None = None
    tokenizer: str
    prepared: str
    corpus: str | None = None

    @model_validator(mode="after")
    def valid_mode(self) -> ExperimentPlanStage:
        if (self.mode == "reference") != (self.lock is not None):
            raise ValueError("experiment_plan lock is required only for reference mode")
        return self


class RuntimeAcceptance(Stage):
    kind: Literal["runtime_acceptance"]
    scope: Literal["runtime"]
    plan: str


class ExperimentRun(Stage):
    kind: Literal["experiment_run"]
    scope: Literal["model"]
    plan: str
    runtime: str
    cell: str

    @field_validator("cell")
    @classmethod
    def valid_cell(cls, value: str) -> str:
        if not value:
            raise ValueError("experiment_run cell must be nonempty")
        return value


class ExperimentCollect(Stage):
    kind: Literal["experiment_collect"]
    scope: Literal["evaluation"]
    plan: str
    run: str


class Evaluation(Stage):
    kind: Literal["evaluation"]
    scope: Literal["evaluation"]
    collect: str
    suite: str

    @field_validator("suite")
    @classmethod
    def valid_suite(cls, value: str) -> str:
        safe_path(Path("."), value)
        return value


class ModelReadiness(Stage):
    kind: Literal["model_readiness"]
    scope: Literal["model"]
    evaluation: str
    policy: str
    review: str | None = None

    @field_validator("review")
    @classmethod
    def valid_review(cls, value: str | None) -> str | None:
        if value is not None:
            operational_path(Path("."), value)
        return value

    @field_validator("policy")
    @classmethod
    def valid_policy(cls, value: str) -> str:
        safe_path(Path("."), value)
        return value


class Approval(Stage):
    kind: Literal["approval"]
    scope: Literal["release", "model"]
    bind: tuple[str, ...] = Field(min_length=1)

    @field_validator("bind")
    @classmethod
    def valid_bind(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _ID.fullmatch(item) for item in value) or len(set(value)) != len(
            value
        ):
            raise ValueError("approval bind must contain distinct safe stage IDs")
        return value


CampaignStage = Annotated[
    ArtifactReference
    | CorpusRelease
    | CorpusReadiness
    | TokenizerReference
    | TokenMeasurement
    | ExperimentPlanStage
    | RuntimeAcceptance
    | ExperimentRun
    | ExperimentCollect
    | Evaluation
    | ModelReadiness
    | Approval,
    Field(discriminator="kind"),
]


class CampaignPlan(StrictModel):
    campaign_version: Literal[1]
    id: str
    stages: tuple[CampaignStage, ...] = Field(min_length=1)
    recovery: str | None = None

    @field_validator("recovery")
    @classmethod
    def valid_recovery(cls, value: str | None) -> str | None:
        if value is not None:
            safe_path(Path("."), value)
        return value

    @model_validator(mode="after")
    def valid_plan(self) -> CampaignPlan:
        if not _ID.fullmatch(self.id):
            raise ValueError("campaign ID must be a safe nonempty identifier")
        by_id = {stage.id: stage for stage in self.stages}
        if len(by_id) != len(self.stages):
            raise ValueError("duplicate campaign stage IDs")
        for stage in self.stages:
            if stage.id in stage.requires:
                raise ValueError(f"stage {stage.id} cannot depend on itself")
            for requirement in stage.requires:
                if requirement not in by_id:
                    raise ValueError(
                        f"stage {stage.id} requires unknown stage {requirement}"
                    )

            def input_is(
                reference: str,
                *allowed: tuple[str, str | None],
                stage: CampaignStage = stage,
            ) -> None:
                if reference not in stage.requires:
                    raise ValueError(
                        f"stage {stage.id} input {reference} must be in requires"
                    )
                upstream = by_id[reference]
                if not any(
                    upstream.kind == kind
                    and (
                        artifact_kind is None or upstream.artifact.kind == artifact_kind
                    )
                    for kind, artifact_kind in allowed
                ):
                    raise ValueError(
                        f"stage {stage.id} input {reference} has wrong producer kind"
                    )

            release = (
                ("corpus_release", None),
                ("artifact_reference", "corpus_release"),
            )
            tokenizer = (
                ("tokenizer_reference", None),
                ("artifact_reference", "tokenizer"),
            )
            if isinstance(stage, (CorpusReadiness, TokenMeasurement)):
                input_is(stage.corpus, *release)
                if stage.tokenizer is not None:
                    input_is(stage.tokenizer, *tokenizer)
                if (
                    isinstance(stage, CorpusReadiness)
                    and (
                        stage.policy.min_unique_train_tokens_by_domain
                        or (
                            stage.policy.passes is not None
                            and stage.policy.passes.basis == "tokens"
                        )
                    )
                    and stage.tokenizer is None
                ):
                    raise ValueError(
                        "token-based readiness requires a tokenizer dependency"
                    )
            elif isinstance(stage, ExperimentPlanStage):
                input_is(stage.tokenizer, *tokenizer)
                input_is(stage.prepared, ("artifact_reference", "prepared_data"))
                if stage.corpus is not None:
                    input_is(stage.corpus, *release)
            elif isinstance(stage, RuntimeAcceptance):
                input_is(stage.plan, ("experiment_plan", None))
            elif isinstance(stage, ExperimentRun):
                input_is(stage.plan, ("experiment_plan", None))
                input_is(stage.runtime, ("runtime_acceptance", None))
                if by_id[stage.runtime].plan != stage.plan:
                    raise ValueError(
                        f"stage {stage.id} runtime belongs to another plan"
                    )
            elif isinstance(stage, ExperimentCollect):
                input_is(stage.plan, ("experiment_plan", None))
                input_is(stage.run, ("experiment_run", None))
                if by_id[stage.run].plan != stage.plan:
                    raise ValueError(f"stage {stage.id} run belongs to another plan")
            elif isinstance(stage, Evaluation):
                input_is(stage.collect, ("experiment_collect", None))
            elif isinstance(stage, ModelReadiness):
                input_is(stage.evaluation, ("evaluation", None))
            elif isinstance(stage, Approval):
                ancestors = set(stage.requires)
                pending = list(stage.requires)
                while pending:
                    parent = pending.pop()
                    for ancestor in by_id[parent].requires:
                        if ancestor not in ancestors:
                            ancestors.add(ancestor)
                            pending.append(ancestor)
                for reference in stage.bind:
                    if reference not in ancestors:
                        raise ValueError(
                            f"stage {stage.id} bound input {reference} must be an ancestor"
                        )
        self.ordered_stages()  # Detect cycles independently of authored order.
        return self

    def ordered_stages(self) -> tuple[CampaignStage, ...]:
        """Stable topological ordering, preserving authored order among ready nodes."""
        remaining = list(self.stages)
        ordered: list[CampaignStage] = []
        completed: set[str] = set()
        while remaining:
            available = next(
                (stage for stage in remaining if set(stage.requires) <= completed), None
            )
            if available is None:
                raise ValueError("campaign stage dependency cycle")
            ordered.append(available)
            completed.add(available.id)
            remaining.remove(available)
        return tuple(ordered)


def load_campaign(path: Path) -> CampaignPlan:
    """Parse and statically validate an authored campaign, without creating files."""
    from pydantic import ValidationError

    path = Path(path)
    try:
        plan = CampaignPlan.model_validate(read_document(path, exact_decimals=True))
        for stage in plan.stages:
            if isinstance(stage, (ArtifactReference, TokenizerReference)):
                assert stage.artifact.path is not None
                operational_path(path.parent, stage.artifact.path)
            elif isinstance(stage, CorpusRelease):
                safe_path(path.parent, stage.project)
            elif isinstance(stage, ExperimentPlanStage):
                safe_path(path.parent, stage.source)
                if stage.lock is not None:
                    operational_path(path.parent, stage.lock)
            elif isinstance(stage, Evaluation):
                safe_path(path.parent, stage.suite)
            elif isinstance(stage, ModelReadiness):
                safe_path(path.parent, stage.policy)
                if stage.review is not None:
                    operational_path(path.parent, stage.review)
        if plan.recovery is not None:
            safe_path(path.parent, plan.recovery)
        return plan
    except (ValidationError, ValueError) as error:
        raise ValueError(f"invalid campaign plan {path}: {error}") from error
