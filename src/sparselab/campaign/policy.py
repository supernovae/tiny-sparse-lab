"""Declared corpus readiness thresholds over verified, unique source evidence."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, WithJsonSchema, field_validator, model_validator

from sparselab.config.models import StrictModel
from sparselab.corpus.release import _rows, verify_release
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.manifest import sha256_file

NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
PositiveWeight = Annotated[
    Decimal,
    Field(gt=0, allow_inf_nan=False),
    WithJsonSchema({"type": "number", "exclusiveMinimum": 0}),
]


class SourcePassRequirement(StrictModel):
    """Maximum projected per-domain reuse of unique source material.

    Overlapping source documents across domains are counted once *per domain*;
    these bounds do not describe train order or actual ingestion passes.
    """

    basis: Literal["bytes", "tokens"]
    requested_total: PositiveInt
    mixture: dict[str, PositiveWeight]
    max_required: PositiveInt

    @field_validator("mixture", mode="before")
    @classmethod
    def require_numeric_weights(cls, values: Any) -> Any:
        if isinstance(values, dict) and any(
            type(weight) not in (float, int, Decimal) for weight in values.values()
        ):
            raise ValueError("source pass mixture weights must be numeric")
        return values

    @model_validator(mode="after")
    def validate_mixture(self) -> SourcePassRequirement:
        if not self.mixture or any(not domain.strip() for domain in self.mixture):
            raise ValueError("source passes require named mixture domains")
        total = sum(
            (Fraction(weight) for weight in self.mixture.values()),
            Fraction(0),
        )
        if total != 1:
            raise ValueError("source pass mixture weights must sum to 1")
        return self


class CorpusReadinessPolicy(StrictModel):
    """Authored minima, independent of the frozen release's mixture."""

    min_unique_train_bytes_by_domain: dict[str, NonnegativeInt] = Field(
        default_factory=dict
    )
    min_unique_train_tokens_by_domain: dict[str, NonnegativeInt] = Field(
        default_factory=dict
    )
    required_nonzero_languages: tuple[str, ...] = ()
    required_nonzero_shapes: tuple[str, ...] = ()
    min_heldout_families: NonnegativeInt = 0
    passes: SourcePassRequirement | None = None

    @model_validator(mode="after")
    def validate_requirements(self) -> CorpusReadinessPolicy:
        if not any(
            (
                self.min_unique_train_bytes_by_domain,
                self.min_unique_train_tokens_by_domain,
                self.required_nonzero_languages,
                self.required_nonzero_shapes,
                self.min_heldout_families,
                self.passes is not None,
            )
        ):
            raise ValueError("corpus readiness needs at least one requirement")
        for labels in (
            self.min_unique_train_bytes_by_domain,
            self.min_unique_train_tokens_by_domain,
            self.required_nonzero_languages,
            self.required_nonzero_shapes,
        ):
            if any(not label.strip() for label in labels):
                raise ValueError("readiness labels must not be empty")
        if len(set(self.required_nonzero_languages)) != len(
            self.required_nonzero_languages
        ):
            raise ValueError("duplicate required language")
        if len(set(self.required_nonzero_shapes)) != len(self.required_nonzero_shapes):
            raise ValueError("duplicate required shape")
        return self


def measure_readiness(
    release: Path, policy: CorpusReadinessPolicy, tokenizer: Path | None = None
) -> dict[str, Any]:
    """Check source-document coverage, never build-report or ingestion estimates.

    A zero-availability mixture domain has no finite projected number of passes.
    The declared mixture only provides weights for this policy; it does not
    modify the release. This estimate is a per-domain upper-bound reuse check,
    not a claim about dataloader passes or overlapping domains' global total.
    """
    release = Path(release)
    manifest = verify_release(release)
    need_tokens = bool(policy.min_unique_train_tokens_by_domain) or (
        policy.passes is not None and policy.passes.basis == "tokens"
    )
    model = (
        load_tokenizer(Path(tokenizer))
        if tokenizer is not None and need_tokens
        else None
    )
    unique_bytes: dict[str, int] = defaultdict(int)
    unique_tokens: dict[str, int] = defaultdict(int)
    seen: dict[str, set[str]] = defaultdict(set)
    languages: dict[str, int] = defaultdict(int)
    language_seen: set[str] = set()
    for row in _rows(release / "documents.jsonl"):
        if row["split"] != "train" or row["drop_reason"] is not None:
            continue
        domains = set(row["domains"])
        if not domains:
            continue
        text = row["text"]
        digest = row["content_sha256"]
        if digest not in language_seen:
            language_seen.add(digest)
            languages[row["language"]] += 1
        novel = [domain for domain in domains if digest not in seen[domain]]
        if not novel:
            continue
        byte_count = len(text.encode("utf-8"))
        token_count = len(model.encode(text).ids) if model is not None else None
        for domain in novel:
            seen[domain].add(digest)
            unique_bytes[domain] += byte_count
            if token_count is not None:
                unique_tokens[domain] += token_count

    # The view lineage identifies selected training shapes, rather than every
    # merely eligible row in the full lineage ledger.
    ledger = {row["record_id"]: row for row in _rows(release / "lineage.jsonl")}
    release_views = manifest["build_identity"]["release"]
    selected_train_ids = {
        link["record_id"]
        for view in ("lm", "chat")
        if release_views[view]["selected"]
        and "train" in release_views[view]["training_splits"]
        for link in _rows(release / view / "train.lineage.jsonl")
    }
    shapes: dict[str, int] = defaultdict(int)
    for record_id in selected_train_ids:
        shapes[ledger[record_id]["shape"]["id"]] += 1
    heldout = {
        family
        for row in ledger.values()
        if row["split"] == "test" and row.get("drop_reason") is None
        for family in row["source_family_ids"]
        if family
    }

    deficits: list[dict[str, Any]] = []

    def deficient(
        dimension: str, observed: int | None, required: int, domain: str | None = None
    ) -> None:
        if observed is None or observed < required:
            deficit: dict[str, Any] = {
                "dimension": dimension,
                "observed": observed,
                "required": required,
            }
            if domain is not None:
                deficit["domain"] = domain
            deficits.append(deficit)

    for domain, threshold in policy.min_unique_train_bytes_by_domain.items():
        deficient("unique_train_bytes", unique_bytes.get(domain), threshold, domain)
    for domain, threshold in policy.min_unique_train_tokens_by_domain.items():
        deficient(
            "unique_train_tokens",
            unique_tokens.get(domain) if model is not None else None,
            threshold,
            domain,
        )
    for language in policy.required_nonzero_languages:
        deficient("train_language", languages.get(language, 0), 1, language)
    for shape in policy.required_nonzero_shapes:
        deficient("train_shape", shapes.get(shape, 0), 1, shape)
    deficient("heldout_families", len(heldout), policy.min_heldout_families)

    projected_passes: dict[str, int | None] = {}
    if policy.passes is not None:
        requirement = policy.passes
        amounts = unique_bytes if requirement.basis == "bytes" else unique_tokens
        for domain, weight in requirement.mixture.items():
            amount = (
                amounts.get(domain, 0)
                if requirement.basis == "bytes" or model is not None
                else None
            )
            if amount is None or amount == 0:
                projected_passes[domain] = None
                deficient("passes", None, requirement.max_required, domain)
                continue
            # A Decimal-authored weight has an exact rational representation;
            # integer ceiling avoids precision loss for arbitrarily large totals.
            numerator, denominator = weight.as_integer_ratio()
            target = requirement.requested_total * numerator
            available = amount * denominator
            passes = (target + available - 1) // available
            projected_passes[domain] = passes
            # Passes are a maximum, unlike the minima handled above.
            if passes > requirement.max_required:
                deficits.append(
                    {
                        "dimension": "passes",
                        "observed": passes,
                        "required": requirement.max_required,
                        "domain": domain,
                    }
                )
    deficits.sort(key=lambda item: (item["dimension"], item.get("domain", "")))
    measurements = {
        "release_id": manifest["release_id"],
        "tokenizer_sha256": sha256_file(Path(tokenizer)) if model is not None else None,
        "unique_train_bytes_by_domain": dict(sorted(unique_bytes.items())),
        "unique_train_tokens_by_domain": dict(sorted(unique_tokens.items()))
        if model is not None
        else None,
        "train_languages": dict(sorted(languages.items())),
        "selected_train_shapes": dict(sorted(shapes.items())),
        "heldout_families": len(heldout),
        "projected_source_passes_by_domain": dict(sorted(projected_passes.items())),
    }
    return {
        "state": "BLOCKED" if deficits else "COMPLETE",
        "outcome": "EXPAND_MORE" if deficits else "READY_FOR_TOKENIZER",
        "deficits": deficits,
        "measurements": measurements,
    }
