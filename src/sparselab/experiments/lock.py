"""Canonical, verified, immutable experiment plans and their local availability bindings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from sparselab.config.models import RunConfig, StrictModel
from sparselab.experiments.artifacts import verify_artifact, verify_inputs
from sparselab.experiments.compiler import (
    apply_patch,
    compare_configs,
    describe_differences,
    expand_axes,
    patchable_config,
)
from sparselab.experiments.plan import (
    Artifact,
    CorpusVariant,
    ExperimentPlan,
    Phase,
    base_run_config,
)
from sparselab.training.manifest import (
    canonical_json,
    config_sha256,
    sha256_file,
    source_identity,
)

_LOCAL_FIELDS = {
    "logging": ("root_dir",),
    "tokenizer": ("path",),
    "dataset": (
        "cache_dir",
        "source_manifest_path",
        "train_path",
        "validation_path",
        "allocation_manifest_path",
        "corpus_release_path",
        "corpus_export_path",
    ),
    "model": ("memory_package_path",),
    "training": ("portability_manifest_path",),
}


class ResolvedCell(StrictModel):
    id: str
    coordinate: dict[str, str]
    phase: str
    config: RunConfig
    config_sha256: str
    requested_runtime: dict[str, Any]
    effective: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]


class ResolvedComparison(StrictModel):
    id: str
    baseline: str
    variant: str
    mode: Literal["controlled", "multi_factor", "descriptive", "none"]
    interventions: tuple[str, ...]
    invariants: tuple[str, ...]
    confounders: tuple[str, ...]
    differences: tuple[dict[str, Any], ...]


class ResolvedPhase(StrictModel):
    id: str
    transition: Literal["fresh", "resume", "extend_budget", "promote"]
    parent: str | None = None
    checkpoint: str | None = None
    selector: str | None = None
    at_step: int | None = None


class ResolvedExperimentPlan(StrictModel):
    lock_version: Literal[1] = 1
    id: str
    source_identity: dict[str, Any]
    cells: tuple[ResolvedCell, ...]
    inputs: dict[str, dict[str, Any]]
    artifacts: dict[str, dict[str, Any]]
    comparisons: tuple[ResolvedComparison, ...]
    phases: tuple[ResolvedPhase, ...]
    evaluations: tuple[dict[str, Any], ...]
    execution: dict[str, Any]
    retention: dict[str, Any]
    scientific_sha256: str
    plan_sha256: str
    # Machine-local paths are deliberately outside both digest domains. They are
    # included in the lock and independently authenticated by the lock's sidecar.
    availability: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_identity(self) -> ResolvedExperimentPlan:
        scientific, full = _identities(self.model_dump(mode="json"))
        if self.scientific_sha256 != scientific or self.plan_sha256 != full:
            raise ValueError(
                "experiment lock canonical scientific/full digest mismatch"
            )
        return self


def _hash(domain: str, value: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + canonical_json(value)
    ).hexdigest()


def _portable(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only known host locations, never their scientific artifact identities."""
    import copy

    value = copy.deepcopy(config)
    for section, keys in _LOCAL_FIELDS.items():
        if section in value:
            for key in keys:
                value[section].pop(key, None)
    return value


def _machine_paths(config: RunConfig) -> dict[str, str]:
    values = config.model_dump(mode="json")
    return {
        f"{section}.{key}": values[section][key]
        for section, keys in _LOCAL_FIELDS.items()
        for key in keys
        if section in values and values[section].get(key) is not None
    }


def _scientific_cell(cell: dict[str, Any]) -> dict[str, Any]:
    config = _portable(cell["config"])
    config["runtime"].pop("device_index", None)
    return {
        "id": cell["id"],
        "coordinate": cell["coordinate"],
        "phase": cell["phase"],
        "config": {
            key: val
            for key, val in config.items()
            if key not in {"name", "logging", "checkpoint", "staging"}
        },
        "requested_runtime": {
            key: val
            for key, val in cell["requested_runtime"].items()
            if key != "device_index"
        },
        "effective": cell["effective"],
        "artifacts": cell["artifacts"],
    }


def _identities(lock: dict[str, Any]) -> tuple[str, str]:
    stable = {
        key: value
        for key, value in lock.items()
        if key not in {"availability", "scientific_sha256", "plan_sha256"}
    }
    science = {
        key: value
        for key, value in stable.items()
        if key not in {"execution", "retention"}
    }
    science["cells"] = [_scientific_cell(cell) for cell in stable["cells"]]
    scientific = _hash("sparselab-experiment-scientific-v1", science)
    full = _hash(
        "sparselab-experiment-lock-v1",
        {
            **stable,
            "execution": {
                key: value
                for key, value in stable["execution"].items()
                if key != "workspace"
            },
            "cells": [
                {**cell, "config": _portable(cell["config"])}
                for cell in stable["cells"]
            ],
        },
    )
    return scientific, full


def _runtime(
    config: RunConfig, selected: str | None
) -> tuple[RunConfig, dict[str, Any]]:
    requested = config.runtime.model_dump(mode="json")
    backend = config.runtime.backend
    if backend == "auto":
        if selected != "cpu":
            raise ValueError(
                "runtime.backend=auto needs a selected supported execution.backend (cpu)"
            )
        backend = "cpu"
    if selected is not None and backend != selected:
        raise ValueError(
            f"runtime backend {backend} disagrees with selected execution.backend {selected}"
        )
    if backend != "cpu" or config.runtime.engine != "pytorch":
        raise ValueError(
            f"backend {backend}/{config.runtime.engine} cannot lock until worker/device capability verification exists"
        )
    precision = config.runtime.precision
    if precision == "auto":
        if backend != "cpu" or config.runtime.engine != "pytorch":
            raise ValueError(
                "runtime.precision=auto cannot be determined for the selected worker/backend"
            )
        precision = "fp32"
    if backend == "cpu" and precision != "fp32":
        raise ValueError("CPU runtime requires fp32 precision")
    runtime = config.runtime.model_copy(
        update={"backend": backend, "precision": precision}
    )
    return RunConfig.model_validate(
        {**config.model_dump(mode="python"), "runtime": runtime}
    ), requested


def _effective(config: RunConfig) -> dict[str, Any]:
    training = config.training
    per_update = (
        training.seq_len * training.micro_batch_size * training.gradient_accumulation
    )
    optimizer = config.optimizer
    return {
        "tokens_per_update": per_update,
        "target_token_exposures": per_update * training.max_steps,
        "optimizer_schedule_kind": (
            "warmup_cosine_floor_v1"
            if optimizer.name == "adamw" and optimizer.decay_steps is not None
            else "warmup_cosine_v1"
            if optimizer.name == "adamw"
            else "adafactor_v1"
        ),
        "optimizer_decay_horizon": (
            optimizer.decay_steps or training.max_steps
            if optimizer.name == "adamw"
            else training.max_steps
        ),
        "runtime_backend": config.runtime.backend,
        "runtime_precision": config.runtime.precision,
    }


def _prepared_variants(
    plan: ExperimentPlan, prepared: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    if not plan.corpus_variants:
        return {}
    if (
        prepared is None
        or prepared.get("format") != "experiment-preparation-v1"
        or prepared.get("id") != plan.id
    ):
        raise ValueError(
            "corpus variants require matching verified prepare_plan output before locking"
        )
    records = prepared.get("variants")
    if not isinstance(records, list) or len(records) != len(plan.corpus_variants):
        raise ValueError("prepared corpus variant inventory differs from declaration")
    result: dict[str, dict[str, Any]] = {}
    for declaration, record in zip(plan.corpus_variants, records, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("id") != declaration.id
            or declaration.id in result
        ):
            raise ValueError(
                "prepared corpus variant IDs/order differ from declaration"
            )
        result[declaration.id] = record
    return result


def _variant_identity(
    record: dict[str, Any],
    declaration: CorpusVariant,
    source: Path,
    authored_artifacts: dict[str, Artifact],
    memo: dict[tuple[object, ...], Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Reverify all materialized dependencies against the declared release."""
    release = Path(record["release_path"])
    export = Path(record["export_path"])
    tokenizer = Path(record["tokenizer_path"])
    packed = Path(record["prepared_path"])
    from sparselab.corpus.project import (
        ReleaseDeclaration,
        load_project,
        release_declaration_payload,
        verify_fraction_tokenizer,
    )
    from sparselab.data.tokenizer import load_tokenizer
    from sparselab.experiments.prepare import verified_reuse_tokenizer

    project_path = Path(declaration.project)
    if not project_path.is_absolute():
        project_path = source.parent / project_path
    project = load_project(project_path)
    expected_release = ReleaseDeclaration.model_validate(
        {
            **project.release.model_dump(mode="python"),
            **declaration.release_set,
        }
    )
    candidate = project.model_copy(update={"release": expected_release})
    verify_fraction_tokenizer(candidate)
    reused = (
        verified_reuse_tokenizer(
            declaration, authored_artifacts, source, expected_release
        )
        if declaration.tokenizer_artifact is not None
        else None
    )
    if reused is None:
        if "tokenizer_artifact" in record:
            raise ValueError("trained variant cannot claim a reused tokenizer")
    elif (
        record.get("tokenizer_artifact") != declaration.tokenizer_artifact
        or reused["kind"] != "tokenizer"
        or reused["version"] != 1
        or reused["identifier"] != tokenizer.parent.name
        or reused["sha256"] != record["tokenizer_sha256"]
        or reused["path"] != str(tokenizer)
    ):
        raise ValueError("prepared variant differs from declared tokenizer artifact")
    release_manifest = json.loads(
        (release / "manifest.json").read_text(encoding="utf-8")
    )
    if release_manifest["build_identity"][
        "project_id"
    ] != project.config.id or release_manifest["build_identity"][
        "release"
    ] != release_declaration_payload(expected_release):
        raise ValueError(
            f"prepared corpus variant {declaration.id} is not the declared project/release"
        )
    manifest = json.loads((packed / "manifest.json").read_text(encoding="utf-8"))
    if record["prepared_manifest_sha256"] != sha256_file(packed / "manifest.json"):
        raise ValueError("prepared variant manifest bytes changed")
    specifications = {
        "corpus_release": (release.name, record["release_id"], release, 1),
        "corpus_export": (export.name, export.name, export, 1),
        "tokenizer": (tokenizer.parent.name, record["tokenizer_sha256"], tokenizer, 1),
        "prepared_data": (
            manifest["settings_sha256"],
            manifest["manifest_sha256"],
            packed,
            1,
        ),
    }
    identities: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for kind, (identifier, digest, path, version) in specifications.items():
        spec = Artifact(
            kind=kind,
            version=version,
            producer="sparselab",
            identifier=identifier,
            sha256=digest,
            path=str(path),
        )
        verified = verify_artifact(spec, source, memo=memo)
        identities[kind] = {
            key: value for key, value in verified.items() if key != "path"
        }
        paths[kind] = verified["path"]
    if record["release_id"] != release.name or record[
        "tokenizer_sha256"
    ] != sha256_file(tokenizer):
        raise ValueError("prepared variant identity mismatch")
    run = RunConfig.model_validate(record["config"])
    if config_sha256(run.model_dump(mode="json")) != record["config_sha256"]:
        raise ValueError("prepared variant config digest mismatch")
    from sparselab.config.loading import load_config

    exported_run = load_config(export / "run.yaml")
    if reused is not None:
        exported_run = RunConfig.model_validate(
            {
                **exported_run.model_dump(mode="python"),
                "tokenizer": {
                    **exported_run.tokenizer.model_dump(mode="python"),
                    "path": tokenizer,
                },
            }
        )
        if run.model.vocab_size != load_tokenizer(tokenizer).get_vocab_size():
            raise ValueError(
                "prepared variant vocabulary differs from verified tokenizer"
            )
    if run.model_dump(mode="json") != exported_run.model_dump(mode="json"):
        raise ValueError(
            "prepared variant config differs from verified export run.yaml"
        )
    if (
        run.dataset.corpus_release_path != release
        or run.dataset.corpus_export_path != export
        or run.tokenizer.path != tokenizer
    ):
        raise ValueError(
            "prepared variant config does not reference verified artifacts"
        )
    return identities, paths


def _check_cell_inputs(
    config: RunConfig,
    identities: dict[str, dict[str, Any]],
    paths: dict[str, str],
    source_sha256: str,
) -> None:
    """Require verified artifacts to be the actual concrete training dependencies."""
    by_kind = {
        item["kind"]: (item, paths[name])
        for name, item in identities.items()
        if name in paths and item.get("sha256") is not None
    }
    required = (
        {"tokenizer", "prepared_data"}
        if config.dataset.source == "synthetic"
        else {"corpus_release", "corpus_export", "tokenizer", "prepared_data"}
    )
    if not required.issubset(by_kind):
        raise ValueError(
            f"cell needs verified {sorted(required)}; missing {sorted(required - by_kind.keys())}"
        )
    _, tokenizer_path = by_kind["tokenizer"]
    _, packed_path = by_kind["prepared_data"]
    if (
        config.tokenizer.path != Path(tokenizer_path)
        or config.dataset.cache_dir != Path(packed_path).parent
        or config.model.memory_package_path is not None
    ):
        raise ValueError(
            "cell config does not match verified tokenizer/prepared paths or uses an unsupported extension package"
        )
    from sparselab.data.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(Path(tokenizer_path))
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        raise ValueError("verified tokenizer vocabulary does not match cell model")
    prepared_identity = json.loads((Path(packed_path) / "manifest.json").read_text())[
        "cache_identity"
    ]
    dataset = config.dataset.model_dump(mode="json")
    for key in (
        "cache_dir",
        "train_path",
        "validation_path",
        "source_manifest_path",
        "corpus_release_path",
        "corpus_export_path",
    ):
        dataset.pop(key, None)
    packing = {
        "memory": config.model.memory,
        "memory_table_size": config.model.memory_table_size,
        "memory_ngram_size": config.model.memory_ngram_size,
    }
    if (
        prepared_identity["dataset"] != dataset
        or prepared_identity["packing"] != packing
        or prepared_identity["tokenizer_sha256"]
        != hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
        or prepared_identity["source_identity_sha256"] != source_sha256
    ):
        raise ValueError(
            "prepared cache does not bind the locked dataset/tokenizer/packing/source"
        )
    if config.dataset.source == "synthetic":
        if (
            config.dataset.corpus_release_path is not None
            or config.dataset.corpus_export_path is not None
        ):
            raise ValueError("synthetic cell cannot claim a Corpus Forge export")
        return
    release, release_path = by_kind["corpus_release"]
    _, export_path = by_kind["corpus_export"]
    if (
        config.dataset.revision != release["sha256"]
        or config.dataset.corpus_release_path != Path(release_path)
        or config.dataset.corpus_export_path != Path(export_path)
    ):
        raise ValueError("cell config does not match verified corpus release/export")
    if config.dataset.train_path is None or config.dataset.validation_path is None:
        raise ValueError("verified corpus export must supply training and validation")
    from sparselab.config.loading import load_config

    exported = load_config(Path(export_path) / "run.yaml")
    if (
        config.dataset.model_dump(mode="json")
        != exported.dataset.model_dump(mode="json")
        or config.model.vocab_size != exported.model.vocab_size
    ):
        raise ValueError("cell dataset/vocabulary differs from verified corpus export")


def _phases(plan: ExperimentPlan) -> tuple[ResolvedPhase, ...]:
    phases = plan.phases or (Phase(id="main"),)
    positions = {phase.id: index for index, phase in enumerate(phases)}
    for index, phase in enumerate(phases):
        if phase.parent is not None:
            if phase.parent not in positions or positions[phase.parent] >= index:
                raise ValueError(
                    f"phase {phase.id} parent must be an earlier declared phase (acyclic DAG)"
                )
            if phase.selector == "terminal" and phase.at_step is None:
                raise ValueError(f"phase {phase.id} terminal parent needs an update")
            if phase.selector == "best_validation" and phase.at_step is not None:
                raise ValueError(
                    f"phase {phase.id} best-validation selection cannot also pin a terminal update"
                )
        if phase.checkpoint is not None:
            artifact = plan.artifacts.get(phase.checkpoint)
            if (
                artifact is None
                or artifact.kind != "checkpoint"
                or artifact.from_phase is not None
            ):
                raise ValueError(
                    f"phase {phase.id} needs a pinned external checkpoint generation"
                )
            if (
                phase.transition in {"resume", "extend_budget"}
                and artifact.state != "full"
            ):
                raise ValueError(
                    f"phase {phase.id} {phase.transition} requires full checkpoint state"
                )
            if phase.transition == "promote" and artifact.state not in {
                "full",
                "weights",
            }:
                raise ValueError(
                    f"phase {phase.id} promotion requires pinned checkpoint weights"
                )
            if phase.selector is not None or phase.at_step is not None:
                raise ValueError(
                    f"phase {phase.id} external checkpoint is already a pinned generation"
                )
        if phase.transition == "fresh" and (
            phase.selector is not None or phase.at_step is not None
        ):
            raise ValueError(
                f"fresh phase {phase.id} cannot select a parent generation"
            )
    return tuple(
        ResolvedPhase(
            id=p.id,
            transition=p.transition,
            parent=p.parent,
            checkpoint=p.checkpoint,
            selector=p.selector,
            at_step=p.at_step,
        )
        for p in phases
    )


def resolve_plan(
    plan: ExperimentPlan,
    source: Path,
    *,
    prepared: dict[str, Any] | None = None,
    max_runs: int = 1000,
) -> ResolvedExperimentPlan:
    """Resolve only verifiable scientific inputs into a frozen executable plan."""
    source = Path(source).resolve()
    package = source_identity()
    base = base_run_config(plan, source)
    phases = _phases(plan)
    variants = _prepared_variants(plan, prepared)
    memo: dict[tuple[object, ...], Any] = {}
    verified_inputs = verify_inputs(plan, source, memo=memo)
    artifacts: dict[str, dict[str, Any]] = {}
    availability: dict[str, Any] = {
        "inputs": {},
        "artifacts": {},
        "variants": {},
        "cell_paths": {},
        "workspace": plan.execution.workspace,
    }
    for name, artifact in plan.artifacts.items():
        if artifact.from_phase is not None:
            if artifact.from_phase not in {phase.id for phase in phases}:
                raise ValueError(f"planned artifact {name} has unknown producing phase")
            artifacts[name] = artifact.model_dump(mode="json", exclude={"path"})
            continue
        verified = verify_artifact(artifact, source, memo=memo)
        artifacts[name] = {
            **{key: val for key, val in verified.items() if key != "path"},
            "producer": artifact.producer,
            "state": artifact.state,
        }
        availability["artifacts"][name] = verified["path"]
    for name, item in verified_inputs.items():
        availability["inputs"][name] = item["path"]
    inputs = {
        name: {key: val for key, val in item.items() if key != "path"}
        for name, item in verified_inputs.items()
    }
    identities: dict[str, dict[str, dict[str, Any]]] = {}
    for declaration in plan.corpus_variants:
        name = declaration.id
        identities[name], availability["variants"][name] = _variant_identity(
            variants[name],
            declaration,
            source,
            plan.artifacts,
            memo,
        )
        for kind, identity in identities[name].items():
            key = f"variant.{name}.{kind}"
            if key in artifacts:
                raise ValueError(
                    f"variant {name} artifact key collides with declared artifact {key}"
                )
            artifacts[key] = {**identity, "producer": "sparselab", "state": None}
            availability["artifacts"][key] = availability["variants"][name][kind]
    # The axis compiler alone validates labels, collisions and bounded products.
    selection_axes = []
    for axis in plan.axes:
        choices = []
        for choice in axis.choices:
            patch = dict(choice.set)
            selection = patch.pop("inputs.corpus_variant", None)
            if selection is not None and selection not in variants:
                raise ValueError(
                    f"axis {axis.name}.{choice.label} selects unprepared corpus variant {selection!r}"
                )
            choices.append(choice.model_copy(update={"set": patch}))
        selection_axes.append(axis.model_copy(update={"choices": tuple(choices)}))
    expanded = expand_axes(
        base, tuple(selection_axes), max_runs=max_runs, base_dir=source.parent
    )
    cells: list[ResolvedCell] = []
    for config, coordinate in expanded:
        selected = [
            choice.set["inputs.corpus_variant"]
            for axis in plan.axes
            for choice in axis.choices
            if coordinate[axis.name] == choice.label
            and "inputs.corpus_variant" in choice.set
        ]
        if len(selected) > 1:
            raise ValueError(
                f"coordinate {coordinate} selects conflicting corpus variants"
            )
        if not selected and len(variants) == 1 and not plan.axes:
            selected = [next(iter(variants))]
        cell_artifacts = dict(inputs)
        if selected:
            record = variants[selected[0]]
            authored = config.model_dump(mode="python")
            exported = record["config"]
            config = RunConfig.model_validate(
                {
                    **exported,
                    **{
                        key: value
                        for key, value in authored.items()
                        if key not in {"dataset", "tokenizer", "model"}
                    },
                    "model": {
                        **exported["model"],
                        **{
                            key: value
                            for key, value in authored["model"].items()
                            if key != "vocab_size"
                        },
                    },
                }
            )
            cell_artifacts.update(identities[selected[0]])
        elif not inputs:
            raise ValueError(
                f"coordinate {coordinate}: no verified external dataset/tokenizer inputs"
            )
        cell_id = (
            ",".join(f"{key}={value}" for key, value in coordinate.items()) or "single"
        )
        previous: dict[str, ResolvedCell] = {}
        for phase in phases:
            declaration = (
                next(item for item in plan.phases if item.id == phase.id)
                if plan.phases
                else None
            )
            config_values = patchable_config(config)
            if declaration is not None and declaration.set:
                config_values = apply_patch(
                    config_values, declaration.set, source=f"phase {phase.id}"
                )
            phased, requested = _runtime(
                RunConfig.model_validate(config_values), plan.execution.backend
            )
            cell_paths = {name: availability["inputs"][name] for name in inputs}
            if selected:
                cell_paths.update(availability["variants"][selected[0]])
            _check_cell_inputs(phased, cell_artifacts, cell_paths, package["sha256"])
            phase_artifacts = dict(cell_artifacts)
            if phase.parent is not None:
                parent = previous[phase.parent]
                if (
                    phase.selector == "terminal"
                    and phase.at_step != parent.config.training.max_steps
                ):
                    raise ValueError(
                        f"phase {phase.id} terminal selection must pin parent terminal update"
                    )
                if phase.transition == "resume":
                    from sparselab.training.continuation import _resume_settings

                    if _resume_settings(parent.config) != _resume_settings(phased):
                        raise ValueError(
                            f"phase {phase.id} full-state resume changes parent scientific settings"
                        )
                if phase.transition == "extend_budget":
                    if (
                        parent.config.optimizer.name != "adamw"
                        or phased.optimizer.name != "adamw"
                        or parent.config.optimizer.decay_steps is not None
                        or phased.optimizer.decay_steps
                        != parent.config.training.max_steps
                        or phased.training.max_steps <= parent.config.training.max_steps
                        or parent.config.training.max_tokens
                        != parent.effective["target_token_exposures"]
                        or phased.training.max_tokens
                        != _effective(phased)["target_token_exposures"]
                    ):
                        raise ValueError(
                            f"phase {phase.id} requires terminal whole-update AdamW budget extension"
                        )
                    from sparselab.training.continuation import _resume_settings

                    comparable = phased.model_dump(mode="json")
                    comparable["training"]["max_steps"] = (
                        parent.config.training.max_steps
                    )
                    comparable["training"]["max_tokens"] = (
                        parent.config.training.max_tokens
                    )
                    comparable["optimizer"].pop("decay_steps")
                    if _resume_settings(
                        RunConfig.model_validate(comparable)
                    ) != _resume_settings(parent.config):
                        raise ValueError(
                            f"phase {phase.id} changes more than budget and decay horizon"
                        )
                if phase.transition == "promote":
                    from sparselab.training.manifest import architecture_sha256

                    if architecture_sha256(
                        parent.config.model_dump(mode="json")
                    ) != architecture_sha256(
                        phased.model_dump(mode="json")
                    ) or parent.artifacts.get("tokenizer") != phase_artifacts.get(
                        "tokenizer"
                    ):
                        raise ValueError(
                            f"phase {phase.id} promotion changes architecture or tokenizer"
                        )
                phase_artifacts["parent_checkpoint"] = {
                    "kind": "checkpoint",
                    "version": 1,
                    "producer": parent.id,
                    "identifier": f"{parent.id}:{phase.selector}:{phase.at_step or 'lowest-loss'}",
                    "from_phase": phase.parent,
                    "selector": phase.selector,
                    "at_step": phase.at_step,
                    "state": "full",  # trainer emits full generations; promotion transfers weights only
                }
            elif phase.checkpoint is not None:
                from sparselab.training.continuation import _resume_settings
                from sparselab.training.manifest import (
                    architecture_sha256,
                    read_manifest,
                )

                generation = Path(availability["artifacts"][phase.checkpoint])
                parent_run = generation.parent.parent
                parent_config = RunConfig.model_validate(
                    read_manifest(parent_run / "manifest.json")["effective_config"]
                )
                checkpoint_record = json.loads(
                    (generation / "manifest.json").read_text()
                )
                if phase.transition == "resume":
                    if _resume_settings(parent_config) != _resume_settings(phased):
                        raise ValueError(
                            f"phase {phase.id} differs from full-state external parent"
                        )
                elif phase.transition == "extend_budget":
                    if (
                        parent_config.optimizer.name != "adamw"
                        or phased.optimizer.name != "adamw"
                        or parent_config.optimizer.decay_steps is not None
                        or phased.optimizer.decay_steps
                        != parent_config.training.max_steps
                        or checkpoint_record["step"] != parent_config.training.max_steps
                        or checkpoint_record["tokens_seen"]
                        != parent_config.training.max_tokens
                        or phased.training.max_steps <= parent_config.training.max_steps
                        or parent_config.training.max_tokens
                        != _effective(parent_config)["target_token_exposures"]
                        or phased.training.max_tokens
                        != _effective(phased)["target_token_exposures"]
                    ):
                        raise ValueError(
                            f"phase {phase.id} needs a terminal whole-update AdamW external parent"
                        )
                    comparable = phased.model_dump(mode="json")
                    comparable["training"]["max_steps"] = (
                        parent_config.training.max_steps
                    )
                    comparable["training"]["max_tokens"] = (
                        parent_config.training.max_tokens
                    )
                    comparable["optimizer"].pop("decay_steps")
                    if _resume_settings(
                        RunConfig.model_validate(comparable)
                    ) != _resume_settings(parent_config):
                        raise ValueError(
                            f"phase {phase.id} changes more than external parent budget and decay horizon"
                        )
                elif architecture_sha256(
                    parent_config.model_dump(mode="json")
                ) != architecture_sha256(phased.model_dump(mode="json")) or sha256_file(
                    parent_run / "tokenizer.json"
                ) != sha256_file(phased.tokenizer.path):
                    raise ValueError(
                        f"phase {phase.id} promotion changes external parent architecture/tokenizer"
                    )
                phase_artifacts["parent_checkpoint"] = artifacts[phase.checkpoint]
            cell = ResolvedCell(
                id=f"{phase.id}:{cell_id}",
                coordinate=coordinate,
                phase=phase.id,
                config=phased,
                config_sha256=config_sha256(phased.model_dump(mode="json")),
                requested_runtime=requested,
                effective=_effective(phased),
                artifacts=phase_artifacts,
            )
            cells.append(cell)
            availability["cell_paths"][cell.id] = _machine_paths(cell.config)
            previous[phase.id] = cell
    comparisons: list[ResolvedComparison] = []
    for comparison in plan.comparisons:
        if set(comparison.baseline) != {axis.name for axis in plan.axes}:
            raise ValueError(
                f"comparison {comparison.id} must uniquely select each declared axis"
            )
        for phase in phases:
            left = next(
                (
                    cell
                    for cell in cells
                    if cell.coordinate == comparison.baseline and cell.phase == phase.id
                ),
                None,
            )
            right = next(
                (
                    cell
                    for cell in cells
                    if cell.coordinate == comparison.variant and cell.phase == phase.id
                ),
                None,
            )
            if left is None or right is None:
                raise ValueError(
                    f"comparison {comparison.id} has unmatched coordinate/phase"
                )
            left_values = {
                **_portable(left.config.model_dump(mode="json")),
                "effective": left.effective,
            }
            right_values = {
                **_portable(right.config.model_dump(mode="json")),
                "effective": right.effective,
            }
            for values in (left_values, right_values):
                for key in ("name", "logging", "checkpoint", "staging"):
                    values.pop(key, None)
                values["runtime"].pop("device_index", None)
            differences = compare_configs(
                left_values,
                right_values,
                expected=comparison.interventions,
                invariants=comparison.invariants,
                artifacts={"base": left.artifacts, "variant": right.artifacts},
            )
            if comparison.mode == "controlled" and len(comparison.interventions) != 1:
                raise ValueError(
                    f"comparison {comparison.id} controlled requires exactly one semantic intervention"
                )
            if comparison.mode == "none" and differences:
                raise ValueError(f"comparison {comparison.id} none has changed fields")
            comparisons.append(
                ResolvedComparison(
                    id=comparison.id,
                    baseline=left.id,
                    variant=right.id,
                    mode=comparison.mode,
                    interventions=comparison.interventions,
                    invariants=comparison.invariants,
                    confounders=comparison.confounders,
                    differences=describe_differences(differences),
                )
            )
    for evaluation in plan.evaluations:
        for reference in (evaluation.checkpoint, evaluation.data, evaluation.cases):
            if reference is not None and reference not in artifacts:
                raise ValueError(
                    f"evaluation {evaluation.id} references unsupported/unverified artifact {reference}"
                )
        if artifacts[evaluation.checkpoint]["kind"] != "checkpoint":
            raise ValueError(
                f"evaluation {evaluation.id} requires a verified checkpoint"
            )
    data = {
        "lock_version": 1,
        "id": plan.id,
        "source_identity": package,
        "cells": tuple(cells),
        "inputs": inputs,
        "artifacts": artifacts,
        "comparisons": tuple(comparisons),
        "phases": phases,
        "evaluations": tuple(item.model_dump(mode="json") for item in plan.evaluations),
        "execution": {**plan.execution.model_dump(mode="json"), "workspace": None},
        "retention": plan.retention.model_dump(mode="json"),
        "availability": availability,
    }
    raw = ResolvedExperimentPlan.model_construct(
        **data, scientific_sha256="", plan_sha256=""
    ).model_dump(mode="json")
    scientific, full = _identities(raw)
    return ResolvedExperimentPlan.model_validate(
        {**raw, "scientific_sha256": scientific, "plan_sha256": full}
    )


def _exclusive_bytes(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".experiment-lock-", delete=False
    ) as stream:
        staging = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
    try:
        os.link(
            staging, path
        )  # atomic, exclusive publication; never truncate an existing ID
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        staging.unlink(missing_ok=True)


def publish_lock(lock: ResolvedExperimentPlan, workspace: Path) -> Path:
    """Publish once; existing IDs must be byte-identical and independently valid."""
    lock = ResolvedExperimentPlan.model_validate(lock.model_dump(mode="json"))
    directory = Path(workspace) / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{lock.plan_sha256}.json"
    sidecar = directory / f"{lock.plan_sha256}.availability.json"
    payload = lock.model_dump(mode="json")
    availability = payload.pop("availability")
    body = canonical_json(payload) + b"\n"
    binding = (
        canonical_json(
            {
                "plan_sha256": lock.plan_sha256,
                "availability": availability,
                "sha256": _hash(
                    "sparselab-experiment-availability-v1",
                    {"plan_sha256": lock.plan_sha256, "availability": availability},
                ),
            }
        )
        + b"\n"
    )
    if path.exists() or sidecar.exists():
        existing = open_lock(path)
        if existing.model_dump(mode="json") != lock.model_dump(mode="json"):
            raise ValueError(
                f"lock {path} already exists with a different availability binding"
            )
        return path
    _exclusive_bytes(sidecar, binding)
    try:
        _exclusive_bytes(path, body)
    except BaseException:
        sidecar.unlink(missing_ok=True)
        raise
    open_lock(path)
    return path


def open_lock(path: Path) -> ResolvedExperimentPlan:
    """Reopen a lock without consulting the authored YAML or unverified aliases."""
    path = Path(path)
    lock_bytes = path.read_bytes()
    binding_bytes = path.with_name(path.stem + ".availability.json").read_bytes()
    raw = json.loads(lock_bytes)
    binding = json.loads(binding_bytes)
    if not isinstance(raw, dict) or not isinstance(binding, dict):
        raise TypeError("invalid experiment lock or availability object")
    if (
        lock_bytes != canonical_json(raw) + b"\n"
        or binding_bytes != canonical_json(binding) + b"\n"
    ):
        raise ValueError("experiment lock or availability is not canonical finite JSON")
    if path.name != f"{raw.get('plan_sha256')}.json" or binding.get(
        "plan_sha256"
    ) != raw.get("plan_sha256"):
        raise ValueError("experiment lock filename/availability ID mismatch")
    availability = binding.get("availability")
    if binding.get("sha256") != _hash(
        "sparselab-experiment-availability-v1",
        {"plan_sha256": raw["plan_sha256"], "availability": availability},
    ):
        raise ValueError("experiment lock availability binding changed")
    lock = ResolvedExperimentPlan.model_validate({**raw, "availability": availability})
    package = source_identity()
    if canonical_json(lock.source_identity) != canonical_json(package):
        raise ValueError("experiment lock source implementation identity changed")
    for name, identity in lock.artifacts.items():
        if identity.get("from_phase") is not None:
            if identity["from_phase"] not in {phase.id for phase in lock.phases}:
                raise ValueError(
                    f"planned checkpoint {name} has unknown producing phase"
                )
            continue
        artifact = Artifact.model_validate(
            {**identity, "path": availability["artifacts"][name]}
        )
        verify_artifact(artifact, path)
    for name, identity in lock.inputs.items():
        if not any(
            identity
            == {
                key: value
                for key, value in artifact.items()
                if key in {"kind", "version", "identifier", "sha256"}
            }
            for artifact in lock.artifacts.values()
        ):
            raise ValueError(f"input {name} no longer references a locked artifact")
    for name, locations in availability["variants"].items():
        for kind, location in locations.items():
            key = f"variant.{name}.{kind}"
            if (
                key not in lock.artifacts
                or availability["artifacts"].get(key) != location
            ):
                raise ValueError(
                    f"variant {name} {kind} has an unbound availability path"
                )
    for cell in lock.cells:
        phase = next((phase for phase in lock.phases if phase.id == cell.phase), None)
        if phase is None:
            raise ValueError(f"cell {cell.id} has an unknown phase")
        if availability["cell_paths"].get(cell.id) != _machine_paths(cell.config):
            raise ValueError(f"cell {cell.id} has changed machine-local paths")
        for name, identity in cell.artifacts.items():
            if name == "parent_checkpoint" and phase.parent is not None:
                parent_id = f"{phase.parent}:{cell.id.split(':', 1)[1]}"
                if (
                    identity.get("producer") != parent_id
                    or identity.get("selector") != phase.selector
                    or identity.get("at_step") != phase.at_step
                ):
                    raise ValueError(
                        f"cell {cell.id} changed parent checkpoint selector"
                    )
            elif (
                identity not in lock.inputs.values()
                and identity not in lock.artifacts.values()
                and not any(
                    identity
                    == {
                        key: value
                        for key, value in artifact.items()
                        if key in {"kind", "version", "identifier", "sha256"}
                    }
                    for artifact in lock.artifacts.values()
                )
            ):
                raise ValueError(f"cell {cell.id} has an unverified artifact identity")
        paths = {
            name: availability["inputs"][name]
            for name in lock.inputs
            if name in cell.artifacts
        }
        for variant, inventory in availability["variants"].items():
            if all(
                cell.artifacts.get(kind)
                == {
                    key: value
                    for key, value in lock.artifacts[
                        f"variant.{variant}.{kind}"
                    ].items()
                    if key in {"kind", "version", "identifier", "sha256"}
                }
                for kind in inventory
            ):
                paths.update(inventory)
        _check_cell_inputs(cell.config, cell.artifacts, paths, package["sha256"])
    return lock


def storage_preview(
    config: RunConfig, retention: dict[str, bool] | None = None
) -> dict[str, Any]:
    """Conservative generation/write planning; validation is not checkpoint cadence."""
    from dataclasses import asdict
    from math import ceil

    from sparselab.model.inspection import inspection_report
    from sparselab.workspace_preflight import (
        projected_data_bytes,
        training_storage_checks,
    )

    policy = retention or {}
    training = config.training
    tokens_per_update = (
        training.seq_len * training.micro_batch_size * training.gradient_accumulation
    )
    steps = min(training.max_steps, ceil(training.max_tokens / tokens_per_update))
    checkpoints = config.checkpoint
    explicit = tuple(step for step in checkpoints.steps if 0 < step < steps)
    scheduled_steps = (
        (steps - 1) // checkpoints.every_steps if checkpoints.every_steps else 0
    )
    scheduled_tokens = (
        (steps * tokens_per_update) // checkpoints.every_tokens
        if checkpoints.every_tokens
        else 0
    )
    validation_steps = (steps - 1) // config.evaluation.every_steps
    # Both step zero and the terminal update are always written. Periodic due
    # watermarks may be reset by best-validation writes, so count all possible
    # writes only once per update; minutes are unbounded by wall time but bounded
    # by the number of optimizer updates.
    lower = 2 + max(len(explicit), scheduled_steps)
    upper = steps + 1
    retained_upper = (
        upper
        if checkpoints.keep_periodic and policy.get("keep_periodic", True)
        else min(upper, 4)
    )
    checkpoint_bytes = int(inspection_report(config)["estimated_checkpoint_bytes"])
    cache_bytes = projected_data_bytes(config)
    checks = [asdict(check) for check in training_storage_checks(config)]
    return {
        "initial_generation": 0,
        "terminal_generation": steps,
        "explicit_checkpoint_steps": explicit,
        "checkpoint_every_steps": checkpoints.every_steps,
        "checkpoint_every_tokens": checkpoints.every_tokens,
        "checkpoint_every_minutes": checkpoints.every_minutes,
        "validation_every_steps": config.evaluation.every_steps,
        "periodic_step_trigger_upper": scheduled_steps,
        "periodic_token_trigger_upper": scheduled_tokens,
        "validation_triggered_best_upper": validation_steps + 1,
        "write_count_lower": min(lower, upper),
        "write_count_upper": upper,
        "retained_generations_lower": 1,
        "retained_generations_upper": retained_upper,
        "estimated_checkpoint_bytes": checkpoint_bytes,
        "estimated_total_write_bytes_upper": ceil(upper * checkpoint_bytes * 1.25),
        "estimated_retained_bytes_upper": ceil(
            retained_upper * checkpoint_bytes * 1.25
        ),
        "estimated_retained_inodes_upper": retained_upper * 16,
        "estimated_packed_cache_bytes": cache_bytes,
        "available_storage": checks,
        "estimated_free_bytes_after_retention": min(
            (
                check["available_bytes"]
                - ceil(retained_upper * checkpoint_bytes * 1.25)
                - cache_bytes
                - check["reserve_bytes"]
                for check in checks
            ),
            default=None,
        ),
    }
