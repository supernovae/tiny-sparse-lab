"""Versioned admission of inventoried prepared-input CPU Campaigns.

The authored native Campaign owns connections; this context authenticates them.
A reviewed commit is a caller assertion, not author attestation or compute credit.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, StrictInt, StrictStr, model_validator

from sparselab.campaign.plan import CampaignPlan
from sparselab.config.models import StrictModel
from sparselab.experiments.compiler import expand_axes
from sparselab.experiments.plan import recipe_run_config
from sparselab.experiments.preparation_execution import _environment
from sparselab.experiments.workflow_recipe import (
    _location,
    load_workflow,
    locked_workflow_context,
)
from sparselab.recovery.provenance import git_provenance, repository_root


class PairLimits(StrictModel):
    """Declared ceilings, not allocation or an operating-system resource quota."""

    max_training_updates: StrictInt = Field(gt=0)
    max_target_positions: StrictInt = Field(gt=0)
    max_eval_batches: StrictInt = Field(gt=0)


class WorkflowAdmission(StrictModel):
    workflow_admission_version: StrictInt = Field(ge=1, le=2)
    reviewed_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    recipe_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cpu_python: Path
    pair_limits: PairLimits | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def versioned_limits(self):
        if (self.workflow_admission_version == 2) != (self.pair_limits is not None):
            raise ValueError("pair_limits are required only for workflow admission v2")
        return self


class WorkflowExecution:
    def __init__(
        self, source: Path, bindings: Path, work_dir: Path, admission: WorkflowAdmission
    ):
        self.source = Path(source).absolute()
        self.bindings = Path(bindings).absolute()
        if not Path(work_dir).is_absolute():
            raise ValueError("workflow requires an explicit absolute work root")
        self.work_dir = _location(self.bindings, str(work_dir))
        self.admission = admission.model_copy(deep=True)
        self.loaded = load_workflow(self.source, self.bindings)
        self.environment = _environment()
        self.identity = {
            "format": "sparselab-workflow-execution-v1",
            "source": str(self.source),
            "bindings": str(self.bindings),
            "work_dir": str(self.work_dir),
            "admission": self.admission.model_dump(mode="json"),
            "environment": self.environment,
        }
        loaded = self.loaded
        if loaded.descriptor.campaign is None:
            raise ValueError("workflow execution requires an inventoried Campaign")
        self.campaign_source = loaded.context.reference(
            loaded.source, loaded.descriptor.campaign
        )
        document = loaded.context.read_document(self.campaign_source)
        # Artifact identity is portable; its location comes exclusively from bindings.
        for stage in document.get("stages", []):
            if stage.get("kind") == "artifact_reference":
                name = stage.get("id")
                if name not in loaded.plan.artifacts or stage.get(
                    "artifact"
                ) != loaded.descriptor.artifacts[name].model_dump(mode="json"):
                    raise ValueError("Campaign artifact differs from workflow identity")
                stage["artifact"] = loaded.plan.artifacts[name].model_dump(mode="json")
        self.campaign = CampaignPlan.model_validate(document)
        if admission.workflow_admission_version == 2:
            self._validate_pair()
        else:
            self._validate_single()
        self.refresh()

    def _validate_single(self):
        loaded = self.loaded
        kinds = [s.kind for s in self.campaign.stages]
        if (
            sorted(kinds)
            != sorted(
                [
                    "artifact_reference",
                    "artifact_reference",
                    "experiment_plan",
                    "runtime_acceptance",
                    "approval",
                    "experiment_run",
                    "experiment_collect",
                    "evaluation",
                ]
            )
            or self.campaign.recovery is not None
        ):
            raise ValueError("workflow supports one prepared-input CPU Campaign")
        stages = {s.kind: s for s in self.campaign.stages}
        plan, runtime, gate, run, collect, evaluation = (
            stages[k]
            for k in (
                "experiment_plan",
                "runtime_acceptance",
                "approval",
                "experiment_run",
                "experiment_collect",
                "evaluation",
            )
        )
        if (
            plan.source != loaded.descriptor.entrypoint
            or plan.mode != "lock"
            or plan.tokenizer != loaded.plan.inputs["tokenizer"]
            or plan.prepared != loaded.plan.inputs["prepared_data"]
            or runtime.plan != plan.id
            or runtime.profile_id is not None
            or runtime.worker is not None
            or set(gate.bind) != {plan.id, runtime.id}
            or run.plan != plan.id
            or run.runtime != runtime.id
            or gate.id not in run.requires
            or run.cell != "main:single"
            or collect.plan != plan.id
            or collect.run != run.id
            or evaluation.collect != collect.id
            or evaluation.backend not in (None, "cpu")
            or evaluation.runtime is not None
            or evaluation.suite != loaded.plan.evaluation_suite
        ):
            raise ValueError("unsupported portable Campaign connections")
        if loaded.plan.axes or loaded.plan.comparisons:
            raise ValueError("workflow execution supports one unexpanded cell")
        self.config = recipe_run_config(
            loaded.plan, loaded.source, context=loaded.context
        )
        if (
            self.config.dataset.source != "synthetic"
            or self.config.runtime.backend != "cpu"
            or self.config.runtime.engine != "pytorch"
            or loaded.plan.execution.backend not in (None, "cpu")
        ):
            raise ValueError("workflow execution supports synthetic CPU PyTorch only")

    def _validate_pair(self):
        loaded = self.loaded
        plan = loaded.plan
        if (
            len(plan.axes) != 1
            or len(plan.axes[0].choices) != 2
            or len(plan.comparisons) != 1
            or plan.comparisons[0].mode != "controlled"
            or len(plan.comparisons[0].interventions) != 1
            or plan.comparisons[0].phases
        ):
            raise ValueError(
                "paired workflow requires two choices and one controlled comparison"
            )
        comparison = plan.comparisons[0]
        # A pair shares its observation protocol and data. Native lock comparison
        # checks the exact actual intervention, invariants and artifact identities.
        intervention = comparison.interventions[0]
        if intervention.split(".")[0] not in {"model", "attention", "optimizer"}:
            raise ValueError(
                "paired workflow intervention must preserve data, seed, runtime and evaluation"
            )
        self.config = recipe_run_config(plan, loaded.source, context=loaded.context)
        expanded = expand_axes(
            self.config, plan.axes, max_runs=2, base_dir=loaded.source.parent
        )
        coordinates = [coordinate for _, coordinate in expanded]
        if (
            comparison.baseline not in coordinates
            or comparison.variant not in coordinates
        ):
            raise ValueError(
                "paired comparison must select both concrete configurations"
            )
        limits = self.admission.pair_limits
        assert limits is not None
        updates = positions = 0
        for config, _ in expanded:
            if (
                self.config.dataset.source != "snapshot"
                or config.dataset != self.config.dataset
                or config.tokenizer != self.config.tokenizer
                or config.dataset.source != "snapshot"
                or config.runtime.backend != "cpu"
                or config.runtime.engine != "pytorch"
                or plan.execution.backend not in (None, "cpu")
            ):
                raise ValueError("paired workflow supports snapshot CPU PyTorch only")
            count = (
                config.training.max_steps
                + config.staging.smoke_steps
                + config.staging.warmup_steps
            )
            updates += count
            positions += (
                count
                * config.training.seq_len
                * config.training.micro_batch_size
                * config.training.gradient_accumulation
            )
            if config.evaluation.max_batches > limits.max_eval_batches:
                raise ValueError("paired workflow exceeds declared evaluation limit")
        if (
            updates > limits.max_training_updates
            or positions > limits.max_target_positions
        ):
            raise ValueError(
                "paired workflow exceeds declared training limits (including pilots)"
            )
        groups = {}
        for stage in self.campaign.stages:
            groups.setdefault(stage.kind, []).append(stage)
        counts = {kind: len(stages) for kind, stages in groups.items()}
        if (
            counts
            != {
                "artifact_reference": 2,
                "experiment_plan": 1,
                "runtime_acceptance": 1,
                "approval": 1,
                "experiment_run": 2,
                "experiment_collect": 2,
                "evaluation": 2,
            }
            or self.campaign.recovery is not None
        ):
            raise ValueError(
                "paired workflow requires one native plan and two run/collect/evaluation branches"
            )
        planning = groups["experiment_plan"][0]
        runtime = groups["runtime_acceptance"][0]
        gate = groups["approval"][0]
        runs = groups["experiment_run"]
        collects = groups["experiment_collect"]
        evaluations = groups["evaluation"]
        run_ids = {run.id for run in runs}
        expected_cells = {
            f"main:{plan.axes[0].name}={choice.label}"
            for choice in plan.axes[0].choices
        }
        base_dependencies = {planning.id, runtime.id, gate.id}
        first = [run for run in runs if set(run.requires) == base_dependencies]
        if (
            planning.source != loaded.descriptor.entrypoint
            or planning.mode != "lock"
            or planning.tokenizer != plan.inputs["tokenizer"]
            or planning.prepared != plan.inputs["prepared_data"]
            or set(planning.requires) != set(plan.inputs.values())
            or runtime.plan != planning.id
            or set(runtime.requires) != {planning.id}
            or runtime.profile_id is not None
            or runtime.worker is not None
            or set(gate.bind) != {planning.id, runtime.id}
            or set(gate.requires) != {planning.id, runtime.id}
            or {run.cell for run in runs} != expected_cells
            or any(run.plan != planning.id or run.runtime != runtime.id for run in runs)
            or len(first) != 1
            or any(
                set(run.requires) != base_dependencies | {first[0].id}
                for run in runs
                if run not in first
            )
            or {collect.run for collect in collects} != run_ids
            or any(
                collect.plan != planning.id
                or set(collect.requires) != {planning.id, *run_ids}
                for collect in collects
            )
            or {evaluation.collect for evaluation in evaluations}
            != {collect.id for collect in collects}
            or any(
                evaluation.backend not in (None, "cpu")
                or evaluation.runtime is not None
                or evaluation.suite != plan.evaluation_suite
                or set(evaluation.requires) != {evaluation.collect}
                for evaluation in evaluations
            )
        ):
            raise ValueError("unsupported paired Campaign connections")

    def refresh(self):
        current = load_workflow(self.source, self.bindings)
        if (
            current != self.loaded
            or current.recipe_sha256 != self.admission.recipe_sha256
            or current.binding_sha256 != self.admission.binding_sha256
        ):
            raise ValueError("WORKFLOW_CONTEXT_CHANGED: recipe or bindings differ")
        if (
            self.admission.cpu_python.absolute() != Path(sys.executable).absolute()
            or _environment() != self.environment
        ):
            raise ValueError(
                "WORKFLOW_CONTEXT_CHANGED: selected CPU environment differs"
            )
        _location(self.bindings, str(self.work_dir))
        if repository_root(self.work_dir) is not None:
            raise ValueError("workflow work root must be outside Git")
        provenance = git_provenance(
            (
                self.source,
                *(current.context.root / name for name in current.context.documents),
            )
        )
        if (
            provenance["status"] != "CLEAN_AND_COMMITTED"
            or provenance["source_commit"] != self.admission.reviewed_commit
        ):
            raise ValueError(
                "workflow requires unchanged clean declarations at the selected reviewed commit"
            )
        return {
            **provenance,
            "allow_uncommitted_declaration": False,
            "workflow_context": self.identity,
            "closure": sorted(
                (
                    {"path": item["path"], "sha256": item["sha256"]}
                    for item in provenance["declarations"]
                ),
                key=lambda item: item["path"],
            ),
        }

    def authorize_lock(self, lock):
        self.refresh()
        loaded = locked_workflow_context(lock, self.loaded.source)
        if loaded != self.loaded:
            raise ValueError("WORKFLOW_CONTEXT_CHANGED: lock context differs")

    def engine(self, **kwargs):
        from sparselab.campaign.engine import CampaignEngine

        kwargs.setdefault("cold_verify", True)
        return CampaignEngine(
            self.campaign_source, self.work_dir, workflow_context=self, **kwargs
        )
