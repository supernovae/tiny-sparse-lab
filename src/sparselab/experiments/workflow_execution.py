"""Explicit admission of one inventoried prepared-input CPU Campaign.

The authored native Campaign owns connections; this context authenticates them.
A reviewed commit is a caller assertion, not author attestation or compute credit.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, StrictInt, StrictStr

from sparselab.campaign.plan import CampaignPlan
from sparselab.config.models import StrictModel
from sparselab.experiments.plan import recipe_run_config
from sparselab.experiments.preparation_execution import _environment
from sparselab.experiments.workflow_recipe import (
    _location,
    load_workflow,
    locked_workflow_context,
)
from sparselab.recovery.provenance import git_provenance, repository_root


class WorkflowAdmission(StrictModel):
    workflow_admission_version: StrictInt = Field(ge=1, le=1)
    reviewed_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    recipe_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cpu_python: Path


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
        self.refresh()

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
