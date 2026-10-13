"""Inventory-backed preparation planning; no acquisition, admission or executor."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from sparselab.campaign.plan import CampaignPlan
from sparselab.campaign.preparation_plan import CompiledPreparation, compile_preparation
from sparselab.config.models import RunConfig, StrictModel, TokenizerTrainConfig
from sparselab.config.recipe_paths import _INPUT_FIELDS, _OUTPUT_FIELDS, _paths
from sparselab.declaration_context import RecipeContext, canonical_reference
from sparselab.documents import parse_document
from sparselab.experiments.recipe import (
    MAX_DOCUMENTS,
    _metadata_bytes,
    snapshot_inventory,
)
from sparselab.experiments.workflow_recipe import _location
from sparselab.training.manifest import canonical_json, source_identity


class PreparationFile(StrictModel):
    path: StrictStr
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["campaign", "run_config", "tokenizer_config", "metadata"]

    @field_validator("path")
    @classmethod
    def valid_path(cls, value):
        return canonical_reference(value)


class PreparationRecipe(StrictModel):
    portable_preparation_version: StrictInt = Field(ge=1, le=1)
    id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    entrypoint: StrictStr
    contracts: dict[StrictStr, StrictInt]
    files: tuple[PreparationFile, ...] = Field(min_length=1, max_length=MAX_DOCUMENTS)

    @model_validator(mode="after")
    def valid_inventory(self):
        canonical_reference(self.entrypoint)
        names = [item.path for item in self.files]
        if len(names) != len(set(names)):
            raise ValueError("duplicate preparation file path")
        if [m.path for m in self.files if m.role == "campaign"] != [self.entrypoint]:
            raise ValueError("entrypoint must name the sole declared campaign")
        if self.contracts != {"campaign": 1, "run_config": 2, "tokenizer_config": 1}:
            raise ValueError("unsupported preparation contracts")
        return self


class PreparationBindings(StrictModel):
    preparation_bindings_version: StrictInt = Field(ge=1, le=1)
    artifacts: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    config_paths: dict[StrictStr, dict[StrictStr, StrictStr]]


@dataclass(frozen=True)
class LoadedPreparation:
    descriptor: PreparationRecipe
    context: RecipeContext
    plan: CampaignPlan
    compiled: CompiledPreparation
    configs: Mapping[str, RunConfig | TokenizerTrainConfig]
    artifact_locations: Mapping[str, Path]
    recipe_sha256: str
    binding_sha256: str


def _config(
    context: RecipeContext,
    source: Path,
    reference: str,
    role: str,
    roles: Mapping[str, str],
) -> RunConfig | TokenizerTrainConfig:
    target = context.reference(source, reference)
    if roles[reference] != role:
        raise ValueError(f"preparation config must reference a {role} member")
    model = RunConfig if role == "run_config" else TokenizerTrainConfig
    document = context.read_document(target)
    config = model.model_validate(document)
    paths = _paths(config)
    bindings = context.config_paths[reference]
    if set(bindings) != set(paths):
        raise ValueError("config_paths must bind exactly the declared native paths")
    allowed = _INPUT_FIELDS | _OUTPUT_FIELDS | {"output_dir"}
    raw = config.model_dump(mode="python")
    for field, value in paths.items():
        if field not in allowed:
            raise ValueError(f"unsupported preparation path field: {field}")
        parts = field.split(".")
        authored = document
        for part in parts:
            authored = authored.get(part) if isinstance(authored, dict) else None
        canonical_reference(authored if authored is not None else value.as_posix())
        container = raw
        for part in parts[:-1]:
            container = container[part]
        container[parts[-1]] = bindings[field]
    return model.model_validate(raw)


def load_preparation_recipe(path: Path, bindings: Path) -> LoadedPreparation:
    path, bindings = Path(path).absolute(), Path(bindings).absolute()
    raw = _metadata_bytes(path)
    descriptor = PreparationRecipe.model_validate(
        parse_document(raw.decode(), source=path)
    )
    context = snapshot_inventory(path, raw, descriptor.files)
    if bindings == path or bindings in [
        context.root / name for name in context.documents
    ]:
        raise ValueError(
            "local bindings must be separate from inventoried declarations"
        )
    local_bytes = _metadata_bytes(bindings)
    local = PreparationBindings.model_validate(
        parse_document(local_bytes.decode(), source=bindings)
    )
    source = context.reference(
        context.root / descriptor.entrypoint, descriptor.entrypoint
    )
    authored = context.read_document(source)
    # Refuse unsupported stages before their native validators can import computation.
    if authored.get("recovery") is not None:
        raise ValueError("preparation recovery context is unavailable")
    stages = authored.get("stages")
    if not isinstance(stages, list):
        raise TypeError("preparation requires a native stages list")
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("kind") not in {
            "tokenizer_train",
            "data_prepare",
            "artifact_reference",
        }:
            raise ValueError("unsupported preparation operation")
        if stage.get("snapshot") is not None:
            raise ValueError(
                "preparation snapshot context is unavailable; acquisition is separate"
            )
        if stage["kind"] == "artifact_reference":
            artifact = stage.get("artifact")
            if not isinstance(artifact, dict) or artifact.get("kind") != "tokenizer":
                raise ValueError(
                    "preparation supports only external tokenizer references"
                )
            canonical_reference(artifact.get("path", ""))
    plan = CampaignPlan.model_validate(authored)
    compiled = compile_preparation(plan)
    if not compiled.operations:
        raise ValueError("preparation requires at least one preparation operation")
    references = {
        stage.id: stage for stage in plan.stages if stage.kind == "artifact_reference"
    }
    if set(local.artifacts) != set(references):
        raise ValueError(
            "artifact bindings must name exactly the external producer stages"
        )
    locations = {
        name: _location(bindings, value) for name, value in local.artifacts.items()
    }
    if set(local.config_paths) != {op.config_reference for op in compiled.operations}:
        raise ValueError(
            "config bindings must name exactly the compiled config references"
        )
    context = RecipeContext(
        context.root,
        context.documents,
        {
            name: {field: _location(bindings, value) for field, value in fields.items()}
            for name, fields in local.config_paths.items()
        },
    )
    roles = {member.path: member.role for member in descriptor.files}
    configs = {}
    for op in compiled.operations:
        config = _config(
            context,
            source,
            op.config_reference,
            "run_config" if op.kind == "data_prepare" else "tokenizer_config",
            roles,
        )
        configs[op.stage_id] = config
    for op in compiled.operations:
        if op.kind != "data_prepare":
            continue
        producer = next(item.producer for item in op.inputs if item.port == "tokenizer")
        expected = (
            locations[producer]
            if producer in locations
            else configs[producer].output_dir / "tokenizer.json"
        )
        if configs[op.stage_id].tokenizer.path != expected:
            raise ValueError(
                "configured tokenizer differs from compiled producer location"
            )
    canonical = descriptor.model_dump(mode="json")
    canonical["files"] = sorted(canonical["files"], key=lambda member: member["path"])
    return LoadedPreparation(
        descriptor,
        context,
        plan,
        compiled,
        MappingProxyType(configs),
        MappingProxyType(locations),
        hashlib.sha256(
            b"sparselab-portable-preparation-v1\0" + canonical_json(canonical)
        ).hexdigest(),
        hashlib.sha256(local_bytes).hexdigest(),
    )


def inspect_preparation_recipe(path: Path, bindings: Path) -> dict:
    loaded = load_preparation_recipe(path, bindings)
    return {
        "format": "sparselab-preparation-inspection-v1",
        "id": loaded.descriptor.id,
        "recipe_sha256": loaded.recipe_sha256,
        "binding_sha256": loaded.binding_sha256,
        "installed_lab": source_identity(),
        "declaration_integrity": "verified_inventory",
        "committed_provenance": "unverified",
        "runtime_capability": "unknown",
        "executable": False,
        "missing_gates": [{"code": "WORKFLOW_EXECUTION_UNAVAILABLE"}],
        "plan": loaded.compiled.model_dump(mode="json"),
        "configs": {
            op.stage_id: {
                "reference": op.config_reference,
                "sha256": hashlib.sha256(
                    loaded.context.documents[op.config_reference]
                ).hexdigest(),
                "bound_paths": {
                    name: str(value)
                    for name, value in _paths(loaded.configs[op.stage_id]).items()
                },
            }
            for op in loaded.compiled.operations
        },
        "artifacts": {
            stage.id: {
                "identity": stage.artifact.model_dump(mode="json", exclude={"path"}),
                "location": str(loaded.artifact_locations[stage.id]),
                "status": "unverified",
            }
            for stage in loaded.plan.stages
            if stage.kind == "artifact_reference"
        },
    }
