"""Explicit local CPU admission for inventoried synthetic or snapshot preparation.

This binds operational context to native Campaign receipts, not scientific hashes.
A caller-selected reviewed commit is an assertion of review, not author attestation.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import Field, StrictInt, StrictStr

from sparselab.config.models import StrictModel, TokenizerTrainConfig
from sparselab.config.recipe_paths import _paths
from sparselab.experiments.preparation_recipe import load_preparation_recipe
from sparselab.experiments.workflow_recipe import _location
from sparselab.recovery.provenance import git_provenance, repository_root
from sparselab.training.manifest import source_identity


class PreparationAdmission(StrictModel):
    preparation_admission_version: StrictInt = Field(ge=1, le=1)
    reviewed_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    recipe_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cpu_python: Path
    max_documents: StrictInt = Field(gt=0)
    max_tokens: StrictInt = Field(gt=0)
    max_vocab_size: StrictInt = Field(ge=260)
    min_free_bytes: StrictInt = Field(gt=0)
    min_free_inodes: StrictInt = Field(gt=0)


class SnapshotPreparationAdmission(PreparationAdmission):
    """Explicit offline snapshot consumption; v1 serialization stays unchanged."""

    preparation_admission_version: StrictInt = Field(ge=2, le=2)
    max_snapshot_bytes: StrictInt = Field(gt=0)
    max_snapshot_records: StrictInt = Field(gt=0)


def preparation_admission(document: dict) -> PreparationAdmission:
    model = (
        SnapshotPreparationAdmission
        if document.get("preparation_admission_version") == 2
        else PreparationAdmission
    )
    return model.model_validate(document)


def _environment() -> dict:
    # Metadata only: do not import tokenizers, numpy, torch or probe devices.
    try:
        packages = {
            name: version(name)
            for name in (
                "tokenizers",
                "numpy",
                "pydantic",
                "PyYAML",
                "datasets",
                "huggingface-hub",
                "pyarrow",
                "psutil",
                "torch",
            )
        }
    except PackageNotFoundError as error:
        raise ValueError(
            "local CPU preparation dependencies are unavailable"
        ) from error
    return {
        "kind": "local-cpu-preparation-v1",
        "python": str(Path(sys.executable).absolute()),
        "python_version": sys.version,
        "prefix": sys.prefix,
        "packages": packages,
        "installed_lab_sha256": source_identity()["sha256"],
    }


class PreparationExecution:
    """One context used by preflight, native adapters and receipt recovery."""

    def __init__(
        self,
        source: Path,
        bindings: Path,
        work_dir: Path,
        admission: PreparationAdmission,
    ):
        self.source = Path(source).absolute()
        self.bindings = Path(bindings).absolute()
        if not Path(work_dir).is_absolute():
            raise ValueError("preparation requires an explicit absolute work root")
        self.work_dir = _location(self.bindings, str(work_dir))
        self.admission = admission.model_copy(deep=True)
        self.loaded = load_preparation_recipe(self.source, self.bindings)
        self.environment = _environment()
        self.identity = {
            "format": "sparselab-preparation-execution-v1",
            "source": str(self.source),
            "bindings": str(self.bindings),
            "work_dir": str(self.work_dir),
            "admission": self.admission.model_dump(mode="json"),
            "environment": self.environment,
        }
        self.refresh()

    def refresh(self) -> dict:
        current = load_preparation_recipe(self.source, self.bindings)
        if (
            current != self.loaded
            or current.recipe_sha256 != self.admission.recipe_sha256
            or current.binding_sha256 != self.admission.binding_sha256
        ):
            raise ValueError("PREPARATION_CONTEXT_CHANGED: recipe or bindings differ")
        if str(self.admission.cpu_python.absolute()) != str(
            Path(sys.executable).absolute()
        ):
            raise ValueError("explicit CPU Python must select the current interpreter")
        if _environment() != self.environment:
            raise ValueError(
                "PREPARATION_CONTEXT_CHANGED: installed environment differs"
            )
        if repository_root(self.work_dir) is not None:
            raise ValueError("preparation work root must be outside Git")
        # Recheck lexical safety on every boundary, including newly created ancestors.
        _location(self.bindings, str(self.work_dir))
        outputs = set()
        tokenizer_outputs = set()
        caches = set()
        for config in current.configs.values():
            data = config.dataset
            caches.add(data.cache_dir)
            if isinstance(config, TokenizerTrainConfig):
                tokenizer_outputs.add(config.output_dir)
            if data.source not in {"synthetic", "snapshot"}:
                raise ValueError(
                    "preparation execution supports synthetic or admitted snapshot inputs only"
                )
            if (
                data.train_max_documents + data.validation_max_documents
                > self.admission.max_documents
                or data.train_max_tokens + data.validation_max_tokens
                > self.admission.max_tokens
                or not isinstance(config, TokenizerTrainConfig)
                and config.model.vocab_size > self.admission.max_vocab_size
                or isinstance(config, TokenizerTrainConfig)
                and (
                    config.max_documents > self.admission.max_documents
                    or config.vocab_size > self.admission.max_vocab_size
                )
            ):
                raise ValueError(
                    "preparation exceeds explicit document/token/vocabulary limits"
                )
            for field, path in _paths(config).items():
                if field in {"output_dir", "dataset.cache_dir", "logging.root_dir"}:
                    if path == self.work_dir or not path.is_relative_to(self.work_dir):
                        raise ValueError(
                            "preparation outputs must be children of explicit work root"
                        )
                    if path.relative_to(self.work_dir).parts[0] in {
                        "campaigns",
                        "scratch",
                        "execution",
                        "ownership",
                    }:
                        raise ValueError(
                            "preparation output overlaps native operational storage"
                        )
                    if field != "logging.root_dir":
                        outputs.add(path)
                elif field != "tokenizer.path" and not (
                    data.source == "snapshot"
                    and field
                    in {
                        "dataset.source_manifest_path",
                        "dataset.train_path",
                        "dataset.validation_path",
                    }
                ):
                    raise ValueError(f"preparation input path unavailable: {field}")
        if tokenizer_outputs & caches:
            raise ValueError("preparation output directories must not overlap")
        for left in outputs:
            for right in outputs:
                if left != right and left.is_relative_to(right):
                    raise ValueError("preparation output directories must not overlap")
        for stage in current.plan.stages:
            if (
                stage.kind == "artifact_reference"
                and stage.artifact.kind == "dataset_snapshot"
            ):
                snapshot_root = current.artifact_locations[stage.id].parent
                for output in outputs | {
                    config.logging.root_dir
                    for config in current.configs.values()
                    if not isinstance(config, TokenizerTrainConfig)
                }:
                    if output.is_relative_to(
                        snapshot_root
                    ) or snapshot_root.is_relative_to(output):
                        raise ValueError(
                            "preparation outputs must not overlap snapshot inputs"
                        )
        paths = (
            self.source,
            *(current.context.root / name for name in current.context.documents),
        )
        provenance = git_provenance(paths)
        if (
            provenance["status"] != "CLEAN_AND_COMMITTED"
            or provenance["source_commit"] != self.admission.reviewed_commit
        ):
            raise ValueError(
                "preparation requires unchanged clean declarations at the selected reviewed commit"
            )
        ancestor = self.work_dir
        while not ancestor.exists():
            ancestor = ancestor.parent
        stats = os.statvfs(ancestor)
        if (
            stats.f_bavail * stats.f_frsize < self.admission.min_free_bytes
            or stats.f_favail < self.admission.min_free_inodes
            or not os.access(ancestor, os.W_OK | os.X_OK)
        ):
            raise ValueError("preparation storage/permission admission failed")
        self._verify_snapshots(current)
        return {
            **provenance,
            "allow_uncommitted_declaration": False,
            "preparation_context": self.identity,
            "closure": sorted(
                (
                    {"path": item["path"], "sha256": item["sha256"]}
                    for item in provenance["declarations"]
                ),
                key=lambda item: item["path"],
            ),
        }

    def _verify_snapshots(self, loaded):
        references = [
            stage
            for stage in loaded.plan.stages
            if stage.kind == "artifact_reference"
            and stage.artifact.kind == "dataset_snapshot"
        ]
        if not references:
            return
        if not isinstance(self.admission, SnapshotPreparationAdmission):
            raise TypeError("dataset snapshots require preparation admission version 2")
        from sparselab.data.sources import verify_snapshot
        from sparselab.experiments.artifacts import verify_artifact

        total_bytes = 0
        total_records = 0
        for stage in references:
            path = loaded.artifact_locations[stage.id]
            _location(self.bindings, str(path))
            if path.name != "manifest.json" or not path.is_file():
                raise ValueError("missing dataset snapshot manifest.json")
            # Bound the complete local flat snapshot before parsing or cold verification.
            for member in path.parent.iterdir():
                info = member.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(
                        "snapshot inputs must be regular files without symlinks"
                    )
                total_bytes += info.st_size
                if total_bytes > self.admission.max_snapshot_bytes:
                    raise ValueError("snapshot exceeds admitted byte limit")
            raw = json.loads(path.read_text())
            if raw.get("format") != "sparselab-dataset-snapshot-v1":
                raise ValueError("snapshot requires a native source-lock manifest")
            # Declared per-split scan ceilings bound verifier work, not just retained rows.
            source = raw["lock"]["source"]
            records = source["resources"]["max_source_records"]
            if type(records) is not int or records <= 0:
                raise ValueError("invalid snapshot source record bound")
            total_records += records * len(source["splits"])
            if total_records > self.admission.max_snapshot_records:
                raise ValueError("snapshot exceeds admitted source record limit")
            verify_artifact(self.artifact(stage), self.source, verification_mode="cold")
            for operation in loaded.compiled.operations:
                if any(
                    item.port == "snapshot" and item.producer == stage.id
                    for item in operation.inputs
                ):
                    verify_snapshot(
                        loaded.configs[operation.stage_id].dataset,
                        verification_mode="cold",
                    )

    def before_dispatch(self, stage, row):
        self.refresh()
        if row.get("attempts"):
            raise ValueError(
                "preparation has an uncommitted attempt; automatic rerun is unavailable"
            )
        if stage.id in self.loaded.configs:
            config = self.loaded.configs[stage.id]
            output = (
                config.output_dir
                if isinstance(config, TokenizerTrainConfig)
                else config.dataset.cache_dir
            )
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError(
                    "preparation output already exists without a committed receipt"
                )

    @contextmanager
    def dispatch_scope(self):
        # Native preparation publishes operational receipts through ensure_work_dir.
        # Scope its process settings to this explicit invocation and restore them.
        from sparselab.workdir import ensure_work_dir

        names = ("SPARSELAB_WORK_DIR", "TMPDIR", "TEMP", "TMP")
        prior = {name: os.environ.get(name) for name in names}
        temporary = tempfile.tempdir
        try:
            ensure_work_dir(self.work_dir)
            yield
        finally:
            tempfile.tempdir = temporary
            for name, value in prior.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def artifact(self, stage):
        return stage.artifact.model_copy(
            update={"path": str(self.loaded.artifact_locations[stage.id])}
        )

    def engine(self, **kwargs):
        from sparselab.campaign.engine import CampaignEngine

        kwargs.setdefault("cold_verify", True)
        return CampaignEngine(
            self.loaded.context.root / self.loaded.descriptor.entrypoint,
            self.work_dir,
            preparation_context=self,
            **kwargs,
        )
