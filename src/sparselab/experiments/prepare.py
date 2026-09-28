"""Explicit, offline corpus variant preparation; never launches training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.config.models import RunConfig
from sparselab.corpus.acquisition import acquire, verify_acquisition
from sparselab.corpus.export import export_release, verify_release_export
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import (
    ReleaseDeclaration,
    load_project,
    verify_fraction_tokenizer,
)
from sparselab.corpus.release import freeze, verify_release
from sparselab.data.packing import load_prepared_data, prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.plan import Artifact, CorpusVariant, ExperimentPlan
from sparselab.training.manifest import config_sha256, sha256_file
from sparselab.workdir import ensure_work_dir
from sparselab.workspace_preflight import require_storage, training_storage_checks


def verified_reuse_tokenizer(
    variant: CorpusVariant,
    artifacts: dict[str, Artifact],
    source: Path,
    release: ReleaseDeclaration,
) -> dict[str, object]:
    """Bind a reused tokenizer and any fraction selector to verified artifacts."""
    name = variant.tokenizer_artifact
    if name is None:
        raise ValueError("variant does not reuse a tokenizer")
    spec = artifacts[name]
    if spec.kind != "tokenizer":
        raise ValueError(f"{name} is not a tokenizer artifact")
    identity = verify_artifact(spec, source)
    fraction = release.fraction
    if variant.fraction_tokenizer is not None and fraction is None:
        raise ValueError("fraction tokenizer override requires a fractional release")
    if fraction is not None:
        selector = identity
        if variant.fraction_tokenizer is not None:
            selector_spec = artifacts[variant.fraction_tokenizer]
            if selector_spec.kind != "tokenizer":
                raise ValueError("fraction selector is not a tokenizer artifact")
            selector = verify_artifact(selector_spec, source)
        if fraction.tokenizer_sha256.lower() != selector["sha256"]:
            raise ValueError("fraction tokenizer digest differs from verified selector")
    return identity


def prepare_variant(
    variant: CorpusVariant,
    source: Path,
    base_run_path: Path,
    workspace: Path,
    *,
    artifacts: dict[str, Artifact] | None = None,
) -> dict[str, Any]:
    """Reuse one project's pinned acquisition and independently freeze its release."""
    if not base_run_path.is_file():
        raise ValueError(f"corpus variant {variant.id} needs a file-backed base_run")
    project_path = Path(variant.project)
    if not project_path.is_absolute():
        project_path = source.parent / project_path
    project = load_project(project_path)
    values = project.release.model_dump(mode="python")
    values.update(variant.release_set)
    release = ReleaseDeclaration.model_validate(values)
    reused = (
        verified_reuse_tokenizer(variant, artifacts or {}, source, release)
        if variant.tokenizer_artifact is not None
        else None
    )
    candidate = project.model_copy(update={"release": release})
    verify_fraction_tokenizer(candidate)
    ensure_work_dir(workspace)
    acquisition_path = workspace / "corpora" / candidate.config.id / "acquisition.json"
    if acquisition_path.exists():
        acquisition = verify_acquisition(candidate, workspace)
    else:
        if any(
            item.kind not in {"local", "deterministic_generator"}
            for item in candidate.sources
            if item.redistribution != "rejected"
        ):
            raise ValueError(
                f"corpus variant {variant.id} needs pre-acquired pinned external sources"
            )
        acquisition = acquire(candidate, workspace)
    built = build(candidate, workspace, offline=True)
    frozen = freeze(built, workspace)
    verified = verify_release(frozen)
    exported = export_release(
        frozen,
        variant.view,
        base_run_path,
        (
            variant.vocab_size
            if reused is None
            else load_tokenizer(Path(reused["path"])).get_vocab_size()
        ),
        workspace,
    )
    config = load_config(exported / "run.yaml")
    export_record = verify_release_export(config.dataset)
    if reused is None:
        tokenizer = train_tokenizer(load_tokenizer_config(exported / "tokenizer.yaml"))
    else:
        tokenizer = Path(reused["path"])
        config = RunConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "tokenizer": {
                    **config.tokenizer.model_dump(mode="python"),
                    "path": tokenizer,
                },
            }
        )
        if config.model.vocab_size != load_tokenizer(tokenizer).get_vocab_size():
            raise ValueError("reused tokenizer vocabulary differs from prepared model")
    prepared = prepare_data(config, load_tokenizer(tokenizer))
    load_prepared_data(prepared.root, byte_enabled=config.model.memory == "byte")
    return {
        "id": variant.id,
        "acquisition": acquisition,
        "release_id": verified["release_id"],
        "release_path": str(frozen),
        "export": export_record,
        "export_path": str(exported),
        "tokenizer_sha256": sha256_file(tokenizer),
        "tokenizer_path": str(tokenizer),
        "prepared_path": str(prepared.root),
        "prepared_manifest_sha256": sha256_file(prepared.root / "manifest.json"),
        "config_sha256": config_sha256(config.model_dump(mode="json")),
        "config": config.model_dump(mode="json"),
        **(
            {"tokenizer_artifact": variant.tokenizer_artifact}
            if reused is not None
            else {}
        ),
    }


def prepare_plan(plan: ExperimentPlan, source: Path, workspace: Path) -> dict[str, Any]:
    """Materialize declared variants only, after storage preflight and pinned sources."""
    ensure_work_dir(workspace)
    from sparselab.experiments.plan import base_run_config

    base = base_run_config(plan, source)
    require_storage(training_storage_checks(base))
    if not isinstance(plan.base_run, str):
        if plan.corpus_variants:
            raise ValueError("corpus variants need a file-backed base_run")
        return {"format": "experiment-preparation-v1", "id": plan.id, "variants": []}
    base_path = Path(plan.base_run)
    if not base_path.is_absolute():
        base_path = source.parent / base_path
    records = [
        prepare_variant(variant, source, base_path, workspace, artifacts=plan.artifacts)
        for variant in plan.corpus_variants
    ]
    return {"format": "experiment-preparation-v1", "id": plan.id, "variants": records}
