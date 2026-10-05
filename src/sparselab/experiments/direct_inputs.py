"""Bind already-prepared inputs to an authored experiment; never materialize data."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from sparselab.config.loading import load_config
from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.lock import _check_cell_inputs, _exclusive_bytes
from sparselab.experiments.plan import ExperimentPlan, load_plan
from sparselab.recovery.provenance import declaration_reference
from sparselab.training.manifest import sha256_file
from sparselab.verification_proofs import ProofStore, VerificationMode


def bind_direct_inputs(
    run_path: Path,
    template_path: Path,
    prepared_root: Path,
    output: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> ExperimentPlan:
    """Authenticate existing input bindings and exclusively publish a full plan."""
    run_path = run_path.absolute()
    template_path = template_path.absolute()
    output = output.absolute()
    prepared_root = prepared_root.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"experiment plan already exists: {output}")
    config = load_config(run_path)
    template = load_plan(template_path)
    if template.inputs or template.corpus_variants:
        raise ValueError(
            "direct phase template must not already bind inputs or variants"
        )
    reserved = {"tokenizer", "packed", "release", "export"}
    if reserved.intersection(template.artifacts):
        raise ValueError("phase template uses reserved direct-input artifact names")
    if template.evaluation_suite is not None and declaration_reference(
        template_path, template.evaluation_suite
    ) != declaration_reference(output, template.evaluation_suite):
        raise ValueError(
            "phase template evaluation suite changes location when exported"
        )
    if config.tokenizer.path.name != "tokenizer.json":
        raise ValueError("run tokenizer must reference tokenizer.json")
    if prepared_root.parent != config.dataset.cache_dir:
        raise ValueError(
            "prepared root must be directly beneath the run dataset cache_dir"
        )
    manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("prepared manifest must be an object")
    artifacts = {
        "tokenizer": {
            "kind": "tokenizer",
            "version": 1,
            "producer": "sparselab tokenizer train",
            "identifier": config.tokenizer.path.parent.name,
            "path": str(config.tokenizer.path),
            "sha256": sha256_file(config.tokenizer.path),
        },
        "packed": {
            "kind": "prepared_data",
            "version": 1,
            "producer": "sparselab data prepare",
            "identifier": manifest["settings_sha256"],
            "path": str(prepared_root),
            "sha256": manifest["manifest_sha256"],
        },
    }
    inputs = {"tokenizer": "tokenizer", "prepared_data": "packed"}
    if config.dataset.corpus_release_path is not None:
        release = config.dataset.corpus_release_path
        export = config.dataset.corpus_export_path
        if export is None:
            raise ValueError("Forge run requires release and export paths")
        artifacts["release"] = {
            "kind": "corpus_release",
            "version": 1,
            "producer": "sparselab corpus freeze",
            "identifier": release.name,
            "path": str(release),
            "sha256": release.name,
        }
        artifacts["export"] = {
            "kind": "corpus_export",
            "version": 1,
            "producer": "sparselab corpus export",
            "identifier": export.name,
            "path": str(export),
            "sha256": export.name,
        }
        inputs.update(corpus_release="release", corpus_export="export")
    for name, artifact in template.artifacts.items():
        payload = artifact.model_dump(mode="json")
        if artifact.path is not None and not Path(artifact.path).is_absolute():
            payload["path"] = str(template_path.parent / artifact.path)
        artifacts[name] = payload
    plan = ExperimentPlan.model_validate(
        {
            **template.model_dump(mode="json"),
            "base_run": config.model_dump(mode="json"),
            "artifacts": artifacts,
            "inputs": inputs,
        }
    )
    # Authenticate the recorded historical source domain as recorded, without
    # claiming that this checkout matches it. Locking separately enforces the
    # current package identity (or explicitly authorized source compatibility).
    identity = manifest.get("cache_identity")
    recorded_source = (
        identity.get("source_identity_sha256") if isinstance(identity, dict) else None
    )
    if not isinstance(recorded_source, str) or len(recorded_source) != 64:
        raise ValueError("prepared manifest lacks a recorded source identity")
    identities = {}
    paths = {}
    for name, reference in plan.inputs.items():
        verified = verify_artifact(
            plan.artifacts[reference],
            output,
            dataset=config.dataset,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        identities[name] = {
            key: value for key, value in verified.items() if key != "path"
        }
        paths[name] = str(verified["path"])
    _check_cell_inputs(config, identities, paths, recorded_source)
    for reference in template.artifacts.values():
        if reference.from_phase is None:
            verify_artifact(
                reference,
                template_path,
                dataset=config.dataset,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
    content = yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False).encode(
        "utf-8"
    )
    _exclusive_bytes(output, content)
    return plan
