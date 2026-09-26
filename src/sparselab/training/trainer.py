# Malformed persisted portability identity is reported as ValueError.
# ruff: noqa: TRY004
"""Engine-neutral single-host training lifecycle with explicit update boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import signal
import socket
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sparselab.config.models import RunConfig
from sparselab.data.allocation import (
    copy_allocation_bundle,
    load_allocation_manifest,
)
from sparselab.data.packing import (
    BatchCursor,
    PreparedData,
    TokenBlockDataset,
    epoch_order,
    load_prepared_data,
    prepare_data,
)
from sparselab.data.tokenizer import load_tokenizer
from sparselab.engines.base import EngineState, Microbatch
from sparselab.engines.pytorch import PyTorchEngine
from sparselab.memory import (
    calibrated_estimate,
    calibration_key,
    parameter_inventory,
    validate_offload_headroom,
)
from sparselab.runtime import seed_everything
from sparselab.training.checkpoints import (
    CheckpointManager,
    CheckpointRecord,
    LineageBest,
    TrainingSnapshot,
    _atomic_json,
    choose_lineage_best,
    existing_run_recovery_report,
)
from sparselab.training.continuation import load_continuation
from sparselab.training.manifest import (
    ArtifactIdentity,
    RunManifest,
    _is_sha256,
    architecture_sha256,
    canonical_json,
    config_sha256,
    sha256_file,
    source_identity,
    write_manifest,
)
from sparselab.training.metrics import ExperimentStore
from sparselab.training.stages import ExperimentStage, StageHistory


class _WallTimeExpired(Exception):
    """Internal signal that discards an incomplete cooperative evaluation."""


def _load_run_data(run: Path, config: RunConfig) -> PreparedData:
    return load_prepared_data(
        run / "data", byte_enabled=config.model.memory in {"byte", "portable"}
    )


def _stack_optional(
    records: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]
    ],
    field: int,
) -> np.ndarray | None:
    values = [record[field] for record in records]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("microbatch sidecars must be present consistently")
    return np.stack([value for value in values if value is not None])


def _portability_manifest_v2(path: Path | None) -> dict[str, object] | None:
    if path is None or path.is_symlink() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(payload, dict)
        and payload.get("format") == "sparselab-portability-run"
        and type(payload.get("version")) is int
        and payload["version"] == 2
    ):
        return payload
    return None


def _bind_learned_inputs(config: RunConfig, manifest_path: Path) -> RunConfig:
    payload = _portability_manifest_v2(manifest_path)
    if payload is None:
        raise ValueError("learned portability run manifest is not v2")
    memory = payload.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("learned portability manifest lacks memory identity")
    artifact = memory.get("artifact")
    package_path: Path | None = None
    if artifact is not None:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError("learned portable memory descriptor is invalid")
        package_path = manifest_path.parent / artifact["path"]
    role = payload["coordinate"]["role"]
    condition = payload["coordinate"]["condition"]
    default_train = {
        "source": "source_train.jsonl",
        "preparation": "preparation_train.jsonl",
        "recipient": "adapter_calibration.jsonl",
        "native": "source_train.jsonl",
        "calibration": (
            "preparation_train.jsonl"
            if condition.startswith("preparation-")
            else "adapter_calibration.jsonl"
            if condition.startswith("adapter-")
            else "source_train.jsonl"
        ),
    }[role]
    train_name = (
        config.dataset.train_path.name
        if config.dataset.train_path is not None
        else default_train
    )
    default_validation = "preparation_validation.jsonl"
    validation_name = (
        config.dataset.validation_path.name
        if config.dataset.validation_path is not None
        else default_validation
    )
    data_root = manifest_path.parent / "portability" / "data"
    return config.model_copy(
        update={
            "tokenizer": config.tokenizer.model_copy(
                update={"path": manifest_path.parent / "tokenizer.json"}
            ),
            "model": config.model.model_copy(
                update={"memory_package_path": package_path}
            ),
            "dataset": config.dataset.model_copy(
                update={
                    "train_path": data_root / train_name,
                    "validation_path": data_root / validation_name,
                }
            ),
            "training": config.training.model_copy(
                update={"portability_manifest_path": manifest_path}
            ),
        }
    )


def _copy_artifacts(
    run: Path, config: RunConfig, data: PreparedData, source_run: Path | None = None
) -> tuple[ArtifactIdentity, ...]:
    artifacts: list[ArtifactIdentity] = []
    tokenizer_source = (
        source_run / "tokenizer.json"
        if source_run is not None
        else config.tokenizer.path
    )
    tokenizer = run / "tokenizer.json"
    shutil.copy2(tokenizer_source, tokenizer)
    if source_run is None:
        for source in (config.tokenizer.path.with_name("tokenizer_manifest.json"),):
            if source.is_file():
                shutil.copy2(source, run / source.name)
    elif (source_run / "tokenizer_manifest.json").is_file():
        shutil.copy2(
            source_run / "tokenizer_manifest.json", run / "tokenizer_manifest.json"
        )
    shutil.copytree(data.root, run / "data")
    manifest_source = (
        source_run / "portability_manifest.json"
        if source_run is not None
        and (source_run / "portability_manifest.json").is_file()
        else config.training.portability_manifest_path
    )
    learned_manifest = _portability_manifest_v2(manifest_source)
    if learned_manifest is not None:
        assert manifest_source is not None
        bundle_source = manifest_source.parent / "portability"
        if bundle_source.is_symlink() or not bundle_source.is_dir():
            raise ValueError("learned portability owned bundle is missing")
        shutil.copytree(bundle_source, run / "portability")
        shutil.copy2(manifest_source, run / "portability_manifest.json")
    package_source = (
        source_run / "portable_package"
        if source_run is not None
        else config.model.memory_package_path
    )
    if (
        learned_manifest is None
        and package_source is not None
        and package_source.exists()
    ):
        destination = run / "portable_package"
        if package_source.is_dir():
            shutil.copytree(package_source, destination)
        else:
            shutil.copy2(package_source, destination)
    allocation_source = config.dataset.allocation_manifest_path
    if allocation_source is not None:
        cache_identity = data.manifest.get("cache_identity")
        allocation_metadata = data.manifest.get("allocation")
        if not isinstance(cache_identity, dict) or not isinstance(
            allocation_metadata, dict
        ):
            raise ValueError("prepared data lacks allocation identity")
        source_identity_sha256 = cache_identity.get("source_identity_sha256")
        tokenizer_sha256 = cache_identity.get("tokenizer_sha256")
        if not isinstance(source_identity_sha256, str) or not isinstance(
            tokenizer_sha256, str
        ):
            raise ValueError("prepared data allocation identity is malformed")
        if source_run is not None:
            allocation_source = source_run / "allocation" / allocation_source.name
        allocation = load_allocation_manifest(
            allocation_source,
            source_identity_sha256=source_identity_sha256,
            tokenizer_sha256=tokenizer_sha256,
        )
        if allocation.sha256 != allocation_metadata.get("manifest_sha256"):
            raise ValueError("prepared data and allocation manifests differ")
        copy_allocation_bundle(allocation, run / "allocation")
    for path in sorted(
        item
        for item in run.rglob("*")
        if item.is_file()
        and item != run / "manifest.json"
        and not item.is_relative_to(run / "checkpoints")
    ):
        artifacts.append(
            ArtifactIdentity(
                str(path.relative_to(run)), sha256_file(path), path.stat().st_size
            )
        )
    return tuple(artifacts)


def _checkpoint_due(
    config: RunConfig,
    step: int,
    tokens: int,
    elapsed: float,
    watermarks: dict[str, float],
) -> bool:
    checks = (
        (config.checkpoint.every_steps, step - watermarks.get("step", 0)),
        (config.checkpoint.every_tokens, tokens - watermarks.get("tokens", 0)),
        (
            config.checkpoint.every_minutes,
            (elapsed - watermarks.get("minutes", 0)) / 60,
        ),
    )
    return step in config.checkpoint.steps or any(
        interval is not None and value >= interval for interval, value in checks
    )


def _save(
    manager: CheckpointManager,
    engine: object,
    config: RunConfig,
    run_id: str,
    cursor: BatchCursor,
    step: int,
    tokens: int,
    watermarks: dict[str, float],
    validation_loss: float | None = None,
    *,
    source_digest: str,
    parent_digest: str | None,
    wall_seconds: float,
    update_seconds: float,
    lineage_best: LineageBest | None = None,
) -> CheckpointRecord:
    state = engine.export_training_state()
    config_payload = config.model_dump(mode="json")
    snapshot = TrainingSnapshot(
        model={},
        weight_source=engine.export_weights(),
        optimizer=state.optimizer,
        schedule={
            "kind": "warmup_cosine_v1",
            "completed_updates": step,
            "max_steps": config.training.max_steps,
            "warmup_steps": config.optimizer.warmup_steps,
            "peak": config.optimizer.peak,
            "floor": config.optimizer.floor,
        },
        step=step,
        tokens_seen=tokens,
        cursor=(cursor.epoch, cursor.next_block),
        config=config_payload,
        run_id=run_id,
        rng=state.rng,
        scaler=state.scaler,
        validation_loss=validation_loss,
        cadence=watermarks.copy(),
        engine=config.runtime.engine,
        backend=config.runtime.backend,
        optimizer_parameter_names=state.optimizer_parameter_names,
        config_sha256=config_sha256(config_payload),
        architecture_sha256=architecture_sha256(config_payload),
        source_identity_sha256=source_digest,
        manifest_sha256=manager.manifest_sha256,
        parent_checkpoint_sha256=parent_digest,
        cumulative_wall_seconds=wall_seconds,
        cumulative_update_seconds=update_seconds,
        lineage_best=lineage_best,
    )
    return manager.save(snapshot, validation_loss)


def _write_validation_report(
    run: Path,
    record: CheckpointRecord,
    result: dict[str, object],
    config: RunConfig,
    source_digest: str,
) -> Path:
    evaluations = run / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    report = (
        evaluations
        / f"validation_step_{record.step:08d}_gen_{record.generation_id:06d}.json"
    )
    identities = {
        "data/validation.npy": sha256_file(run / "data" / "validation.npy"),
        "tokenizer.json": sha256_file(run / "tokenizer.json"),
        "source_identity_sha256": source_digest,
        "protocol": "next-token-cross-entropy-v1",
    }
    for name in ("validation_supervision.npy", "validation_byte_addresses.npy"):
        artifact = run / "data" / name
        if artifact.is_file():
            identities[f"data/{name}"] = sha256_file(artifact)
    payload: dict[str, object] = {
        "kind": "held_out_validation_v2",
        "checkpoint": record.relative_path,
        "checkpoint_sha256": record.manifest_sha256,
        "step": record.step,
        "tokens_seen": record.tokens_seen,
        "identities": identities,
        "batch_size": config.training.micro_batch_size,
        "max_batches": config.evaluation.max_batches,
        "seq_len": config.training.seq_len,
        **result,
    }
    payload["sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    content = canonical_json(payload) + b"\n"
    if report.exists():
        if report.read_bytes() != content:
            raise FileExistsError(
                f"immutable validation report already exists: {report}"
            )
        return report
    temporary = report.with_name(report.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        __import__("os").fsync(handle.fileno())
    temporary.replace(report)
    return report


def train(
    config: RunConfig,
    *,
    resume: Path | None = None,
    promote: Path | None = None,
    recover: Path | None = None,
    run_id: str | None = None,
    stop_after_step: int | None = None,
    worker_id: str | None = None,
    allow_runtime_drift: bool = False,
    max_wall_seconds: float | None = None,
    cancel_path: Path | None = None,
    experiment_id: str | None = None,
    attempt_id: str | None = None,
    dispatch_metadata: dict[str, object] | None = None,
    stage_bundle: Path | None = None,
    requested_config_override: dict[str, object] | None = None,
) -> str:
    """Run one independent experiment, optionally bound to a stage bundle."""
    if max_wall_seconds is not None and (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(max_wall_seconds)
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be positive and finite")
    return _train_impl(
        config,
        resume=resume,
        promote=promote,
        recover=recover,
        run_id=run_id,
        stop_after_step=stop_after_step,
        worker_id=worker_id,
        allow_runtime_drift=allow_runtime_drift,
        stage_bundle=stage_bundle,
        cancel_path=cancel_path,
        purpose="training",
        experiment_id=experiment_id,
        attempt_id=attempt_id,
        dispatch_metadata=dispatch_metadata,
        requested_config_override=requested_config_override,
        max_wall_seconds=max_wall_seconds,
    )


def _train_impl(
    config: RunConfig,
    *,
    resume: Path | None = None,
    promote: Path | None = None,
    recover: Path | None = None,
    run_id: str | None = None,
    stop_after_step: int | None = None,
    worker_id: str | None = None,
    allow_runtime_drift: bool = False,
    stage_bundle: Path | None = None,
    cancel_path: Path | None = None,
    purpose: str = "training",
    experiment_id: str | None = None,
    attempt_id: str | None = None,
    dispatch_metadata: dict[str, object] | None = None,
    requested_config_override: dict[str, object] | None = None,
    max_wall_seconds: float | None = None,
) -> str:

    def wall_expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    if max_wall_seconds is not None and (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(max_wall_seconds)
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be positive and finite")
    deadline = None if max_wall_seconds is None else time.monotonic() + max_wall_seconds
    with ExitStack() as resources:
        if (experiment_id is None) != (attempt_id is None):
            raise ValueError("experiment_id and attempt_id must be supplied together")
        for identity in (experiment_id, attempt_id):
            if identity is not None and (
                not isinstance(identity, str)
                or not identity
                or identity in {".", ".."}
                or Path(identity).name != identity
            ):
                raise ValueError("execution identities must be nonempty identifiers")
        dispatch_decisions: tuple[dict[str, object], ...] = ()
        if dispatch_metadata is not None:
            if experiment_id is None or not isinstance(dispatch_metadata, dict):
                raise ValueError(
                    "dispatch metadata requires experiment and attempt identities"
                )
            if not {"spec_digest", "bundle_digest"} <= dispatch_metadata.keys() or (
                dispatch_metadata.keys() - {"spec_digest", "bundle_digest", "matrix"}
            ):
                raise ValueError("unsupported dispatch metadata fields")
            if not all(
                _is_sha256(dispatch_metadata[key])
                for key in ("spec_digest", "bundle_digest")
            ):
                raise ValueError("dispatch identities must be SHA-256 digests")
            matrix = dispatch_metadata.get("matrix")
            if matrix is not None and (
                not isinstance(matrix, dict)
                or set(matrix) != {"matrix_sha256", "coordinate"}
                or not _is_sha256(matrix["matrix_sha256"])
                or not isinstance(matrix["coordinate"], dict)
                or not matrix["coordinate"]
                or not all(
                    isinstance(key, str) and key and isinstance(value, str) and value
                    for key, value in matrix["coordinate"].items()
                )
            ):
                raise ValueError("invalid dispatch matrix identity")
            if len(canonical_json(dispatch_metadata)) > 16 * 1024:
                raise ValueError("dispatch metadata exceeds 16 KiB")
            dispatch_decisions = (
                {"kind": "worker_dispatch", "schema_version": 1, **dispatch_metadata},
            )
        requested_config = config.model_dump(mode="json")
        if requested_config_override is not None:
            original = RunConfig.model_validate(requested_config_override)
            comparable = original.model_dump(mode="json")
            if original.runtime.backend == "auto":
                comparable["runtime"]["backend"] = config.runtime.backend
            if (
                original.runtime.precision == "auto"
                and config.runtime.precision != "auto"
            ):
                comparable["runtime"]["precision"] = "fp32"
            if config_sha256(comparable) != config_sha256(requested_config):
                raise ValueError(
                    "requested execution override changes scientific configuration"
                )
            requested_config = original.model_dump(mode="json")
        choices = [path for path in (resume, promote, recover) if path is not None]
        if len(choices) > 1:
            raise ValueError("resume, promote, and recover are mutually exclusive")
        if purpose not in {"training", "smoke", "warmup"}:
            raise ValueError("invalid execution purpose")
        if purpose != "training" and (choices or stage_bundle is None):
            raise ValueError(
                "pilots require immutable staged inputs and fresh initialization"
            )
        staged = None
        if stage_bundle is not None:
            from sparselab.staging import verify_stage_bundle

            staged = verify_stage_bundle(stage_bundle, config, purpose=purpose)
        history = StageHistory()
        history.start(ExperimentStage.CONFIGURED)
        history.finish()
        history.start(ExperimentStage.INSPECTED)
        inventory = parameter_inventory(config)
        history.finish()
        history.start(ExperimentStage.VALIDATED)
        run_id = run_id or uuid.uuid4().hex
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a single nonempty directory name")
        run = config.logging.root_dir / run_id
        if run.exists():
            report = existing_run_recovery_report(run)
            raise FileExistsError(
                f"run exists: {run_id}; explicit recovery creates a child; "
                f"read-only recovery report: {json.dumps(report, sort_keys=True)}"
            )
        if config.runtime.engine == "pytorch":
            engine = PyTorchEngine()
        elif config.runtime.engine == "mlx":
            from sparselab.engines.mlx import MLXEngine

            engine = MLXEngine()
        else:
            raise ValueError(f"unsupported execution engine: {config.runtime.engine}")
        runtime = engine.validate(config)
        config = config.model_copy(
            update={
                "runtime": config.runtime.model_copy(
                    update={
                        "backend": runtime.backend,
                        "precision": "fp32"
                        if config.runtime.precision == "auto"
                        else config.runtime.precision,
                    }
                )
            }
        )
        current_source = source_identity()
        if promote is None and (resume is not None or recover is not None):
            continuation_root = (
                recover.resolve()
                if recover is not None
                else resume.parent.parent.resolve()  # type: ignore[union-attr]
            )
            owned_manifest = continuation_root / "portability_manifest.json"
            if _portability_manifest_v2(owned_manifest) is not None:
                config = _bind_learned_inputs(config, owned_manifest)
        continuation_state = load_continuation(
            config,
            runtime,
            current_source,
            resume=resume,
            promote=promote,
            recover=recover,
            allow_runtime_drift=allow_runtime_drift,
        )
        snapshot = continuation_state.snapshot
        source_run = continuation_state.source_run
        continuation = continuation_state.kind
        artifact_source = source_run if continuation == "RESUMED" else None
        if staged is not None and continuation != "RESUMED":
            artifact_source = Path(staged["assets_root"])
            config = config.model_copy(
                update={
                    "tokenizer": config.tokenizer.model_copy(
                        update={"path": artifact_source / "tokenizer.json"}
                    ),
                    "model": config.model.model_copy(
                        update={
                            "memory_package_path": artifact_source / "portable_package"
                            if config.model.memory_package_path is not None
                            else None,
                        }
                    ),
                }
            )
        seed_everything(config.seed, deterministic_cpu=config.training.deterministic)
        if continuation == "RESUMED":
            assert source_run is not None
            data = _load_run_data(source_run, config)
        elif artifact_source is not None:
            data = _load_run_data(artifact_source, config)
        else:
            tokenizer = load_tokenizer(config.tokenizer.path)
            data = prepare_data(config, tokenizer)
        dataset = TokenBlockDataset(
            data.train,
            config.training.seq_len,
            data.train_byte_addresses,
            data.train_supervision,
            data.train_owner_ids,
            data.train_semantic_queries,
            data.train_semantic_mask,
        )
        validation_dataset = TokenBlockDataset(
            data.validation,
            config.training.seq_len,
            data.validation_byte_addresses,
            data.validation_supervision,
            data.validation_owner_ids,
            data.validation_semantic_queries,
            data.validation_semantic_mask,
        )
        if not len(dataset):
            raise ValueError("training data contains no supervised next-token targets")
        if not len(validation_dataset):
            raise ValueError(
                "validation data contains no supervised next-token targets"
            )
        calibration_identity = calibration_key(
            config, runtime, source_digest=str(current_source["sha256"])
        )
        observations = ExperimentStore.get_calibration(
            config.logging.root_dir, calibration_identity
        )
        if staged is not None and stage_bundle is not None:
            for pilot_purpose in ("smoke", "warmup"):
                observations.extend(
                    ExperimentStore.get_calibration(
                        stage_bundle / "pilots" / pilot_purpose,
                        calibration_identity,
                    )
                )
        memory_estimate = calibrated_estimate(config, runtime, inventory, observations)
        if memory_estimate.result == "LIKELY_TO_EXCEED":
            raise MemoryError(
                "estimated training memory exceeds the safe ceiling; inspect --write-proposal before training"
            )
        offload_headroom = validate_offload_headroom(config, runtime, memory_estimate)
        history.finish()
        if (
            continuation == "RESUMED"
            and config.model.memory_package_path is not None
            and (
                config.training.portability_manifest_path is None
                or _portability_manifest_v2(config.training.portability_manifest_path)
                is None
            )
        ):
            assert source_run is not None
            owned_package = source_run / "portable_package"
            if not owned_package.exists():
                raise ValueError("resume run lacks immutable portable package")
            config = config.model_copy(
                update={
                    "model": config.model.model_copy(
                        update={"memory_package_path": owned_package}
                    )
                }
            )
        initial_weights = snapshot.model if snapshot is not None else None
        first_update_step = (
            snapshot.step + 1
            if snapshot is not None and continuation == "RESUMED"
            else 1
        )
        if isinstance(engine, PyTorchEngine):
            engine.initialize(
                config,
                initial_weights=initial_weights,
                resume_portability=(
                    continuation == "RESUMED"
                    and _portability_manifest_v2(
                        config.training.portability_manifest_path
                    )
                    is not None
                ),
            )
        else:
            engine.initialize(config, initial_weights=initial_weights)
        resources.callback(engine.close)
        step = tokens = 0
        cursor = BatchCursor()
        parent_run_id = continuation_state.parent_run_id
        lineage_best = continuation_state.lineage_best
        cumulative_wall = 0.0
        cumulative_updates = 0.0
        watermarks: dict[str, float] = {}
        if snapshot is not None and continuation == "RESUMED":
            engine.restore_training_state(
                EngineState(
                    snapshot.optimizer,
                    snapshot.rng or {},
                    snapshot.scaler,
                    snapshot.optimizer_parameter_names or [],
                )
            )
            step, tokens = snapshot.step, snapshot.tokens_seen
            cursor = BatchCursor(*snapshot.cursor)
            watermarks = snapshot.cadence or {}
            cumulative_wall = snapshot.cumulative_wall_seconds or 0.0
            cumulative_updates = snapshot.cumulative_update_seconds or 0.0
        if snapshot is not None:
            snapshot.model.clear()
            snapshot.optimizer.clear()
            snapshot.rng = None
            snapshot.scaler = None
        del initial_weights, snapshot
        if continuation == "RESUMED" and (
            step >= config.training.max_steps or tokens >= config.training.max_tokens
        ):
            raise ValueError("cannot resume a completed training budget")
        if stop_after_step is not None and stop_after_step <= step:
            raise ValueError("stop-after-step must exceed current step")
        run.mkdir(parents=True)
        manager = CheckpointManager(run, keep_periodic=config.checkpoint.keep_periodic)
        resources.enter_context(manager.writer_lease())
        artifacts = _copy_artifacts(run, config, data, artifact_source)
        owned_manifest = run / "portability_manifest.json"
        if _portability_manifest_v2(owned_manifest) is not None:
            config = _bind_learned_inputs(config, owned_manifest)
            from sparselab.research.portability import load_portability_manifest

            assert isinstance(engine, PyTorchEngine)
            engine.portability_run = load_portability_manifest(config)
            engine.config = config
        pilot_reports = tuple(staged["pilot_reports"]) if staged else ()
        stage_digest = str(staged["sha256"]) if staged else None
        if staged is not None:
            evidence = run / "stage_evidence.json"
            _atomic_json(
                evidence,
                {
                    "stage_bundle_sha256": stage_digest,
                    "pilot_reports": pilot_reports,
                    "stages": staged["stages"],
                },
            )
            artifacts = (
                *artifacts,
                ArtifactIdentity(
                    evidence.name,
                    sha256_file(evidence),
                    evidence.stat().st_size,
                ),
            )
        (run / "resolved_config.yaml").write_text(
            json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
        )
        manifest = RunManifest(
            run_id,
            config.name,
            runtime,
            requested_config,
            config.model_dump(mode="json"),
            architecture_sha256(config.model_dump(mode="json")),
            current_source,
            worker_id or socket.gethostname(),
            continuation,
            experiment_id=experiment_id,
            attempt_id=attempt_id,
            parent_run_id=parent_run_id,
            checkpoint_sha256=continuation_state.parent_checkpoint_sha256,
            purpose=purpose,
            stage_bundle_sha256=stage_digest,
            pilot_reports=pilot_reports,
            artifacts=artifacts,
            resource_decisions=continuation_state.decisions
            + dispatch_decisions
            + (
                (
                    {
                        "requested": "activation_offload",
                        "effective": True,
                        "reason": "conservative host headroom checked before model allocation",
                        "host_budget": offload_headroom,
                    },
                )
                if offload_headroom is not None
                else ()
            ),
        )
        manifest_digest = write_manifest(run / "manifest.json", manifest)
        manager.manifest_sha256 = manifest_digest
        store = ExperimentStore(config.logging.root_dir)
        store.create_run(
            run_id,
            config.model_dump(mode="json"),
            {
                "inspection": asdict(inventory),
                "runtime": runtime.as_dict(),
                "manifest_sha256": manifest_digest,
                "memory_estimate": asdict(memory_estimate),
                "purpose": purpose,
            },
            parent_run_id,
        )
        store.register_manifest(run_id, manifest_digest, manifest.payload())
        for sequence, stage_record in enumerate(history.records):
            store.record_stage(
                run_id,
                sequence,
                stage_record.stage,
                stage_record.status,
                stage_record.step,
                stage_record.tokens_seen,
                stage_record.started_at,
                stage_record.finished_at,
                stage_record.payload,
            )
        for decision in continuation_state.decisions:
            store.log_event(
                run_id, step, tokens, cumulative_wall, str(decision["kind"]), decision
            )
        started = time.perf_counter()
        interrupted = False
        latest_record: CheckpointRecord | None = None
        local_best: CheckpointRecord | None = None
        observed_peaks: dict[str, float] = {}

        def elapsed_seconds() -> float:
            return cumulative_wall + time.perf_counter() - started

        def progress(status: str = "running") -> None:
            _atomic_json(
                run / "progress.json",
                {
                    "manifest_sha256": manifest_digest,
                    "status": status,
                    "step": step,
                    "tokens_seen": tokens,
                    "cumulative_wall_seconds": elapsed_seconds(),
                    "cumulative_update_seconds": cumulative_updates,
                    "current_stage": asdict(history.current)
                    if history.current
                    else None,
                    "stages": [asdict(item) for item in history.records],
                    "latest": asdict(latest_record) if latest_record else None,
                    "best": asdict(local_best) if local_best else None,
                    "lineage_best": lineage_best.as_dict() if lineage_best else None,
                },
            )

        def finish_stage(status: str = "complete", reason: str | None = None) -> None:
            record = history.finish(
                status, reason=reason, step=step, tokens_seen=tokens
            )
            payload = dict(record.payload or {})
            if reason is not None:
                payload["reason"] = reason
            store.record_stage(
                run_id,
                len(history.records) - 1,
                record.stage,
                record.status,
                step,
                tokens,
                record.started_at,
                record.finished_at,
                payload,
            )

        def enter_stage(
            stage: ExperimentStage, payload: dict[str, object] | None = None
        ) -> None:
            if history.current is not None:
                finish_stage()
            current = history.start(
                stage, step=step, tokens_seen=tokens, payload=payload
            )
            store.log_event(
                run_id,
                step,
                tokens,
                elapsed_seconds(),
                "stage_started",
                asdict(current),
            )
            progress()

        def save_boundary(validation: dict[str, object] | None) -> CheckpointRecord:
            nonlocal latest_record, local_best, lineage_best, watermarks
            enter_stage(ExperimentStage.CHECKPOINTED)
            elapsed = elapsed_seconds()
            watermarks = {
                "step": float(step),
                "tokens": float(tokens),
                "minutes": elapsed,
            }
            loss = float(validation["loss"]) if validation else None
            record = _save(
                manager,
                engine,
                config,
                run_id,
                cursor,
                step,
                tokens,
                watermarks,
                loss,
                source_digest=str(current_source["sha256"]),
                parent_digest=continuation_state.parent_checkpoint_sha256,
                wall_seconds=elapsed,
                update_seconds=cumulative_updates,
                lineage_best=lineage_best,
            )
            latest_record = record
            if loss is not None:
                if local_best is None or loss < float(local_best.validation_loss):
                    local_best = record
                lineage_best = choose_lineage_best(
                    lineage_best,
                    LineageBest(run_id, record.manifest_sha256, step, loss),
                )
                _write_validation_report(
                    run, record, validation, config, str(current_source["sha256"])
                )
            store.record_checkpoint(run_id, record)
            store.log_event(
                run_id,
                step,
                tokens,
                elapsed_seconds(),
                "checkpoint_saved",
                asdict(record),
            )
            progress()
            return record

        def finish_run(status: str, reason: str | None = None) -> None:
            if isinstance(engine, PyTorchEngine) and engine.portability_run is not None:
                audit_path = run / "portability_audit.json"
                audit_version = (
                    2 if engine.portability_run.payload.get("version") == 2 else 1
                )
                try:
                    audit: dict[str, object] = {
                        "format": "sparselab-portability-audit",
                        "version": audit_version,
                        "valid": True,
                        "run_status": status,
                        **engine.portability_audit(completed=status == "completed"),
                    }
                # Any audit exception invalidates the run and must be recorded.
                except Exception as error:  # noqa: BLE001
                    audit = {
                        "format": "sparselab-portability-audit",
                        "version": 1,
                        "valid": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    status = "failed"
                    reason = f"portability integrity audit failed: {error}"
                _atomic_json(audit_path, audit)
                store.log_event(
                    run_id,
                    step,
                    tokens,
                    elapsed_seconds(),
                    "portability_audit_recorded",
                    {
                        "valid": audit["valid"],
                        "sha256": sha256_file(audit_path),
                    },
                )
            if status == "failed" and history.current is not None:
                finish_stage("failed", reason)
            enter_stage(
                {
                    "completed": ExperimentStage.COMPLETE,
                    "interrupted": ExperimentStage.INTERRUPTED,
                    "failed": ExperimentStage.FAILED,
                }[status]
            )
            finish_stage("complete" if status == "completed" else status, reason)
            store.finish_run(
                run_id,
                status,
                str(manager.root / "latest.json") if latest_record else None,
            )
            store.log_event(
                run_id,
                step,
                tokens,
                elapsed_seconds(),
                f"run_{status}",
                {"reason": reason},
            )
            if latest_record is not None and status != "completed":
                store.log_event(
                    run_id,
                    step,
                    tokens,
                    elapsed_seconds(),
                    "resumable",
                    {
                        "checkpoint": latest_record.relative_path,
                        "checkpoint_digest": latest_record.manifest_sha256,
                        "resume_level": latest_record.resume_level,
                    },
                )
            if observed_peaks:
                store.record_calibration(
                    calibration_identity,
                    run_id,
                    {
                        "estimate": asdict(memory_estimate),
                        "observed": observed_peaks,
                        "peak_method": "native"
                        if "memory/device_peak_allocated_bytes" in observed_peaks
                        else "sampled_lower_bound",
                        "runtime": runtime.as_dict(),
                    },
                )
            progress(status)

        def evaluation_batches():
            limit = config.evaluation.max_batches
            for emitted, offset in enumerate(
                range(0, len(validation_dataset), config.training.micro_batch_size),
                start=1,
            ):
                if limit is not None and emitted > limit:
                    break
                if wall_expired():
                    raise _WallTimeExpired
                records = [
                    validation_dataset.numpy_microblock(index)
                    for index in range(
                        offset,
                        min(
                            offset + config.training.micro_batch_size,
                            len(validation_dataset),
                        ),
                    )
                ]
                yield Microbatch(
                    np.stack([record[0] for record in records]),
                    np.stack([record[1] for record in records]),
                    _stack_optional(records, 2),
                    _stack_optional(records, 3),
                    _stack_optional(records, 4),
                    _stack_optional(records, 5),
                )

        def evaluate_and_record(elapsed: float) -> dict[str, object] | None:
            if wall_expired():
                return None
            enter_stage(ExperimentStage.EVALUATING)
            try:
                result = engine.evaluate(evaluation_batches()).to_report()
            except _WallTimeExpired:
                finish_stage("interrupted", "wall_time_limit")
                return None
            metrics: dict[str, float] = {"validation/loss": float(result["loss"])}
            if isinstance(result.get("perplexity"), (int, float)):
                metrics["validation/perplexity"] = float(result["perplexity"])
            store.log_metrics(run_id, step, tokens, elapsed, metrics)
            store.log_event(
                run_id, step, tokens, elapsed, "validation_completed", result
            )
            return result

        enter_stage(ExperimentStage.TRAINING)

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal interrupted
            interrupted = True

        old_int, old_term = (
            signal.signal(signal.SIGINT, request_stop),
            signal.signal(signal.SIGTERM, request_stop),
        )
        try:
            initial_validation = evaluate_and_record(cumulative_wall)
            save_boundary(initial_validation)
            if wall_expired():
                finish_run("interrupted", "wall_time_limit")
                return run_id
            enter_stage(ExperimentStage.TRAINING)
            while (
                step < config.training.max_steps and tokens < config.training.max_tokens
            ):
                if cancel_path is not None and cancel_path.exists():
                    interrupted = True
                if wall_expired():
                    interrupted = True
                if interrupted:
                    if latest_record is None or latest_record.step != step:
                        validation = evaluate_and_record(elapsed_seconds())
                        if (
                            validation is None
                            and wall_expired()
                            and history.current is not None
                        ):
                            finish_stage("interrupted", "wall_time_limit")
                        save_boundary(validation)
                    reason = (
                        "wall_time_limit"
                        if wall_expired()
                        else "cancelled"
                        if cancel_path is not None and cancel_path.exists()
                        else "signal"
                    )
                    finish_run("interrupted", reason)
                    return run_id
                order = epoch_order(len(dataset), config.seed, cursor.epoch)
                remaining = config.training.max_tokens - tokens
                window_size = (
                    config.training.micro_batch_size
                    * config.training.gradient_accumulation
                )
                candidate_indices = order[
                    cursor.next_block : cursor.next_block + window_size
                ].tolist()
                if not candidate_indices:
                    cursor = BatchCursor(cursor.epoch + 1, 0)
                    continue
                records = []
                valid_targets = 0
                for index in candidate_indices:
                    inputs, targets, addresses, owners, queries, query_mask = (
                        dataset.numpy_microblock(index)
                    )
                    labels = targets.copy()
                    available = int(np.count_nonzero(labels != -100))
                    allowed = min(remaining - valid_targets, available)
                    if allowed < available:
                        labels[np.flatnonzero(labels != -100)[allowed:]] = -100
                    records.append(
                        (inputs, labels, addresses, owners, queries, query_mask)
                    )
                    valid_targets += allowed
                    if valid_targets == remaining:
                        break
                if valid_targets == 0:
                    raise RuntimeError(
                        "selected training window has no supervised targets"
                    )
                microbatches = [
                    Microbatch(
                        np.stack(
                            [
                                record[0]
                                for record in records[
                                    offset : offset + config.training.micro_batch_size
                                ]
                            ]
                        ),
                        np.stack(
                            [
                                record[1]
                                for record in records[
                                    offset : offset + config.training.micro_batch_size
                                ]
                            ]
                        ),
                        _stack_optional(
                            records[offset : offset + config.training.micro_batch_size],
                            2,
                        ),
                        _stack_optional(
                            records[offset : offset + config.training.micro_batch_size],
                            3,
                        ),
                        _stack_optional(
                            records[offset : offset + config.training.micro_batch_size],
                            4,
                        ),
                        _stack_optional(
                            records[offset : offset + config.training.micro_batch_size],
                            5,
                        ),
                    )
                    for offset in range(
                        0, len(records), config.training.micro_batch_size
                    )
                ]
                retry_seconds = 0.0
                for retries in range(3):
                    update = engine.train_update(microbatches, step + 1, valid_targets)
                    retry_seconds += update.elapsed_seconds
                    if update.outcome == "APPLIED":
                        break
                else:
                    raise FloatingPointError(
                        "GradScaler overflow persisted after three attempts"
                    )
                metric_values = dict(update.metrics)
                metric_values["optimizer/overflow_retries"] = float(retries)
                metric_values["performance/step_seconds"] = retry_seconds
                metric_values["performance/tokens_per_second"] = valid_targets / max(
                    retry_seconds, 1e-9
                )
                cumulative_updates += retry_seconds
                step, tokens = step + 1, tokens + update.committed_targets
                cursor = BatchCursor(cursor.epoch, cursor.next_block + len(records))
                elapsed = cumulative_wall + time.perf_counter() - started
                store.log_metrics(run_id, step, tokens, elapsed, metric_values)
                if isinstance(engine, PyTorchEngine):
                    row_evidence = engine.portability_update_evidence()
                    if row_evidence is not None:
                        store.log_event(
                            run_id,
                            step,
                            tokens,
                            elapsed,
                            "learned_engram_gradient_rows",
                            row_evidence,
                        )
                for name, value in metric_values.items():
                    if name.startswith("memory/") and "peak" in name:
                        observed_peaks[name] = max(observed_peaks.get(name, 0.0), value)
                unavailable = getattr(engine, "unavailable_memory_reasons", {})
                if unavailable and step == first_update_step:
                    store.log_event(
                        run_id,
                        step,
                        tokens,
                        elapsed,
                        "memory_measurements_unavailable",
                        unavailable,
                    )
                progress()
                if cancel_path is not None and cancel_path.exists():
                    interrupted = True
                terminal = (
                    step >= config.training.max_steps
                    or tokens >= config.training.max_tokens
                    or step == stop_after_step
                    or interrupted
                )
                deadline_stopping = wall_expired()
                evaluation_due = not deadline_stopping and (
                    terminal or step % config.evaluation.every_steps == 0
                )
                validation = evaluate_and_record(elapsed) if evaluation_due else None
                terminal = terminal or wall_expired()
                new_best = validation is not None and (
                    local_best is None
                    or float(validation["loss"]) < float(local_best.validation_loss)
                )
                if (
                    _checkpoint_due(config, step, tokens, elapsed, watermarks)
                    or terminal
                    or new_best
                ):
                    save_boundary(validation)
                if terminal:
                    status = (
                        "completed"
                        if step >= config.training.max_steps
                        or tokens >= config.training.max_tokens
                        else "interrupted"
                    )
                    finish_run(
                        status,
                        None
                        if status == "completed"
                        else "wall_time_limit"
                        if wall_expired()
                        else "cancelled"
                        if cancel_path is not None and cancel_path.exists()
                        else "signal"
                        if interrupted
                        else "stop_after_step",
                    )
                    return run_id
                if (
                    history.current is not None
                    and history.current.stage != ExperimentStage.TRAINING
                ):
                    enter_stage(ExperimentStage.TRAINING)
            raise RuntimeError(
                "training stopped without completing a valid target window"
            )
        except BaseException as error:
            finish_run("failed", str(error))
            raise
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
