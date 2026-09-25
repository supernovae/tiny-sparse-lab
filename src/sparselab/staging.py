"""Portable, immutable preflight evidence; pilots execute only in subprocesses."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from sparselab.config.models import RunConfig
from sparselab.data.allocation import (
    copy_allocation_bundle,
    load_allocation_manifest,
    load_semantic_retriever,
)
from sparselab.data.packing import (
    _tokenizer_sha256,
    load_prepared_data,
    prepare_data,
)
from sparselab.data.tokenizer import load_tokenizer
from sparselab.memory import (
    calibrated_estimate,
    calibration_key,
    estimate_memory,
    parameter_inventory,
    plan_memory,
    write_resource_proposal,
)
from sparselab.model.inspection import named_tensor_inventory
from sparselab.model.portable_engram import load_portable_engram
from sparselab.runtime import (
    RuntimeInfo,
    discover_runtimes,
    select_device,
    validate_runtime,
)
from sparselab.training.checkpoints import _atomic_json, _safe_member
from sparselab.training.manifest import (
    canonical_json,
    config_sha256,
    sha256_file,
    source_identity,
)
from sparselab.training.metrics import SCHEMA_VERSION, ExperimentStore
from sparselab.training.stages import ExperimentStage, StageHistory

_LEVELS = {"inspect": 1, "validate": 2, "smoke": 3, "warmup": 4}
_MAX_REPORT_BYTES = 16 * 1024 * 1024


def inspect_runtime(config: RunConfig) -> RuntimeInfo:
    """Resolve passive readings without validating or allocating model tensors."""
    runtimes = discover_runtimes()
    backend = config.runtime.backend
    if backend == "auto":
        selected = str(select_device("auto"))
        backend = next(
            info.backend
            for info in runtimes
            if info.engine == "pytorch" and info.torch_device == selected
        )
    runtime = next(
        info
        for info in runtimes
        if info.engine == config.runtime.engine and info.backend == backend
    )
    if config.runtime.device_index != runtime.device_index:
        runtime = replace(
            runtime,
            device_index=config.runtime.device_index,
            torch_device=None,
            device_name=None,
            physical_device_id=None,
            device_total_bytes=None,
            device_free_bytes=None,
            device_recommended_bytes=None,
            device_driver_allocated_bytes=None,
            limitations=(
                *runtime.limitations,
                "passive memory readings for this device index are unavailable",
            ),
        )
    return runtime


def _seal(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    value = {**payload, "sha256": hashlib.sha256(canonical_json(payload)).hexdigest()}
    _atomic_json(path, value)
    return value


def _read_sealed(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > _MAX_REPORT_BYTES
    ):
        raise ValueError(f"invalid or oversized stage manifest: {path}")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate stage key: {key}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    if not isinstance(payload, dict):
        raise TypeError("stage manifest must be an object")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != hashlib.sha256(canonical_json(body)).hexdigest():
        raise ValueError("stage manifest digest mismatch")
    if payload.get("format_version") != 1:
        raise ValueError("unsupported stage bundle version")
    return payload


def _inventory(root: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in immutable stage assets: {path}")
        if path.is_file():
            result.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return result


def _verify_inventory(root: Path, inventory: object) -> None:
    if not isinstance(inventory, list):
        raise TypeError("stage inventory must be a list")
    seen: set[str] = set()
    for item in inventory:
        if not isinstance(item, dict):
            raise TypeError("invalid stage inventory row")
        name = item.get("relative_path")
        path = _safe_member(root, name)
        if (
            not isinstance(name, str)
            or path is None
            or not path.is_file()
            or name in seen
        ):
            raise ValueError(f"unsafe or duplicate stage member: {name}")
        seen.add(name)
        if (
            type(item.get("size_bytes")) is not int
            or path.stat().st_size != item["size_bytes"]
        ):
            raise ValueError(f"stage member length mismatch: {name}")
        if sha256_file(path) != item.get("sha256"):
            raise ValueError(f"stage member digest mismatch: {name}")


def pilot_config(config: RunConfig, purpose: str, root: Path) -> RunConfig:
    if purpose not in {"smoke", "warmup"}:
        raise ValueError("invalid pilot purpose")
    steps = getattr(config.staging, f"{purpose}_steps")
    payload = config.model_dump(mode="json")
    payload["training"].update(
        max_steps=steps,
        max_tokens=steps
        * config.training.micro_batch_size
        * config.training.gradient_accumulation
        * config.training.seq_len,
    )
    payload["optimizer"]["warmup_steps"] = min(config.optimizer.warmup_steps, steps - 1)
    payload["logging"]["root_dir"] = str(root / "pilots" / purpose)
    if config.dataset.allocation_manifest_path is not None:
        payload["dataset"]["allocation_manifest_path"] = str(
            root
            / "assets"
            / "allocation"
            / config.dataset.allocation_manifest_path.name
        )
    return RunConfig.model_validate(payload)


def verify_stage_bundle(
    root: Path,
    config: RunConfig,
    *,
    purpose: str = "training",
    allow_runtime_drift: bool = False,
) -> dict[str, Any]:
    """Verify frozen inputs; a pilot may read only the sealed pre-pilot input manifest."""
    root = root.resolve(strict=True)
    inputs = _read_sealed(root / "inputs.json")
    base = RunConfig.model_validate(inputs["requested_config"])
    expected = base if purpose == "training" else pilot_config(base, purpose, root)
    if config_sha256(config.model_dump(mode="json")) != config_sha256(
        expected.model_dump(mode="json")
    ):
        raise ValueError("stage bundle configuration differs from requested execution")
    if (
        inputs.get("source_identity_sha256") != source_identity()["sha256"]
        and not allow_runtime_drift
    ):
        raise ValueError(
            "stage bundle executable source identity differs; restage explicitly"
        )
    _verify_inventory(root / "assets", inputs.get("artifacts"))
    data = load_prepared_data(
        root / "assets" / "data", byte_enabled=base.model.memory in {"byte", "portable"}
    )
    tokenizer = load_tokenizer(root / "assets" / "tokenizer.json")
    if tokenizer.get_vocab_size() != base.model.vocab_size:
        raise ValueError("staged tokenizer vocabulary does not match model")
    _verify_allocation_assets(
        root / "assets",
        config,
        data,
        inputs.get("source_identity_sha256"),
        tokenizer,
    )
    if not len(data.train) or not len(data.validation):
        raise ValueError("staged data is empty")
    if purpose == "training":
        bundle = _read_sealed(root / "bundle.json")
        if bundle.get("status") != "complete" or bundle.get("through") not in {
            "validate",
            "smoke",
            "warmup",
        }:
            raise ValueError("stage bundle has not completed asset validation")
        _verify_inventory(root, bundle.get("artifacts"))
        if bundle.get("inputs_sha256") != inputs["sha256"]:
            raise ValueError("stage bundle input identity mismatch")
    else:
        bundle = {"sha256": inputs["sha256"], "pilot_reports": [], "stages": []}
    return {**bundle, "assets_root": str(root / "assets")}


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"symlink asset root: {source}")
    if source.is_dir():
        _inventory(source)  # Reject symlinks before following any source members.
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _verify_allocation_assets(
    assets: Path,
    config: RunConfig,
    data: Any,
    source_sha256: object,
    tokenizer: Any,
) -> None:
    allocation_path = config.dataset.allocation_manifest_path
    metadata = data.manifest.get("allocation")
    if allocation_path is None:
        if metadata is not None or (assets / "allocation").exists():
            raise ValueError("prepared inputs contain an unexpected allocation")
        return
    if not isinstance(source_sha256, str):
        raise TypeError("prepared allocation source identity must be a string")
    if not isinstance(metadata, dict):
        raise TypeError("prepared allocation metadata must be an object")
    allocation = load_allocation_manifest(
        assets / "allocation" / allocation_path.name,
        source_identity_sha256=source_sha256,
        tokenizer_sha256=_tokenizer_sha256(tokenizer),
    )
    if allocation.sha256 != metadata.get("manifest_sha256"):
        raise ValueError("prepared data and allocation manifests differ")
    for split, values in (("train", data.train), ("validation", data.validation)):
        allocation.split(split, token_count=len(values))
    retriever = load_semantic_retriever(allocation)
    if (retriever is None) != (config.model.semantic_memory_dim is None):
        raise ValueError("model.semantic_memory_dim must match the allocation pack")
    if (
        retriever is not None
        and retriever.memory_dim != config.model.semantic_memory_dim
    ):
        raise ValueError("model.semantic_memory_dim differs from semantic pack width")


def materialize_prepared_inputs(
    config: RunConfig,
    destination: Path,
    *,
    prepared_inputs: Path | None = None,
) -> Path:
    """Publish immutable, engine-neutral training inputs without probing a runtime.

    ``prepared_inputs`` is a prior materialized root (or its ``assets`` directory).
    It is verified before copying, so callers never use controller-local source paths
    after this boundary.
    """
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"prepared input destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        work = Path(temporary)
        assets = work / "assets"
        source_digest = source_identity()["sha256"]
        if prepared_inputs is None:
            tokenizer = load_tokenizer(config.tokenizer.path)
            if tokenizer.get_vocab_size() != config.model.vocab_size:
                raise ValueError("tokenizer vocabulary does not match model")
            data = prepare_data(config, tokenizer)
            assets.mkdir()
            shutil.copy2(config.tokenizer.path, assets / "tokenizer.json")
            tokenizer_manifest = config.tokenizer.path.with_name(
                "tokenizer_manifest.json"
            )
            if tokenizer_manifest.is_file():
                shutil.copy2(tokenizer_manifest, assets / tokenizer_manifest.name)
            _copy_tree(data.root, assets / "data")
            allocation_source = config.dataset.allocation_manifest_path
            if allocation_source is not None:
                allocation = load_allocation_manifest(
                    allocation_source,
                    source_identity_sha256=str(source_digest),
                    tokenizer_sha256=_tokenizer_sha256(tokenizer),
                )
                prepared_allocation = data.manifest.get("allocation")
                if (
                    not isinstance(prepared_allocation, dict)
                    or prepared_allocation.get("manifest_sha256") != allocation.sha256
                ):
                    raise ValueError("prepared data differs from allocation manifest")
                copy_allocation_bundle(allocation, assets / "allocation")
            if config.model.memory_package_path is not None:
                _copy_tree(
                    config.model.memory_package_path, assets / "portable_package"
                )
                load_portable_engram(
                    assets / "portable_package",
                    expected_shape=(
                        config.model.memory_table_size,
                        config.model.memory_dim,
                    ),
                    expected_ngram_size=config.model.memory_ngram_size,
                )
        else:
            source = prepared_inputs.resolve(strict=True)
            if source.name == "assets":
                source = source.parent
            verify_prepared_inputs(source, config)
            _copy_tree(source / "assets", assets)
        _seal(
            work / "inputs.json",
            {
                "format_version": 1,
                "requested_config": config.model_dump(mode="json"),
                "source_identity_sha256": source_digest,
                "artifacts": _inventory(assets),
            },
        )
        os.rename(work, destination)
    return destination


def _prepared_config_matches(saved: RunConfig, requested: RunConfig) -> bool:
    """Allow only a worker's recorded auto-runtime resolution over frozen inputs."""
    if config_sha256(saved.model_dump(mode="json")) == config_sha256(
        requested.model_dump(mode="json")
    ):
        return True
    base = saved.model_dump(mode="json")
    effective = requested.model_dump(mode="json")
    runtime = base["runtime"]
    resolved = effective["runtime"]
    if (
        runtime["engine"] != resolved["engine"]
        or runtime["device_index"] != resolved["device_index"]
    ):
        return False
    if runtime["backend"] == "auto":
        runtime["backend"] = resolved["backend"]
    if runtime["precision"] == "auto":
        runtime["precision"] = resolved["precision"]
    return config_sha256(base) == config_sha256(effective)


def verify_prepared_inputs(
    root: Path, config: RunConfig, *, allow_runtime_drift: bool = False
) -> dict[str, Any]:
    """Verify a materialized input root without acquiring a target accelerator."""
    root = root.resolve(strict=True)
    inputs = _read_sealed(root / "inputs.json")
    saved = RunConfig.model_validate(inputs["requested_config"])
    if not _prepared_config_matches(saved, config):
        raise ValueError(
            "prepared inputs configuration differs from requested execution"
        )
    if (
        inputs.get("source_identity_sha256") != source_identity()["sha256"]
        and not allow_runtime_drift
    ):
        raise ValueError("prepared inputs executable source identity differs")
    _verify_inventory(root / "assets", inputs.get("artifacts"))
    data = load_prepared_data(
        root / "assets" / "data",
        byte_enabled=config.model.memory in {"byte", "portable"},
    )
    tokenizer = load_tokenizer(root / "assets" / "tokenizer.json")
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        raise ValueError("prepared tokenizer vocabulary does not match model")
    if not len(data.train) or not len(data.validation):
        raise ValueError("prepared data is empty")
    _verify_allocation_assets(
        root / "assets",
        config,
        data,
        inputs.get("source_identity_sha256"),
        tokenizer,
    )
    if config.model.memory_package_path is not None:
        load_portable_engram(
            root / "assets" / "portable_package",
            expected_shape=(config.model.memory_table_size, config.model.memory_dim),
            expected_ngram_size=config.model.memory_ngram_size,
        )
    return {**inputs, "assets_root": str(root / "assets")}


def _run_pilot(
    root: Path,
    purpose: str,
    *,
    inherited_fds: tuple[int, ...] = (),
    cancel_path: Path | None = None,
) -> dict[str, Any]:
    if cancel_path is not None and cancel_path.exists():
        raise InterruptedError("staging cancelled before pilot initialization")
    directory = root / "pilots" / purpose
    directory.mkdir(parents=True)
    log_path = directory / "execution.log"
    with log_path.open("xb") as log:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "sparselab.training.pilot",
                str(root),
                purpose,
                *([] if cancel_path is None else ["--cancel-path", str(cancel_path)]),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
            pass_fds=inherited_fds,
        )
    if cancel_path is not None and cancel_path.exists():
        raise InterruptedError("staging cancelled at pilot boundary")
    if completed.returncode:
        with log_path.open("rb") as log:
            detail = log.read(65536).decode("utf-8", errors="replace")
        failure_path = directory / "failure.json"
        if failure_path.is_file():
            failure = _read_sealed(failure_path)
            if failure.get("kind") == "out_of_memory":
                raise MemoryError(
                    f"{purpose} pilot exhausted memory: {failure['message']}"
                )
        raise RuntimeError(f"{purpose} pilot failed ({completed.returncode}): {detail}")
    report = _read_sealed(directory / "report.json")
    if report.get("purpose") != purpose or report.get("status") != "complete":
        raise ValueError("pilot returned incomplete or mismatched evidence")
    return {
        "run_id": report["run_id"],
        "purpose": purpose,
        "relative_path": (directory / "report.json").relative_to(root).as_posix(),
        "sha256": report["sha256"],
        "report": report,
    }


def stage(
    config: RunConfig,
    output: Path,
    through: str = "smoke",
    *,
    prepared_inputs: Path | None = None,
    allow_runtime_drift: bool = False,
    inherited_fds: tuple[int, ...] = (),
    cancel_path: Path | None = None,
) -> Path:
    if through not in _LEVELS:
        raise ValueError("through must be inspect, validate, smoke, or warmup")
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(f".{output.name}.stage.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FileExistsError("another writer owns this stage output") from error
        if output.exists():
            bundle = _read_sealed(output / "bundle.json")
            if bundle.get("status") != "complete" or bundle.get("through") != through:
                raise FileExistsError("stage output is conflicting or incomplete")
            if bundle.get("config_sha256") != config_sha256(
                config.model_dump(mode="json")
            ):
                raise FileExistsError("stage output belongs to another configuration")
            if (
                bundle.get("source_identity_sha256") != source_identity()["sha256"]
                and not allow_runtime_drift
            ):
                raise FileExistsError(
                    "stage output belongs to another executable source"
                )
            _verify_inventory(output, bundle.get("artifacts"))
            if through != "inspect":
                verify_stage_bundle(
                    output, config, allow_runtime_drift=allow_runtime_drift
                )
                if prepared_inputs is not None:
                    verify_prepared_inputs(
                        prepared_inputs, config, allow_runtime_drift=allow_runtime_drift
                    )
                elif config.tokenizer.path.exists() and sha256_file(
                    config.tokenizer.path
                ) != sha256_file(output / "assets" / "tokenizer.json"):
                    raise FileExistsError("requested tokenizer changed since staging")
            return output
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.", dir=output.parent
        ) as temporary:
            work = Path(temporary)
            history = StageHistory()
            reports: list[dict[str, Any]] = []
            runtime = estimate = None
            report: dict[str, Any] = {
                "format_version": 1,
                "requested_config": config.model_dump(mode="json"),
                "effective_config": config.model_dump(mode="json"),
                "through": through,
                "metric_schema_version": SCHEMA_VERSION,
            }
            try:
                history.start(ExperimentStage.CONFIGURED)
                history.finish()
                history.start(ExperimentStage.INSPECTED)
                runtime = inspect_runtime(config)
                inventory = parameter_inventory(config)
                estimate = estimate_memory(config, runtime, inventory)
                source_digest = source_identity()["sha256"]
                report.update(
                    inventory=asdict(inventory),
                    runtime=runtime.as_dict(),
                    estimate=asdict(estimate),
                    source_identity_sha256=source_digest,
                    tensor_inventory={
                        name: asdict(spec)
                        for name, spec in named_tensor_inventory(
                            config.model,
                            config.attention,
                            trainable_parameters=config.training.trainable_parameters,
                        ).items()
                    },
                )
                history.finish()
                inputs_digest = None
                if _LEVELS[through] >= 2:
                    history.start(ExperimentStage.VALIDATED)
                    runtime = validate_runtime(config)
                    calibration_identity = calibration_key(
                        config, runtime, source_digest=str(source_digest)
                    )
                    observations = ExperimentStore.get_calibration(
                        config.logging.root_dir, calibration_identity
                    )
                    estimate = calibrated_estimate(
                        config, runtime, inventory, observations
                    )
                    report.update(runtime=runtime.as_dict(), estimate=asdict(estimate))
                    effective = config.model_dump(mode="json")
                    effective["runtime"]["backend"] = runtime.backend
                    if effective["runtime"]["precision"] == "auto":
                        effective["runtime"]["precision"] = "fp32"
                    report["effective_config"] = effective
                    if estimate.result == "LIKELY_TO_EXCEED":
                        write_resource_proposal(
                            plan_memory(config, runtime, estimate),
                            work / "proposal.yaml",
                        )
                        raise MemoryError(
                            "estimated training memory exceeds the explicit safe ceiling"
                        )
                    if prepared_inputs is None:
                        prepared_root = materialize_prepared_inputs(
                            config, work / "prepared"
                        )
                    else:
                        prepared_root = prepared_inputs.resolve(strict=True)
                        if prepared_root.name == "assets":
                            prepared_root = prepared_root.parent
                        verify_prepared_inputs(
                            prepared_root,
                            config,
                            allow_runtime_drift=allow_runtime_drift,
                        )
                    _copy_tree(prepared_root / "assets", work / "assets")
                    prepared_identity = _read_sealed(prepared_root / "inputs.json")
                    inputs = _seal(
                        work / "inputs.json",
                        {
                            "format_version": 1,
                            "requested_config": config.model_dump(mode="json"),
                            "source_identity_sha256": source_digest,
                            "parent_inputs_sha256": prepared_identity["sha256"],
                            "artifacts": _inventory(work / "assets"),
                        },
                    )
                    inputs_digest = inputs["sha256"]
                    history.finish()
                for purpose, level, stage_name in (
                    ("smoke", 3, ExperimentStage.SMOKE_TEST),
                    ("warmup", 4, ExperimentStage.WARMUP),
                ):
                    if _LEVELS[through] < level:
                        continue
                    history.start(stage_name)
                    pilot = _run_pilot(
                        work,
                        purpose,
                        inherited_fds=inherited_fds,
                        cancel_path=cancel_path,
                    )
                    reports.append(pilot)
                    observations.extend(
                        ExperimentStore.get_calibration(
                            work / "pilots" / purpose, calibration_identity
                        )
                    )
                    estimate = calibrated_estimate(
                        config, runtime, inventory, observations
                    )
                    report["estimate"] = asdict(estimate)
                    peak = pilot["report"].get("observed_peak_bytes")
                    if estimate.result == "LIKELY_TO_EXCEED" or (
                        peak is not None
                        and estimate.capacity_ceiling_bytes is not None
                        and peak > estimate.capacity_ceiling_bytes
                    ):
                        write_resource_proposal(
                            plan_memory(config, runtime, estimate),
                            work / "proposal.yaml",
                        )
                        raise MemoryError(
                            f"{purpose} measured or calibrated memory exceeds safe ceiling"
                        )
                    history.finish(
                        payload={
                            "pilot_run_id": pilot["run_id"],
                            "report_sha256": pilot["sha256"],
                        }
                    )
                report.update(
                    status="complete",
                    stages=[asdict(item) for item in history.records],
                    pilot_reports=reports,
                )
                _seal(work / "stage.json", report)
                _seal(
                    work / "bundle.json",
                    {
                        "format_version": 1,
                        "status": "complete",
                        "through": through,
                        "config_sha256": config_sha256(config.model_dump(mode="json")),
                        "source_identity_sha256": source_digest,
                        "inputs_sha256": inputs_digest,
                        "stages": report["stages"],
                        "pilot_reports": reports,
                        "artifacts": _inventory(work),
                    },
                )
            except BaseException as error:
                if history.current is not None:
                    history.finish("failed", reason=str(error))
                if history.records:
                    history.start(ExperimentStage.FAILED)
                    history.finish("failed", reason=str(error))
                report.update(
                    status="failed",
                    reason=str(error),
                    stages=[asdict(item) for item in history.records],
                    pilot_reports=reports,
                )
                _seal(work / "stage.json", report)
                if (
                    isinstance(error, MemoryError)
                    and runtime is not None
                    and estimate is not None
                    and not (work / "proposal.yaml").exists()
                ):
                    write_resource_proposal(
                        plan_memory(config, runtime, estimate), work / "proposal.yaml"
                    )
                os.rename(work, output)
                raise
            os.rename(work, output)
    return output
