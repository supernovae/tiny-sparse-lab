"""Load one frozen PyTorch checkpoint using only its run-owned artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from sparselab.config.models import RunConfig
from sparselab.data.packing import TokenBlockDataset
from sparselab.data.tokenizer import load_tokenizer
from sparselab.model.inspection import inspect_model
from sparselab.model.transformer import DenseLM
from sparselab.runtime import discover_runtimes, select_device, torch_device_for
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import canonical_json, read_manifest, sha256_file


@dataclass(frozen=True)
class InferenceRun:
    run: Path
    config: RunConfig
    model: DenseLM
    tokenizer: Tokenizer
    device: torch.device
    identity: dict[str, Any]

    def validation_dataset(self) -> TokenBlockDataset:
        root = self.run / "data"
        addresses = root / "validation_byte_addresses.npy"
        return TokenBlockDataset(
            np.load(root / "validation.npy", mmap_mode="r", allow_pickle=False),
            self.config.training.seq_len,
            np.load(addresses, mmap_mode="r", allow_pickle=False)
            if self.config.model.memory in {"byte", "portable"}
            else None,
        )


def load_run(
    run_id: str,
    runs_dir: Path,
    checkpoint: str | None = None,
    backend: str | None = None,
) -> InferenceRun:
    run = (runs_dir / run_id).resolve()
    config = RunConfig.model_validate_json((run / "resolved_config.yaml").read_text())
    if config.runtime.engine != "pytorch":
        raise ValueError(
            "chat, generation and capability evaluation currently require a PyTorch run; "
            "MLX native checkpoints are training-only and are not silently converted"
        )
    manifest = read_manifest(run / "manifest.json")
    manifest_digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    if RunConfig.model_validate(manifest["effective_config"]) != config:
        raise ValueError("resolved config differs from the verified run manifest")
    artifacts: dict[str, str] = {}
    for entry in manifest["artifacts"]:
        relative = Path(entry["relative_path"])
        path = run / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(run)
            or not path.is_file()
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"run artifact integrity failure: {relative}")
        artifacts[str(relative)] = entry["sha256"]
    required = {"tokenizer.json", "data/train.npy", "data/validation.npy"}
    if config.model.memory in {"byte", "portable"}:
        required.update(
            {"data/train_byte_addresses.npy", "data/validation_byte_addresses.npy"}
        )
    if not required <= artifacts.keys():
        raise ValueError(
            f"run lacks verified artifacts: {sorted(required - artifacts.keys())}"
        )

    selected = Path(checkpoint or "latest.json")
    if not selected.is_absolute() and not selected.exists():
        selected = run / "checkpoints" / selected
    selected = selected.resolve()
    expected_digest = None
    if selected.name in {"latest.json", "best.json"}:
        pointer = json.loads(selected.read_text())
        relative = pointer.get("relative_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("inference requires a validated v2 checkpoint pointer")
        expected_digest = pointer["manifest_sha256"]
        selected = (selected.parent / relative).resolve()
    if selected.parent != run / "checkpoints" or not selected.is_dir():
        raise ValueError(
            "selected checkpoint must be a generation belonging to this run"
        )
    metadata = json.loads((selected / "manifest.json").read_text())
    if expected_digest is not None and metadata["sha256"] != expected_digest:
        raise ValueError("checkpoint pointer digest does not match selected generation")
    manager = CheckpointManager(run, manifest_sha256=manifest_digest)
    report = manager.verify(
        selected, expected_manifest=manifest_digest, require_training_state=False
    )
    if not report.valid:
        raise ValueError(f"invalid checkpoint: {report.errors}")
    if metadata["engine"] != "pytorch":
        raise ValueError("checkpoint engine differs from the PyTorch run")

    requested = backend or metadata["backend"]
    selected_device = select_device(requested)
    actual_backend = (
        "rocm"
        if selected_device.type == "cuda" and torch.version.hip
        else selected_device.type
    )
    device = torch_device_for(actual_backend, config.runtime.device_index)
    tokenizer = load_tokenizer(run / "tokenizer.json")
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        raise ValueError("run tokenizer vocabulary differs from model configuration")
    model_config = config.model
    if model_config.memory_package_path is not None:
        if "portable_package" not in artifacts:
            raise ValueError("run lacks a verified portable memory package")
        model_config = model_config.model_copy(
            update={"memory_package_path": run / "portable_package"}
        )
    # Constructor initialization is irrelevant to loaded weights; don't perturb caller RNG.
    with torch.random.fork_rng(devices=[]):
        model = DenseLM(model_config, config.attention)
    weights = manager.load(selected, "promote").model
    model.load_state_dict(weights)
    del weights
    model.to(device).eval()
    runtime = next(
        info
        for info in discover_runtimes()
        if info.engine == "pytorch" and info.backend == actual_backend
    )
    device_name = runtime.device_name
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "xpu":
        device_name = torch.xpu.get_device_name(device)
    identity = {
        "run_id": run_id,
        "checkpoint_sha256": metadata["sha256"],
        "checkpoint_relative_path": str(selected.relative_to(run)),
        "step": metadata["step"],
        "tokens_seen": metadata["tokens_seen"],
        "config": config.model_dump(mode="json"),
        "tokenizer_sha256": artifacts["tokenizer.json"],
        "data_sha256": {
            key: artifacts[f"data/{key}.npy"] for key in ("train", "validation")
        },
        "source_identity_sha256": manifest["source_identity"]["sha256"],
        "runtime": {
            "engine": "pytorch",
            "backend": actual_backend,
            "device_index": config.runtime.device_index,
            "device_name": device_name,
            "framework_version": runtime.framework_version,
            "os": runtime.os,
            "precision": "fp32",
        },
        "training_runtime": {
            key: manifest["runtime"][key]
            for key in (
                "engine",
                "backend",
                "device_index",
                "device_name",
                "framework_version",
                "os",
            )
        },
        "parameter_inventory": inspect_model(model),
    }
    return InferenceRun(run, config, model, tokenizer, device, identity)


def write_inference_result(run: Path, kind: str, result: dict[str, Any]) -> Path:
    """Content-addressed results preserve earlier checkpoints and observations."""
    content = canonical_json(result) + b"\n"
    digest = hashlib.sha256(content).hexdigest()
    path = run / "evaluations" / f"{kind}-{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"result artifact conflict: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(content)
    return path
