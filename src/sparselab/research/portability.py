"""Verified inputs and model initialization for Engram portability runs."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from safetensors.torch import load_file

from sparselab.config.models import RunConfig
from sparselab.engram.packs import verify_pack
from sparselab.model.inspection import named_tensor_inventory
from sparselab.training.manifest import (
    architecture_sha256,
    canonical_json,
    read_manifest,
    sha256_file,
)

_FORMAT = "sparselab-portability-run"
_VERSION = 1
_SEEDS = (17, 41, 73)
_MEMORY_MODEL_FIELDS = frozenset(
    {
        "memory",
        "memory_injection",
        "memory_table_size",
        "memory_ngram_size",
        "memory_dim",
        "memory_package_path",
        "memory_ngram_orders",
        "memory_hash_heads",
        "semantic_memory_dim",
    }
)


@dataclass(frozen=True, slots=True)
class PortabilityRun:
    path: Path
    root: Path
    payload: dict[str, Any]
    backbone: dict[str, Any] | None
    memory: dict[str, Any]


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("portability asset path must be a nonempty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("portability asset path is unsafe")
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("portability asset path traverses a symlink")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("portability asset escapes its manifest directory") from error
    return path


def _verify_file(root: Path, descriptor: object) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("file descriptor must contain path, sha256, and size_bytes")
    path = _relative_path(root, descriptor["path"])
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("portability file asset is not a regular file")
    expected_size = descriptor["size_bytes"]
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError("portability file size_bytes must be a nonnegative integer")
    expected_digest = _digest(descriptor["sha256"], "file sha256")
    if info.st_size != expected_size or sha256_file(path) != expected_digest:
        raise ValueError(f"portability file asset failed integrity verification: {path}")
    return path


def _verify_directory(root: Path, descriptor: object) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "files"}:
        raise ValueError("directory descriptor must contain path and files")
    directory = _relative_path(root, descriptor["path"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("portability directory asset is not a regular directory")
    files = descriptor["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("directory descriptor requires a nonempty file inventory")
    expected: dict[str, object] = {}
    for member in files:
        if not isinstance(member, dict) or set(member) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("directory member descriptor is invalid")
        member_path = member["path"]
        if not isinstance(member_path, str) or member_path in expected:
            raise ValueError("directory member path is invalid or duplicated")
        expected[member_path] = member
        _verify_file(directory, member)
    actual: set[str] = set()
    for item in directory.rglob("*"):
        if item.is_symlink():
            raise ValueError("portability directory contains a symlink")
        if item.is_file():
            actual.add(item.relative_to(directory).as_posix())
    if actual != set(expected):
        raise ValueError("portability directory inventory differs from its descriptor")
    return directory


def _verify_asset(root: Path, descriptor: object) -> Path:
    if isinstance(descriptor, dict) and set(descriptor) == {
        "path",
        "sha256",
        "size_bytes",
    }:
        return _verify_file(root, descriptor)
    return _verify_directory(root, descriptor)


def _verify_world_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("world manifest is invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("format") != "sparselab-portability-worlds":
        raise ValueError("unsupported portability world manifest")
    expected = _digest(payload.get("sha256"), "world manifest sha256")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if hashlib.sha256(canonical_json(content)).hexdigest() != expected:
        raise ValueError("portability world manifest hash mismatch")
    files = payload.get("files")
    if not isinstance(files, list):
        raise TypeError("world manifest file inventory must be an array")
    expected_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise TypeError("world manifest file descriptor is invalid")
        _verify_file(path.parent, entry)
        expected_paths.add(entry["path"])
    actual_paths: set[str] = set()
    for item in path.parent.rglob("*"):
        if item.is_symlink():
            raise ValueError("world assets contain a symlink")
        if item.is_file() and item != path:
            actual_paths.add(item.relative_to(path.parent).as_posix())
    if actual_paths != expected_paths:
        raise ValueError("world manifest file inventory differs from its directory")
    return payload


def load_portability_manifest(config: RunConfig) -> PortabilityRun:
    """Validate the complete seed-selected portability input contract before training."""
    manifest_path = config.training.portability_manifest_path
    if manifest_path is None:
        raise ValueError("training.portability_manifest_path is not configured")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("portability run manifest must be a regular nonsymlink file")
    root = manifest_path.parent.resolve(strict=True)
    path = manifest_path.resolve(strict=True)
    if path.parent != root:
        raise ValueError("portability run manifest must reside in its asset root")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("portability run manifest is invalid JSON") from error
    required = {
        "format",
        "version",
        "protocol",
        "world_manifest",
        "coordinate",
        "seeds",
        "initial_backbone",
        "memory",
        "observations",
        "training_fact_ids",
        "sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("portability run manifest has an invalid top-level schema")
    if payload["format"] != _FORMAT or type(payload["version"]) is not int or payload["version"] != _VERSION:
        raise ValueError("unsupported portability run manifest version")
    expected = _digest(payload["sha256"], "portability run manifest sha256")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if hashlib.sha256(canonical_json(content)).hexdigest() != expected:
        raise ValueError("portability run manifest hash mismatch")
    if payload["seeds"] != list(_SEEDS) or config.seed not in _SEEDS:
        raise ValueError("portability seed axis must be exactly [17, 41, 73]")

    _verify_file(root, payload["protocol"])
    world_path = _verify_file(root, payload["world_manifest"])
    world = _verify_world_manifest(world_path)
    coordinate = payload["coordinate"]
    if not isinstance(coordinate, dict) or set(coordinate) != {
        "recipient",
        "representation",
        "condition",
    }:
        raise ValueError(
            "portability coordinate must name recipient, representation, and condition"
        )
    if not all(
        isinstance(coordinate[key], str) and coordinate[key]
        for key in coordinate
    ):
        raise ValueError("portability coordinate fields must be nonempty strings")
    representation = coordinate["representation"]
    condition = coordinate["condition"]
    if representation not in {"token", "byte", "semantic"}:
        raise ValueError("unknown portability representation")
    if condition not in {
        "adapter-tuned",
        "frozen-only",
        "joint",
        "native",
        "random",
        "corrupt",
        "disabled",
    }:
        raise ValueError("unsupported portability training condition")

    seed_key = str(config.seed)
    backbones = payload["initial_backbone"]
    memories = payload["memory"]
    seed_keys = {str(seed) for seed in _SEEDS}
    if not isinstance(backbones, dict) or set(backbones) != seed_keys:
        raise ValueError(
            "initial_backbone must be keyed by all declared decimal seed strings"
        )
    if not isinstance(memories, dict) or set(memories) != seed_keys:
        raise ValueError("memory must be keyed by all declared decimal seed strings")
    for candidate in backbones.values():
        if candidate is None:
            continue
        if not isinstance(candidate, dict) or set(candidate) != {
            "checkpoint",
            "checkpoint_sha256",
            "architecture_sha256",
            "tokenizer_sha256",
        }:
            raise ValueError("initial backbone entry has an invalid schema")
        _verify_directory(root, candidate["checkpoint"])
        for field in ("checkpoint_sha256", "architecture_sha256", "tokenizer_sha256"):
            _digest(candidate[field], field)
    backbone = backbones[seed_key]
    if (condition == "native") != (backbone is None):
        raise ValueError("only native-memory conditions may omit preparation backbones")

    evaluation_world_ids = {
        item["world_id"] for item in world["worlds"] if item["partition"] != "train"
    }
    for candidate in memories.values():
        if not isinstance(candidate, dict) or set(candidate) != {
            "kind",
            "artifact",
            "pack_id",
            "tensor_sha256",
            "addressing",
            "encoder_contract",
            "replacements",
        }:
            raise ValueError("memory entry has an invalid schema")
        if candidate["kind"] != representation:
            raise ValueError("memory kind differs from the declared representation")
        artifact_value = candidate["artifact"]
        native_local_table = condition == "native" and representation in {"token", "byte"}
        disabled = condition == "disabled"
        if (condition == "native" and representation == "semantic") or (
            artifact_value is None and not native_local_table and not disabled
        ):
            raise ValueError("portability memory artifact is required for this condition")
        memory_path = (
            None if artifact_value is None else _verify_asset(root, artifact_value)
        )
        if disabled:
            if any(
                candidate[field] is not None
                for field in ("pack_id", "tensor_sha256", "addressing", "encoder_contract")
            ):
                raise ValueError("disabled conditions cannot declare an attached memory asset")
        elif representation == "semantic":
            if (
                memory_path is None
                or candidate["pack_id"] is None
                or candidate["tensor_sha256"] is not None
                or not isinstance(candidate["encoder_contract"], dict)
                or candidate["addressing"] is not None
            ):
                raise ValueError("semantic memory must bind a verified pack and encoder contract")
            pack_report = verify_pack(
                memory_path,
                expected_pack_id=_digest(candidate["pack_id"], "pack_id"),
            )
            if not pack_report.valid:
                raise ValueError("portability semantic pack failed verification")
        else:
            if candidate["pack_id"] is not None or candidate["encoder_contract"] is not None:
                raise ValueError("lexical memory must not claim a semantic pack contract")
            if not isinstance(candidate["addressing"], dict):
                raise ValueError("lexical memory must bind its addressing identity")
            if artifact_value is None:
                if candidate["tensor_sha256"] is not None:
                    raise ValueError("native local memory must not pin a frozen table digest")
            else:
                _digest(candidate["tensor_sha256"], "memory tensor_sha256")
        replacements = candidate["replacements"]
        if not isinstance(replacements, dict):
            raise TypeError("memory replacements must be an object")
        if not disabled and condition != "native" and set(replacements) != evaluation_world_ids:
            raise ValueError("replacement memory inventory must cover every evaluation world")
        if (disabled or condition == "native") and replacements:
            raise ValueError("conditions without transferred memory cannot declare replacements")
        for replacement in replacements.values():
            if not isinstance(replacement, dict) or set(replacement) != {
                "artifact",
                "pack_id",
                "tensor_sha256",
            }:
                raise ValueError("replacement memory descriptor has an invalid schema")
            _verify_asset(root, replacement["artifact"])
            if representation == "semantic":
                _digest(replacement["pack_id"], "replacement pack_id")
                if replacement["tensor_sha256"] is not None:
                    raise ValueError("semantic replacements cannot declare tensor digests")
            else:
                if replacement["pack_id"] is not None:
                    raise ValueError("lexical replacements cannot declare pack ids")
                _digest(replacement["tensor_sha256"], "replacement tensor_sha256")
    memory = memories[seed_key]
    if condition != "native" and backbone is None:
        raise ValueError("transferred portability conditions require a preparation checkpoint")

    observations = payload["observations"]
    if not isinstance(observations, dict) or set(observations) != {
        "development",
        "final",
    }:
        raise ValueError("observations must reference development and final case manifests")
    for descriptor in observations.values():
        _verify_file(root, descriptor)
    training_fact_ids = payload["training_fact_ids"]
    if (
        not isinstance(training_fact_ids, list)
        or any(not isinstance(value, str) or not value for value in training_fact_ids)
        or len(training_fact_ids) != len(set(training_fact_ids))
    ):
        raise ValueError("training_fact_ids must be unique nonempty strings")
    training_world_ids = {
        fact_id
        for item in world["worlds"]
        if item["partition"] == "train"
        for fact_id in item["producer_fact_ids"]
    }
    if not set(training_fact_ids) <= training_world_ids:
        raise ValueError("portability manifest includes non-training-world facts")
    permitted_ids = {
        fact_id
        for item in world["worlds"]
        if item["partition"] == "train"
        for fact_id in (
            item["eligible_adapter_fact_ids"]
            if condition != "native"
            else item["producer_fact_ids"]
        )
    }
    if not set(training_fact_ids) <= permitted_ids:
        raise ValueError("portability manifest includes recipient-ineligible fact associations")
    tokenizer_digest = sha256_file(config.tokenizer.path)
    if backbone is not None and tokenizer_digest != backbone["tokenizer_sha256"]:
        raise ValueError(
            "recipient tokenizer differs from its preparation checkpoint identity"
        )
    return PortabilityRun(path, root, payload, backbone, memory)


def _nonmemory_name(name: str) -> bool:
    return not name.startswith(("memory.", "semantic_memories."))


def initialize_recipient_backbone(
    model: torch.nn.Module, config: RunConfig, run: PortabilityRun
) -> dict[str, torch.Tensor] | None:
    """Load only verified non-memory tensors from this recipient's own checkpoint."""
    if run.backbone is None:
        return None
    from sparselab.training.checkpoints import CheckpointManager

    descriptor = run.backbone["checkpoint"]
    checkpoint = _verify_directory(run.root, descriptor)
    manager = CheckpointManager(checkpoint.parents[1])
    snapshot = manager.load(checkpoint, mode="promote")
    if snapshot.checkpoint_sha256 != run.backbone["checkpoint_sha256"]:
        raise ValueError("recipient preparation checkpoint digest mismatch")
    checkpoint_manifest = json.loads(
        (checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    run_manifest_path = checkpoint.parents[1] / "manifest.json"
    run_manifest = read_manifest(run_manifest_path)
    run_manifest_digest = json.loads(run_manifest_path.read_text(encoding="utf-8"))[
        "sha256"
    ]
    if checkpoint_manifest.get("manifest_sha256") != run_manifest_digest:
        raise ValueError("recipient checkpoint is not bound to its run manifest")
    prepared_config = run_manifest.get("effective_config")
    if not isinstance(prepared_config, dict):
        raise TypeError("recipient run manifest has no effective configuration")
    if (
        architecture_sha256(prepared_config) != run.backbone["architecture_sha256"]
        or checkpoint_manifest.get("architecture_sha256")
        != run.backbone["architecture_sha256"]
    ):
        raise ValueError("recipient preparation architecture identity mismatch")
    checkpoint_config = RunConfig.model_validate(prepared_config)
    if checkpoint_config.attention.model_dump(mode="json") != config.attention.model_dump(mode="json"):
        raise ValueError("recipient preparation attention configuration mismatch")
    prepared_model = checkpoint_config.model.model_dump(mode="json")
    recipient_model = config.model.model_dump(mode="json")
    for field in _MEMORY_MODEL_FIELDS:
        prepared_model.pop(field, None)
        recipient_model.pop(field, None)
    if prepared_model != recipient_model:
        raise ValueError("recipient preparation backbone configuration mismatch")
    expected = {
        name: spec
        for name, spec in named_tensor_inventory(config.model, config.attention).items()
        if _nonmemory_name(name)
    }
    prepared = {name: tensor for name, tensor in snapshot.model.items() if _nonmemory_name(name)}
    if set(prepared) != set(expected):
        raise ValueError("recipient preparation non-memory tensor names or aliases differ")
    for name, spec in expected.items():
        if tuple(prepared[name].shape) != spec.shape:
            raise ValueError(f"recipient preparation tensor shape mismatch: {name}")
    model_state = model.state_dict()
    current_nonmemory = {name for name in model_state if _nonmemory_name(name)}
    if current_nonmemory != set(expected):
        raise ValueError("recipient model non-memory tensor inventory differs")
    result = model.load_state_dict(prepared, strict=False)
    allowed_missing = {name for name in model_state if not _nonmemory_name(name)}
    if result.unexpected_keys or set(result.missing_keys) != allowed_missing:
        raise ValueError("recipient checkpoint attachment inventory differs")
    return prepared


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def initialize_memory_artifact(
    model: torch.nn.Module, config: RunConfig, run: PortabilityRun
) -> None:
    """Load or verify the manifest-selected frozen memory attachment."""
    memory = run.memory
    kind = memory["kind"]
    descriptor = memory["artifact"]
    if descriptor is None:
        condition = run.payload["coordinate"]["condition"]
        if condition == "disabled":
            if config.model.memory != "none" or model.memory is not None:
                raise ValueError("disabled condition must not attach lexical memory")
            semantic_modules = getattr(model, "semantic_memories", None)
            if semantic_modules is not None and len(semantic_modules):
                raise ValueError("disabled condition must not attach semantic memory")
            return
        if condition != "native" or kind not in {"token", "byte"}:
            raise ValueError("this portability condition requires a frozen memory artifact")
        if kind == "token":
            if config.model.memory != "ngram" or model.memory is None:
                raise ValueError("native token memory requires a token n-gram attachment")
            expected = {
                "tokenizer_sha256": sha256_file(config.tokenizer.path),
                "order": config.model.memory_ngram_size,
                "hash_heads": config.model.memory_hash_heads,
                "rows": config.model.memory_table_size,
                "embedding_dim": config.model.memory_dim,
            }
        else:
            if config.model.memory != "byte" or model.memory is None:
                raise ValueError("native byte memory requires a local byte-table attachment")
            expected = {
                "format_version": 1,
                "normalization": "raw-utf8-v1",
                "hashing": "poly257-terminal-v1",
                "ngram_size": config.model.memory_ngram_size,
                "table_size": config.model.memory_table_size,
                "embedding_dim": config.model.memory_dim,
            }
        if memory["addressing"] != expected:
            raise ValueError("native memory addressing identity differs from recipient config")
        return

    artifact = _verify_asset(run.root, descriptor)
    if kind == "token":
        if config.model.memory != "ngram" or model.memory is None:
            raise ValueError("token portability requires a token n-gram memory attachment")
        loaded = load_file(artifact, device="cpu")
        if set(loaded) != {"memory.table.weight"}:
            raise ValueError("token memory safetensors must contain only memory.table.weight")
        table = loaded["memory.table.weight"]
        module_table = model.memory.table.weight
        if table.dtype != module_table.dtype or tuple(table.shape) != tuple(module_table.shape):
            raise ValueError("token memory table shape or dtype differs from recipient configuration")
        if _tensor_digest(table) != memory["tensor_sha256"]:
            raise ValueError("token memory table tensor digest mismatch")
        with torch.no_grad():
            module_table.copy_(table.to(device=module_table.device))
        expected = {
            "tokenizer_sha256": sha256_file(config.tokenizer.path),
            "order": config.model.memory_ngram_size,
            "hash_heads": config.model.memory_hash_heads,
            "rows": config.model.memory_table_size,
            "embedding_dim": config.model.memory_dim,
        }
        if memory["addressing"] != expected:
            raise ValueError("token memory addressing identity differs from recipient config")
        return
    if kind == "byte":
        from sparselab.model.portable_engram import load_portable_engram

        if config.model.memory != "byte" or model.memory is None:
            raise ValueError("transferred byte memory requires a byte-table attachment")
        package = load_portable_engram(
            artifact,
            expected_shape=(config.model.memory_table_size, config.model.memory_dim),
            expected_ngram_size=config.model.memory_ngram_size,
        )
        if package.manifest.table_sha256 != memory["tensor_sha256"]:
            raise ValueError("portable byte table tensor digest differs from manifest")
        table = model.memory.table.weight
        if tuple(table.shape) != tuple(package.table.shape) or table.dtype != package.table.dtype:
            raise ValueError("portable byte table shape or dtype differs from recipient configuration")
        if _tensor_digest(package.table) != memory["tensor_sha256"]:
            raise ValueError("portable byte table tensor digest mismatch")
        with torch.no_grad():
            table.copy_(package.table.to(device=table.device))
        expected = {
            "format_version": 1,
            "normalization": "raw-utf8-v1",
            "hashing": "poly257-terminal-v1",
            "ngram_size": config.model.memory_ngram_size,
            "table_size": config.model.memory_table_size,
            "embedding_dim": config.model.memory_dim,
        }
        if memory["addressing"] != expected:
            raise ValueError("portable byte addressing identity differs from recipient config")
        return
    if kind == "semantic":
        if "allocation" not in model.semantic_memories:
            raise ValueError("semantic portability requires the verified allocation attachment")
        adapter = model.semantic_memories["allocation"]
        retriever = adapter.retriever
        if retriever.pack_id != memory["pack_id"]:
            raise ValueError("attached semantic pack differs from manifest identity")
        contract = memory["encoder_contract"]
        actual = {
            "key_encoder": retriever.key_encoder.model_dump(mode="json"),
            "value_encoder": retriever.value_encoder.model_dump(mode="json"),
            "key_dim": retriever.key_dim,
            "value_dim": retriever.memory_dim,
            "normalization": retriever.key_normalization,
            "representation_space_id": retriever.space_id,
        }
        if contract != actual:
            raise ValueError("semantic encoder contract differs from verified pack")
        return
    raise ValueError(f"unsupported portability memory kind: {kind}")


def apply_trainable_parameter_filter(
    model: torch.nn.Module,
    config: RunConfig,
    run: PortabilityRun | None = None,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Freeze every nonselected storage and return optimizer parameters by canonical name."""
    names = config.training.trainable_parameters
    if names is None:
        return tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    inventory = named_tensor_inventory(
        config.model,
        config.attention,
        trainable_parameters=names,
    )
    if (
        run is not None
        and run.memory["kind"] == "token"
        and run.memory["artifact"] is not None
        and run.payload["coordinate"]["condition"] != "joint"
        and "memory.table.weight" in names
    ):
        raise ValueError("a transferred token table is trainable only in the joint condition")
    parameters = dict(model.named_parameters())
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name in names:
        if name not in parameters or not inventory[name].trainable:
            raise ValueError(
                f"configured trainable parameter is not a canonical model storage: {name}"
            )
        selected.append((name, parameters[name]))
    selected_names = set(names)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in selected_names)
    return tuple(selected)


def verify_portability_assets_unchanged(run: PortabilityRun) -> None:
    """Reverify every pinned run input after training or attachment-only evaluation."""
    payload = run.payload
    _verify_file(run.root, payload["protocol"])
    _verify_world_manifest(_verify_file(run.root, payload["world_manifest"]))
    for entry in payload["initial_backbone"].values():
        if entry is not None:
            _verify_directory(run.root, entry["checkpoint"])
    for entry in payload["memory"].values():
        if entry["artifact"] is not None:
            path = _verify_asset(run.root, entry["artifact"])
            if entry["pack_id"] is not None and not verify_pack(
                path, expected_pack_id=entry["pack_id"]
            ).valid:
                raise ValueError("pinned semantic pack changed after training")
        for replacement in entry["replacements"].values():
            path = _verify_asset(run.root, replacement["artifact"])
            if replacement["pack_id"] is not None and not verify_pack(
                path, expected_pack_id=replacement["pack_id"]
            ).valid:
                raise ValueError("pinned replacement semantic pack changed after training")
    for descriptor in payload["observations"].values():
        _verify_file(run.root, descriptor)
