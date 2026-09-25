"""PyTorch implementation of the engine-neutral training boundary."""

from __future__ import annotations

import hashlib
import random
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from typing import cast

import numpy as np
import torch
from torch.nn import functional

from sparselab.config.models import RunConfig
from sparselab.data.allocation import (
    OWNER_HYBRID,
    OWNER_LEXICAL,
    OWNER_NEURAL,
    OWNER_SEMANTIC,
    load_allocation_manifest,
    load_semantic_retriever,
)
from sparselab.data.packing import _tokenizer_sha256
from sparselab.data.tokenizer import load_tokenizer
from sparselab.engines.base import (
    CanonicalTensor,
    EngineNonFiniteError,
    EngineOutOfMemory,
    EngineState,
    EvaluationResult,
    Microbatch,
    UpdateResult,
    WeightSource,
)
from sparselab.engram.semantic import SemanticQueryBatch
from sparselab.memory import MemoryMonitor
from sparselab.model.inspection import architecture_metrics
from sparselab.model.transformer import DenseLM
from sparselab.research.portability import (
    PortabilityRun,
    apply_trainable_parameter_filter,
    initialize_memory_artifact,
    initialize_recipient_backbone,
    load_portability_manifest,
    verify_portability_assets_unchanged,
)
from sparselab.runtime import (
    make_grad_scaler,
    precision_context,
    synchronize,
    torch_device_for,
    validate_runtime,
)
from sparselab.training.manifest import source_identity
from sparselab.training.offload import ActivationOffload
from sparselab.training.optimizer import (
    learning_rate_for_step,
    make_adafactor,
    make_optimizer,
)


def _rng_state(device: torch.device, backend: str) -> dict[str, object]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    if backend in {"cuda", "rocm"}:
        device_rng = torch.cuda.get_rng_state(device)
    elif backend == "mps":
        device_rng = torch.mps.get_rng_state()
    elif backend == "xpu":
        device_rng = torch.xpu.get_rng_state(device)
    else:
        device_rng = None
    return {
        "python": python_state,
        "numpy_kind": numpy_state[0],
        "numpy_keys": torch.from_numpy(numpy_state[1].copy()),
        "numpy_pos": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch": torch.get_rng_state(),
        "device_type": backend,
        "device_index": device.index or 0,
        "device_rng": device_rng,
    }


def _restore_rng(state: Mapping[str, object], device: torch.device) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    keys = state["numpy_keys"]
    np.random.set_state(
        (
            str(state["numpy_kind"]),
            keys.numpy(),  # type: ignore[union-attr]
            int(state["numpy_pos"]),
            int(state["numpy_has_gauss"]),
            float(state["numpy_cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    backend = state.get("device_type", "cpu")
    if backend in {"cuda", "rocm"}:
        torch.cuda.set_rng_state(state["device_rng"], device)  # type: ignore[arg-type]
    elif backend == "mps":
        torch.mps.set_rng_state(state["device_rng"])  # type: ignore[arg-type]
    elif backend == "xpu":
        torch.xpu.set_rng_state(state["device_rng"], device)  # type: ignore[arg-type]


def _is_allocation_failure(error: BaseException) -> bool:
    """Recognize only PyTorch/backend allocation failures, preserving other errors."""
    if isinstance(error, (torch.OutOfMemoryError, MemoryError)):
        return True
    return isinstance(error, RuntimeError) and bool(
        re.search(
            r"\b(out of memory|cannot allocate memory|can't allocate memory|"
            r"allocation (?:failed|failure))\b",
            str(error),
            re.IGNORECASE,
        )
    )


@contextmanager
def _translate_allocation_failures() -> Iterator[None]:
    try:
        yield
    except (RuntimeError, MemoryError) as error:
        if _is_allocation_failure(error):
            raise EngineOutOfMemory(f"PyTorch allocation failed: {error}") from error
        raise


def _named_optimizer_state(
    model: DenseLM,
    optimizer: torch.optim.Optimizer,
    saved_state: dict[str, object],
    saved_parameter_names: dict[int, str] | list[list[str]],
) -> dict[str, object]:
    """Validate and remap saved slots by canonical parameter name before loading.

    ``Optimizer.load_state_dict`` otherwise binds state positionally.  Reordered
    registration is permissible under recorded runtime drift, but equal-shaped
    AdamW/Adafactor slots must never silently follow that positional order.
    """
    if not isinstance(saved_parameter_names, dict):
        raise TypeError("PyTorch optimizer parameter names must be an ID-to-name map")
    saved_names: dict[int, str] = {}
    for saved_id, name in saved_parameter_names.items():
        if isinstance(saved_id, bool) or not isinstance(saved_id, (int, str)):
            raise TypeError("saved optimizer parameter ID is invalid")
        try:
            normalized_id = int(saved_id)
        except ValueError as error:
            raise ValueError("saved optimizer parameter ID is invalid") from error
        if normalized_id in saved_names or not isinstance(name, str) or not name:
            raise ValueError("saved optimizer parameter names are invalid")
        saved_names[normalized_id] = name

    if len(set(saved_names.values())) != len(saved_names):
        raise ValueError("saved optimizer parameter names are not unique")
    saved_groups = saved_state.get("param_groups")
    saved_slots = saved_state.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_slots, dict):
        raise TypeError("saved optimizer state is malformed")
    live_state = optimizer.state_dict()
    live_groups = live_state.get("param_groups")
    if not isinstance(live_groups, list) or len(saved_groups) != len(live_groups):
        raise ValueError("saved optimizer parameter group count differs")

    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    live_ids_to_names: dict[int, str] = {}
    live_names_to_ids: dict[str, int] = {}
    live_names_to_parameters: dict[str, torch.nn.Parameter] = {}
    for serialized_group, live_group in zip(
        live_groups, optimizer.param_groups, strict=True
    ):
        serialized_ids = serialized_group.get("params")
        parameters = live_group.get("params")
        if not isinstance(serialized_ids, list) or not isinstance(parameters, list):
            raise TypeError("live optimizer parameter group is malformed")
        if len(serialized_ids) != len(parameters):
            raise ValueError("live optimizer parameter group is inconsistent")
        for serialized_id, parameter in zip(serialized_ids, parameters, strict=True):
            name = parameter_names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer includes an unnamed parameter")
            live_ids_to_names[int(serialized_id)] = name
            live_names_to_ids[name] = int(serialized_id)
            live_names_to_parameters[name] = parameter

    if set(saved_names.values()) != set(live_ids_to_names.values()):
        raise ValueError("saved optimizer parameter names differ from the live model")
    saved_ids_in_groups: set[int] = set()
    saved_group_names: list[set[str]] = []
    for group in saved_groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise TypeError("saved optimizer parameter group is malformed")
        group_ids = group["params"]
        if any(not isinstance(value, int) for value in group_ids):
            raise TypeError("saved optimizer parameter group IDs are invalid")
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("saved optimizer parameter group repeats a parameter")
        saved_ids_in_groups.update(group_ids)
        try:
            saved_group_names.append({saved_names[value] for value in group_ids})
        except KeyError as error:
            raise ValueError("saved optimizer group lacks a parameter name") from error
    if saved_ids_in_groups != set(saved_names):
        raise ValueError("saved optimizer names and groups disagree")

    unused_live_groups = set(range(len(live_groups)))
    saved_to_live_group: list[int] = []
    for names in saved_group_names:
        matches = [
            index
            for index in unused_live_groups
            if {live_ids_to_names[int(value)] for value in live_groups[index]["params"]}
            == names
        ]
        if len(matches) != 1:
            raise ValueError("saved optimizer parameter group membership differs")
        match = matches[0]
        unused_live_groups.remove(match)
        saved_to_live_group.append(match)

    remapped_slots: dict[int, object] = {}
    for saved_id, slots in saved_slots.items():
        if not isinstance(saved_id, int) or saved_id not in saved_names:
            raise ValueError("saved optimizer state has an unknown parameter")
        if not isinstance(slots, dict):
            raise TypeError("saved optimizer parameter state is malformed")
        name = saved_names[saved_id]
        parameter = live_names_to_parameters[name]
        factor_shapes = {}
        if parameter.ndim >= 2:
            factor_shapes = {
                "row_var": (*parameter.shape[:-1], 1),
                "col_var": (*parameter.shape[:-2], 1, parameter.shape[-1]),
            }
        for slot_name, value in slots.items():
            if not isinstance(slot_name, str):
                raise TypeError("saved optimizer state key is invalid")
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            shape = tuple(value.shape)
            if shape == tuple(parameter.shape):
                continue
            if factor_shapes.get(slot_name) == shape:
                continue
            raise ValueError(
                f"optimizer state {slot_name!r} shape {shape} does not match {name!r}"
            )
        live_id = live_names_to_ids[name]
        remapped_slots[live_id] = slots

    remapped_groups: list[dict[str, object]] = []
    for live_group_index in range(len(live_groups)):
        saved_group_index = saved_to_live_group.index(live_group_index)
        remapped_group = dict(saved_groups[saved_group_index])
        remapped_group["params"] = list(live_groups[live_group_index]["params"])
        remapped_groups.append(remapped_group)
    return {"state": remapped_slots, "param_groups": remapped_groups}


class _PyTorchWeights(WeightSource):
    def __init__(self, model: DenseLM) -> None:
        self._model = model
        self._trainable = {
            name: parameter.requires_grad
            for name, parameter in model.named_parameters(remove_duplicate=False)
        }
        aliases: dict[str, str] = {}
        seen: dict[tuple[int, int, tuple[int, ...], torch.dtype], str] = {}
        for name, tensor in model.state_dict().items():
            key = (
                tensor.data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tensor.dtype,
            )
            if key in seen:
                aliases[name] = seen[key]
            else:
                seen[key] = name
        self._aliases = aliases

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._aliases

    def tensors(self) -> Iterator[CanonicalTensor]:
        for name, tensor in self._model.state_dict().items():
            if name in self._aliases:
                continue
            array = tensor.detach().cpu().contiguous().numpy()
            yield CanonicalTensor(name, array, self._trainable.get(name, False))


def _parameter_sha256(parameter: torch.nn.Parameter) -> str:
    raw = parameter.detach().to(device="cpu").contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


class PyTorchEngine:
    """The former trainer compute path, without lifecycle or checkpoint ownership."""

    def __init__(self) -> None:
        self.config: RunConfig | None = None
        self.runtime = None
        self.device: torch.device | None = None
        self.model: DenseLM | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.monitor: MemoryMonitor | None = None
        self.offload: ActivationOffload | None = None
        self.semantic_encoder = None
        self.portability_run: PortabilityRun | None = None
        self._portability_parameters: tuple[tuple[str, torch.nn.Parameter], ...] = ()
        self._last_gradient_parameter_names: tuple[str, ...] = ()
        self._last_update_parameter_names: tuple[str, ...] = ()
        self._portability_initial_digests: dict[str, str] = {}
        self._portability_update_history: list[dict[str, object]] = []

    def validate(self, config: RunConfig):
        self.runtime = validate_runtime(config)
        return self.runtime

    def initialize(
        self, config: RunConfig, initial_weights: Mapping[str, object] | None = None
    ) -> None:
        runtime = self.runtime if self.runtime is not None else self.validate(config)
        portability_run = (
            load_portability_manifest(config)
            if config.training.portability_manifest_path is not None
            else None
        )
        if portability_run is not None and initial_weights is not None:
            raise ValueError(
                "portability initialization must come from its verified manifest"
            )
        device = torch_device_for(runtime.backend, config.runtime.device_index)
        model = DenseLM(config.model, config.attention).to(device)
        self.semantic_encoder = None
        if config.dataset.allocation_manifest_path is not None:
            allocation = load_allocation_manifest(
                config.dataset.allocation_manifest_path,
                source_identity_sha256=cast(str, source_identity()["sha256"]),
                tokenizer_sha256=_tokenizer_sha256(
                    load_tokenizer(config.tokenizer.path)
                ),
            )
            retriever = load_semantic_retriever(allocation)
            if (retriever is None) != (config.model.semantic_memory_dim is None):
                raise ValueError(
                    "model.semantic_memory_dim must match the verified allocation pack"
                )
            if retriever is not None:
                if retriever.memory_dim != config.model.semantic_memory_dim:
                    raise ValueError(
                        "model.semantic_memory_dim differs from semantic pack value width"
                    )
                model.add_semantic_memory(
                    "allocation",
                    retriever,
                    site="final",
                    min_score=3.5 if portability_run is not None else None,
                )
                self.semantic_encoder = retriever.key_encoder
        if portability_run is not None:
            initialize_recipient_backbone(model, config, portability_run)
            initialize_memory_artifact(model, config, portability_run)
        elif initial_weights is not None:
            model.load_state_dict(initial_weights)  # type: ignore[arg-type]
        trainable_parameters = apply_trainable_parameter_filter(
            model, config, portability_run
        )
        optimizer = (
            make_optimizer(
                model,
                config.optimizer.peak,
                config.optimizer.weight_decay,
                config.optimizer.betas,
                config.optimizer.eps,
            )
            if config.optimizer.name == "adamw"
            else make_adafactor(
                model,
                config.optimizer.peak,
                config.optimizer.weight_decay,
                config.optimizer.beta2_decay,
                config.optimizer.eps,
                config.optimizer.d,
            )
        )
        selected_ids = {id(parameter) for _, parameter in trainable_parameters}
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != selected_ids:
            raise RuntimeError(
                "optimizer parameters differ from the explicit trainable inventory"
            )
        self.config, self.runtime, self.device = config, runtime, device
        self.model, self.optimizer = model, optimizer
        self.portability_run = portability_run
        self._portability_parameters = trainable_parameters
        self._last_gradient_parameter_names = ()
        self._portability_initial_digests = (
            {
                name: _parameter_sha256(parameter)
                for name, parameter in model.named_parameters()
            }
            if portability_run is not None
            else {}
        )
        self.scaler = make_grad_scaler(config, device)
        self.monitor = MemoryMonitor(device, runtime_info=runtime)
        self.offload = (
            ActivationOffload(device, model)
            if config.runtime.memory.activation_offload.enabled
            else None
        )

    def _ready(
        self,
    ) -> tuple[RunConfig, torch.device, DenseLM, torch.optim.Optimizer, MemoryMonitor]:
        if (
            self.config is None
            or self.device is None
            or self.model is None
            or self.optimizer is None
            or self.monitor is None
        ):
            raise RuntimeError("PyTorchEngine is not initialized")
        return self.config, self.device, self.model, self.optimizer, self.monitor

    def portability_audit(self) -> dict[str, object]:
        if self.portability_run is None or self.model is None:
            raise RuntimeError("no portability run is initialized")
        final_digests = {
            name: _parameter_sha256(parameter)
            for name, parameter in self.model.named_parameters()
        }
        selected = {name for name, _ in self._portability_parameters}
        changed = {
            name
            for name, digest in final_digests.items()
            if self._portability_initial_digests.get(name) != digest
        }
        frozen_changes = sorted(changed - selected)
        if frozen_changes:
            raise RuntimeError(
                f"frozen parameters changed during portability run: {frozen_changes}"
            )
        backbone_names = sorted(
            name
            for name in self._portability_initial_digests
            if not name.startswith(("memory.", "semantic_memories."))
        )
        backbone_unchanged = not (changed & set(backbone_names))
        verify_portability_assets_unchanged(self.portability_run)
        return {
            "trainable_parameter_names": sorted(selected),
            "gradient_update_history": self._portability_update_history.copy(),
            "updated_parameter_names": sorted(changed),
            "frozen_parameters_unchanged": True,
            "backbone_unchanged": backbone_unchanged,
            "initial_parameter_sha256": self._portability_initial_digests.copy(),
            "final_parameter_sha256": final_digests,
            "assets_unchanged": True,
        }



    def train_update(
        self, microbatches: list[Microbatch], update_index: int, valid_targets: int
    ) -> UpdateResult:
        with _translate_allocation_failures():
            return self._train_update(microbatches, update_index, valid_targets)

    def _train_update(
        self, microbatches: list[Microbatch], update_index: int, valid_targets: int
    ) -> UpdateResult:
        config, device, model, optimizer, monitor = self._ready()
        chunk_counts = [
            int(np.count_nonzero(batch.targets != -100)) for batch in microbatches
        ]
        observed_targets = sum(chunk_counts)
        if observed_targets <= 0:
            raise ValueError("update needs at least one supervised target")
        if valid_targets != observed_targets:
            raise ValueError(
                "update valid-target count differs from the supplied microbatches"
            )
        window_rng = _rng_state(device, config.runtime.backend) if self.scaler else None
        prior_lrs = [group["lr"] for group in optimizer.param_groups]
        lr = learning_rate_for_step(
            update_index,
            config.training.max_steps,
            config.optimizer.warmup_steps,
            config.optimizer.peak,
            config.optimizer.floor,
        )
        synchronize(device)
        started = time.perf_counter()
        monitor.begin_update()
        if self.offload is not None and not self.offload.reset():
            raise RuntimeError("previous update retained offloaded activation storage")
        optimizer.zero_grad(set_to_none=True)
        language_sum = torch.zeros((), device=device)
        aux_sum = torch.zeros((), device=device)
        diagnostic_sums: dict[str, torch.Tensor] = {}
        losses_finite = torch.ones((), device=device, dtype=torch.bool)
        recomputed = 0.0
        executed_microbatches = 0
        allocation_enabled = config.dataset.allocation_manifest_path is not None
        portability_enabled = self.portability_run is not None
        allocation_raw_counts = {
            code: 0
            for code in (OWNER_NEURAL, OWNER_LEXICAL, OWNER_SEMANTIC, OWNER_HYBRID)
        }
        allocation_target_counts = dict(allocation_raw_counts)
        allocation_groups: dict[int, list[torch.nn.Parameter]] | None = None
        if allocation_enabled:
            neural, lexical, semantic = [], [], []
            seen: set[int] = set()
            for name, parameter in model.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if not parameter.requires_grad:
                    continue
                if name.startswith("memory."):
                    lexical.append(parameter)
                elif name.startswith("semantic_memories."):
                    semantic.append(parameter)
                else:
                    neural.append(parameter)
            allocation_groups = {
                OWNER_NEURAL: neural,
                OWNER_LEXICAL: lexical,
                OWNER_SEMANTIC: semantic,
            }
        for batch, chunk_valid in zip(microbatches, chunk_counts, strict=True):
            if allocation_enabled:
                if (
                    batch.owner_ids is None
                    or batch.owner_ids.shape != batch.targets.shape
                    or batch.owner_ids.dtype != np.uint8
                    or np.any(batch.owner_ids > OWNER_HYBRID)
                ):
                    raise ValueError("owner sidecar shape, dtype, or codes are invalid")
                for owner in allocation_raw_counts:
                    allocation_raw_counts[owner] += int(
                        np.count_nonzero(batch.owner_ids == owner)
                    )
            if not chunk_valid:
                continue
            executed_microbatches += 1
            x, y = (
                torch.from_numpy(batch.inputs).to(device),
                torch.from_numpy(batch.targets).to(device),
            )
            addresses = (
                None
                if batch.byte_addresses is None
                else torch.from_numpy(batch.byte_addresses).to(device)
            )
            owners = None
            owner_tensor = None
            memory_mask = None
            semantic_queries = None
            if allocation_enabled:
                if batch.owner_ids is None:
                    raise ValueError("allocation training requires owner sidecars")
                owners = batch.owner_ids
                if (
                    owners.shape != batch.targets.shape
                    or owners.dtype != np.uint8
                    or np.any(owners > OWNER_HYBRID)
                ):
                    raise ValueError("owner sidecar shape, dtype, or codes are invalid")
                owner_tensor = torch.from_numpy(owners).to(device)
                for owner in allocation_target_counts:
                    allocation_target_counts[owner] += int(
                        np.count_nonzero((owners == owner) & (batch.targets != -100))
                    )
                semantic_width = (
                    next(iter(model.semantic_memories.values())).retriever.key_dim
                    if model.semantic_memories
                    else None
                )
                assert allocation_groups is not None
                if (
                    allocation_target_counts[OWNER_LEXICAL]
                    and not allocation_groups[OWNER_LEXICAL]
                ):
                    raise ValueError(
                        "lexical ownership requires a lexical memory module"
                    )
                if (
                    allocation_target_counts[OWNER_SEMANTIC]
                    and not allocation_groups[OWNER_SEMANTIC]
                ):
                    raise ValueError(
                        "semantic ownership requires a verified semantic memory"
                    )
                if allocation_target_counts[OWNER_HYBRID] and (
                    not allocation_groups[OWNER_LEXICAL]
                    or not allocation_groups[OWNER_SEMANTIC]
                ):
                    raise ValueError(
                        "hybrid ownership requires lexical and semantic memory"
                    )
                memory_mask = (owner_tensor == OWNER_LEXICAL) | (
                    owner_tensor == OWNER_HYBRID
                )
                if (
                    batch.semantic_queries is not None
                    or batch.semantic_mask is not None
                ):
                    if (
                        self.semantic_encoder is None
                        or batch.semantic_queries is None
                        or batch.semantic_mask is None
                        or batch.semantic_queries.ndim != 3
                        or batch.semantic_queries.shape
                        != (*batch.inputs.shape, semantic_width)
                        or batch.semantic_queries.dtype != np.float32
                        or batch.semantic_mask.shape != batch.inputs.shape
                        or batch.semantic_mask.dtype != bool
                        or not np.isfinite(batch.semantic_queries).all()
                    ):
                        raise ValueError("semantic allocation sidecars are invalid")
                    semantic_mask = torch.from_numpy(batch.semantic_mask).to(device)
                    allowed = (owner_tensor == OWNER_SEMANTIC) | (
                        owner_tensor == OWNER_HYBRID
                    )
                    if bool(torch.any(semantic_mask & ~allowed)):
                        raise ValueError(
                            "semantic query mask may select only semantic or hybrid ownership"
                        )
                    semantic_queries = SemanticQueryBatch(
                        self.semantic_encoder,
                        torch.from_numpy(batch.semantic_queries).to(device),
                        semantic_mask,
                    )
                elif allocation_groups[OWNER_SEMANTIC]:
                    raise ValueError("semantic ownership requires query sidecars")
            elif batch.semantic_queries is not None:
                if self.semantic_encoder is None or batch.semantic_mask is None:
                    raise ValueError(
                        "semantic query sidecars require a verified attached pack"
                    )
                semantic_queries = SemanticQueryBatch(
                    self.semantic_encoder,
                    torch.from_numpy(batch.semantic_queries).to(device),
                    torch.from_numpy(batch.semantic_mask).to(device),
                )
            valid_target_mask = y != -100
            neural_aux_targets = 0
            if allocation_enabled:
                assert owner_tensor is not None
                if not portability_enabled:
                    neural_target_mask = (owner_tensor == OWNER_NEURAL) | (
                        owner_tensor == OWNER_HYBRID
                    )
                    valid_target_mask = valid_target_mask & neural_target_mask
                neural_aux_targets = int(valid_target_mask.sum())
            with (
                self.offload.hooks() if self.offload is not None else nullcontext(),
                precision_context(config, device),
            ):
                logits, auxiliary = model.forward_with_aux(
                    x,
                    byte_addresses=addresses,
                    semantic_queries=semantic_queries,
                    memory_mask=memory_mask,
                    valid_target_mask=valid_target_mask,
                    activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                    diagnostics=config.logging.architecture_diagnostics,
                )
            diagnostics = architecture_metrics(model)
            monitor.sample("forward")
            ce_sum = functional.cross_entropy(
                logits.float().flatten(0, 1),
                y.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
            auxiliary = auxiliary.float()
            losses_finite = (
                losses_finite & torch.isfinite(ce_sum) & torch.isfinite(auxiliary)
            )
            if allocation_enabled and not portability_enabled:
                assert owner_tensor is not None and allocation_groups is not None
                terms: list[tuple[torch.Tensor, list[torch.nn.Parameter]]] = []
                for owner, parameters in allocation_groups.items():
                    mask = (owner_tensor == owner) | (owner_tensor == OWNER_HYBRID)
                    count = int(mask.logical_and(y != -100).sum())
                    weight = (
                        config.training.neural_loss_weight
                        if owner == OWNER_NEURAL
                        else 1.0
                    )
                    if count and parameters and weight:
                        labels = y.masked_fill(~mask, -100)
                        component = functional.cross_entropy(
                            logits.float().flatten(0, 1),
                            labels.flatten(),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        terms.append((component * (weight / valid_targets), parameters))
                if (
                    config.training.neural_loss_weight
                    and neural_aux_targets
                    and auxiliary.requires_grad
                ):
                    terms.append(
                        (
                            auxiliary
                            * neural_aux_targets
                            * config.training.neural_loss_weight
                            / valid_targets,
                            allocation_groups[OWNER_NEURAL],
                        )
                    )
                for index, (term, parameters) in enumerate(terms):
                    if self.scaler is not None:
                        term = self.scaler.scale(term)
                    torch.autograd.backward(
                        term,
                        inputs=parameters,
                        retain_graph=index + 1 < len(terms),
                    )
            else:
                backward_loss = (ce_sum + auxiliary * chunk_valid) / valid_targets
                if self.scaler is not None:
                    backward_loss = self.scaler.scale(backward_loss)
                torch.autograd.backward(
                    backward_loss,
                    inputs=[
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                )
            monitor.sample("backward")
            recomputed += model.recomputed_block_call_ratio
            language_sum = language_sum + ce_sum.detach()
            if allocation_enabled and not portability_enabled:
                aux_sum = (
                    aux_sum
                    + auxiliary.detach()
                    * neural_aux_targets
                    * config.training.neural_loss_weight
                )
            else:
                aux_sum = aux_sum + auxiliary.detach() * chunk_valid
            for name, value in diagnostics.items():
                diagnostic_sums[name] = (
                    diagnostic_sums.get(name, value.new_zeros(())) + value * chunk_valid
                )
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        named_parameters = dict(model.named_parameters())
        gradient_names = tuple(
            name
            for name, parameter in named_parameters.items()
            if parameter.grad is not None
        )
        if config.training.trainable_parameters is not None:
            allowed_names = set(config.training.trainable_parameters)
            unexpected_gradients = set(gradient_names) - allowed_names
            if unexpected_gradients:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "gradients escaped the explicit trainable inventory: "
                    f"{sorted(unexpected_gradients)}"
                )
        self._last_gradient_parameter_names = gradient_names
        if portability_enabled:
            norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in self._portability_parameters],
                config.training.grad_clip_norm,
                error_if_nonfinite=False,
                foreach=False,
            )
        elif allocation_enabled:
            assert allocation_groups is not None
            norms = [
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    config.training.grad_clip_norm,
                    error_if_nonfinite=False,
                    foreach=False,
                )
                for parameters in allocation_groups.values()
                if parameters
            ]
            norm = torch.linalg.vector_norm(
                torch.stack([value.float() for value in norms])
            )
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.grad_clip_norm,
                error_if_nonfinite=False,
                foreach=False,
            )
        gradients_finite = torch.isfinite(norm) if self.scaler is None else None
        # One pre-step boundary checks the accumulated loss and gradients before
        # optimizer mutation. No diagnostic scalar is read per microbatch.
        synchronize(device)
        if not bool(losses_finite):
            optimizer.zero_grad(set_to_none=True)
            raise EngineNonFiniteError("nonfinite training loss")
        if gradients_finite is not None and not bool(gradients_finite):
            optimizer.zero_grad(set_to_none=True)
            raise EngineNonFiniteError("nonfinite unscaled gradients")
        for group in optimizer.param_groups:
            group["lr"] = lr
        overflow = False
        if self.scaler is None:
            optimizer.step()
        else:
            old_scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            overflow = self.scaler.get_scale() < old_scale
        monitor.sample("optimizer")
        synchronize(device)
        memory_metrics = monitor.end_update()
        elapsed = time.perf_counter() - started
        if overflow:
            optimizer.zero_grad(set_to_none=True)
            for group, old_lr in zip(optimizer.param_groups, prior_lrs, strict=True):
                group["lr"] = old_lr
            assert window_rng is not None
            _restore_rng(window_rng, device)
            return UpdateResult("OVERFLOW", None, valid_targets, 0, elapsed)
        self._last_update_parameter_names = gradient_names
        if portability_enabled:
            self._portability_update_history.append(
                {
                    "step": update_index,
                    "gradient_parameter_names": list(gradient_names),
                    "update_parameter_names": list(gradient_names),
                }
            )
        metrics = {
            "train/loss": float(language_sum) / valid_targets,
            "moe/router_auxiliary_loss": float(aux_sum) / valid_targets,
            "optimizer/learning_rate": lr,
            "optimizer/grad_norm": float(norm),
            "performance/step_seconds": elapsed,
            "performance/tokens_per_second": valid_targets / max(elapsed, 1e-9),
            "batch/micro_batch_size": float(config.training.micro_batch_size),
            "batch/accumulation_steps": float(len(microbatches)),
            "batch/effective_batch_size": float(
                sum(batch.inputs.shape[0] for batch in microbatches)
            ),
            "batch/effective_tokens_per_update": float(valid_targets),
            "recompute/block_call_ratio": recomputed / executed_microbatches,
            **{
                name: float(value) / valid_targets
                for name, value in diagnostic_sums.items()
            },
            **memory_metrics,
        }
        if allocation_enabled:
            neural_targets = (
                allocation_target_counts[OWNER_NEURAL]
                + allocation_target_counts[OWNER_HYBRID]
            )
            metrics.update(
                {
                    "allocation/raw_tokens": float(sum(allocation_raw_counts.values())),
                    "allocation/valid_targets": float(
                        sum(allocation_target_counts.values())
                    ),
                    "allocation/weighted_neural_supervision_mass": (
                        config.training.neural_loss_weight * neural_targets
                    ),
                    "allocation/owner/neural_raw_tokens": float(
                        allocation_raw_counts[OWNER_NEURAL]
                    ),
                    "allocation/owner/lexical_raw_tokens": float(
                        allocation_raw_counts[OWNER_LEXICAL]
                    ),
                    "allocation/owner/semantic_raw_tokens": float(
                        allocation_raw_counts[OWNER_SEMANTIC]
                    ),
                    "allocation/owner/hybrid_raw_tokens": float(
                        allocation_raw_counts[OWNER_HYBRID]
                    ),
                    "allocation/owner/neural_targets": float(
                        allocation_target_counts[OWNER_NEURAL]
                    ),
                    "allocation/owner/lexical_targets": float(
                        allocation_target_counts[OWNER_LEXICAL]
                    ),
                    "allocation/owner/semantic_targets": float(
                        allocation_target_counts[OWNER_SEMANTIC]
                    ),
                    "allocation/owner/hybrid_targets": float(
                        allocation_target_counts[OWNER_HYBRID]
                    ),
                }
            )
        if self.scaler is not None:
            metrics["optimizer/loss_scale"] = self.scaler.get_scale()
        if self.offload is not None:
            metrics.update(self.offload.metrics.as_metrics(elapsed))
        return UpdateResult(
            "APPLIED",
            float(language_sum),
            valid_targets,
            valid_targets,
            elapsed,
            metrics,
        )

    def evaluate(self, batches: Iterable[Microbatch]) -> EvaluationResult:
        with _translate_allocation_failures():
            return self._evaluate(batches)

    def _evaluate(self, batches: Iterable[Microbatch]) -> EvaluationResult:
        config, device, model, _, _ = self._ready()
        was_training = model.training
        rng = _rng_state(device, config.runtime.backend)
        total_loss, total_targets, count = 0.0, 0, 0
        try:
            model.eval()
            with torch.no_grad(), precision_context(config, device):
                for batch in batches:
                    x = torch.from_numpy(batch.inputs).to(device)
                    y = torch.from_numpy(batch.targets).to(device)
                    addresses = (
                        None
                        if batch.byte_addresses is None
                        else torch.from_numpy(batch.byte_addresses).to(device)
                    )
                    semantic_queries = None
                    memory_mask = None
                    if config.dataset.allocation_manifest_path is not None:
                        if (
                            batch.owner_ids is None
                            or batch.owner_ids.shape != batch.targets.shape
                            or batch.owner_ids.dtype != np.uint8
                            or np.any(batch.owner_ids > OWNER_HYBRID)
                        ):
                            raise ValueError(
                                "allocation evaluation requires valid owner sidecars"
                            )
                        owners = torch.from_numpy(batch.owner_ids).to(device)
                        memory_mask = (owners == OWNER_LEXICAL) | (
                            owners == OWNER_HYBRID
                        )
                        semantic_width = (
                            next(
                                iter(model.semantic_memories.values())
                            ).retriever.key_dim
                            if model.semantic_memories
                            else None
                        )
                        if (
                            batch.semantic_queries is not None
                            or batch.semantic_mask is not None
                        ):
                            if (
                                self.semantic_encoder is None
                                or batch.semantic_queries is None
                                or batch.semantic_mask is None
                                or batch.semantic_queries.ndim != 3
                                or batch.semantic_queries.shape
                                != (*batch.inputs.shape, semantic_width)
                                or batch.semantic_queries.dtype != np.float32
                                or batch.semantic_mask.shape != batch.inputs.shape
                                or batch.semantic_mask.dtype != bool
                                or not np.isfinite(batch.semantic_queries).all()
                            ):
                                raise ValueError(
                                    "semantic allocation sidecars are invalid"
                                )
                            semantic_mask = torch.from_numpy(batch.semantic_mask).to(
                                device
                            )
                            allowed = (owners == OWNER_SEMANTIC) | (
                                owners == OWNER_HYBRID
                            )
                            if bool(torch.any(semantic_mask & ~allowed)):
                                raise ValueError(
                                    "semantic query mask may select only semantic or hybrid ownership"
                                )
                            semantic_queries = SemanticQueryBatch(
                                self.semantic_encoder,
                                torch.from_numpy(batch.semantic_queries).to(device),
                                semantic_mask,
                            )
                    elif batch.semantic_queries is not None:
                        if self.semantic_encoder is None or batch.semantic_mask is None:
                            raise ValueError(
                                "semantic query sidecars require a verified attached pack"
                            )
                        semantic_queries = SemanticQueryBatch(
                            self.semantic_encoder,
                            torch.from_numpy(batch.semantic_queries).to(device),
                            torch.from_numpy(batch.semantic_mask).to(device),
                        )
                    logits = model(
                        x,
                        byte_addresses=addresses,
                        semantic_queries=semantic_queries,
                        memory_mask=memory_mask,
                    )
                    total_loss += float(
                        functional.cross_entropy(
                            logits.float().flatten(0, 1),
                            y.flatten(),
                            ignore_index=-100,
                            reduction="sum",
                        )
                    )
                    total_targets += int(np.count_nonzero(batch.targets != -100))
                    count += 1
        finally:
            model.train(was_training)
            _restore_rng(rng, device)
        return EvaluationResult(total_loss, total_targets, count)

    def export_weights(self) -> WeightSource:
        _, _, model, _, _ = self._ready()
        return _PyTorchWeights(model)

    def export_training_state(self) -> EngineState:
        _, device, model, optimizer, _ = self._ready()
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        optimizer_state = optimizer.state_dict()
        parameter_names = {
            stored_id: names[id(parameter)]
            for stored_group, live_group in zip(
                optimizer_state["param_groups"], optimizer.param_groups, strict=True
            )
            for stored_id, parameter in zip(
                stored_group["params"], live_group["params"], strict=True
            )
        }
        return EngineState(
            optimizer_state,
            _rng_state(device, self.config.runtime.backend),
            self.scaler.state_dict() if self.scaler is not None else None,
            parameter_names,  # type: ignore[arg-type]
        )

    def restore_training_state(self, state: EngineState) -> None:
        _, device, model, optimizer, _ = self._ready()
        from sparselab.training.checkpoints import restore_optimizer_state

        remapped_optimizer_state = _named_optimizer_state(
            model,
            optimizer,
            state.optimizer,
            state.optimizer_parameter_names,
        )
        restore_optimizer_state(optimizer, remapped_optimizer_state)
        if self.scaler is not None:
            if state.scaler is None:
                raise ValueError("fp16 resume requires saved GradScaler state")
            self.scaler.load_state_dict(state.scaler)
        _restore_rng(state.rng, device)

    def memory_snapshot(self) -> dict[str, float]:
        _, _, _, _, monitor = self._ready()
        return {
            key: float(value)
            for key, value in monitor.sample("snapshot").items()
            if key.startswith("memory/") and isinstance(value, (int, float))
        }

    def synchronize(self) -> None:
        _, device, _, _, _ = self._ready()
        synchronize(device)

    @property
    def unavailable_memory_reasons(self) -> dict[str, str]:
        return {} if self.monitor is None else self.monitor.unavailable_reasons

    def close(self) -> None:
        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None
