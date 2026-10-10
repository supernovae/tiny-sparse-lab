"""Model explorer: look inside a small checkpoint (data layer, no Streamlit).

What it shows and where each part comes from:

* **architecture** (layers, attention heads, MoE experts, memory tables) and
  parameter counts: the run's resolved config through the shape-only
  :mod:`sparselab.model.inspection` schema, so it is available before (and
  without) loading any weights;
* **weights**: per-tensor norms, spread and histograms of the checkpoint
  loaded by the verified loader (``sparselab.probes.cli.load_target``, the same
  one ``sparselab probe`` uses);
* **tokens**: per-token loss and top-k next-token predictions over a text,
  from the shared scoring path (:func:`sparselab.probes.scoring.log_probs`);
* **attention** patterns per layer and head
  (:meth:`DenseAttention.attention_probabilities`, the reference forward's
  own weights), **expert routing** per token (the MoE layer's ``full``
  diagnostics) and **memory lookups** per token (the addresses each memory
  table was read at, and the gate), captured with forward hooks during one
  forward pass.

Big checkpoints are refused before loading (:func:`size_check`). Results are
cached per (checkpoint digest, text) under ``LAB/explorer`` as sealed JSON; a
cancel sentinel and an optional resource envelope are honored between stages
through :class:`~sparselab.lab_context.LabContext`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sparselab.lab_context import LabContext, is_out_of_memory, release_memory
from sparselab.lab_records import is_sealed, jsonable, seal, write_json_atomic

EXPLORER_FORMAT = "sparselab-explorer-v1"
# Loading and hooking a model this size on a laptop CPU takes seconds; beyond
# it the explorer refuses and says so (attention maps grow with heads x T^2).
MAX_PARAMETERS = 60_000_000
MAX_TOKENS = 128
TOP_K = 5
HISTOGRAM_BINS = 40
HISTOGRAM_SAMPLE = 200_000
DEFAULT_TEXT = (
    "Once upon a time, a little dog found a ball in the park. "
    "The dog ran to the ball and picked it up."
)


class ExplorerUnavailable(ValueError):
    """This checkpoint cannot be explored (too big, unsupported engine...)."""


# --- Architecture (shape-only, before loading) --------------------------------


def run_config(run: Path) -> Any:
    from sparselab.config.models import RunConfig

    return RunConfig.model_validate_json(
        (run / "resolved_config.yaml").read_text(encoding="utf-8")
    )


def architecture(config: Any) -> dict[str, Any]:
    """Layers, heads, experts and memory tables with their parameter counts."""
    from sparselab.model.inspection import inspection_report, named_tensor_inventory

    m, attention = config.model, config.attention
    tensors = named_tensor_inventory(m, attention)

    def count(prefix: str) -> int:
        return sum(
            spec.numel
            for name, spec in tensors.items()
            if name.startswith(prefix) and spec.alias_of is None
        )

    layers = []
    for index in range(m.num_layers):
        block = f"blocks.{index}"
        layer: dict[str, Any] = {
            "index": index,
            "attention": {
                "kind": attention.kind,
                "heads": m.num_heads,
                "kv_heads": m.num_kv_heads or m.num_heads,
                "head_dim": m.hidden_dim // m.num_heads,
                "window": attention.window_size,
                "parameters": count(f"{block}.attention."),
            },
            "ffn": {
                "kind": m.ffn,
                "experts": m.num_experts if m.ffn == "moe" else 0,
                "experts_per_token": m.experts_per_token if m.ffn == "moe" else 0,
                "shared_expert": bool(m.ffn == "moe" and m.shared_expert),
                "parameters": count(f"{block}.ffn."),
                "per_expert": count(f"{block}.ffn.experts.0.") if m.ffn == "moe" else 0,
            },
            "norm_parameters": count(f"{block}.norm"),
        }
        layers.append(layer)
    memory = None
    if m.memory != "none":
        streams = (
            (len(m.memory_ngram_orders) or 1) * m.memory_hash_heads
            if m.memory == "ngram"
            else 1
        )
        memory = {
            "kind": m.memory,
            "injection": m.memory_injection,
            "tables": streams,
            "rows": m.memory_table_size,
            "dim": m.memory_dim,
            "ngram_orders": list(m.memory_ngram_orders or (m.memory_ngram_size,))
            if m.memory == "ngram"
            else None,
            "parameters": count("memory."),
        }
    return {
        "vocab_size": m.vocab_size,
        "hidden_dim": m.hidden_dim,
        "ffn_dim": m.ffn_dim,
        "max_seq_len": m.max_seq_len,
        "tie_embeddings": m.tie_embeddings,
        "embedding_parameters": count("embedding."),
        "output_parameters": 0 if m.tie_embeddings else count("output."),
        "layers": layers,
        "memory": memory,
        "inventory": inspection_report(config),
    }


def size_check(arch: Mapping[str, Any]) -> str | None:
    """Why this model is too big to explore here (None when it fits)."""
    total = int(arch["inventory"]["total"])
    if total > MAX_PARAMETERS:
        return (
            f"{total:,} parameters is above the explorer's {MAX_PARAMETERS:,} "
            "limit (attention maps and per-tensor histograms would not fit an "
            "interactive budget). The architecture and parameter counts below "
            "come from the config alone; probe it with `sparselab probe` instead."
        )
    return None


# --- Measurements on a loaded checkpoint --------------------------------------


def weight_stats(model: Any) -> list[dict[str, Any]]:
    """Per stored tensor: shape, norm, spread and a histogram of its values."""
    out = []
    for name, parameter in model.named_parameters():
        values = parameter.detach().float().flatten().cpu().numpy()
        step = max(1, values.size // HISTOGRAM_SAMPLE)
        sample = values[::step]
        counts, edges = np.histogram(sample, bins=HISTOGRAM_BINS)
        out.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(values.size),
                "norm": float(np.linalg.norm(values)),
                "rms": float(np.sqrt(np.mean(values**2))) if values.size else 0.0,
                "mean": float(values.mean()) if values.size else 0.0,
                "std": float(values.std()) if values.size else 0.0,
                "abs_max": float(np.abs(values).max()) if values.size else 0.0,
                "trainable": bool(parameter.requires_grad),
                "histogram": {
                    "counts": counts.tolist(),
                    "edges": edges.tolist(),
                    "sampled_every": step,
                },
            }
        )
    return out


def _token_text(loaded: Any, token: int) -> str:
    text = str(loaded.tokenizer.decode([int(token)]))
    if text:
        return text
    name = loaded.tokenizer.id_to_token(int(token))
    return f"⟨{name}⟩" if name else f"⟨{int(token)}⟩"


def tokenize(loaded: Any, text: str) -> tuple[list[int], bool]:
    from sparselab.probes import scoring

    ids = [scoring.eot(loaded), *scoring.encode(loaded, text)]
    limit = min(MAX_TOKENS, int(loaded.config.model.max_seq_len))
    return ids[:limit], len(ids) > limit


def token_view(loaded: Any, ids: list[int]) -> list[dict[str, Any]]:
    """Per position: the token, its loss given the prefix, the top-k guesses
    for the next token (from the shared scoring path)."""
    from sparselab.probes import scoring

    rows = scoring.log_probs(loaded, ids)
    out = []
    for position, token in enumerate(ids):
        loss = None if position == 0 else float(-rows[position - 1, token])
        top = np.argsort(-rows[position])[:TOP_K]
        rank = (
            None
            if position == 0
            else int((rows[position - 1] > rows[position - 1, token]).sum()) + 1
        )
        out.append(
            {
                "position": position,
                "id": int(token),
                "text": _token_text(loaded, token),
                "loss": loss,
                "rank": rank,
                "next_top": [
                    {
                        "id": int(t),
                        "text": _token_text(loaded, t),
                        "p": float(np.exp(rows[position, t])),
                    }
                    for t in top
                ],
            }
        )
    return out


def _forward_with_hooks(loaded: Any, ids: list[int]) -> dict[str, Any]:
    """One forward pass; capture attention inputs, routing and memory reads."""
    import torch

    from sparselab.model.attention.dense import DenseAttention
    from sparselab.model.moe import TopKMoE
    from sparselab.probes import scoring

    model = loaded.model
    captured: dict[str, Any] = {"attention": {}, "memory": {}}
    handles = []

    def keep_input(key: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...], _output: Any) -> None:
            captured["attention"][key] = args[0].detach()

        return hook

    def keep_lookup(name: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...], _output: Any) -> None:
            captured["memory"][name] = args[0].detach()

        return hook

    for index, block in enumerate(model.blocks):
        if isinstance(block.attention, DenseAttention):
            handles.append(
                block.attention.register_forward_hook(keep_input(str(index)))
            )
    memory = getattr(model, "memory", None)
    if memory is not None:
        for name, module in memory.named_modules():
            if isinstance(module, torch.nn.Embedding):
                handles.append(module.register_forward_hook(keep_lookup(name)))
        handles.append(
            memory.gate.register_forward_hook(
                lambda _m, _a, output: captured.__setitem__("gate", output.detach())
            )
        )
    x = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    try:
        with torch.inference_mode():
            model(x, **scoring._byte_inputs(loaded, ids), diagnostics="full")
            attention = {}
            for index, block in enumerate(model.blocks):
                key = str(index)
                if key in captured["attention"]:
                    weights = block.attention.attention_probabilities(
                        captured["attention"][key]
                    )
                    attention[key] = weights[0].cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    routing = {}
    for index, block in enumerate(model.blocks):
        if isinstance(block.ffn, TopKMoE) and block.ffn.last_diagnostics is not None:
            d = block.ffn.last_diagnostics
            routing[str(index)] = {
                "selected": d.selected_experts.cpu().numpy().tolist(),
                "weights": d.selected_weights.float().cpu().numpy().tolist(),
                "counts": d.counts.cpu().numpy().tolist(),
                "entropy": float(d.entropy),
            }
    return {**captured, "attention": attention, "routing": routing}


def attention_view(
    arch: Mapping[str, Any], attention: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    layers = {}
    for key, weights in attention.items():
        # Rounded to keep the cached payload small; rows still sum to ~1.
        layers[key] = np.round(weights.astype(np.float64), 4).tolist()
    missing = [
        layer["index"]
        for layer in arch["layers"]
        if str(layer["index"]) not in attention
    ]
    note = None
    if missing:
        kinds = sorted({arch["layers"][i]["attention"]["kind"] for i in missing})
        note = (
            f"attention weights are shown for dense attention only; "
            f"{', '.join(kinds)} layer(s) {missing} select or compress keys "
            "and expose no per-key weights"
        )
    return {"layers": layers, "note": note}


def _single_token_offset(
    addresses: list[int], ids: list[int], loaded: Any
) -> int | None:
    """The offset k when every address is just ``token[p-k] mod rows``.

    An n-gram table whose rows depend on one token only is a unigram table in
    disguise (e.g. a hash multiplier that is a multiple of the table size).
    Checked from the observed reads; the hash itself is not re-implemented.
    """
    rows = int(getattr(loaded.config.model, "memory_table_size", 0) or 0)
    if rows <= 0 or len(addresses) < 4 or loaded.config.model.memory != "ngram":
        return None
    for offset in range(4):
        expected = [
            ids[p - offset] % rows if p >= offset else None
            for p in range(len(addresses))
        ]
        tail = [
            (a, e) for a, e in zip(addresses, expected, strict=True) if e is not None
        ]
        if len(tail) >= 3 and all(a == e for a, e in tail):
            return offset
    return None


def memory_view(
    loaded: Any, ids: list[int], captured: Mapping[str, Any]
) -> dict | None:
    lookups = captured.get("memory") or {}
    if not lookups:
        return None
    streams = []
    for name, addresses in sorted(lookups.items()):
        row = addresses[0].cpu().numpy().tolist()
        first: dict[int, int] = {}
        reuse = []
        for position, address in enumerate(row):
            seen = first.setdefault(address, position)
            reuse.append(None if seen == position else seen)
        streams.append(
            {
                "table": name,
                "addresses": row,
                "first_seen_at": reuse,
                "single_token_offset": _single_token_offset(row, ids, loaded),
            }
        )
    gate = captured.get("gate")
    gates = (
        None
        if gate is None
        else [float(v) for v in 1 / (1 + np.exp(-gate[0, :, 0].float().cpu().numpy()))]
    )
    return {
        "kind": loaded.config.model.memory,
        "streams": streams,
        "gate": gates,
        "note": "addresses are the table rows each token read; a repeat points "
        "at the first position that read the same row (the same n-gram, or a "
        "hash collision between different n-grams)",
    }


# --- Orchestration and cache ---------------------------------------------------


def cache_key(checkpoint_sha256: str, text: str) -> str:
    payload = json.dumps(
        {"format": EXPLORER_FORMAT, "checkpoint": checkpoint_sha256, "text": text},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_path(lab_dir: Path, checkpoint_sha256: str, text: str) -> Path:
    return (
        lab_dir
        / "explorer"
        / checkpoint_sha256[:16]
        / f"{cache_key(checkpoint_sha256, text)[:24]}.json"
    )


def read_cached(path: Path) -> dict[str, Any] | None:
    """A cached exploration, or None when absent, edited or another format."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if not is_sealed(value) or value.get("format") != EXPLORER_FORMAT:
        return None
    return value


def cached_explorations(lab_dir: Path) -> list[dict[str, Any]]:
    """Every readable cached exploration, newest first."""
    out = []
    for path in sorted((lab_dir / "explorer").glob("*/*.json")):
        value = read_cached(path)
        if value is not None:
            out.append({**value, "path": str(path)})
    return sorted(out, key=lambda v: v.get("created_at", ""), reverse=True)


def explore_loaded(
    loaded: Any,
    text: str,
    *,
    context: LabContext | None = None,
    arch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every explorer view of a loaded SparseLab checkpoint."""
    if getattr(loaded, "engine", None) is not None:
        raise ExplorerUnavailable(
            "the explorer needs the PyTorch engine (MLX not supported)"
        )
    if getattr(loaded.model, "semantic_memories", None):
        raise ExplorerUnavailable(
            "the explorer does not support attached semantic packs"
        )
    started = time.monotonic()
    arch = dict(arch or architecture(loaded.config))
    state: dict[str, Any] = {}

    def checkpoint(stage: str) -> None:
        if context is not None:
            context.enter("explorer", stage)
            context.checkpoint()

    ids, truncated = tokenize(loaded, text)
    if len(ids) < 2:
        raise ExplorerUnavailable("the text needs at least one token")
    stages: list[tuple[str, Callable[[], Any]]] = [
        ("weights", lambda: weight_stats(loaded.model)),
        ("tokens", lambda: token_view(loaded, ids)),
        ("forward", lambda: _forward_with_hooks(loaded, ids)),
    ]
    was_training = loaded.model.training
    loaded.model.eval()
    try:
        for stage, work in stages:
            checkpoint(stage)
            state[stage] = work()
    finally:
        loaded.model.train(was_training)
    forward = state["forward"]
    identity = loaded.identity
    return {
        "format": EXPLORER_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "target": {
            "run_id": identity.get("run_id"),
            "checkpoint": identity.get("checkpoint_relative_path"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "step": identity.get("step"),
            "tokens_seen": identity.get("tokens_seen"),
        },
        "text": text,
        "truncated": truncated,
        "architecture": arch,
        "weights": state["weights"],
        "tokens": state["tokens"],
        "attention": attention_view(arch, forward["attention"]),
        "routing": forward["routing"] or None,
        "memory": memory_view(loaded, ids, forward),
        "seconds": round(time.monotonic() - started, 3),
    }


def explore(
    target: str,
    *,
    lab_dir: Path,
    runs_dir: Path | None = None,
    text: str = DEFAULT_TEXT,
    backend: str | None = "cpu",
    authorization: Any = None,
    resource_envelope: Any = None,
    refresh: bool = False,
    load: Callable[[], Any] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Explore TARGET (run id, run dir or checkpoint), cached per checkpoint+text.

    Returns ``(exploration, cache path)``. A too-big model returns only its
    architecture with ``unavailable`` set (nothing is loaded).
    """
    from sparselab.probes.cli import load_target, resolve_target

    run, _ = resolve_target(target, lab_dir=lab_dir, runs_dir=runs_dir)
    arch = architecture(run_config(run))
    too_big = size_check(arch)
    if too_big is not None:
        return (
            {
                "format": EXPLORER_FORMAT,
                "target": {"run_id": run.name},
                "architecture": arch,
                "unavailable": too_big,
            },
            None,
        )
    folder = lab_dir / "explorer"
    folder.mkdir(parents=True, exist_ok=True)
    # Touch LAB/explorer/CANCEL to stop at the next stage; a sentinel left by
    # an earlier exploration never cancels a new one.
    (folder / "CANCEL").unlink(missing_ok=True)
    context = LabContext(
        folder / "CANCEL", resource_envelope=resource_envelope, workspace=lab_dir
    )
    loaded = (
        load
        or (
            lambda: load_target(
                target,
                lab_dir=lab_dir,
                runs_dir=runs_dir,
                backend=backend,
                authorization=authorization,
            )
        )
    )()
    try:
        sha = str(loaded.identity.get("checkpoint_sha256"))
        path = cache_path(lab_dir, sha, text)
        cached = None if refresh else read_cached(path)
        if cached is not None:
            return cached, path
        try:
            result = explore_loaded(loaded, text, context=context, arch=arch)
        except Exception as error:
            if not is_out_of_memory(error):
                raise
            raise ExplorerUnavailable(
                f"out of memory while exploring: {type(error).__name__}"
            ) from None
    finally:
        del loaded
        release_memory()
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal(jsonable(result))
    write_json_atomic(path, sealed)
    return sealed, path
