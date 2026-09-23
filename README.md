# Tiny Sparse Lab

**A small-model workbench for learning mechanisms, training bounded local capabilities, and keeping the evidence with each experiment.**

Tiny Sparse Lab is a reference laboratory, not a production training service. One run uses one process and one selected CPU or accelerator. A local controller can queue whole independent experiments on registered local or SSH workers. PyTorch is canonical; optional MLX is a separate FP32 Apple-Metal engine with explicit feature boundaries. There is no distributed process group, expert sharding, or multi-host backward pass.

**New to training models? Start with [From flashcards to a useful local assistant](docs/from-toy-to-useful.md).** The `amber → lumen` exercise is deliberately invented recall, not general intelligence; the guide moves from that bounded task to licensed local conversation data and honest evaluation.

## What is implemented

| Area | Current boundary |
|---|---|
| Decoder and chat | RMSNorm/RoPE causal decoder, SwiGLU, local generation/chat, and checkpoint-bound transcripts. Supported PyTorch attention paths use a verified request-local KV cache; MLX uses full-prefix decoding. |
| Architectural experiments | Dense, sliding-window, block-sparse, and MLA reference attention; local Top-K MoE; token, byte-addressed, and portable Engram memory. Native MLX sparse kernels have bounded component measurements, not a general speed claim. |
| Local training | Exact gradient accumulation, assistant-only objectives, block recomputation, evaluation, common immutable generations, recovery, and promotion on both engines. Adafactor is an explicit PyTorch alternative to AdamW. |
| Runtime/staging | Passive discovery, disposable capability probes, conservative memory planning, and isolated smoke/warmup pilots. Hardware claims require that hardware's own evidence. |
| Independent workers | Versioned local/SSH endpoints, capability-filtered queues, host-wide device leases, sealed input bundles, acknowledged cancellation, explicit child resume, and controller-local metric/artifact ingestion. Real remote hardware acceptance remains separately gated. |
| Boundaries | No distributed backward, expert all-to-all, shared optimizer, optimizer-state offload, or blanket MLX/PyTorch feature parity. |

## Start with a bounded local path

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). Run commands from the repository root.

```sh
uv sync --locked --dev
uv run sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run sparselab inspect configs/runtime_smoke_cpu.yaml --json
uv run sparselab stage configs/runtime_smoke_cpu.yaml --through warmup --output /tmp/sparselab-stage-runtime
uv run sparselab train configs/runtime_smoke_cpu.yaml --run-id runtime-full
uv run sparselab eval runtime-full
uv run sparselab generate runtime-full --prompt "Once upon a time" --max-new-tokens 24
uv run sparselab chat runtime-full --max-new-tokens 12
uv run sparselab dashboard --runs-dir runs
```

The smoke configuration is intentionally tiny. It exercises tokenizer, prepared data, training, evaluation, checkpointing, and local inference; it does not demonstrate fluent generation, broad capability, a performance win, or hardware capacity at larger scale. `stage --through warmup` runs disposable pilot subprocesses and seals their evidence; it does not alter the full run's weights, optimizer, schedule, counters, cursor, or RNG.

### Safely stop, verify, and resume

`--stop-after-step` finishes a successful update boundary and leaves a durable checkpoint. Verify it before continuing into a new child run:

```sh
uv run sparselab train configs/runtime_smoke_cpu.yaml --run-id runtime-part --stop-after-step 10
uv run sparselab checkpoint inspect runs/runtime-part/checkpoints/latest.json --json
uv run sparselab checkpoint verify runs/runtime-part/checkpoints/latest.json --json
uv run sparselab train configs/runtime_smoke_cpu.yaml --run-id runtime-resumed \
  --resume runs/runtime-part/checkpoints/latest.json
```

Full resume requires the same compatible experiment and engine/backend, and restores training state. `--promote CHECKPOINT` instead starts fresh optimizer/schedule/RNG/cursor state from compatible weights. Neither path resizes a model. A separate validated weights importer supports historical SparseLab checkpoints and a narrow compatible Llama safetensor mapping, not arbitrary downloaded models. See [checkpointing](docs/checkpointing.md) and [reproducibility](docs/reproducibility.md).

### Queue independent experiments

```sh
uv run sparselab worker register local-cpu --backend cpu --store /tmp/sparselab-controller
uv run sparselab experiment submit configs/runtime_smoke_cpu.yaml --worker local-cpu --store /tmp/sparselab-controller
uv run sparselab controller run --store /tmp/sparselab-controller
```

The controller runs in the foreground; use another terminal for `experiment list`, `experiment cancel RUN_ID`, or `experiment resume RUN_ID` with the same `--store`. An interrupted controller does not stop an already launched worker or authorize another optimizer execution. `sparselab run CONFIG --store ROOT` registers a local endpoint and composes dispatch/warmup/training without requiring a separate controller terminal. See [worker operation and failure semantics](docs/workers.md), including vendor-provisioned SSH environments, transfer deadlines, explicit matrices, and hardware limits.

## Why does training use so much memory?

Training holds more than weights: gradients, optimizer moments, retained activations for backward, attention working tensors, runtime buffers, temporary workspace, and safety headroom all matter. Inactive MoE experts and frozen memory tables remain resident even if a token does not select them. The estimator keeps these categories disjoint and conservative; it is a plan, not a measured peak.

There are four different levers:

- **Disk checkpoints** preserve durable restart state.
- **Activation recomputation** reruns selected forward blocks during backward to retain fewer activations.
- **Gradient accumulation** combines small microbatches into one optimizer update.
- **Activation offload** copies autograd-saved tensors to host storage, with explicit transfer and host-headroom accounting. Unified-memory capacity gain remains zero.

None of these creates distributed training. Inference KV caches are request-local and receive no training-memory discount. Supported PyTorch decoding can be compared against the full-prefix reference with the Python `generate(..., use_cache=False)` API; unsupported paths use full-prefix evaluation. Read [memory accounting](docs/memory.md), [activation recomputation](docs/activation-checkpointing.md), [gradient accumulation](docs/gradient-accumulation.md), and [activation offload](docs/offload.md) before interpreting a memory number.

## Learn with controlled comparisons

1. Establish the CPU baseline and inspect its shape-only parameter/memory inventory.
2. Change one explicit mechanism—attention, MoE, or Engram—while matching data, tokenizer, budget, seed, optimizer, and runtime conditions.
3. Verify checkpoint-bound evaluation and retain negative results: a zero delta, failed fit, or unavailable capability is evidence.
4. Scale only after the smaller task learns; a synthetic or narrow score does not certify broad usefulness.

The existing [project review and measured example](docs/project-review.md) retains the recorded local learning observations, including failed controls and their limits; it does not turn them into broad quality claims. The [capability workflow](docs/capabilities.md) explains held-out narrow claims; [instruction training](docs/instruction-training.md) explains licensed local conversation input.

## Documentation

- [Using SparseLab](docs/using-sparselab.md) — commands, local artifacts, and dashboard.
- [Runtime policy](docs/runtime.md) — selection, probes, stages, measurements, and proposals.
- [Memory accounting](docs/memory.md) — disjoint categories, capacity uncertainty, and observations.
- [Checkpointing](docs/checkpointing.md) and [reproducibility](docs/reproducibility.md) — verified state, recovery, resume, promotion, and identity.
- [Gradient accumulation](docs/gradient-accumulation.md), [activation recomputation](docs/activation-checkpointing.md), and [activation offload](docs/offload.md) — resource mechanisms and their distinct contracts.
- [Independent workers](docs/workers.md) — queue, local/SSH operation, cancellation, recovery, verified ingestion, and hardware gates.
- [Architecture](docs/architecture.md), [MoE](docs/moe.md), [sparse attention](docs/sparse-attention.md), [MLA](docs/mla.md), and [Engram](docs/engram.md) — reference mechanisms.
- [Training and resume](docs/training.md), [metrics](docs/metrics.md), [experiments](docs/experiments.md), and [evidence](docs/evidence.md) — local lifecycle and comparison practice.
- [Completion backlog](TODO.md) — recorded acceptance and separately blocked hardware/distributed work.

## Data and contributions

Synthetic data is an offline fixture. TinyStories artifacts retain pinned source/revision and license metadata locally; other downloaded datasets retain their own terms. Do not commit downloaded corpora, prepared arrays, run directories, or checkpoints.

Contributions should preserve explicit causal, configuration, artifact, and comparison contracts. Read [CONTRIBUTING.md](CONTRIBUTING.md), verify the affected local behavior, and add a focused regression only when it protects an observable contract.
