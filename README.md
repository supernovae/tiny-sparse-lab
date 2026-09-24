# Tiny Sparse Lab

**A small-model workbench for learning mechanisms, training bounded local capabilities, and keeping the evidence with each experiment.**

Tiny Sparse Lab is a reference laboratory, not a production training service. One run uses one process and one selected CPU or accelerator. A local controller can queue whole independent experiments on registered local or SSH workers. PyTorch is canonical; optional MLX is a separate FP32 Apple-Metal engine with explicit feature boundaries. There is no distributed process group, expert sharding, or multi-host backward pass.

**New to training models? Start with [From flashcards to a useful local assistant](docs/from-toy-to-useful.md).** The `amber → lumen` exercise is deliberately invented recall, not general intelligence; the guide moves from that bounded task to licensed local conversation data and honest evaluation.

## What is implemented

| Area | Current boundary |
|---|---|
| Decoder and inference | RMSNorm/RoPE causal decoder, SwiGLU, tied/untied embeddings, generation/chat, and checkpoint-bound transcripts. Supported PyTorch paths use a bounded request-local KV cache; MLX uses full-prefix decoding. |
| Architectural experiments | PyTorch dense, sliding-window, block-sparse, and MLA attention; local Top-K MoE; token, byte-addressed, and portable Engram memory. MLX supports dense and native block-sparse attention, not blanket architecture parity. |
| Training and objectives | Exact target-counted gradient accumulation, whole-transcript or assistant-only conversation loss, block recomputation, periodic validation, and safe-boundary interruption. Versioned tool-call transcripts are training data, not executable tools. |
| Checkpoints and continuation | Immutable, hash-verified generations; offline verification; explicit recovery; full-state same-engine/backend child resume; compatible weight promotion with fresh training state. Validated legacy and narrow Llama safetensor import are separate from resume. |
| Runtime and memory | Discovery, disposable precision probes, shape-only capacity estimates, explicit config proposals, isolated smoke/warmup pilots, and measured memory/timing. PyTorch supports capability-checked mixed precision, activation offload, and Adafactor alongside default AdamW. |
| Independent workers | Local/SSH stdio protocol, capability-filtered durable queues, physical-device leases, sealed inputs, cancellation, explicit child resume, verified artifact/metric ingestion, and explicit Cartesian matrices. |
| Evidence and dashboard | Checkpoint-bound evaluation/capability cards, recorded comparisons, and read-only live Training, Evaluation, Architecture, Runtime, Memory, Checkpoints, Stages, Learn, and Research pages. |
| Boundaries | No distributed backward, expert all-to-all, shared optimizer, optimizer-state/parameter/expert offload, automatic tool execution, or production-serving guarantee. |

## Verified status and remaining gates

The **2026-09-22/23 acceptance records** cover **335 passing tests**, Ruff lint/format checks, installed-wheel execution outside the checkout, and real CPU, MPS, and MLX/Metal scenarios. These are implementation checks, not a claim that a tiny checkpoint is a useful general assistant.

| Evidence | What was actually exercised |
|---|---|
| [Single-host acceptance](artifacts/acceptance/single_host_gate_2026_09_22.json) | CPU FP32/BF16 and Adafactor continuation, actual MPS/MLX execution, safe interruption/recovery, promotion, offline artifacts, staging, and populated dashboard checks. |
| [Independent-worker acceptance](artifacts/acceptance/independent_workers_2026_09_23.json) | Three overlapping logical CPU workers, progress through controller loss, idempotent launch replay, cancellation and bitwise child-resume comparison, forced executor loss, CLI matrices, source/capacity rejection, and an actual MLX worker. |
| [Scientific studies](artifacts/acceptance/scientific_studies_2026_09_22.json) | Preregistered multi-seed/multi-budget context/Engram comparisons and domain adaptation with retention checks. Untouched context overrides remained **0/8** at every endpoint; adaptation did not establish reliable held-out domain behavior and caused severe forgetting. |

[The completion ledger](TODO.md) records **31 completed tasks and five hardware-blocked gates**: native CUDA sparse attention, native HIP sparse attention, actual ROCm acceptance, actual XPU acceptance, and overlapping real Mac/AMD/Intel execution. CPU worker labels do not prove foreign hardware support; SSH protocol tests are not remote driver validation.

The latest worker UI check captured actual rendered curve pixels; standard browser screenshots stalled, so runtime-table values were additionally verified through Streamlit's app harness. The earlier populated single-host dashboard screenshots remain in the acceptance record.

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

Use new run IDs and stage output directories for another experiment; existing artifacts are not silently overwritten. Source-checkout installs intentionally use the PyTorch CPU index on Linux. CUDA/ROCm/XPU workers need a vendor-provisioned environment and the project wheel, not a blind CPU-locked `uv sync`; see [worker installation boundaries](docs/workers.md#user-provisioned-ssh-workers).

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

### Optional MLX on Apple Silicon

After preparing the smoke tokenizer above:

```sh
uv sync --locked --dev --extra mlx
uv run --extra mlx sparselab train configs/smoke_mlx.yaml --run-id mlx-smoke
uv run --extra mlx sparselab eval mlx-smoke
```

Keep `--extra mlx` on subsequent `uv run` commands, or invoke the installed `.venv/bin/sparselab` directly. MLX is a separate FP32 Metal engine with AdamW, dense/native block-sparse attention, and block recomputation. It does not support MoE, Engram, MLA, sliding-window attention, mixed precision, activation offload, or Adafactor. Core-only installations can verify native MLX checkpoints without the SDK but cannot execute them. See [runtime support](docs/runtime.md).

### Queue independent experiments

```sh
uv run sparselab worker register local-cpu --backend cpu --store /tmp/sparselab-controller
uv run sparselab experiment submit configs/runtime_smoke_cpu.yaml --worker local-cpu --store /tmp/sparselab-controller
uv run sparselab controller run --store /tmp/sparselab-controller
```

The controller runs in the foreground; use another terminal for `experiment list`, `experiment cancel RUN_ID`, or `experiment resume RUN_ID` with the same `--store`. An interrupted controller does not stop an already launched worker or authorize another optimizer execution. `sparselab run CONFIG --store ROOT` registers a local endpoint and composes dispatch/warmup/training without requiring a separate controller terminal. See [worker operation and failure semantics](docs/workers.md), including vendor-provisioned SSH environments, transfer deadlines, explicit matrices, and hardware limits.

Inspect the checked-in three-seed matrix without preparing data or changing the queue:

```sh
uv run sparselab experiment submit --matrix tests/fixtures/runtime-matrix.yaml \
  --dry-run --store /tmp/sparselab-controller
```

Remove `--dry-run` to enqueue its three independent runs. Launch the dashboard with `--runs-dir /tmp/sparselab-controller` to inspect imported results; worker databases and WAL files stay on their own hosts.

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


## Research workbench

Learn one mechanism or scaffold a declared controlled study without downloading data or starting a run: [research workbench](docs/research/README.md). It distinguishes configuration-only scaffolding from the explicit tokenizer/data/training commands, and explains static reports and read-only dashboard browsing.

Published reports: [FFN-substitution smoke](artifacts/research-reports/fbdb00217f8e952e12bd07e053d97d85796f889748ee77d3bc81b21bb3c98c0b/index.html) and [nano/offline follow-up](artifacts/research-reports/a444e2869973568e28315aa2cac1a97454d7f9d7b8742fd69de8358e867e7daf/index.html); see [outcomes and interpretation](docs/research/sample-report.md).

The [project review](docs/project-review.md) preserves the original local learning observations and failed controls. The completed [context/Engram study](docs/context-engram-study.md#execution-results--2026-09-22) and [domain adaptation study](docs/path-domain-corpus.md#2026-09-22-execution-record) add multi-seed outcomes, collision measurements, and retention checks without selecting favorable endpoints. The [capability workflow](docs/capabilities.md) explains held-out narrow claims; [instruction training](docs/instruction-training.md) explains licensed local conversations and assistant-only/tool-transcript supervision.

## Documentation

- [Using SparseLab](docs/using-sparselab.md) — commands, local artifacts, and dashboard.
- [Runtime policy](docs/runtime.md) — selection, probes, stages, measurements, and proposals.
- [Memory accounting](docs/memory.md) — disjoint categories, capacity uncertainty, and observations.
- [Checkpointing](docs/checkpointing.md) and [reproducibility](docs/reproducibility.md) — verified state, recovery, resume, promotion, and identity.
- [Gradient accumulation](docs/gradient-accumulation.md), [activation recomputation](docs/activation-checkpointing.md), and [activation offload](docs/offload.md) — resource mechanisms and their distinct contracts.
- [Independent workers](docs/workers.md) — queue, local/SSH operation, cancellation, recovery, verified ingestion, and hardware gates.
- [Architecture](docs/architecture.md), [MoE](docs/moe.md), [sparse attention](docs/sparse-attention.md), [MLA](docs/mla.md), and [Engram](docs/engram.md) — reference mechanisms.
- [Training and resume](docs/training.md), [metrics](docs/metrics.md), [experiments](docs/experiments.md), and [evidence](docs/evidence.md) — local lifecycle and comparison practice.
- [Context/Engram study](docs/context-engram-study.md) and [domain corpus/adaptation](docs/path-domain-corpus.md) — frozen inputs, executed comparisons, negative results, and limitations.
- [Completion backlog](TODO.md) — recorded acceptance and separately blocked hardware/distributed work.

## Data and contributions

Synthetic data is an offline fixture. TinyStories artifacts retain pinned source/revision and license metadata locally; other downloaded datasets retain their own terms. The checked-in curated fixtures, provenance, study reports, and bounded acceptance artifacts are intentional. Do not commit downloaded corpora, prepared arrays, full run directories, or model checkpoints.

Contributions should preserve explicit causal, configuration, artifact, and comparison contracts. Read [CONTRIBUTING.md](CONTRIBUTING.md), verify the affected local behavior, and add a focused regression only when it protects an observable contract.
