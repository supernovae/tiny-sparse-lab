# Runtime policy

Each SparseLab experiment runs in one process on one host/device. An optional controller schedules multiple whole independent experiments on local or SSH workers; it never shares their optimizer state or gradients. A run records its selected engine, backend, precision, device index, available memory readings, framework/runtime versions, and probe result in its manifest. That record describes the machine that ran it; it is not evidence that another machine has the same capability.

## Host environment and compute backend

Host OS, execution environment, and compute backend are independent dimensions.
WSL2 is a Linux environment; it does not select AMD, NVIDIA, Intel, or CPU.
macOS likewise does not imply that a run must use its GPU. Runtime discovery
records `host_os`, `host_environment`, and `host_architecture` alongside the
existing OS description and backend/device metadata. Historical records without
these fields keep them unknown rather than inheriting the reader's current host.
Linux kernel WSL2 markers identify `wsl2`; a WSL hint without a known generation
is recorded as `unknown-wsl`. Host detection never initializes a GPU.

| Dimension | Examples | Role |
| --- | --- | --- |
| Host OS/environment | Native Linux, Linux under WSL2, macOS | OS behavior and installation context |
| Host architecture | `x86_64`, `arm64` | CPU instruction architecture, independent of GPU vendor |
| Engine/backend | PyTorch CPU, CUDA, ROCm, XPU, MPS; MLX Metal | Actual execution APIs and device capabilities |
| Workspace | `runtime-acceptance`, a named research campaign | Task-owned storage shared across backend coordinates |

CPU execution uses the same path on Linux, WSL2, and macOS. NVIDIA CUDA, AMD
ROCm, and Intel XPU use their installed framework APIs on native Linux or WSL2
where the vendor supports that host/device combination. Apple MPS and MLX use
Metal on supported Macs. Discovery and explicit validation decide availability;
a host label never certifies an accelerator or enables a fallback.

WSL2 uses the Linux process, locking, and filesystem paths in SparseLab. Keep
Linux training data, caches, and checkpoints in the Linux filesystem where
practical, and measure the actual chosen storage; Microsoft's
[filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
explains the cost of crossing into Windows-mounted storage. Memory readings
describe the Linux environment and available device APIs, not an assumed share
of all physical Windows host RAM. Missing driver or device telemetry remains
unavailable, including under WSL2.

Provision GPU drivers and framework wheels for the actual host and device.
For WSL2, follow the vendor's WSL instructions: NVIDIA's
[CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/) uses the Windows host
driver and explicitly excludes installing a Linux GPU driver inside WSL;
PyTorch's [Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
lists supported XPU host/device combinations; AMD's
[installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
covers its framework packages. SparseLab does not install or infer drivers from
the presence of WSL. Native Windows execution is not established by WSL2 tests.

## Choose an execution target

A schema-v2 run config has a `runtime` section:

```yaml
runtime:
  engine: pytorch       # pytorch | mlx
  backend: auto         # auto | cpu | mps | cuda | rocm | xpu | metal
  device_index: 0
  precision: fp32       # auto | fp32 | bf16 | fp16
```

For the PyTorch engine, `auto` prefers an available local MPS device, then CUDA or ROCm, then XPU, then CPU. An explicit unavailable backend is an error—SparseLab does not retry it on CPU. ROCm uses PyTorch's `cuda` device API internally but is recorded as `rocm`; a CUDA request does not silently select a HIP build. CPU and MPS have index 0 only; CUDA/ROCm/XPU validate the requested index against the local runtime.

`precision: auto` resolves to FP32. BF16 and FP16 are not merely labels: validation starts a disposable subprocess that performs a tiny forward, backward, and optimizer update at the requested precision. CPU rejects FP16. FP16 is accepted only on backends with the current GradScaler path; BF16 is passed through PyTorch autocast when the requested probe succeeds. Parameters, gradients, reductions, and the conservative estimator remain accounted as FP32 even when autocast is used.

The optional MLX engine is separate from PyTorch: use `engine: mlx` and
`backend: metal`, never a `torch.device`. The pinned `mlx==0.32.2` runtime
supports FP32 dense and native block-sparse attention, block recomputation,
common immutable generations, same-engine resume/recovery, canonical-weight
promotion, evaluation, generation, chat, and capability/evidence reports.
MLX decoding currently uses full-prefix evaluation, not a KV cache. MoE,
Engram memory, MLA, sliding-window attention, mixed precision, activation
offload, and Adafactor remain explicitly unsupported on this engine.

Install the Apple-arm64 extra with `uv sync --extra mlx`; retain it when using
`uv run --extra mlx ...`, or invoke the installed `.venv/bin/sparselab` directly.
A core-only installation can inspect and verify native checkpoint files without
the MLX SDK, but cannot execute native training or inference.

The project defaults to CPython 3.14 on CPU and macOS as well as ROCm. The
locked environment includes Apple Silicon wheel resolution; use
`uv sync --locked --dev --extra mlx` on Apple Silicon to install the optional
MLX runtime and run the native MLX tests.

## Discovery, validation, and measurements

Discovery is passive: it inventories CPU, MPS, CUDA, ROCm, XPU, and MLX availability without creating a model. It records unavailable runtimes and API limitations rather than guessing. Validation separately exercises a disposable forward/backward/optimizer probe for the requested engine and precision. Discovery or a vendor specification is not a hardware acceptance result.

Per-update timing synchronizes at update boundaries. `performance/step_seconds` and `performance/tokens_per_second` cover successful update work; validation, checkpointing, staging, and setup are outside that interval. Memory monitoring samples process RSS and supported allocator readings at update phases. CUDA/ROCm/XPU-shaped allocator APIs may supply native peak counters; MPS has a driver allocation reading and an observed sampled peak, not an allocator high-water guarantee. Unsupported readings are omitted and recorded as unavailable rather than emitted as zero.

MPS and MLX use unified memory. Their device recommendation/driver allocation and host RAM are not independent pools, so reports and estimates must never add them together.

## Inspect and stage before a run

`inspect` builds a shape-only parameter inventory and a conservative memory estimate; it does not construct the model or load tokenizer/data/package assets. The estimate uses disjoint categories—resident weights, registered runtime buffers, gradients, optimizer state, retained activations, attention working tensors, workspace, and headroom. It is a planning model, not a measured peak. Missing capacity information produces `UNKNOWN`; an artificial `budget_bytes` can demonstrate an exceedance but cannot prove physical fit.

```sh
uv run sparselab inspect configs/runtime_smoke_cpu.yaml --json
uv run sparselab stage configs/runtime_smoke_cpu.yaml --through smoke --output sparselab-work/stages/runtime-smoke
uv run sparselab stage configs/runtime_smoke_cpu.yaml --through warmup --output sparselab-work/stages/runtime-warmup
```

A stage bundle is an immutable, verified copy of the effective inputs. `inspect` records only preflight inspection; `validate` additionally validates backend and artifacts; `smoke` and `warmup` run short, disposable subprocess pilots. Pilots do not advance the eventual training run's optimizer, schedule, cursor, counters, or RNG. A stage output must be new, or an identical complete bundle is re-verified and reused. If estimation or a pilot exceeds its safe ceiling, staging writes a proposal and stops; it does not rewrite the requested config.

A direct `train` performs configured, inspected, and validated stages, but never silently runs pilots. Supplying `--stage-bundle DIR` adds verified pilot linkage to the resulting manifest; it does not use pilot weights as initialization.

## Runtime forecasting and progress

Runtime forecasts are operational aids, not scientific results or performance
guarantees. SparseLab keeps four records distinct: a planning estimate from
compatible historical optimizer throughput, an optional warmup-calibrated
estimate from a disposable pilot, a live target-based ETA, and a final
observation of actual phase costs. Missing evidence is `null`/unavailable, not
zero.

```sh
uv run --locked sparselab inspect CONFIG --estimate-runtime --json
uv run --locked sparselab stage CONFIG --through warmup --output sparselab-work/stages/forecast
uv run --locked sparselab train --runs-dir sparselab-work/runs CONFIG --stage-bundle sparselab-work/stages/forecast
uv run --locked sparselab runtime status RUN_ID --runs-dir sparselab-work/runs --json
```

`inspect --estimate-runtime` is read-only. Planning uses up to the 25 most
recent observations whose full runtime signatures match: engine, backend and
device identity; OS, framework/runtime and driver versions; precision; model
and attention architecture; optimizer; sequence length, microbatch and
accumulation; recomputation and offload. The JSON reports matching dimensions
and rejection reasons. It does not convert theoretical work or parameter counts
into wall time. Without compatible observations—or when device identity is
unknown—the planning estimate remains unavailable.

`stage --through warmup` adds a separate estimate from measured disposable
optimizer updates. It does not change the requested training state. The first
pilot update is excluded as initialization; the staged estimate is used only
when its runtime signature exactly matches the eventual run. Without a warmup
bundle, no warmup estimate is implied.

Training progress counts completed supervised targets against the configured
target budget and reports steps separately. Live throughput uses robust recent
and longer windows; ETA is named `optimizer_only_eta`. Initial updates are not
used as completed-run throughput calibration. Disagreeing windows are marked
unstable. When targets stop advancing for the stall interval, state becomes
`NO_PROGRESS` and ETA is suspended; the process is not killed.

Versioned progress JSON Lines go to stderr, leaving existing stdout result
payloads unchanged. Runtime status reads the latest bounded progress snapshot
and the separate `runtime_final_observation` event. SQLite retains one
replaceable progress snapshot per run outside the replication outbox. Status is
read-only; if the snapshot is stale, it marks progress `NO_PROGRESS` and clears
ETA in the returned view without rewriting stored state.

The final observation keeps preparation, planning, optimizer updates,
validation, checkpointing, reporting, and end-to-end wall time separate. Phase
times with no observation, including evaluation or generation that did not run,
remain unavailable rather than appearing as zero cost. The dashboard reads the
same records and does not schedule runs or tune configuration.

**Illustrative planning miss (hypothetical, not a recorded run):** compatible
history predicts 2 hours of optimizer work, but the completed run observes 2.5
hours. New contention or thermal throttling can change device throughput without
changing the signature, so a compatible historical rate is not a guarantee.
The final observation makes that error visible; only an eligible completed run
with enough post-initialization updates can contribute to later calibration.

In an implementation CPU smoke, no compatible history meant planning was
unavailable. A separate warmup forecast for 640 targets was 0.0588 seconds
(0.0582–0.0614); the 20-update run observed 0.0750 seconds of optimizer-update
time. The bounded pilot had four measured update intervals after discarding its
first update. This tiny smoke demonstrates forecast error, not general
throughput or model quality.

The existing `fast` resource proposal can preserve effective batch while
changing microbatch/accumulation based on capacity rules. It is not an empirical
throughput search. Measured candidate ranking and batch recommendations remain
tracked separately in the [implementation backlog](../TODO.md#throughput-and-resource-proposals).

## Resource proposals

Memory policy can propose an explicit complete config; it never changes the config supplied to training. `fast`, `balanced`, `low_memory`, and `max_fit` order candidate choices differently, but unsupported actions receive no imagined savings. In particular, changing micro-batch/accumulation can preserve an effective example batch, while changing precision or sequence length is scientifically significant and must remain visible.

Transformer-block recomputation is implemented for both engines. PyTorch
activation offload requires an actual saved-tensor roundtrip/backward probe,
records synchronized transfer metrics, and checks incremental host headroom
before run creation. MPS has zero estimated physical-capacity gain; discrete
device estimates are conditional and need paired hardware measurements before
a capacity claim. Optimizer-state offload remains rejected.

## Optimizer semantics

AdamW is the default. PyTorch Adafactor is an explicit alternative with
shape-factored second-moment state, `foreach=False`, and recorded algorithm
settings. The shared peak/floor schedule controls Adafactor's **relative
step-size cap**, not an AdamW-equivalent absolute learning rate. Parameter RMS
scaling and its own update clipping still apply. An optimizer change requires
a fresh or promoted run, never full resume.

## Runtime acceptance on a provisioned host

Use the same acceptance workflow for native Linux, WSL2, and macOS. Select a
config with an explicit backend and precision supported by the provisioned
environment. The CPU and ROCm runtime smoke configs use the same 20-step,
640-target workload; a CUDA or XPU candidate can copy that config and change
`runtime.backend` in its own source config. Do not change a running or frozen
experiment's config. Backend names in configs and run IDs are useful coordinates;
host/backend combinations do not define separate storage layouts.

```sh
WORK=sparselab-work/experiments/runtime-acceptance
export SPARSELAB_WORK_DIR="$WORK"
CONFIG=configs/runtime_smoke_cpu.yaml
RUN_ID="runtime-acceptance-$(date -u +%Y%m%dT%H%M%SZ)"
STAGE_DIR="$WORK/staging/$RUN_ID"
# --no-sync preserves the already provisioned worker framework.
uv run --locked --no-sync sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run --locked --no-sync sparselab inspect "$CONFIG" --json
uv run --locked --no-sync sparselab stage "$CONFIG" --through warmup --output "$STAGE_DIR"
uv run --locked --no-sync sparselab train --runs-dir "$WORK/runs" "$CONFIG" --run-id "$RUN_ID"
uv run --locked --no-sync sparselab eval "$RUN_ID" --runs-dir "$WORK/runs"
```

The stage, training manifest, and evaluation must record the requested backend;
training must commit 20 steps / 640 targets for these runtime smoke configs, and
evaluation must use the committed checkpoint. Inspect the measured host/device
identity and probe result. An explicit unavailable backend is a failed
acceptance. Each new host/device combination requires actual execution evidence.

## Recorded local acceptance

The [Astra CLI record](../artifacts/acceptance/host_cli_2026_09_22.json) contains
fresh-process CPU FP32 20/640 versus 10+resume, CPU BF16 and Adafactor 4/128
versus 2+resume, and native MLX FP32 2/64 versus 1+resume. Model, optimizer,
schedule, scaler, cursor, and RNG match bitwise within each pair; parent
generations remain unchanged. Native logits differed from canonical PyTorch
by at most `5.97e-7` on the fixed probe, and a PyTorch-to-MLX promotion committed
one fresh update. These are bounded implementation checks, not cross-engine
training equivalence, model-quality results, or foreign-hardware acceptance.

The [integrated single-host gate](../artifacts/acceptance/single_host_gate_2026_09_22.json) also retains actual MPS continuation/promotion, corruption and signal recovery, installed-wheel/offline checks, and populated dashboard evidence. The [independent-worker gate](../artifacts/acceptance/independent_workers_2026_09_23.json) adds three overlapping CPU workers, controller disconnect/replay, acknowledged cancellation, explicit recovery after executor loss, offline promotion, actual CLI matrix execution, genuine source-mismatch rejection, and a real MLX/Metal worker.

Native CUDA sparse kernels, actual XPU acceptance, and overlapping real Mac/AMD/Intel execution remain open in [the implementation backlog](../TODO.md). Native HIP sparse attention has been exercised and benchmarked on the RX 7900 XTX; ROCm runtime acceptance remains limited to one WSL2 host and does not establish cross-host support.

See [memory accounting](memory.md), [activation recomputation](activation-checkpointing.md),
and [activation offload](offload.md) for the estimate/measurement boundaries.
