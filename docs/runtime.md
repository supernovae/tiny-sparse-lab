# Runtime policy

Each SparseLab experiment runs in one process on one host/device. An optional controller schedules multiple whole independent experiments on local or SSH workers; it never shares their optimizer state or gradients. A run records its selected engine, backend, precision, device index, available memory readings, framework/runtime versions, and probe result in its manifest. That record describes the machine that ran it; it is not evidence that another machine has the same capability.

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

## Discovery, validation, and measurements

Discovery is passive: it inventories CPU, MPS, CUDA, ROCm, XPU, and MLX availability without creating a model. It records unavailable runtimes and API limitations rather than guessing. Validation separately exercises a disposable forward/backward/optimizer probe for the requested engine and precision. Discovery or a vendor specification is not a hardware acceptance result.

Per-update timing synchronizes at update boundaries. `performance/step_seconds` and `performance/tokens_per_second` cover successful update work; validation, checkpointing, staging, and setup are outside that interval. Memory monitoring samples process RSS and supported allocator readings at update phases. CUDA/ROCm/XPU-shaped allocator APIs may supply native peak counters; MPS has a driver allocation reading and an observed sampled peak, not an allocator high-water guarantee. Unsupported readings are omitted and recorded as unavailable rather than emitted as zero.

MPS and MLX use unified memory. Their device recommendation/driver allocation and host RAM are not independent pools, so reports and estimates must never add them together.

## Inspect and stage before a run

`inspect` builds a shape-only parameter inventory and a conservative memory estimate; it does not construct the model or load tokenizer/data/package assets. The estimate uses disjoint categories—resident weights, registered runtime buffers, gradients, optimizer state, retained activations, attention working tensors, workspace, and headroom. It is a planning model, not a measured peak. Missing capacity information produces `UNKNOWN`; an artificial `budget_bytes` can demonstrate an exceedance but cannot prove physical fit.

```sh
uv run sparselab inspect configs/runtime_smoke_cpu.yaml --json
uv run sparselab stage configs/runtime_smoke_cpu.yaml --through smoke --output /tmp/sparselab-stage
uv run sparselab stage configs/runtime_smoke_cpu.yaml --through warmup --output /tmp/sparselab-stage-warmup
```

A stage bundle is an immutable, verified copy of the effective inputs. `inspect` records only preflight inspection; `validate` additionally validates backend and artifacts; `smoke` and `warmup` run short, disposable subprocess pilots. Pilots do not advance the eventual training run's optimizer, schedule, cursor, counters, or RNG. A stage output must be new, or an identical complete bundle is re-verified and reused. If estimation or a pilot exceeds its safe ceiling, staging writes a proposal and stops; it does not rewrite the requested config.

A direct `train` performs configured, inspected, and validated stages, but never silently runs pilots. Supplying `--stage-bundle DIR` adds verified pilot linkage to the resulting manifest; it does not use pilot weights as initialization.

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

Native CUDA/HIP sparse kernels, actual ROCm/XPU acceptance, and overlapping real Mac/AMD/Intel execution remain hardware-blocked in [the completion ledger](../TODO.md). Provisioned vendor-compatible runtimes and those physical targets are prerequisites; declaring a capability or assigning a CPU worker a platform name does not close a hardware gate.

See [memory accounting](memory.md), [activation recomputation](activation-checkpointing.md),
and [activation offload](offload.md) for the estimate/measurement boundaries.
