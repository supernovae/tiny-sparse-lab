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

Install the Apple-arm64 extra with `uv sync --locked --extra mlx`; retain it when using
`uv run --locked --extra mlx ...`, or invoke the installed `.venv/bin/sparselab` directly.
A core-only installation can inspect and verify native checkpoint files without
the MLX SDK, but cannot execute native training or inference.

The project defaults to CPython 3.14 on CPU and macOS as well as ROCm. The
locked environment includes Apple Silicon wheel resolution; use
`uv sync --locked --dev --extra mlx` on Apple Silicon to install the optional
MLX runtime and run the native MLX tests.

## Executable runtime profiles

Hardware inventory does not select a Python environment. Accelerator execution
requires either an explicit runtime profile or a registered worker with an
explicit absolute Python interpreter. Ambient CPU execution remains supported.
`auto` resolving to an accelerator does not authorize it.

```yaml
runtime_profile_version: 1
id: vendor-runtime
python: /absolute/vendor-environment/bin/python
engine: pytorch
backend: rocm
device_index: 0
requirements:
  torch_hip: true
  bf16: true
  device_name_regex: "Radeon"
```

Profiles are frozen, reject unknown fields/versions, and require an executable
absolute Python path and a safe ID. PyTorch accepts CPU/ROCm/CUDA/MPS/XPU;
MLX accepts Metal. Requirement fields are optional; `torch_hip: true` is valid
only for ROCm. BF16 requirements need a disposable optimizer update, not just
a passive support declaration. Device-name matching uses regular-expression
search against the selected device.

`sparselab runtime probe PROFILE --json` queries the selected interpreter and
records its real executable, Python version and prefixes, Torch path/version,
HIP/CUDA/XPU versions, selected device and available names, source identity,
and host fields. Missing optional APIs remain null.
MLX profiles record the loaded `mlx.core` module's path and version, not the
`mlx` namespace package, and do not import Torch.
Profiles never install or synchronize environments. A mismatched installed
source fails before execution; provision the vendor environment and install
this source before retrying.

Direct `train`, `stage`, `eval`, `generate`, and `chat` accept
`--runtime-profile PROFILE`; `run` accepts it only when auto-registering a local
worker. A different interpreter replaces the CLI once with that Python, preserving
arguments and environment. Authorization happens before workspace initialization.
`inspect` and `stage --through inspect` remain passive. Worker dispatch performs
fresh interpreter/runtime/source validation, and pilot children rederive
authorization from sealed operational evidence. Persisted evidence or an arbitrary
worker ID does not authorize acceleration. Profiles and probe records are
operational evidence, not scientific config or prepared-cache identity.

Controller scheduling binds the successful validation reply's tested runtime
to the freshly discovered registered-worker identity. A subsequent passive
discovery cannot replace that probe evidence. Campaign tick deadlines also
bound worker discovery and validation RPCs.

## Discovery, validation, and measurements

Discovery is passive: it inventories CPU, MPS, CUDA, ROCm, XPU, and MLX availability without creating a model. It records unavailable runtimes and API limitations rather than guessing. Validation separately exercises a disposable forward/backward/optimizer probe for the requested engine and precision. Discovery or a vendor specification is not a hardware acceptance result.

Per-update timing synchronizes at update boundaries. `performance/step_seconds` and `performance/tokens_per_second` cover successful update work; validation, checkpointing, staging, and setup are outside that interval. Memory monitoring samples process RSS and supported allocator readings at update phases. CUDA/ROCm/XPU-shaped allocator APIs may supply native peak counters; MPS has a driver allocation reading and an observed sampled peak, not an allocator high-water guarantee. Unsupported readings are omitted and recorded as unavailable rather than emitted as zero.

MPS and MLX use unified memory. Their device recommendation/driver allocation and host RAM are not independent pools, so reports and estimates must never add them together.

## Inspect and stage before a run

`inspect` builds a shape-only parameter inventory and a conservative memory estimate; it does not construct the model or load tokenizer/data/package assets. The estimate uses disjoint categories—resident weights, registered runtime buffers, gradients, optimizer state, retained activations, attention working tensors, workspace, and headroom. It is a planning model, not a measured peak. Missing capacity information produces `UNKNOWN`; an artificial `budget_bytes` can demonstrate an exceedance but cannot prove physical fit.

```sh
uv run --locked sparselab inspect configs/runtime_smoke_cpu.yaml --json
uv run --locked sparselab stage configs/runtime_smoke_cpu.yaml --through smoke --output sparselab-work/stages/runtime-smoke
uv run --locked sparselab stage configs/runtime_smoke_cpu.yaml --through warmup --output sparselab-work/stages/runtime-warmup
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

## Operational resource envelopes

`data prepare`, `stage`, `train`, `run`, and `experiment prepare` accept
`--resource-envelope PATH`. Policies are strict, frozen YAML mappings:

```yaml
resource_envelope_version: 1
max_rss_bytes: 2147483648
max_host_memory_fraction: 0.5
min_available_ram_bytes: 134217728
min_swap_bytes: 0
min_disk_bytes: 268435456
min_inodes: 128
max_workers: 4
max_queue_depth: 1
spill_to_disk: true
```

All numeric limits are optional. Maxima are positive; minima are nonnegative;
host fraction is in `(0, 1]`. Booleans are not integers. Unknown fields and
versions fail. RAM/swap come from psutil, storage from the actual filesystem's
existing ancestor without creating the destination. Missing measurements are
null and fail closed only when a requested limit needs them. Host fraction is
process RSS divided by host RAM, not accelerator memory. Policies never rewrite
batch size, sequence length, source, precision, or scientific config.

Limits are checked before outputs, at bounded streaming boundaries, and before
publication. Streaming preparation rejects `spill_to_disk: false`.
`max_workers` bounds tokenizer workers; `max_queue_depth` bounds pending work.
Version-1 worker specs optionally carry the policy; old specs omit it unchanged.

Preparation progress on stderr and a sibling `<cache>.preparation.json` receipt
record records, source UTF-8 bytes, output tokens, elapsed seconds, source MB/s,
tokens/s, sampled current/peak process RSS, available host RAM, logical input/output
bytes, disk free bytes/inodes, and six independently timed counters:
`source_iteration_seconds`, `tokenizer_encoding_seconds`,
`python_bookkeeping_seconds`, `spool_write_seconds`, `finalize_fsync_seconds`,
and `deep_verification_hash_seconds`. Peak RSS is the maximum observed psutil
sample, not a guaranteed OS lifetime high-water mark. Unavailable measurements
remain null. These receipts and timings are operational evidence, outside the
scientific manifest and cache identity; they are not model-quality metrics.

## Prepared data: bounded encoding and verification

For `local_text` and `local_stories`, `data prepare`, `stage`, `train`, `run`,
and `experiment prepare` accept `--tokenizer-batch-documents` (1–256, default 256) and
`--tokenizer-batch-source-bytes` (1–4194304, default 1048576). Both bounds apply.
Oversized documents fail before encoding; token limits never truncate a selected
story. Batch 1 uses scalar `Tokenizer.encode`; larger batches use Rust
`encode_batch` in source order, preserving EOS and byte-address semantics.
Preparation uses a fresh, Torch-free tokenizer child with
`TOKENIZERS_PARALLELISM=true` and a Rayon cap installed before its first encode:
`min(4, os.cpu_count() or 1)`, or the explicit envelope's `max_workers`.
Tokenizer training remains scalar with its existing parallelism policy.
Document and byte caps, resource policies, timings, and profiles do not enter
the scientific manifest or cache identity.

New `contiguous-eos-v6` caches describe supervision explicitly:
`{kind: all_tokens}` omits loss-mask arrays and loads masks as `None`;
`{kind: token-loss-mask-v1}` requires exact bool masks for both splits.
Historical v4 no-mask and v5 mandatory-mask artifacts retain their meaning and
are never rewritten. Loss, evaluation, and batching treat implicit all-token
supervision identically to an explicit all-true mask.

Final NPY headers and data feed SHA-256 during their original writes.
`PreparedData.receipt` is sealed **in-process** evidence binding the canonical
root, scientific manifest SHA, file digests/sizes, validated header dtype/shape,
and device/inode/size/mtime_ns fingerprints. Structural reload rejects changed
files, symlinks, unsigned descriptors, and unexpected inventory. The sibling
JSON preparation receipt records measurements only; copying it to a new
process never permits skipping cold array hashes.

The deep-scan call graph changed as follows:

| Path | Before | V6 behavior |
| --- | --- | --- |
| `_prepare_data` fresh write | `_array_metadata` re-read, then postpublish deep load | SHA during write, sealed structural postpublish load |
| `prepare_variant` | Immediate second deep load | Reuse returned `PreparedData` |
| Existing cache | Discovery/deep load, possible caller re-load | One cold deep load |
| `verify_artifact` / `resolve_plan` | Repeated artifact references re-scanned | Operation-local memo, canonical path and fingerprint guarded |
| `verify_prepared_inputs` | Inventory hash then loader hash | Inventory's verified digests feed structural loader |
| `verify_stage_bundle` | Inventory, loader, final bundle inventory could repeat | Same-call digest evidence reused after fingerprint checks |
| External `load_prepared_data` | Deep scan | Still deep by default; no persisted skip authority |

On the generated 2,351-token CPU fixture, v5 creation re-read four arrays
twice: 8 SHA file reads / 24,534 bytes. V6 all-token creation re-read 0 arrays;
cold load, existing-cache hit, and prepared-input verification each read the
two arrays once (2 / 9,660 bytes). The stage bundle has two distinct copies of
each split, so it reads 4 physical array files once each—not one shared file
four times. IDs were exact-byte equal, and omitted masks saved 2,607 bytes.
These are fixture measurements, not predictions for other data or devices.

### Interrupted local preparation

Streaming preparation keeps task-owned `<identity>.tmp` chunks with at most
4096 acquired records and 16 MiB total raw output bytes, ending at whole
documents. One encoded document larger than the raw cap fails with an actionable
error. IDs and byte-address sidecars carry ordered, checksummed receipts with
raw SHA/length/dtype and acquired/retained/token counts. Raw files are fsynced;
the receipt commits last, followed by directory fsync.

A restart verifies source/tokenizer/packing identity and every sealed chunk's
raw bytes. It discards only the known next unsealed chunk, rejects unexpected
files and changed sealed data, re-iterates the verified source, and skips exactly
the sealed acquired count without encoding those records. Completed train
chunks survive validation interruption. Final arrays remain ordinary contiguous
NPYs; interrupted final arrays are recomputed from verified chunks. Cache
preparation is locked even outside the managed workspace, so a live producer is
never mistaken for a crashed one.

Manifest fsync precedes atomic no-replace publication. Owned chunks are retired
only afterward; a durable cleanup intent and owner-last removal make cleanup
itself resumable. Recovery of an already published cache still cold-hashes its
final arrays once and never retokenizes them. Persisted cleanup metadata does
not authorize structural loading without those hashes.

The generated recovery regression has 13 train and 7 validation records with
byte-address sidecars and a forced four-record test chunk. SIGKILL after the
fifth train record's unsealed write preserved records 1–4; restart encoded only
train records 5–13 and all 7 validation records. All four final NPY digests and
the full scientific manifest matched uninterrupted preparation. Interrupting
validation after its first sealed chunk encoded only the remaining 3 validation
records, not the completed train split. Tests also reject changed sealed bytes,
source/tokenizer/owner identity, unexpected entries, and replacement publication.

### Offline performance evidence

Run the standalone utility offline:

```sh
uv run --locked --offline python benchmarks/preparation_benchmark.py \
  --workspace sparselab-work/runtime-prep-v1/benchmarks
```

It generates 1/8/32 MiB mixed prose/code JSONL, one local BPE tokenizer,
scalar and 16/64/256-document cases, and controlled 1 versus bounded multi-thread Rayon.
Raw outputs stay in
the named ignored workspace; the durable result is
[`artifacts/benchmarks/runtime-prep-v1.json`](../artifacts/benchmarks/runtime-prep-v1.json).
Wall time includes imports/child startup and batch-bound instrumentation;
sampled process-tree RSS includes the tokenizer child. This is a standalone
measurement utility, not a CI speed assertion or a model-quality result.

The recorded 32 MiB, four-Rayon-thread cases selected the final default **256**
documents: 5.404 MB/s and 458.2 MiB sampled process-tree peak, versus scalar
1.638 MB/s / 391.4 MiB. The observed input batches stayed below 1 MiB and tree
peak stayed below 1.5 times scalar. At one Rayon thread, batch 16 was faster
than 64/256 on this fixture; the declared default selection uses the bounded
four-thread configuration. Results are one fresh execution per case, not a
statistical portability or causal Rust-only speedup claim. Source re-iteration
on resume, scalar tokenizer training, JSON IPC, and the contiguous final copy
remain measured-performance questions.

The [runtime-prep-v1 acceptance record](../artifacts/acceptance/runtime-prep-v1.json)
binds the actual CPU profile probe, preparation/hash measurements, bounded
recovery checks, CLI smoke, and final focused/broader suite results. It is
operational implementation evidence, not a research finding or a hardware
portability claim.

The [Campaign integration rebase record](../artifacts/acceptance/runtime-prep-v1-rebase.json)
adds actual worker-dispatch/ingestion CLI proof and the combined CPU suites
after integrating Campaign v1. Original benchmark measurements retain their
original source identities; they were not rerun or relabeled as rebase measurements.


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
