# Independent workers

One worker executes a whole experiment with its own optimizer, RNG, local database, checkpoints, and immutable inputs. The controller owns a separate local queue and read-only-result projection. Local and SSH workers use the same versioned, finite stdio endpoint; neither is a persistent GPU daemon. This is independent experiment scheduling, not distributed training.

## Local operation

Prepare the tokenizer and use a concrete configuration whose backend and precision the worker supports:

```sh
sparselab worker register cpu-one --backend cpu --store /tmp/sparselab-controller
sparselab worker status cpu-one --json --store /tmp/sparselab-controller
sparselab experiment submit configs/runtime_smoke_cpu.yaml --worker cpu-one --store /tmp/sparselab-controller
sparselab controller run --store /tmp/sparselab-controller
```

Run the controller in its own terminal. From another terminal, use the same `--store`:

```sh
sparselab experiment list --json --store /tmp/sparselab-controller
sparselab experiment cancel RUN_ID --store /tmp/sparselab-controller
sparselab experiment resume RUN_ID --worker cpu-one --store /tmp/sparselab-controller
```

`--worker NAME` is a hard binding. Without one, queued submissions use eligible idle registered workers; a preferred worker is only a tie-breaker. Eligibility checks engine, backend, features, tested precision, required schema/codecs, source identity, and measured memory requirements. Unknown capacity never satisfies a numeric minimum. No eligible worker means `QUEUED` with a reason, not a silently changed configuration.

For a single composed local experiment, `sparselab run CONFIG --store ROOT` registers a controller-scoped local endpoint and drives the same queue, isolated smoke/warmup, training, and ingestion path. `run CONFIG --worker NAME` targets an existing worker. If another controller owns the store lock, `run` observes it rather than competing for assignments. `train CONFIG` remains the direct single-host entry point and does not silently run staging pilots.

`controller run` and composed `run` accept `--transfer-timeout SECONDS` (default 1800). It is a finite per-RPC deadline for bundle and artifact transfers, including hashing and transmission; increase it explicitly for a slow link or large input. Metadata requests have shorter bounded deadlines. Killing the controller never authorizes re-execution of an assigned optimizer attempt.

## User-provisioned SSH workers

The following is an example for an already provisioned host, not evidence that such a host is available:

```sh
sparselab worker register amd --backend rocm --ssh amd-host \
  --python /opt/sparselab/bin/python --root /srv/sparselab/amd \
  --store /tmp/sparselab-controller
```

The SSH alias must already be configured in the user's SSH environment. Transport uses strict host-key checking and `BatchMode=yes`; it does not store passwords, accept unknown host keys, install software remotely, or execute commands from experiment configurations. Interpreter and worker-root paths are absolute. The endpoint runs a fixed, shell-quoted agent command. Worker roots are private and bound to an immutable registration.

Provision each target with its vendor-compatible PyTorch/runtime and the project wheel. The source checkout's Linux CPU package index is for CPU CI; do not blindly apply that CPU-locked environment to accelerator workers. Record actual driver/framework/runtime versions and perform the worker's concrete validation. PyTorch MPS and optional MLX Metal are different engines sharing the same physical Apple GPU lease.

## Preparation and durable identity

Submission verifies and seals an engine-neutral dispatch bundle before atomically reserving experiment, attempt, and run IDs. It contains exact tokenizer/data/package bytes, configuration and source identities, supported version requirements, and selected continuation artifacts. Original input paths are not needed after successful preparation. Promotion must match the requested tokenizer/package identity before any path is rebased; full resume can prepare from verified parent-owned inputs after the originals are gone.

The worker checks its content-addressed cache, installs only missing verified assets, revalidates its actual runtime, and runs disposable smoke/warmup pilots. Pilots are separate runs and never advance the experiment's optimizer, schedule, cursor, RNG, or target budget. Memory failure produces a proposal rather than silently reducing scientific settings.

A durable `PREPARED` receipt precedes spawning. A nonblocking attempt claim prevents duplicate wrappers from training. A claimed execution records PID, process-start identity, boot identity, start token, and phase before training. Replaying the same launch returns the same receipt; conflicting identities fail. `RUNNING` and terminal attempts are never restarted under their old IDs.

Each configured worker has one logical slot. Host-wide OS leases additionally serialize aliases of the same accelerator across scratch roots; an unknown accelerator identity conservatively excludes other accelerator work. Isolated pilots inherit the lease, so killing their wrapper cannot release hardware while a child still owns it. This namespace assumes one service account per host, not coordination across unrelated user accounts.

## Cancellation, disconnects, and explicit recovery

- Cancelling a queued experiment creates no executor. An assigned cancellation remains durable through a lost RPC and is redelivered until evidence arrives.
- A marker arriving before preparation is retained without inventing a receipt or acknowledgement. Once preparation is known and unclaimed, cancellation can acknowledge without initializing a model.
- Staging checks cancellation at pilot boundaries; an executing pilot and the trainer stop at committed update boundaries. After model initialization, even a step-zero interruption has a real full checkpoint. There is no fabricated pre-initialization checkpoint.
- `CANCEL_REQUESTED` is intent. Acknowledged interruption becomes `CANCELLED`; natural completion already committed at the budget wins a late request.
- A controller disconnect does not stop worker updates. Reconnection resumes record/artifact ingestion, not optimizer execution.
- A lost executor is finalized only after its attempt claim and inherited resource holders are gone. It remains `UNKNOWN`; verified finalized generations are available for an explicit child resume. Missing crash-window manifest/checkpoint projections can be reconstructed from those verified files without creating an echo outbox. Conflicting metadata is rejected, never overwritten.
- Resume creates a new visible child with the selected full-state digest. It requires compatible same-engine/backend training state and a remaining scientific budget. Complete budgets cannot resume. Source/runtime drift requires explicit `--allow-runtime-drift` and remains best-effort, not an architecture/data compatibility bypass.

Execution state and `ingestion_status` are separate durable facts. A remotely complete receipt does not mean its checkpoints are already locally available. Composed `run` waits for `COMPLETE` or `NOT_REQUIRED` ingestion for every terminal outcome, including cancellation and failure. Queued cancellation and failures with no run artifacts require no transfer; they do not fabricate a checkpoint. Full resume is admitted only after verified local ingestion. Artifact-transfer errors remain visible and can be retried without rerunning the optimizer.

## Local aggregation and integrity

Every worker's database/WAL stays local. The controller imports contiguous `(origin_id, sequence)` envelopes, rejects conflicting duplicates or gaps, and never re-emits imported records into its own outbox. Live runs therefore have real local metric series before completion.

Terminal artifacts are downloaded into a private temporary directory, checked against the receipt and sealed dispatch, then atomically published. Checks bind attempt/run/experiment/worker IDs, requested and effective configurations, source policy, continuation lineage, matrix coordinate, exact input identities, full-native checkpoint state, terminal counters, and replicated manifest/checkpoint metadata. Checkpoint generations use their embedded canonical digest; a file's transport hash is not that identity. Logs are sealed snapshots rather than mutable live files. Failed pre-training attempts with only diagnostic logs do not claim a resumable run.

Protocol v1 uses a bounded canonical JSON header and length/hash-framed attachments. Unknown versions/operations, duplicate JSON keys, nonfinite/coerced scalar values, unsafe names, symlink destinations, excess sizes, truncated bodies, and conflicting hashes fail closed. Error responses cannot carry attachments. A `cancel` response may contain no receipt only while its marker precedes preparation; acknowledgement is then false.

Native state codecs are engine-specific: PyTorch uses `pytorch_native/v2`; MLX uses `mlx_native/v1`. Capability advertisement and dispatch requirements derive these versions from their implemented codecs rather than treating a generic checkpoint schema version as every engine's native-state version.

## Explicit matrices

A version-1 matrix names a base configuration and insertion-ordered axes. Each option has a label and explicit dotted-path `set` patch. Expansion rejects duplicate labels, overlapping paths, unknown keys, invalid resulting configurations, and more than 1000 coordinates unless `--max-runs` is explicitly increased.

```sh
sparselab experiment submit --matrix MATRIX.yaml --dry-run --store /tmp/sparselab-controller
sparselab experiment submit --matrix MATRIX.yaml --store /tmp/sparselab-controller
```

Dry-run resolves the complete coordinates, configuration hashes, and scheduling constraints without creating the controller store, preparing data, probing devices, or enqueueing. Non-dry-run preparation must succeed for every coordinate before the queue transaction. Each coordinate receives independent IDs and retains the matrix digest and labels in its spec and resulting manifest.

## Comparison and hardware limits

The read-only dashboard can use the controller root as `--runs-dir`. Compare actual observed losses/counters and keep differences in optimizer, precision, architecture, data/tokenizer, effective batch, backend, and budget visible. Equal worker names or similar parameter counts do not make a controlled scientific comparison.

Logical workers named for Mac, AMD, or Intel roles must truthfully report CPU when executed on CPU. Real ROCm, XPU, and overlapping Mac/AMD/Intel acceptance remain blocked without those actual provisioned hosts. SSH framing and quoting tests do not establish remote driver or kernel performance.

No worker shares optimizer state or exchanges gradients/expert tokens with another. There is no process group, all-reduce, expert all-to-all, tensor/model sharding, or heterogeneous distributed backward. The hardware and distributed research gates remain separate in `TODO.md`.

## Observed acceptance — 2026-09-23

[Astra's gate record](../artifacts/acceptance/independent_workers_2026_09_23.json) binds the execution wheel, source identity, commands, native checkpoints, counters, independent checks and retained evidence hashes. The final suite passed 335 tests, Ruff lint and formatting checks.

- Three genuine CPU workers overlapped with FP32 AdamW, FP32 Adafactor and BF16 AdamW. Their updates continued while the controller was stopped; reconnect imported contiguous, deduplicated records with no echo outbox.
- Replaying launch retained the original process identity. Cancellation stopped at update 26,943; an explicit child reached 32,768 updates / 2,097,152 targets. Its model, optimizer, RNG, cursor, scaler, schedule and counters matched an uninterrupted baseline bitwise. Wall-clock checkpoint cadence was not compared bitwise.
- A separate identity-checked executor SIGKILL left an immutable UNKNOWN attempt. Its verified step-256 checkpoint resumed only through a new child, reaching exactly 32,768 updates / 1,048,576 targets without changing parent files.
- Installed CLI matrix submission executed all three coordinates at exactly 2 updates / 64 targets. Actual incompatible engine/capacity and a worker with genuinely different installed source remained queued without an executor.
- Core-only installed execution after removing owned original inputs proved offline promotion, fresh optimizer/RNG/counters, evaluation, generation, unchanged parent files, and composed local `run` without prior registration. An actual MLX/Metal worker completed 4 updates / 128 targets; its full native checkpoint also verified from the core-only installation without MLX.
- [Captured dashboard curves](../artifacts/acceptance/workers_training_2026_09_23.svg) contain actual browser WebGL pixels and SVG overlays, not regenerated series. Standard screenshot capture and runtime-table rasterization stalled, including independent browser attempts. The real Runtime view was additionally exercised through Streamlit's app harness: all three CPU/worker/precision/optimizer rows were correct and the database data version was unchanged.

These synthetic workloads establish orchestration and state-transition behavior, not model quality. Worker labels do not establish AMD/Intel hardware execution. No actual SSH target, CUDA/HIP kernel, ROCm/XPU acceptance, or overlapping real three-host result is claimed.
