# Optional hosted GPU environments

SparseLab can execute a prepared run on a user-provisioned CUDA host, including
an explicitly supplied Google Colab session or a full-SSH machine. Placement is
operational: it does not change the declaration's scientific identity or make a
hosted backend the default. The feature does **not** create, upgrade, stop, or
delete provider instances, and it never authorizes paid allocation.

This guide is an operational procedure, not evidence that a particular account,
GPU, provider, relay, or kernel is available. All paths below are task-owned and
outside the checkout unless stated otherwise.

## Boundaries and gates

- GPU work uses the bounded `cuda-cu126-v1` environment recipe. It requires the
  selected CUDA wheel, a usable CUDA device, and a real FP16
  forward/backward/GradScaler optimizer update. A CPU or HIP wheel is not a
  substitute.
- Dense SDPA is opt-in. Calling SDPA does not itself prove FlashAttention or a
  fused kernel. The runtime attention probe reports the actual selected backend
  and performs backward for each supported candidate.
- A free T4 acceptance requires observed NVIDIA T4/SM75 identity, a CUDA build,
  FP16 scaler evidence, and successful forward **and backward** evidence for
  the memory-efficient SDPA candidate. If unrestricted SDPA selects math, retain
  that result and do not claim fused acceptance.
- TPUs are detected and reported as `TPU_UNSUPPORTED`; SparseLab has no
  executable TPU backend. PyTorch/XLA is a separate backlog item, not a CPU
  fallback.
- Relay publication is synchronous at finalized checkpoint boundaries. It
  protects the last verified committed checkpoint, not in-flight optimizer work.
  A user-mounted Drive filesystem is not a training/runtime root and is not
  equivalent to an independently verified Drive API copy.

The CPU-only native readiness command verifies parent/child full-state checkpoint
and resume wiring. It does **not** certify CUDA, Colab, SSH-provider access,
cloud storage, throughput, or model quality.

[PyTorch SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
can select math or another implementation. The external
[FlashAttention project](https://github.com/Dao-AILab/flash-attention) is not
integrated: its FA2 CUDA path excludes T4/Turing, while FA3 is a separate
H100/H800 CUDA >=12.3 build.


## Sample and local CPU preparation

The copyable declaration set is
[`experiments/samples/hosted-tinystories/`](../experiments/samples/hosted-tinystories/).
It retains the TinyStories microlab's exact pinned source and train-only
2,048-token tokenizer declarations. The acceptance declaration is a 6-layer,
256-hidden, 4-head dense model with FFN 704 and max sequence 256. It uses SDPA,
FP16 CUDA, sequence length 128, `(microbatch, accumulation) = (8, 1)`, 20
steps / 20,480 targets, AdamW peak/floor `0.0003`/`0.00003`, four warmup steps,
and checkpoints/evaluation every five steps.

The declaration explicitly sets `training.deterministic: false` for this CUDA
performance sample. It is not a bitwise-reproducibility claim. The checked-in
sample remains `(8, 1)`. The two matrix declarations are pilots only: reference
versus SDPA changes only `attention.implementation`; `(4, 2)` versus `(8, 1)`
changes only the divisible batch pair. Preserve OOM and unavailable-kernel
observations; select a proposal only after a successful measurement at at most
90% observed VRAM, recording targets/second, post-initialization step
distribution, memory, and failures.

Prepare inputs locally on CPU; these commands must not initialize CUDA:

```sh
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
WORK="$SPARSELAB_WORK_DIR/experiments/hosted-environments-acceptance"
mkdir -p "$WORK/inputs"
cp experiments/samples/hosted-tinystories/source.yaml "$WORK/inputs/"
cp experiments/samples/hosted-tinystories/tokenizer.yaml "$WORK/inputs/"
cp experiments/samples/hosted-tinystories/run-t4.yaml "$WORK/inputs/"
uv run --locked --extra cpu sparselab workspace preflight "$WORK/inputs/tokenizer.yaml" --tokenizer
uv run --locked --extra cpu sparselab inspect "$WORK/inputs/run-t4.yaml" --json
uv run --locked --extra cpu sparselab workspace preflight "$WORK/inputs/run-t4.yaml"
uv run --locked --extra cpu sparselab data lock "$WORK/inputs/source.yaml" --output "$WORK/inputs/source.lock.json"
uv run --locked --extra cpu sparselab data snapshot "$WORK/inputs/source.lock.json" --output "$WORK/snapshot"
uv run --locked --extra cpu sparselab tokenizer train "$WORK/inputs/tokenizer.yaml"
uv run --locked --extra cpu sparselab data prepare "$WORK/inputs/run-t4.yaml"
```

Retain the prepared-data provenance and native output. Reuse already verified
immutable inputs through the native commands; do not redownload or retrain the
tokenizer merely because execution changes host.

## Relay profile

Copy either `relay-drive.example.yaml` or `relay-s3.example.yaml` from the
sample directory to `$WORK/relay.yaml`. Both sides must name the same namespace
and remote prefix. The profile holds only local rclone configuration *paths*,
never credentials. Configure rclone interactively or through the provider's
least-privilege environment credentials; do not copy a controller credential
file to a VM. Missing rclone, config, or authorization is an actionable
`RELAY_UNAVAILABLE` condition, not permission to weaken verification.

The relay is a content-addressed, verified copy protocol. It does not mount
cloud storage as a live training filesystem, synchronize SQLite/WAL files, or
trust provider success text, mtimes, sizes, or multipart ETags.

Every attempt also has a controller-private, exact 32-byte HMAC relay key.
It authenticates immutable commit descriptors but is delivered separately to
the private worker attempt directory; it is never uploaded to the relay or
included in a profile, source bundle, scientific declaration, receipt, or log.


## Colab: supplied free T4 only

Use the installed public `colab` CLI, not private token files or undocumented
HTTP APIs. First inspect the authenticated account without creating a session:

```sh
colab version
colab sessions
colab usage
# SESSION must identify an existing, idle, user-approved free T4 session.
colab status -s "$SESSION"
```

If account/CLI information cannot establish a zero-charge allocation, do not
spend credits or probe repeatedly. In the browser, use **Runtime → Change
runtime type → T4 GPU**, verify it is a free supplied session, and provide that
session name. Do not use high-memory, paid upgrades, a paid fallback, or stop an
unrelated session. `colab new -s sparselab-t4-acceptance --gpu T4` is only an
exact manual allocation command after free availability is established; a T4
request alone does not prove it is free.

Before inspection or setup, enroll the dedicated notebook manually. This creates
the boot-bound nonce and the event-driven occupancy observer; generating the
cell itself submits no provider work:

```sh
ENROLLMENT="$WORK/colab-enrollment.py" # an absolute, fresh local path
uv run --locked --extra cpu sparselab hosted notebook-cell \
  --root /content/sparselab --output "$ENROLLMENT" --json
```

Open the supplied dedicated notebook and run the generated fixed cell once.
It must finish before `hosted inspect` or `hosted setup`: an observer state of
`BUSY`, missing state, changed boot/nonce, stale observation, or an unverifiable
live PID/start time is refused rather than queued behind unknown work. Custom
task roots must be passed consistently with `--root`; omitted Colab inspection
observes `/content` but enrollment remains `/content/sparselab`.

From the fixed, committed feature revision, create a source bundle and discover
actual host properties before setup:

```sh
git bundle create "$WORK/source.bundle" HEAD
SHA="$(git rev-parse HEAD)"
uv run --locked --extra cpu sparselab hosted inspect \
  --colab-session "$SESSION" --root /content/sparselab --json
uv run --locked --extra cpu sparselab hosted setup \
  --colab-session "$SESSION" --source-bundle "$WORK/source.bundle" \
  --source-commit "$SHA" --root /content/sparselab \
  --runtime-root /content/sparselab-runtimes --recipe cuda-cu126-v1 --json
```

`inspect` reports requested/session accelerator metadata separately from the
observed GPU name, capability, VRAM, driver, framework build, and storage
measurements. A request is not hardware proof. Use setup's returned
`source_root`, `worker_root`, `runtime_id`, absolute `python`,
`environment_path`, and native provision receipt; do not assume the notebook
interpreter or `/content` is durable. Setup verifies the immutable source
commit, uses private task-owned tools/cache and a separate managed Python 3.14
vendor environment, checks capacity before writes, and refuses an incomplete,
mismatched, or stale-device receipt rather than overwriting it. It does not
install drivers, replace the notebook kernel, or sync the CPU extra.

With `REMOTE_PYTHON` and `REMOTE_WORKER_ROOT` copied from setup's structured
result, register and stage the worker:

```sh
uv run --locked --extra cpu sparselab worker register colab-t4-acceptance \
  --colab-session "$SESSION" --backend cuda --engine pytorch \
  --python "$REMOTE_PYTHON" --root "$REMOTE_WORKER_ROOT" \
  --relay-profile "$WORK/relay.yaml" --store "$WORK/runs"
uv run --locked --extra cpu sparselab worker relay check "$WORK/relay.yaml" \
  --worker colab-t4-acceptance --store "$WORK/runs" --json
uv run --locked --extra cpu sparselab runtime attention "$WORK/inputs/run-t4.yaml" \
  --worker colab-t4-acceptance --store "$WORK/runs" --json
uv run --locked --extra cpu sparselab run "$WORK/inputs/run-t4.yaml" \
  --worker colab-t4-acceptance --store "$WORK/runs"
```

The foreground Colab execution occupies one notebook kernel for real work.
While it is occupied, status, records, artifacts, and cancellation use bounded
Contents/relay paths; they do not issue another kernel `exec`. Ctrl-C or a
controller exit is passive: it neither cancels a delivered execution nor
authorizes resubmission. Reconnect and collect existing evidence instead.
`experiment cancel RUN_ID --store "$WORK/runs"` uploads one authenticated
cancellation intent; only the worker's genuine terminal receipt acknowledges
it. A missing session, changed nonce, stale status, failed CLI process, or
disconnected VM is `UNKNOWN`, not completion, death, or a reason to run on CPU.
Fresh Contents observations and live-attempt heartbeats must remain within the
30-second operational freshness bound. A terminal receipt is immutable: its
completion timestamp may precede a slow final relay upload and does not expire
merely because that upload takes longer than 30 seconds.

When the terminal receipt or a verified durable checkpoint is available,
collect without rerunning the executor:

```sh
uv run --locked --extra cpu sparselab experiment collect "$RUN_ID" \
  --relay-profile "$WORK/relay.yaml" --store "$WORK/runs" --json
uv run --locked --extra cpu sparselab checkpoint verify "$GENERATION" --json
uv run --locked --extra cpu sparselab evidence "$RUN_ID" --runs-dir "$WORK/runs" --json
uv run --locked --extra cpu sparselab triage "$RUN_ID" --runs-dir "$WORK/runs" --json
```

Recovery from a confirmed lost worker is explicit and only selects a fully
verified committed checkpoint:

```sh
uv run --locked --extra cpu sparselab experiment collect "$RUN_ID" \
  --relay-profile "$WORK/relay.yaml" --store "$WORK/runs" \
  --recover --confirm-worker-lost --json
```

This keeps the original attempt `UNKNOWN` and records a recovery observation
while preserving the original receipt; it does not fabricate terminal evidence.
Resume, if eligible, is a new child attempt on a newly registered worker. An
operator confirmation fences further dispatch but cannot prove provider
termination: a later original receipt or relay commit is retained as a
split-brain conflict, never merged into the recovered child. With no full
verified generation, retain `UNKNOWN` with `NO_DURABLE_CHECKPOINT`.

### Manual notebook fallback

If CLI execution is unavailable but an explicitly supplied session is usable,
use [`colab-fallback.ipynb`](../experiments/samples/hosted-tinystories/colab-fallback.ipynb).
It contains only the native worker execution invocation. Fill its values from
the fixed setup and already-prepared attempt receipts; do not write a separate
Python trainer, hashing loop, relay client, or orchestrator. First complete
local CPU preparation, bundle transfer, `hosted setup`, worker registration,
relay check, and attention probe above. Collect through the relay afterward.
The same T4/FP16/SDPA evidence gate applies: a notebook cell cannot turn a CPU
result or an unchecked API call into GPU acceptance.

## User-provisioned full SSH: Runpod and Vast

Runpod, Vast, and comparable hosts reuse the CUDA recipe, relay, native worker
protocol, and collection flow. SparseLab does not allocate them. For Runpod,
obtain an existing pod's full-TCP SSH endpoint from its console/official CLI;
its restricted proxy SSH is insufficient. For Vast, obtain the assigned host
and port from its console/official CLI and inspect its storage and bandwidth
before staging. In both cases use key-authenticated full SSH with strict
host-key verification. Do not disable host-key checking or auto-accept a
recycled host.

```sh
uv run --locked --extra cpu sparselab hosted inspect \
  --ssh "$SSH_HOST" --root /absolute/task/root --json
uv run --locked --extra cpu sparselab hosted setup \
  --ssh "$SSH_HOST" --source-bundle "$WORK/source.bundle" \
  --source-commit "$SHA" --root /absolute/task/root \
  --runtime-root /absolute/task/runtimes --recipe cuda-cu126-v1 --json
uv run --locked --extra cpu sparselab worker register ssh-cuda-host \
  --ssh "$SSH_HOST" --backend cuda --engine pytorch \
  --python "$REMOTE_PYTHON" --root "$REMOTE_WORKER_ROOT" \
  --relay-profile "$WORK/relay.yaml" --store "$WORK/runs"
```

Run the same relay check, attention probe, run, and offline collection commands
as the Colab procedure, substituting `ssh-cuda-host`. Never treat a provider
root as durable solely because the VM remains running.

## Selecting a provider

- [Google Colab](https://research.google.com/colaboratory/faq.html) is for an
  interactive, opportunistic session explicitly supplied by the user.
- [Runpod's SSH guide](https://docs.runpod.io/pods/configuration/use-ssh)
  distinguishes its basic proxy (no SCP/SFTP) from the required full public-IP
  SSH endpoint.
  Check [Runpod's current pricing](https://www.runpod.io/pricing) for on-demand
  GPU and storage rates, rather than assuming that a quoted GPU is available.
- [Vast storage types](https://docs.vast.ai/guides/instances/storage/types.md)
  describe instance-destruction loss and stopped-storage billing that relay
  recovery must not obscure.
  [Vast's live pricing](https://vast.ai/pricing) varies with host reliability,
  bandwidth, storage, region, and currently rentable offers.
- [Thunder Compute pricing](https://www.thundercompute.com/pricing) is another
  full-SSH/per-minute reference.

These links report live pricing and availability; this document deliberately
does not rank a "cheapest" host or promise inventory. Compare warmup/setup,
transfer, persistent storage, and egress with GPU time. No lifecycle adapter in
this repository authorizes purchasing, upgrading, stopping, or deleting an
instance.

Runpod, Vast, Thunder Compute, and future SSH providers remain
user-provisioned only: require key-authenticated, strict-host-key,
noninteractive full SSH, an explicit private root, and the native setup
receipt. Do not use Docker requirements, auto-allocation, password/proxy SSH,
or automatic host-key acceptance.

## Recording acceptance

A successful local CPU preparation, a CUDA recipe installation, and a relay
round-trip are separate observations. For the T4 sample, retain actual device,
CUDA build, probe report, staged runtime identity, targets/second, VRAM,
checkpoint/relay timings, collected immutable checkpoint, and terminal
ingestion evidence. Do not claim model quality from this 20-update smoke.

For a controlled loss demonstration, use a new task-owned run with a verified
step-five relay checkpoint and an owned executor interruption before the next
durable boundary. Verify relay recovery before any loss action; never delete the
sole good copy or an unrelated VM.
