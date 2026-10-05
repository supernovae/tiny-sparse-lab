# Build a TinyStories microlab

Train a small story model, inspect its held-out loss, try a completion, and
compare two feed-forward widths using a YAML experiment matrix. The checked-in
[sample files](../experiments/samples/tinystories-microlab/) use a pinned
TinyStories revision, a train-only 2,048-token BPE, and a small dense decoder.
The default is 40 optimizer updates and 10,240 supervised targets per model:
enough to learn the workflow, not enough to expect fluent stories. Follow one
baseline through evaluation and a larger-exposure child before the optional
fresh architecture matrix. No Python scripting is required.

For an authored training program with immutable input bindings and checkpoint
dependencies, continue with [Chain a training program](experiment-programs.md).
Both guides describe teaching examples; they do not register research results.

## Set up a named workspace

Run from the checkout root with Python 3.14 and [uv](https://docs.astral.sh/uv/).
On CPU or macOS, install the locked environment:

```sh
uv sync --locked --extra cpu --dev
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
WORK="$SPARSELAB_WORK_DIR/experiments/tinystories-microlab"
SAMPLE=experiments/samples/tinystories-microlab
mkdir -p "$WORK"
# Choose a new task name if inputs already exists; preserve prior experiments.
mkdir "$WORK/inputs"
cp "$SAMPLE/run.yaml" "$SAMPLE/tokenizer.yaml" "$SAMPLE/continued.yaml" \
  "$SAMPLE/matrix.yaml" "$WORK/inputs/"
```

On a provisioned ROCm/CUDA/XPU worker, use its documented vendor environment
and `uv run --locked --no-sync` for every command below. Running `uv sync` or
`uv run` without `--no-sync` can replace vendor PyTorch with the Linux CPU
lock. See [worker provisioning](workers.md#user-provisioned-ssh-workers).

Keep downloaded source, tokenizers, prepared data, checkpoints and outputs in
this external workspace. The preparation commands bound the selected text, but
the upstream download cache can be larger. Check actual available bytes and
inodes before downloading; allow several GB for the source cache in addition to
the small models. The preflight commands estimate known local growth and do not
bound the upstream download.

Paths in the copied YAML are anchored to `$WORK/inputs`: `../tokenizer`,
`../data` and `../runs` stay within this task workspace. Run the copies, not
the templates in the checkout. No environment substitution inside YAML is
required. Select another backend by editing `runtime.backend` in both copied
run configs before compute; keep every other scientific field matched.

These examples use Bash. See [the iteration guide](iteration.md) for PowerShell
setup and equivalent variable syntax; native commands are the same.
For a generated mechanism lesson instead, use the existing `learn scaffold dense`
with `--scale nano --data tinystories --backend cpu` and a fresh `--output`
directory, then follow its generated README. That is a separate configuration,
not this sample's baseline.

| Host/device | Configuration | Current boundary |
|---|---|---|
| macOS or Linux CPU | `engine: pytorch`, `backend: cpu`, `precision: fp32` | Portable default for this guide. |
| Apple Silicon | `engine: pytorch`, `backend: mps`, `precision: fp32` | Tested Apple GPU path; inspect and stage on your device. |
| Linux AMD GPU | `engine: pytorch`, `backend: rocm`, `precision: fp32` | Measured ROCm training exists; vendor provisioning and a config-specific pilot are required. |
| NVIDIA / Intel GPU | `backend: cuda` / `xpu` | Runtime hooks exist; hardware acceptance is pending. |
| Apple MLX | `engine: mlx`, `backend: metal`, `precision: fp32` | Optional separate engine; see [runtime support](runtime.md). AdamW budget extension below is PyTorch-only. |

Host OS and compute backend are independent. A working CPU example does not
establish acceptance on every accelerator. Keep sequence length, effective batch,
seed, data, optimizer and precision fixed when comparing the two matrix cells.

For MLX on Apple Silicon, install with `uv sync --locked --extra cpu --extra mlx --dev`,
set `engine: mlx` and `backend: metal` in the editable run config, and retain
`--extra cpu --extra mlx` on every `uv run --locked` invocation. Register its worker with
`--engine mlx --backend metal`. The story training and dense-width matrix use
supported MLX features; skip the PyTorch-only budget extension below.

## Prepare and pilot

```sh
df -h "$WORK"
df -i "$WORK"
uv run --locked --extra cpu sparselab workspace preflight "$WORK/inputs/tokenizer.yaml" --tokenizer
uv run --locked --extra cpu sparselab workspace preflight "$WORK/inputs/run.yaml"
uv run --locked --extra cpu sparselab inspect "$WORK/inputs/run.yaml" --json
uv run --locked --extra cpu sparselab tokenizer train "$WORK/inputs/tokenizer.yaml"
uv run --locked --extra cpu sparselab data prepare "$WORK/inputs/run.yaml"
uv run --locked --extra cpu sparselab stage "$WORK/inputs/run.yaml" \
  --through warmup --output "$WORK/stages/baseline"
```

The first preparation requires network access to TinyStories. Later runs reuse
the pinned source/cache; tokenizer fitting uses training text only. `stage`
launches disposable smoke/warmup pilots, records timing and memory, and leaves
the full training run's initial state untouched. Use a new stage output path
for another pilot.

## Train, evaluate and generate

```sh
uv run --locked --extra cpu sparselab train "$WORK/inputs/run.yaml" \
  --runs-dir "$WORK/runs" --run-id stories-base \
  --stage-bundle "$WORK/stages/baseline"
uv run --locked --extra cpu sparselab checkpoint verify \
  "$WORK/runs/stories-base/checkpoints/latest.json" --json
uv run --locked --extra cpu sparselab eval stories-base --runs-dir "$WORK/runs"
uv run --locked --extra cpu sparselab evidence stories-base --runs-dir "$WORK/runs" --json
uv run --locked --extra cpu sparselab triage stories-base --runs-dir "$WORK/runs"
uv run --locked --extra cpu sparselab generate stories-base --runs-dir "$WORK/runs" \
  --prompt "Once upon a time, a little rabbit" --max-new-tokens 32 --seed 42
uv run --locked --extra cpu sparselab dashboard --runs-dir "$WORK/runs"
```

The dashboard runs in the foreground; stop it or use another terminal for the
next steps. This is language-model training, so story continuations are the
natural first interaction. Record held-out loss and actual target counts beside
the generated text; neither a short completion nor lower loss establishes a
general assistant. Use new run IDs for repetitions. For a single fresh run
without separate staging, the existing composed
`sparselab run "$WORK/inputs/run.yaml" --store "$WORK/runs"` prepares,
queues, pilots, trains and waits for ingestion. Choose that route or staged
`train` above; running both creates another baseline.

## Iterate on exposure while preserving the parent

The copied [continued.yaml](../experiments/samples/tinystories-microlab/continued.yaml)
is a complete child config. Its intended scientific delta is:

| Field | Baseline | Child |
|---|---:|---:|
| `training.max_steps` | 40 | 80 cumulative |
| `training.max_tokens` | 10,240 | 20,480 cumulative |
| Effective decay horizon | 40 (implicit) | 40 (explicit `optimizer.decay_steps`) |

The child adds 40 updates / 10,240 targets and retains other scientific
settings. It restores full optimizer state, cursor, counters and RNG and
continues at the original learning-rate floor. This is PyTorch AdamW budget
extension; skip it for MLX. If you edited the baseline backend or other settings,
make the identical edits in the copied child config before compute.

Inspect the terminal parent using native commands:

```sh
uv run --locked --extra cpu sparselab checkpoint inspect \
  "$WORK/runs/stories-base/checkpoints/latest.json" --json
ls "$WORK/runs/stories-base/checkpoints"
```

Copy the full finalized `step_00000040_gen_...` directory name corresponding
to the inspected terminal checkpoint. Replace `ACTUAL` below with that suffix.
Do not select a verified parent by sorting filenames. Verify and inspect the
exact directory: its SHA must match the inspected pointer and its counters
must be 40 updates and 10,240 targets.

```sh
GENERATION="$WORK/runs/stories-base/checkpoints/step_00000040_gen_ACTUAL"
uv run --locked --extra cpu sparselab checkpoint verify "$GENERATION" \
  --config "$WORK/inputs/run.yaml" --json
uv run --locked --extra cpu sparselab checkpoint inspect "$GENERATION" --json
uv run --locked --extra cpu sparselab inspect "$WORK/inputs/continued.yaml"
uv run --locked --extra cpu sparselab workspace preflight "$WORK/inputs/continued.yaml"
uv run --locked --extra cpu sparselab train "$WORK/inputs/continued.yaml" \
  --runs-dir "$WORK/runs" --run-id stories-continued --extend-budget "$GENERATION"
uv run --locked --extra cpu sparselab checkpoint verify \
  "$WORK/runs/stories-continued/checkpoints/latest.json" --json
uv run --locked --extra cpu sparselab eval stories-continued --runs-dir "$WORK/runs"
uv run --locked --extra cpu sparselab evidence stories-continued --runs-dir "$WORK/runs" --json
uv run --locked --extra cpu sparselab triage stories-continued --runs-dir "$WORK/runs"
uv run --locked --extra cpu sparselab generate stories-continued --runs-dir "$WORK/runs" \
  --prompt "Once upon a time, a little rabbit" --max-new-tokens 32 --seed 42
uv run --locked --extra cpu sparselab checkpoint verify "$GENERATION" --json
```

Full-state extension validates scientific compatibility before creating a child.
Read the native evidence's actual counters and parent digest. Compare heldout
loss and identical prompt/decoder observations; retain repetitive or worse
outputs. Select the child immutable generation the same way and use
`--checkpoint` for retained endpoint evaluation/generation. Lower loss does
not establish better prose. Triage reads retained advice; missing advice does
not authorize another run.

An interrupted child uses `--resume` with its unchanged child config, verified
checkpoint and a new run ID. Preserve the interrupted attempt. `--promote`
resets training state and answers another question. Source/runtime drift fails
unless explicitly authorized by the native contract; do not bypass it with
Python. See [checkpointing](checkpointing.md).

This direct teaching route still asks the operator to copy an immutable
generation path. ExperimentPlan's `parent`, `selector: terminal` and `at_step`
already bind that selection declaratively. Native `experiment bind-inputs`
authenticates existing direct story inputs for a phase template, and
`iteration check` reports the continuation's gates without executing it.
A complete declared Campaign demo remains
[tracked implementation work](../TODO.md#rapid-iteration). Use
[the iteration guide](iteration.md) for current checks between runs.

## Optional fresh architecture comparison with the matrix DSL

The sample matrix changes one field, `model.ffn_dim`, from 256 to 384:

```yaml
matrix_version: 1
base_config: run.yaml
axes:
  architecture:
    - label: baseline
      set: {}
    - label: wider-ffn
      set: {model.ffn_dim: 384}
```

Inspect its coordinates before queueing. This trains two fresh runs; the earlier
direct run is not imported into the matrix. Registration must use the backend
you chose in `run.yaml`; replace `cpu` below when appropriate.

```sh
uv run --locked --extra cpu sparselab experiment submit --matrix "$WORK/inputs/matrix.yaml" \
  --dry-run --store "$WORK/runs"
uv run --locked --extra cpu sparselab worker register stories-local --backend cpu --store "$WORK/runs"
uv run --locked --extra cpu sparselab experiment submit --matrix "$WORK/inputs/matrix.yaml" \
  --worker stories-local --store "$WORK/runs"
uv run --locked --extra cpu sparselab controller run --store "$WORK/runs"
```

The controller stays running and executes the queued models independently on
one selected device. In a second terminal with the same `WORK`, check both
execution and ingestion status:

```sh
uv run --locked --extra cpu sparselab experiment list --store "$WORK/runs"
```

Once both have `status: COMPLETE` and `ingestion_status: COMPLETE`, use the
returned run IDs for `eval`, `evidence`, `generate` and the dashboard commands
above. Compare at matching actual steps/targets with the same tokenizer and
held-out source. This one-seed width comparison is descriptive; it does not
establish a universal winning architecture. Add labeled seed choices for a
larger declared matrix. [Controlled comparisons](experiments.md) explains
study receipts, task-specific capability cards and collection when your
experiment has a separately defined behavioral test.

All mutable output belongs to this task's workspace. Preserve interrupted and
failed attempts while investigating them; do not edit or prune active runs.
For a larger program, copy these inputs into a new named workspace, inspect
the new budget and storage estimate, and repeat the pilot before training.
