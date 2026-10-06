# Build a TinyStories microlab

Learn the whole loop: start with a tiny model, inspect its predictions, continue
training, compare a new model, and plan a larger-data exercise. A tokenizer turns
text into token IDs; the model learns to predict the next ID; held-out loss
measures those predictions on text not used for optimizer updates. The checked-in
[sample files](../experiments/samples/tinystories-microlab/) use a pinned
TinyStories revision, a train-only 2,048-token BPE, and a small dense decoder.
The default is 40 optimizer updates and 10,240 supervised targets per model:
enough to learn the workflow, not enough to expect fluent stories. Follow one
baseline through evaluation and a larger-exposure child before the optional
fresh architecture matrix. No Python scripting is required.

## Your route through the exercise

| Step | What you learn | Additional work |
| --- | --- | --- |
| [Set up](#set-up-a-named-workspace) and [pilot](#prepare-and-pilot) | Inputs, storage and execution checks | Preparation and disposable pilot updates. |
| [First model](#train-evaluate-and-generate) | Loss, checkpoints and story continuations | One 40-update baseline. |
| [Continue](#iterate-on-exposure-while-preserving-the-parent) | More exposure with the parent preserved | Another 40 updates; 80 cumulative. |
| [Validate each iteration](#what-to-check-between-iterations) | Decide what the evidence supports | Read-only inspection and selected evaluation. |
| [Compare models](#optional-fresh-architecture-comparison-with-the-matrix-dsl) | A controlled width change | Two fresh runs; earlier runs are not imported. |
| [Use a declared program](#optional-declare-the-comparison-and-continuation) | Bound inputs, comparisons and checkpoint phases | A separate four-run practice program. |
| [Scale up](#grow-the-model-and-the-data-deliberately) | More optimization, source variety or capacity | A new budgeted experiment. |
| [Full-data exercise](#an-advanced-full-tinystories-exercise) | Source coverage versus actual training exposure | Potentially substantial preparation and training. |

Start with the first three steps on CPU. Later branches are optional. This is a
teaching reference for the lab's general iteration pattern, not a dedicated
TinyStories training engine. Current dataset-specific ingestion and Campaign
gaps are identified below and in [TODO.md](../TODO.md#rapid-iteration).

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
| Apple Silicon | `engine: pytorch`, `backend: mps`, `precision: fp32` | PyTorch execution on the Apple GPU; inspect and stage your config. |
| Linux AMD GPU | `engine: pytorch`, `backend: rocm`, `precision: fp32` | PyTorch execution on an AMD GPU; requires a provisioned vendor environment. |
| NVIDIA / Intel GPU | `backend: cuda` / `xpu` | Requires a compatible vendor framework and drivers; check workload compatibility with local staging. |
| Apple MLX | `engine: mlx`, `backend: metal`, `precision: fp32` | Optional separate engine; see [runtime support](runtime.md). AdamW budget extension below is PyTorch-only. |

Host OS and compute backend are independent. Before selecting a GPU, register
and doctor its interpreter using [runtime environments](runtime.md#machine-local-runtime-environments),
then add `--runtime YOUR_RUNTIME_ID` to `stage`, `train`, `eval` and `generate`.
Changing the YAML backend alone does not authorize GPU execution.
Keep sequence length, effective batch,
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
PREPARED=$(uv run --locked --extra cpu sparselab data prepare "$WORK/inputs/run.yaml")
uv run --locked --extra cpu sparselab stage "$WORK/inputs/run.yaml" \
  --through warmup --output "$WORK/stages/baseline"
```

`PREPARED` is the actual cache directory printed by the native command. Keep it
for later input binding; do not guess its hash-based name. See [Hub account
setup](huggingface-access.md) if authentication or rate limits block acquisition.

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
uv run --locked --extra cpu sparselab eval stories-base --runs-dir "$WORK/runs" --checkpoint latest.json
uv run --locked --extra cpu sparselab evidence stories-base --runs-dir "$WORK/runs" --json
uv run --locked --extra cpu sparselab triage stories-base --runs-dir "$WORK/runs"
uv run --locked --extra cpu sparselab generate stories-base --runs-dir "$WORK/runs" --checkpoint latest.json \
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
uv run --locked --extra cpu sparselab eval stories-continued --runs-dir "$WORK/runs" --checkpoint latest.json
uv run --locked --extra cpu sparselab evidence stories-continued --runs-dir "$WORK/runs" --json
uv run --locked --extra cpu sparselab triage stories-continued --runs-dir "$WORK/runs"
uv run --locked --extra cpu sparselab generate stories-continued --runs-dir "$WORK/runs" --checkpoint latest.json \
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
The [optional program below](#optional-declare-the-comparison-and-continuation)
uses that route. Campaign integration has a separate
[current boundary](#where-campaigns-fit).

## What to check between iterations

One full update here supervises `64 × 2 × 2 = 256` targets: sequence length ×
microbatch × accumulation. The prepared-data cap is 200,000 tokens; the first
run uses only 10,240 target exposures. More updates may repeat prepared blocks;
they do not acquire more stories.

| Check | Native evidence | What to establish |
| --- | --- | --- |
| Intended change | YAML, `inspect`, resolved plan | Name the changed fields and controls held fixed. |
| Inputs and parent | Preparation, `checkpoint verify`, `evidence` | Intended source revision, tokenizer, data and parent identities. |
| Fit | `workspace preflight`, actual-config warmup | Enough storage and measured memory headroom; successful execution. |
| Completion | `evidence`; `experiment list` for queued work | Actual steps/targets; queued runs also need complete ingestion. |
| Learning | `eval` and the dashboard loss curve | Finite loss and whether it improves, regresses or remains inconclusive. |
| Text behavior | Same prompts, decoding and completion budget | Keep improvements and failures, not just a favorite completion. |
| Parent preservation | Verify the immutable generation again | The child's execution did not replace or modify its parent. |

The starter evaluates only four batches per checkpoint, not the whole validation
split. Before a larger comparison, declare a larger fixed evaluation budget for
both arms. Keep the held-out input, evaluator and checkpoint-selection rule fixed.
Use a small development prompt panel and reserve separate prompts for final review.
Do not tune against the final review set after viewing its results.

Keep a short table of run ID, parent digest, changed fields, actual targets,
held-out loss, prompt observations, memory and timing scope. Native evidence
supplies observations; interpretation still requires judgment. A failed input
check needs repair. Flat loss or worse generation can be a valid negative result.

| Next question | Appropriate transition |
| --- | --- |
| More exposure to unchanged data/model? | Full-state budget extension, as above. |
| Wider/deeper model or different attention? | New config and fresh run; ordinary resume cannot resize weights. |
| More data, new tokenizer, context or precision? | New declared inputs/settings, preparation as needed, and a new pilot. |
| Compatible weights on changed data? | Explicit weights-only promotion with fresh optimizer state; see [checkpointing](checkpointing.md). |
| Interrupted attempt? | Verify its last generation and explicitly resume with unchanged scientific settings. |

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

For a different architectural question on PyTorch, save a separate matrix with
the following axis instead of the width axis. Keep the base config unchanged:

```yaml
matrix_version: 1
base_config: run.yaml
axes:
  attention:
    - label: dense
      set: {}
    - label: local-window
      set:
        attention.kind: sliding_window
        attention.window_size: 32
```

Use the same dry-run, submission and evaluation sequence with that filename.
This asks whether restricting attention changes learning at this budget; it
does not presume a quality or speed improvement. MLX does not support this
attention variant. Additional axes multiply the number of runs, so inspect
the expansion before submission.

## Optional: declare the comparison and continuation

ExperimentPlan can express both widths and their checkpoint-bound continuations
without a Python orchestration script. This is a separate practice exercise:
two fresh parents and two children, not an import of the direct runs above.
Use the original 40-update CPU/PyTorch sample for this declaration. If you have
changed its budget or runtime, restore a separate copy of the original inputs
and prepare it first; do not overwrite an active experiment.

Save this as `$WORK/inputs/story-template.yaml`:

```yaml
plan_version: 1
id: stories-program
base_run: run.yaml
axes:
  - name: architecture
    choices:
      - label: baseline
        set: {}
      - label: wider
        set: {model.ffn_dim: 384}
comparisons:
  - id: wider-versus-baseline
    baseline: {architecture: baseline}
    variant: {architecture: wider}
    interventions: [model.ffn_dim]
    invariants: [seed, dataset, tokenizer, training, optimizer, runtime]
phases:
  - id: pretrain
    transition: fresh
  - id: continue
    transition: extend_budget
    parent: pretrain
    selector: terminal
    at_step: 40
    set:
      training.max_steps: 80
      training.max_tokens: 20480
      optimizer.decay_steps: 40
execution:
  backend: cpu
```

The parent is selected within each architecture coordinate. The child retains
its parent's optimizer state and original decay horizon. Architecture changes
occur between fresh parents, not during resume. Bind the already prepared
inputs; `PREPARED` is the exact path returned earlier by `data prepare`:

```sh
uv run --locked --extra cpu sparselab experiment bind-inputs \
  "$WORK/inputs/run.yaml" "$WORK/inputs/story-template.yaml" \
  --prepared-root "$PREPARED" --output "$WORK/inputs/story-bound.yaml" --json
uv run --locked --extra cpu sparselab experiment validate "$WORK/inputs/story-bound.yaml" --json
uv run --locked --extra cpu sparselab experiment inspect "$WORK/inputs/story-bound.yaml" --json
LOCK=$(uv run --locked --extra cpu sparselab experiment lock "$WORK/inputs/story-bound.yaml" --json | jq -r '.lock')
uv run --locked --extra cpu sparselab experiment explain "$LOCK" --json
PROGRAM="$SPARSELAB_WORK_DIR/experiments/stories-program"
```

`jq` only selects the returned path; you can instead copy the printed `lock`
value. Output declarations must be new files. Use a new plan ID and workspace
for repeats. Keep the source revision and bound inputs fixed through execution.
The lock verifies identities and declared controls; it does not prove that
your chosen comparison answers a useful scientific question.

Before dispatch, use the cell IDs from `explain` with `experiment export-config`
and pilot each resolved configuration as described in
[training programs](experiment-programs.md#pilot-the-resolved-configuration).
Then submit the parents and start the controller:

```sh
uv run --locked --extra cpu sparselab experiment run "$LOCK" --phase pretrain --json
uv run --locked --extra cpu sparselab controller run --store "$PROGRAM/controller"
```

In a second terminal, set the same `PROGRAM` and `LOCK`. Wait until **both**
parents have `status: COMPLETE` and `ingestion_status: COMPLETE` before submitting
the children. The running controller will execute them:

```sh
uv run --locked --extra cpu sparselab experiment list --store "$PROGRAM/controller"
# Run only after both parents have completed and ingested.
uv run --locked --extra cpu sparselab experiment run "$LOCK" --phase continue --json
uv run --locked --extra cpu sparselab experiment list --store "$PROGRAM/controller"
```

After all four runs complete and ingest, collect and reconstruct their lineage:

```sh
uv run --locked --extra cpu sparselab experiment collect "$LOCK" --json > "$PROGRAM/collection-result.json"
INDEX=$(jq -r '.index_path' "$PROGRAM/collection-result.json")
uv run --locked --extra cpu sparselab experiment reconstruct "$LOCK" --index "$INDEX" --json
uv run --locked --extra cpu sparselab dashboard --runs-dir "$PROGRAM/controller"
```

Use each returned run ID with the earlier `checkpoint verify`, `eval`, `evidence`
and `generate` commands, replacing their runs directory with
`$PROGRAM/controller`. Compare the two parents at 40 updates, the two children
at 80, and each parent with its own child. Preserve failures and unavailable
measurements. Collection records execution and lineage; it does not execute
an arbitrary behavioral evaluation or select a winning model for you.

## Where Campaigns fit

The [Campaign DSL](campaigns.md) coordinates declared dependencies across input
preparation, runtime admission, locked experiments, collection and evaluation.
Its value is that the tested parameter choices and gates remain in reviewable
declarations rather than in external Python control flow. A matrix expands
choices; ExperimentPlan adds checkpoint phases and comparison contracts;
Campaigns coordinate the surrounding workflow.

There is currently an integration gap: Campaign plan binding requires a Corpus
Forge release for every non-synthetic cell, while ExperimentPlan supports the
direct TinyStories inputs used here. Therefore this guide uses the supported
native ExperimentPlan path. A Forge-backed study can use Campaigns today; do
not invent a release to wrap these direct inputs. [Rapid iteration work](../TODO.md#rapid-iteration)
tracks a shared typed input contract and a complete declared teaching example.

TinyStories is also specialized at ingestion today: `dataset.source: tinystories`
selects a fixed Hub source, and `data snapshot` invokes a TinyStories-specific
producer. Training, matrices and checkpoint phases are general lab operations.
The backlog calls for declarative Hub acquisition and snapshots so this lesson
becomes a reference source configuration, with the same pattern usable on other
datasets, rather than an expanding collection of dataset-specific functions.

## Grow the model and the data deliberately

Make each step a new named workspace/config and run ID. Copy the inputs and
review relative paths; reference the existing frozen tokenizer by its absolute
path when sharing it. Keep prior data, checkpoints and evidence intact.

| Exercise | Changes to declare | What to compare |
| --- | --- | --- |
| Learn beyond the wiring budget | Fresh run with `training.max_steps: 1024`, `training.max_tokens: 262144`, `optimizer.warmup_steps: 100` | Learning curve and fixed prompts against the short run; this is a new schedule, not the previous full-state child. |
| Add story variety | For example, `dataset.train_max_documents: 20000` and `dataset.train_max_tokens: 2000000`; keep tokenizer and held-out selection fixed | Fresh matched runs with the same supervised-target budget, differing only in training-data selection. Reprepare the larger input. |
| Increase capacity | For example, `model.hidden_dim: 256`, `model.num_layers: 4`, `model.num_heads: 4`, `model.ffn_dim: 512` | A fresh model at matching exposure; this changes several capacity fields, not one isolated mechanism. |

These are proposed learning exercises, not measured quality or fit claims.
Keep vocabulary size and tokenizer identity matched. Leave context length,
effective batch, precision and seed fixed unless that is the declared question.
Use `inspect` to catch invalid fields and inspect actual parameter counts;
`workspace preflight` and a new warmup gate each larger configuration.
Do not carry the sample's 10,240-target cap into a proposed longer run: the first
step or token limit reached stops training. Give each fresh schedule an explicit
decay horizon when comparing schedules.

Judge progress using the iteration checklist above. Lower loss alone does not
establish coherent stories. Repeat promising contrasts with declared seeds
before drawing broader conclusions. Runtime throughput and memory describe
execution on the chosen CPU/GPU; they are separate from story quality.

## An advanced full-TinyStories exercise

Define “full” first: this lesson uses the pinned Hub repository's default
`train` and `validation` splits. The [source dataset card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md)
also describes other archives and versions; training the default split does
not mean combining every file or reproducing the paper. Keep validation out
of tokenizer fitting and optimizer updates.

The current CLI has no `--all` or `--epochs` switch. Dataset document/token
limits are positive bounds. Direct Hub preparation also accumulates token
arrays in host memory before writing them: a successful tiny GPU pilot says
nothing about full-source preparation RAM. Plan storage, host RAM, download
cache and checkpoint growth before attempting this on a suitably sized machine.
`data snapshot` is not an all-records workaround: it requires exact retained
counts and fails when the source ends early. Scalable source-exhaustion
preparation and native coverage/budget reporting are recorded in
[TODO.md](../TODO.md#rapid-iteration).

For an adequately provisioned host, the existing bounded path is:

1. Create a separate workspace and copy the original run config into its
   `inputs/`. Set a distinct run name, absolute path to the frozen 2,048-token
   tokenizer, and task-local data/runs destinations. Full-data training does
   not require refitting a tokenizer on every story. If studying a new
   tokenizer, that is a separate comparison requiring fresh preparation/models.
2. Raise both document and token ceilings above the selected source size,
   following resource admission. For example, 10,000,000 training documents /
   2,000,000,000 training tokens and 100,000 validation documents / 100,000,000
   validation tokens are **admission ceilings, not measured dataset sizes or
   a memory-fit recommendation**. Smaller ceilings may intentionally select a
   subset; never call that full-source coverage.
3. Run the same `inspect`, `workspace preflight` and `data prepare` commands on
   the new config. Retain the returned prepared path. Open its `manifest.json`
   in an editor or with `cat`; under `train` and `validation`, inspect acquired,
   retained, skipped and truncated document counts and the array `shape`.
   Successful preparation below both document and token caps indicates source
   exhaustion on this direct loader. If either cap was reached, coverage is
   unestablished. Report skipped records and require zero truncation for a claim
   that all retained source text was prepared. Do not infer coverage from a
   successful command alone.
4. Derive exposure from the actual prepared training array, not the ceilings.
   For this all-token objective, let `T` be its `shape[0]`, `L` the sequence
   length, and `B` microbatch × accumulation. There are
   `N = floor((T - 1) / L)` complete training blocks. One pass over those blocks
   uses `training.max_tokens = N × L` and
   `training.max_steps = ceil(N / B)`. Use a calculator and author the values
   in the new config. For multiple passes, multiply the target budget first,
   then derive steps. A trailing incomplete block is excluded; “one pass”
   does not mean every raw source token became a supervised target.
5. Declare a fresh optimizer schedule, checkpoint retention and fixed evaluation
   budget appropriate to the larger run. Inspect and stage the actual config,
   then train a fresh run using the earlier native commands. Preserve its
   receipts and verify the actual counters afterwards. These budgets describe
   a fresh run; a full-state extension uses cumulative counters and its existing
   schedule, so do not transplant them blindly into a child.
6. Evaluate the selected verified endpoint and the same prompt panel. Record
   source coverage, prepared blocks and supervised exposure separately, along
   with quality observations and execution cost. A full data pass is an
   educational milestone, not evidence of paper-level performance.

This pathway needs no external Python, but coverage inspection and budget
authoring are still manual. The missing native report and declarative pass-count
proposal should serve any dataset; they should not become a TinyStories-only
training command.
