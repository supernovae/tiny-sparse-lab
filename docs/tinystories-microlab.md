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
TinyStories training engine. The same pinned dataset and snapshot interfaces work for other standard Hub
text datasets; only the source declaration changes. See [datasets](datasets.md).

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
cp "$SAMPLE/"*.yaml "$WORK/inputs/"
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
uv run --locked --extra cpu sparselab data lock "$WORK/inputs/source.yaml" \
  --output "$WORK/inputs/source.lock.json"
uv run --locked --extra cpu sparselab data snapshot "$WORK/inputs/source.lock.json" \
  --output "$WORK/snapshot"
uv run --locked --extra cpu sparselab tokenizer train "$WORK/inputs/tokenizer.yaml"
PREPARED=$(uv run --locked --extra cpu sparselab data prepare "$WORK/inputs/run.yaml")
uv run --locked --extra cpu sparselab stage "$WORK/inputs/run.yaml" \
  --through warmup --output "$WORK/stages/baseline"
```

`PREPARED` is the actual cache directory printed by the native command. Keep it
for later input binding; do not guess its hash-based name. See [Hub account
setup](huggingface-access.md) if authentication or rate limits block acquisition.

The first lock/snapshot requires Hub access. The declaration pins the repository,
revision, splits, text field and selection policy; the lock and immutable snapshot
retain those identities. Later model iterations reuse that snapshot and frozen
tokenizer. Acquisition keeps validation first, excludes exact duplicates and
training overlap, and records exclusions. This policy is explicit in `source.yaml`.
It differs from historical direct TinyStories prefixes; do not relabel old results.
Tokenizer fitting uses training text only. `stage`
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
uses that route, and the [Campaign](#where-campaigns-fit) composes acquisition
through evaluation with the same verified inputs.

## What to check between iterations

One full update here supervises `64 × 2 × 2 = 256` targets: sequence length ×
microbatch × accumulation. The prepared-data safety cap is 2,000,000 tokens; the first
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

The sample `plan.yaml` declares two fresh widths and their checkpoint-bound
continuations. Its baseline and wider cells use identical data, tokenizer, seed,
optimizer and exposure; `model.ffn_dim` is the intervention. Each child retains
its own parent's full AdamW state and original decay horizon. Architecture
changes occur between fresh parents, not during resume.

Bind the prepared inputs from the first exercise without downloading or fitting
again. Use a new output declaration and plan ID for repetitions:

```sh
uv run --locked --extra cpu sparselab experiment bind-inputs \
  "$WORK/inputs/run.yaml" "$WORK/inputs/plan.yaml" \
  --prepared-root "$PREPARED" --output "$WORK/inputs/bound-plan.yaml" --json
uv run --locked --extra cpu sparselab experiment validate "$WORK/inputs/bound-plan.yaml" --json
uv run --locked --extra cpu sparselab experiment inspect "$WORK/inputs/bound-plan.yaml" --json
LOCK=$(uv run --locked --extra cpu sparselab experiment lock "$WORK/inputs/bound-plan.yaml" --json | jq -r '.lock')
uv run --locked --extra cpu sparselab experiment explain "$LOCK" --json
```

`jq` only selects the emitted lock path; copying that value manually is equivalent.
Follow [training programs](experiment-programs.md#pilot-the-resolved-configuration)
for resolved-cell pilots, phase submission, the foreground controller and
collection. This standalone route executes four extra runs if both phases are
submitted; it does not import your earlier direct run. The suite declaration
binds held-out evaluation but standalone collection does not execute it for you.
The Campaign below makes evaluation and the prompt panel explicit stages.

## Where Campaigns fit

Use the checked-in `campaign.yaml` to run acquisition, tokenizer fitting,
preparation, input binding, runtime admission, baseline training, a verified
continuation and a separate fresh wider model. Each endpoint is collected and
evaluated on the fixed suite, then receives the same descriptive prompt panel.
That is three runs: the Campaign explicitly selects the baseline child and
leaves the plan's wider child unselected. These are learning comparisons, not
an automatic quality gate or model promotion.

Start a **separate** workspace for this alternative end-to-end exercise. Its
snapshot and tokenizer outputs must not already exist. The same upstream Hub
cache can be reused. Preserve any earlier direct exercise:

```sh
CAMPAIGN="$SPARSELAB_WORK_DIR/experiments/stories-learning"
mkdir "$CAMPAIGN"
mkdir "$CAMPAIGN/inputs"
cp "$SAMPLE/"*.yaml "$CAMPAIGN/inputs/"
cp "$SAMPLE/campaign.yaml" "$CAMPAIGN/campaign.yaml"
uv run --locked --extra cpu sparselab data lock "$CAMPAIGN/inputs/source.yaml" \
  --output "$CAMPAIGN/inputs/source.lock.json"
uv run --locked --extra cpu sparselab campaign validate "$CAMPAIGN/campaign.yaml" --json
uv run --locked --extra cpu sparselab campaign status "$CAMPAIGN/campaign.yaml" --json
uv run --locked --extra cpu sparselab campaign next "$CAMPAIGN/campaign.yaml" --json
uv run --locked --extra cpu sparselab campaign apply "$CAMPAIGN/campaign.yaml" \
  --execute-runs --allow-uncommitted-declaration --max-wait-seconds 60 --json
```

The explicit uncommitted-declaration flag authorizes your copied teaching inputs;
it does not waive source, data, runtime or checkpoint verification. In a reviewed
research program, commit its declarations and use the ordinary provenance gate.
Keep the lab code revision fixed while a Campaign runs.

`--max-wait-seconds` bounds each run's controller wait, not the complete command:
input verification, preparation, evaluation and reconciliation add time.
Inspect `status`, `next`
and `explain` for the actual next action; use native `campaign resume` after
interruption instead of inventing a fresh retry. Reuse the same declaration and
workspace. Completed stages are verified before reuse, and a child cannot run
before its parent has completed and ingested. Do not edit a locked Campaign to
change widths, source bounds or budgets: author a new one.

The shipped sample uses CPU. For PyTorch MPS, edit the copied run and plan runtime
selection together **before locking** and register/doctor the interpreter as in
[runtime environments](runtime.md#machine-local-runtime-environments). Bind the
Campaign runtime-acceptance stage to that profile ID and its evaluations to that
runtime stage. Pilot matching CPU/MPS configurations and choose by measured
throughput and memory headroom; a tiny model can run faster on CPU. MLX is a
separate engine and does not support this full-state continuation exercise.

Review all three run counters, immutable parent identities, held-out loss and
panel outputs. Keep empty, failed and poor completions visible. Campaigns enforce
that the declared cells and inputs ran; scientific interpretation remains yours.
The [Campaign reference](campaigns.md) explains existing-artifact reuse, stage
contracts and reconciliation.

## Grow the model and the data deliberately

Make each step a new named workspace/config and run ID. Copy the inputs and
review relative paths. With the same snapshot, reference the frozen tokenizer
by its absolute path when sharing it. A changed snapshot currently needs matching
tokenizer provenance; do not describe refitting as an isolated data-only contrast.
Keep prior data, checkpoints and evidence intact.

| Exercise | Changes to declare | What to compare |
| --- | --- | --- |
| Learn beyond the wiring budget | Fresh run with `training.max_steps: 1024`, `training.max_tokens: 262144`, `optimizer.warmup_steps: 100` | Learning curve and fixed prompts against the short run; this is a new schedule, not the previous full-state child. |
| Add story variety | Increase the source declaration's training-document target, acquire a new snapshot and reprepare with adequate whole-document bounds | Fresh matched-exposure runs; retain held-out selection and report any tokenizer change as a confounder. |
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

Define “full” as the pinned default train/validation splits, not every archive
mentioned by the [source card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md).
A complete acquisition reports both excluded records and retained text; a model
pass covers complete prepared blocks, not every raw token. Keep validation out
of tokenizer fitting and optimizer updates.

1. Create a separate named workspace and copy the declarations. In its source
   declaration, set `selection: {mode: exhaustion}` and remove the bounded
   document targets. Set explicit resource ceilings after storage/RAM admission;
   resource ceilings are safety stops, not successful selection targets. Use a
   new lock and snapshot. `data snapshot --resume` authenticates interrupted
   work before replay; never delete its journal to conceal a failed attempt.
2. Reuse the frozen tokenizer by absolute path only when its source provenance
   matches the new training contract. A changed snapshot has a new identity:
   the present direct-input contract requires tokenizer provenance for that
   snapshot, so fit a new train-only tokenizer for the full-source exercise and
   train a fresh model. Do not silently claim it is the same-tokenizer comparison.
3. Set preparation document bounds high enough for the acquired counts and token
   bounds high enough for every selected whole document. Packing is disk-backed
   and resumable; an inadequate cap fails instead of truncating a story. Use
   `data prepare` to obtain the exact prepared root, then inspect native coverage:

```sh
uv run --locked --extra cpu sparselab data coverage \
  --prepared-root "$PREPARED" --config "$FULL/inputs/run.yaml" --json
uv run --locked --extra cpu sparselab data budget \
  --prepared-root "$PREPARED" --config "$FULL/inputs/run.yaml" \
  --passes 1 --require-full --output "$FULL/inputs/one-pass.yaml" --json
```

Here `FULL` names that separate workspace and `PREPARED` its returned cache path.
The report separates source exhaustion, exclusions, truncation, packed tokens,
complete blocks, dropped tails and supervised targets. Budget authoring writes
new paths/configuration and leaves prior settings untouched. It rejects unproven
full coverage; `--no-require-full` explicitly requests passes over a bounded
prepared selection instead.

Choose a fresh optimizer schedule and checkpoint/evaluation cadence for the
proposed budget, then repeat `inspect`, preflight and warmup before training.
Pass-count proposals are for fresh runs, not cumulative continuation counters.
Review actual targets and steps after completion. A full data pass is an
educational exercise, not evidence of paper-level performance.

No full-corpus training is needed to learn these commands. The small exercise
and offline exhaustion/restart tests validate the workflow; a real full-corpus
run still requires its own resource admission and retained evidence.
