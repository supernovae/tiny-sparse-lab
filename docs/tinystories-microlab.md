# Build a TinyStories microlab

Learn one loop: prepare story inputs, change one setting, then read the result.
The existing [sample declarations](../experiments/samples/tinystories-microlab/)
pin TinyStories, fit a train-only 2,048-token BPE and define a small dense decoder.
Each arm runs at most **40 updates / 10,240 supervised targets**. This is a
workflow lesson, not a useful story model or a quality benchmark.

## Set up

Use the [README installation](../README.md#first-run) from the checkout root.
These are Bash commands; [iteration](iteration.md#start-a-runner-session) explains
PowerShell syntax. Keep CPU for this first exercise. An AMD 7900 XTX uses the
ROCm path, not CUDA; accelerator setup belongs in the [runtime guide](runtime.md).
A provisioned vendor environment uses `uv run --locked --no-sync` instead of
syncing the CPU extra into it.

```sh
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
WORK="$SPARSELAB_WORK_DIR/experiments/tinystories-microlab"
SAMPLE=experiments/samples/tinystories-microlab
mkdir -p "$WORK"
mkdir "$WORK/inputs"
cp "$SAMPLE/source.yaml" "$SAMPLE/tokenizer.yaml" "$SAMPLE/run.yaml" "$WORK/inputs/"
df -h "$WORK"
df -i "$WORK"
```

Choose a fresh `WORK` if `inputs` already exists. Paths in the copied YAML resolve
relative to `inputs`, so tokenizer, snapshot and prepared data stay outside Git.
Do not run the templates in place. The Hub cache can require several GB beyond
the small selected text; the snapshot's text/resource limits do not cap upstream
downloads. Preserve earlier inputs and runs.

## Prepare the inputs once

The snapshot command downloads the pinned source. Tokenizer fitting uses training
text only; validation is selected first and overlap is excluded. No model trains
in this block. See [Hub access](huggingface-access.md) if authentication is needed.

```sh
uv run --locked --extra cpu sparselab data lock "$WORK/inputs/source.yaml" \
  --output "$WORK/inputs/source.lock.json"
uv run --locked --extra cpu sparselab data snapshot "$WORK/inputs/source.lock.json" \
  --output "$WORK/snapshot"
uv run --locked --extra cpu sparselab tokenizer train "$WORK/inputs/tokenizer.yaml"
```

Reuse these inputs for subsequent width changes. `try` handles native data
preparation; there is no separate pilot, worker, plan lock or Campaign to create.

## Try one change

The baseline has `model.ffn_dim: 256`. This delta changes only that width to 384:

```sh
cat > "$WORK/wider-ffn.yaml" <<'YAML'
question: Does a wider FFN lower held-out loss at the same token budget?
set:
  model.ffn_dim: 384
YAML
uv run --locked --extra cpu sparselab try "$WORK/wider-ffn.yaml" \
  --vs "$WORK/inputs/run.yaml"
```

This starts model work: two training arms, shared held-out scoring, then fast
probes. Runs and sealed records live in `SPARSELAB_WORK_DIR/lab`, independently
of the sample config's direct-training `logging.root_dir`. Both arms retain the
same seed, tokenizer, validation input and evaluation protocol. The baseline
scores four batches of 64-token windows with batch size two, not the full split.
The 2,000,000-token preparation cap is not the 10,240-target training budget.

The command prints a **try ID** and **baseline/candidate run IDs**. Copy those
values into the placeholders below; they are different kinds of identifier.

## Read, then choose the next step

```sh
uv run --locked --extra cpu sparselab report TRY_ID
uv run --locked --extra cpu sparselab compare TRY_ID --references
```

These only read records. Look at actual targets, loss delta and probe verdict.
Reference comparisons may report `missing_evidence` or `not_comparable`: a tiny
story model's loss cannot be ranked against a different tokenizer/corpus. The
[probe guide](probe-battery.md) explains the optional full-tier lm-eval path.
Do not download reference models just to read their packaged results.

For more screening or an inside view, these commands **load and execute the
saved checkpoints**, but do not train:

```sh
uv run --locked --extra cpu sparselab probe CANDIDATE_RUN --vs BASELINE_RUN \
  --backend cpu --tier standard
uv run --locked --extra cpu sparselab explore CANDIDATE_RUN --text "Once upon a time"
uv run --locked --extra cpu sparselab dashboard
```

`probe` needs run IDs, not `TRY_ID`; it inherits the saved backend unless explicitly
overridden. `--backend cpu` also makes this example usable for compatible
checkpoints trained on a GPU. `explore` defaults to CPU. The dashboard runs in
the foreground; stop it with Ctrl-C or use another terminal.

An incomplete probe is missing evidence, not a pass. A failed or worse candidate
is a useful retained result. Change one setting in a new delta and run `try`
again; identical completed baselines can be reused. Use `--seed N` to repeat
both arms at another seed. One seed and a small evaluation budget cannot
establish an architecture advantage. Keep an untouched final evaluation instead
of repeatedly tuning against the probe items.

## Go further

- [Lab mode](lab-mode.md): reuse, cancellation, resource limits and record semantics.
- [Training and checkpoints](training.md): direct training and verified continuation.
- [Training programs](experiment-programs.md): matrices and checkpoint phases.
- [Campaigns](campaigns.md): release orchestration and explicit approval boundaries.
- [Datasets](datasets.md): larger snapshots, coverage and pass budgets.

The sample directory also retains matrix, continuation and Campaign declarations
and their original [acceptance evidence](../experiments/samples/tinystories-microlab/acceptance.md).
They are separate advanced exercises, not additional steps in this lab loop.
