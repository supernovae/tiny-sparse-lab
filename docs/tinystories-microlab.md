# Build a TinyStories microlab

Train a small story model, inspect its held-out loss, try a completion, and
compare two feed-forward widths using a YAML experiment matrix. The checked-in
[sample files](../experiments/samples/tinystories-microlab/) use a pinned
TinyStories revision, a train-only 2,048-token BPE, and a small dense decoder.
The default is 40 optimizer updates and 10,240 supervised targets per model:
enough to learn the workflow, not enough to expect fluent stories.

For an authored training program with immutable input bindings and checkpoint
dependencies, continue with [Chain a training program](experiment-programs.md).
Both guides describe teaching examples; they do not register research results.

## Set up a named workspace

Run from the checkout root with Python 3.14 and [uv](https://docs.astral.sh/uv/).
On CPU or macOS, install the locked environment:

```sh
uv sync --locked --dev
export SPARSELAB_WORK_DIR="$PWD/sparselab-work"
WORK="$SPARSELAB_WORK_DIR/experiments/tinystories-microlab"
SAMPLE=experiments/samples/tinystories-microlab
mkdir -p "$WORK/inputs"
```

On a provisioned ROCm/CUDA/XPU worker, use its documented vendor environment
and `uv run --locked --no-sync` for every command below. Running `uv sync` or
`uv run` without `--no-sync` can replace vendor PyTorch with the Linux CPU
lock. See [worker provisioning](workers.md#user-provisioned-ssh-workers).

Keep downloaded source, tokenizers, prepared data, checkpoints and outputs in
this ignored workspace. The preparation commands bound the selected text, but
the upstream download cache can be larger. Check actual available bytes and
inodes before downloading; allow several GB for the source cache in addition to
the small models. The preflight commands estimate known local growth and do not
bound the upstream download.

Make editable inputs with absolute workspace paths. This also gives you a place
to select your backend without changing the checked-in sample:

```sh
uv run --locked python - <<'PY'
import os
from pathlib import Path
import yaml
from sparselab.config import load_config, load_tokenizer_config

sample = Path("experiments/samples/tinystories-microlab")
work = Path(os.environ["SPARSELAB_WORK_DIR"]) / "experiments/tinystories-microlab"
inputs = work / "inputs"
run = load_config(sample / "run.yaml").model_dump(mode="json")
tokenizer = load_tokenizer_config(sample / "tokenizer.yaml").model_dump(mode="json")
run["tokenizer"]["path"] = str(work / "tokenizer/tokenizer.json")
run["dataset"]["cache_dir"] = str(work / "data")
run["logging"]["root_dir"] = str(work / "runs")
tokenizer["output_dir"] = str(work / "tokenizer")
tokenizer["dataset"]["cache_dir"] = str(work / "data")
# Select "mps" or "rocm" here when that backend is available.
run["runtime"]["backend"] = "cpu"
for name, value in (("run.yaml", run), ("tokenizer.yaml", tokenizer)):
    (inputs / name).write_text(yaml.safe_dump(value, sort_keys=False))
(inputs / "matrix.yaml").write_text((sample / "matrix.yaml").read_text())
PY
```

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

For MLX on Apple Silicon, install with `uv sync --locked --dev --extra mlx`,
set `engine: mlx` and `backend: metal` in the editable run config, and retain
`--extra mlx` on every `uv run --locked` invocation. Register its worker with
`--engine mlx --backend metal`. The story training and dense-width matrix use
supported MLX features; skip the PyTorch-only budget extension below.

## Prepare and pilot

```sh
df -h "$WORK"
df -i "$WORK"
uv run --locked sparselab workspace preflight "$WORK/inputs/tokenizer.yaml" --tokenizer
uv run --locked sparselab workspace preflight "$WORK/inputs/run.yaml"
uv run --locked sparselab inspect "$WORK/inputs/run.yaml" --json
uv run --locked sparselab tokenizer train "$WORK/inputs/tokenizer.yaml"
uv run --locked sparselab data prepare "$WORK/inputs/run.yaml"
uv run --locked sparselab stage "$WORK/inputs/run.yaml" \
  --through warmup --output "$WORK/stages/baseline"
```

The first preparation requires network access to TinyStories. Later runs reuse
the pinned source/cache; tokenizer fitting uses training text only. `stage`
launches disposable smoke/warmup pilots, records timing and memory, and leaves
the full training run's initial state untouched. Use a new stage output path
for another pilot.

## Train, evaluate and generate

```sh
uv run --locked sparselab train "$WORK/inputs/run.yaml" \
  --runs-dir "$WORK/runs" --run-id stories-base \
  --stage-bundle "$WORK/stages/baseline"
uv run --locked sparselab checkpoint verify \
  "$WORK/runs/stories-base/checkpoints/latest.json" --json
uv run --locked sparselab eval stories-base --runs-dir "$WORK/runs"
uv run --locked sparselab evidence stories-base --runs-dir "$WORK/runs" --json
uv run --locked sparselab generate stories-base --runs-dir "$WORK/runs" \
  --prompt "Once upon a time, a little rabbit" --max-new-tokens 32
uv run --locked sparselab dashboard --runs-dir "$WORK/runs"
```

The dashboard runs in the foreground; stop it or use another terminal for the
next steps. This is language-model training, so story continuations are the
natural first interaction. Record held-out loss and actual target counts beside
the generated text; neither a short completion nor lower loss establishes a
general assistant. Use new run IDs for repetitions.

## Compare two architectures with the matrix DSL

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
uv run --locked sparselab experiment submit --matrix "$WORK/inputs/matrix.yaml" \
  --dry-run --store "$WORK/runs"
uv run --locked sparselab worker register stories-local --backend cpu --store "$WORK/runs"
uv run --locked sparselab experiment submit --matrix "$WORK/inputs/matrix.yaml" \
  --worker stories-local --store "$WORK/runs"
uv run --locked sparselab controller run --store "$WORK/runs"
```

The controller stays running and executes the queued models independently on
one selected device. In a second terminal with the same `WORK`, check both
execution and ingestion status:

```sh
uv run --locked sparselab experiment list --store "$WORK/runs"
```

Once both have `status: COMPLETE` and `ingestion_status: COMPLETE`, use the
returned run IDs for `eval`, `evidence`, `generate` and the dashboard commands
above. Compare at matching actual steps/targets with the same tokenizer and
held-out source. This one-seed width comparison is descriptive; it does not
establish a universal winning architecture. Add labeled seed choices for a
larger declared matrix. [Controlled comparisons](experiments.md) explains
study receipts, task-specific capability cards and collection when your
experiment has a separately defined behavioral test.

## Continue a verified checkpoint

For a completed PyTorch AdamW run, explicitly extend the budget to 80 updates
and 20,480 cumulative targets while retaining the original 40-update decay
horizon. Make a new config:

```sh
uv run --locked python - <<'PY'
import os
from pathlib import Path
import yaml

work = Path(os.environ["SPARSELAB_WORK_DIR"]) / "experiments/tinystories-microlab"
run = yaml.safe_load((work / "inputs/run.yaml").read_text())
run["name"] = "tinystories-microlab-continued"
run["training"].update(max_steps=80, max_tokens=20480)
run["optimizer"]["decay_steps"] = 40
(work / "inputs/continued.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
PY
uv run --locked sparselab checkpoint inspect \
  "$WORK/runs/stories-base/checkpoints/latest.json" --json
uv run --locked sparselab inspect "$WORK/inputs/continued.yaml" --json
```

Resolve the verified pointer once to an immutable generation directory, then
verify that directory and pass it as `GENERATION`:

```sh
GENERATION=$(uv run --locked python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.parent / json.loads(p.read_text())["relative_path"])' "$WORK/runs/stories-base/checkpoints/latest.json")
uv run --locked sparselab checkpoint verify "$GENERATION" --json
uv run --locked sparselab train "$WORK/inputs/continued.yaml" \
  --runs-dir "$WORK/runs" --run-id stories-continued --extend-budget "$GENERATION"
uv run --locked sparselab eval stories-continued --runs-dir "$WORK/runs"
```

The parent must be terminal at both 40 updates and 10,240 targets; inspect its
counters first. The child restores optimizer, cursor, counters and RNG and
continues at the original learning-rate floor. `--resume` instead continues
an interrupted run with unchanged budget/settings, and `--promote` starts fresh
training state from compatible weights. See [checkpointing](checkpointing.md).

All mutable output belongs to this task's workspace. Preserve interrupted and
failed attempts while investigating them; do not edit or prune active runs.
For a larger program, copy these inputs into a new named workspace, inspect
the new budget and storage estimate, and repeat the pilot before training.
