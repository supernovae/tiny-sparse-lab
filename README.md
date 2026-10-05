<p align="center">
  <img src="docs/assets/sparselab-banner.svg" alt="Tiny Sparse Lab — Small models. Real experiments." width="100%">
</p>

<p align="center">
  <a href="https://github.com/supernovae/tiny-sparse-lab/actions/workflows/ci.yml"><img src="https://github.com/supernovae/tiny-sparse-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&amp;logoColor=white" alt="Python 3.14"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-7ceac4" alt="MIT license"></a>
  <a href="docs/experiment-programs.md"><img src="https://img.shields.io/badge/Experiments-YAML%20DSL-a7a2ff" alt="YAML experiment DSL"></a>
</p>

<p align="center">
  <a href="docs/tinystories-microlab.md">Build your first microlab</a> ·
  <a href="docs/experiment-programs.md">Write a training program</a> ·
  <a href="docs/using-sparselab.md">CLI guide</a> ·
  <a href="docs/research/roadmap.md">Research roadmap</a>
</p>

Tiny Sparse Lab is a workbench for building small language models and finding
out what makes them learn. Start with a story-writing decoder, add sparse
attention, routed experts, or Engram memory, and compare what changed in quality,
speed, and memory. We're building a lab where a training program is readable
YAML, each checkpoint carries its inputs and history, and the evidence stays
with the experiment—from your first laptop run to a controlled research campaign.

## What will you build?

| Try this | What the lab gives you |
| --- | --- |
| **A TinyStories microlab** | Fit a tokenizer, train a small decoder, measure held-out loss, and generate your own story continuations. [Walkthrough →](docs/tinystories-microlab.md) |
| **An architecture comparison** | Dense, sliding-window, block-sparse, or MLA attention; local Top-K MoE; token, byte, and portable Engram memory. Declare changes and matched controls in YAML. [Architecture →](docs/architecture.md) |
| **A training program with a history** | Bounded parameter sweeps, verified checkpoints, explicit resume, budget extension, and weight promotion between phases. [Experiment DSL →](docs/experiment-programs.md) |
| **A local task model** | Train on licensed conversations with whole-transcript or assistant-only loss, then evaluate a specific held-out capability. [Instruction training →](docs/instruction-training.md) |
| **A corpus you can trace** | Acquire, shape, freeze, and export data with Corpus Forge; retain source, rights, tokenizer, and preparation identities. [Offline recipe →](corpora/devmind-sample-v0/README.md) |

Watch loss curves, throughput, memory, and checkpoint history in the local
dashboard. Queue whole independent experiments on local or SSH workers when
one machine isn't enough for your sweep. Each training run uses one process and
one device; the controller schedules independent runs.

## First run

You need **Python 3.14** and [uv](https://docs.astral.sh/uv/). Clone the repository,
then check the training lifecycle with this offline fixture before downloading data:

```sh
git clone https://github.com/supernovae/tiny-sparse-lab.git
cd tiny-sparse-lab
uv sync --locked --extra cpu --dev
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
uv run --locked --extra cpu sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run --locked --extra cpu sparselab inspect configs/runtime_smoke_cpu.yaml --json
uv run --locked --extra cpu sparselab stage configs/runtime_smoke_cpu.yaml \
  --through warmup --output "$SPARSELAB_WORK_DIR/first-run/stage"
uv run --locked --extra cpu sparselab train configs/runtime_smoke_cpu.yaml \
  --runs-dir "$SPARSELAB_WORK_DIR/first-run/runs" --run-id first-run \
  --stage-bundle "$SPARSELAB_WORK_DIR/first-run/stage"
uv run --locked --extra cpu sparselab eval first-run --runs-dir "$SPARSELAB_WORK_DIR/first-run/runs"
uv run --locked --extra cpu sparselab dashboard --runs-dir "$SPARSELAB_WORK_DIR/first-run/runs"
```

The base installation is Torch-free. Add `--extra cpu` for this CPU workflow and
the full CLI (which still imports a backend framework); `sparselab runtime env`
commands use a lightweight Torch-free entry route.

This tiny synthetic run checks that the lab works. For real story data, follow
the **[TinyStories microlab](docs/tinystories-microlab.md)**: a ~590K-parameter
starter, a two-cell YAML comparison, generation, and a continued training run.
Choose your backend explicitly and keep expensive outputs on an adequately sized
external filesystem (`SPARSELAB_WORK_DIR=/data/sparselab` is recommended for
substantial campaigns). Example configurations with explicit output/cache locations
retain those destinations even when the global root changes. Use fresh run IDs and
stage directories when repeating an experiment; check free storage and runtime
warmup measurements before scaling.

**Machine-local runtimes** are independent of the scientific work root and lock.
Discover installed interpreters and host hardware separately, then register and
doctor a CPU interpreter before selecting its logical ID for execution:

```sh
uv run --locked --extra cpu sparselab runtime env discover --json
uv run --locked --extra cpu sparselab runtime env register cpu-py314 \
  --python "$PWD/.venv/bin/python" --backend cpu --json
uv run --locked --extra cpu sparselab runtime env doctor cpu-py314 --json
uv run --locked --extra cpu sparselab stage configs/runtime_smoke_cpu.yaml \
  --through inspect --runtime cpu-py314 --output "$SPARSELAB_WORK_DIR/first-run/inspect"
```

The registry is host-local; a detected GPU alone does not authorize execution.
The runtime root defaults to `~/.local/share/sparselab/runtimes` (or
`$XDG_DATA_HOME/sparselab/runtimes`, overridden by `SPARSELAB_RUNTIME_DIR`);
the registry defaults to `~/.config/sparselab/runtimes.yaml` (or
`$XDG_CONFIG_HOME/sparselab/runtimes.yaml`). Neither is
`SPARSELAB_WORK_DIR`. See [machine-local runtime environments](docs/runtime.md#machine-local-runtime-environments)
for provisioning, other backends, and authorization.

**Already have a provisioned ROCm/CUDA/XPU environment?** Select its interpreter
directly or set `UV_PROJECT_ENVIRONMENT` to its environment prefix and use
`uv run --locked --no-sync …` throughout. The Linux `cpu` extra uses CPU PyTorch;
never sync it into a vendor environment. Follow the [worker setup guide](docs/workers.md#user-provisioned-ssh-workers).

## Experiments as programs

The YAML DSL makes the question, changed settings, and checkpoint lineage
reviewable before training. A matrix expands concrete configs; an authored plan
adds artifact identities, comparison contracts, immutable locks, and phases:

```yaml
# Phase excerpt; see the guide for a complete runnable plan.
phases:
  - id: first-pass
    transition: fresh
  - id: longer-run
    transition: extend_budget
    parent: first-pass
    selector: terminal
    at_step: 20
    set:
      training.max_steps: 40
      training.max_tokens: 1280
      optimizer.decay_steps: 20
```

Prepare inputs → validate and lock the plan → dispatch the first phase → verify
its ingested checkpoint → dispatch the child → collect and reconstruct evidence.
Child phases are submitted explicitly after their parent completes. Full-state
continuation and fresh-state weight promotion have different semantics.

**[Run a complete training program →](docs/experiment-programs.md)**
The guide covers working commands, checkpoint chaining, and current implementation
gaps tracked in [TODO.md](TODO.md#experiment-ergonomics).

[Campaign v1](docs/campaigns.md) adds declared DAGs, verified corpus readiness,
input-bound approvals, and recoverable runtime-bound execution above those plans.
Try the [tiny corpus/readiness/approval example](examples/tiny-campaign.yaml) in
an isolated workspace; its outcomes are declared-policy results, not model quality.

For durable declaration-to-artifact-to-checkpoint work, use a normal branch in
the same checkout: `git switch main; git pull --ff-only; git switch -c feat/<task>`.
Commit authored inputs before acquisition, tokenizer fit, preparation or training;
there is no automatic worktree, commit, push or relocation of legacy
`sparselab-work/`. The [lifecycle and recovery protocol](docs/research/lifecycle-recovery.md)
explains how a shared external persistent root, compact SHA evidence, read-only
recovery, declared evaluations, human readiness review and rights-aware archives
fit together. Branch names and checkout paths are not scientific identity.

## Bring your machine

| Platform / backend | Status today |
| --- | --- |
| **macOS** | CPU and Apple Silicon PyTorch MPS exercised; optional MLX/Metal is a separate engine with a smaller feature set. |
| **Linux / WSL2** | CPU path and Linux CI; accelerator execution requires the matching vendor framework and drivers. |
| **AMD ROCm** | Training studies and native HIP sparse attention exercised on an RX 7900 XTX under WSL2. Other host/device combinations need their own checks. |
| **NVIDIA CUDA** | Runtime selection and reference execution paths implemented; hardware validation and native CUDA sparse kernels are coming next. |
| **Intel XPU** | Runtime selection implemented; hardware acceptance is coming next. |

Host OS and compute backend are separate choices. PyTorch is the reference engine;
MLX supports FP32 dense/native block-sparse training, AdamW, and block recomputation,
with explicit limits on other mechanisms. See [runtime support](docs/runtime.md)
for setup, precision, feature boundaries, and acceptance evidence.

## Explore the lab

- **Learn:** [From flashcards to a local assistant](docs/from-toy-to-useful.md) · [Architecture](docs/architecture.md) · [Memory and fit](docs/memory.md)
- **Operate:** [CLI](docs/using-sparselab.md) · [Lifecycle recovery](docs/research/lifecycle-recovery.md) · [Training](docs/training.md) · [Checkpoints](docs/checkpointing.md) · [Workers](docs/workers.md)
- **Investigate:** [Papers we Love](papers.md) · [Research workbench](docs/research/README.md) · [Learning cycle](docs/research/experiment-learning-cycle.md) · [Evidence and results](docs/lab-status.md) · [Retained checkpoint exploration](docs/checkpoint-exploration.md)

This is a reference lab under active development. Small runs help you test a
mechanism; fluent language, reliable task behavior, and architecture advantages
need their own evidence. We retain negative results and compare checkpoints
under recorded conditions—lower validation loss alone doesn't settle text quality.

Contributions are welcome: start with [CONTRIBUTING.md](CONTRIBUTING.md),
the [code backlog](TODO.md), or the [research roadmap](docs/research/roadmap.md).
Code is [MIT licensed](LICENSE); datasets retain their own terms. Keep downloaded
corpora, prepared arrays, checkpoints, immutable receipts and logs in an external
persistent state root rather than Git. Git retains checked-in declarations and
compact verified evidence; their digests establish identity, not quality.
