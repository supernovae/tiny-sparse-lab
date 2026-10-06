# Contributing

Use Python 3.14 (selected by `.python-version`) and
`uv sync --locked --extra cpu --dev`. For the Linux CPU checks:

```sh
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --locked --extra cpu pytest -q -n auto --maxprocesses=4 \
  --dist worksteal --durations=25 \
  -m "not mps and not mlx and not cuda and not rocm and not xpu and not network"
```

CI runs on pull requests and pushes to `main`; use the workflow's manual dispatch
for a branch without a PR. New commits cancel superseded runs on the same ref.
Both Linux x64 and macOS arm64 run the full CPU suite, with up to four pytest
workers (limited by physical cores) and one BLAS/OpenMP thread per worker.
Work stealing redistributes pending tests when one worker falls behind; module
fixtures can be instantiated on more than one worker and must use isolated paths.
The slowest 25 tests are printed in each job log. Full-suite jobs have a 30-minute
limit so a hung test cannot occupy a runner for six hours. Hardware and network
tests retain their separate opt-in gates. PR checkout uses GitHub's synthetic
merge commit to test integration with the base branch; detached HEAD is expected.

Base installation is Torch-free: the lightweight `sparselab runtime env` CLI
works without Torch, but the full CLI currently imports a backend framework.
For a provisioned vendor interpreter use it directly or set
`UV_PROJECT_ENVIRONMENT` to its environment prefix and run with
`uv run --locked --no-sync`; never sync the CPU extra into it. Do not commit
downloaded corpora, prepared arrays, run directories, or checkpoints. Reports
should include the resolved configuration and relevant run evidence.

For a local CPU example, run `sparselab runtime env discover --json`, then
`sparselab runtime env register cpu-py314 --python "$PWD/.venv/bin/python" --backend cpu --json`
and `sparselab runtime env doctor cpu-py314 --json`. Select it with
`sparselab stage configs/runtime_smoke_cpu.yaml --through inspect --runtime cpu-py314 --output /absolute/disposable/stage`.
The XDG runtime root and host-local registry are separate from
`SPARSELAB_WORK_DIR` and scientific locks; never copy runtime IDs as capability
claims to another host. See [machine-local runtime environments](docs/runtime.md#machine-local-runtime-environments).

Work on an ordinary branch in the existing checkout, not an automatically created
worktree:

```sh
git switch main
git pull --ff-only
git switch -c feat/<task>
export SPARSELAB_WORK_DIR=/data/sparselab
```

Choose an external persistent filesystem with enough space for actual campaigns.
Commit corpus/source-rights pins, ExperimentPlan/base run, evaluation suite and
runtime requirement **before** acquisition, tokenizer fitting, preparation or
training. Inspect `sparselab research snapshot <plan-or-campaign> --json` before
expensive execution. Retain a compact verified evidence reference in Git instead
of copying datasets/checkpoints into the repository. `scratch/` and optional
`cache/` under the persistent root are disposable; neither is proof. A reviewed
decision about model quality requires an explicit human receipt. Read the
[recovery and lineage operating protocol](docs/research/lifecycle-recovery.md)
before changing paths or archiving rights-bound corpus content.

## Architecture experiments

The [research contribution workflow](docs/research/contributing.md) explains validated local catalog/recipe forks, declared controls and variations, train-only data boundaries, retained negative outcomes, and evidence identities. Do not commit downloaded corpora, prepared arrays, run directories, or checkpoints; a static report bundle is small evidence, not a replacement for its stated limitations.

## Documentation

Write for someone running their own experiment. Use native `sparselab` commands
and YAML/JSON declarations, through the locked environment; keep Python snippets
out of user workflows. If a required operation is API-only, describe the missing
CLI/DSL input, output and failure criteria in TODO.md instead of inventing a
command or teaching a private script. Label placeholders and prerequisites, and
use fresh task directories under the persistent work root. A config with an
explicit in-checkout destination keeps it; setting the root does not relocate it.

Keep feature availability in the [README matrix](README.md#feature-matrix),
runtime restrictions in [runtime](docs/runtime.md), and scientific lessons in the
[experiment ledger](docs/research/experiment-ledger.md). Link to the owner of a
topic instead of copying status summaries. Describe features by behavior rather
than the branch, release phase or author who introduced them. Retain schema and
protocol versions, source pins, checkpoint IDs, license attribution and historical
paths where they identify evidence. Never rewrite a prior result to modernize a
tutorial, or erase a failed result as obsolete documentation.

For documentation-only changes, check local links and heading anchors, code-fence
balance, CLI parser support and explicit path semantics. Do not launch a research
run to validate prose. Record unavailable hardware or execution checks at handoff.
