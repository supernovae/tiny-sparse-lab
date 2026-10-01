# Contributing

Use Python 3.14 (selected by `.python-version`) and
`uv sync --locked --extra cpu --dev`. For the Linux CPU checks:

```sh
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu ruff format --check .
uv run --locked --extra cpu pytest -q -n 2 --dist loadfile \
  -m "not mps and not mlx and not cuda and not rocm and not xpu and not network"
```

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
