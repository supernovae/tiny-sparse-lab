# Contributing

Use Python 3.14 (selected by `.python-version`) and `uv sync --locked --group dev`. Run CPU checks with `uv run pytest -m "not mps and not network"` when tests exist. Do not commit downloaded corpora, prepared arrays, run directories, or checkpoints. Reports should include the resolved configuration and relevant run evidence.

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
