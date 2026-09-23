# Contributing

Use Python 3.12 and `uv sync --locked --dev`. Run CPU checks with `uv run pytest -m "not mps and not network"` when tests exist. Do not commit downloaded corpora, prepared arrays, run directories, or checkpoints. Reports should include the resolved configuration and relevant run evidence.

## Architecture experiments

The [research contribution workflow](docs/research/contributing.md) explains validated local catalog/recipe forks, declared controls and variations, train-only data boundaries, retained negative outcomes, and evidence identities. Do not commit downloaded corpora, prepared arrays, run directories, or checkpoints; a static report bundle is small evidence, not a replacement for its stated limitations.
