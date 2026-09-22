# Contributing

Use Python 3.12 and `uv sync --locked --dev`. Run CPU checks with `uv run pytest -m "not mps and not network"` when tests exist. Do not commit downloaded corpora, prepared arrays, run directories, or checkpoints. Reports should include the resolved configuration and relevant run evidence.
