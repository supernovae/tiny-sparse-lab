# Tiny Sparse Lab

Tiny Sparse Lab is an installable PyTorch **architecture-learning laboratory**. Milestone 0.1 implements a transparent dense causal decoder and reproducible local training; it is not a production training framework and does not implement MoE, Engram, or other sparse mechanisms.

## Local workflow

```sh
uv sync --locked --dev
uv run sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run sparselab data prepare configs/smoke_cpu.yaml
uv run sparselab inspect configs/micro_dense.yaml --json
uv run sparselab train configs/smoke_cpu.yaml --run-id smoke
uv run sparselab eval smoke
uv run sparselab generate smoke --prompt "Once upon a time" --max-new-tokens 24
uv run sparselab dashboard --runs-dir runs
```

Use `--stop-after-step N` to make a durable interrupted checkpoint, then resume into a new child run:

```sh
uv run sparselab train configs/smoke_cpu.yaml --run-id part --stop-after-step 20
uv run sparselab train configs/smoke_cpu.yaml --run-id resumed --resume runs/part/checkpoints/step_00000020.pt
```

The dashboard is read-only and binds to `127.0.0.1`. It shows actual stored metric observations, events, ancestry, and metric explanations; it never launches training. Loss is not an architecture benchmark: compare runs only when data, tokenizer, budget, and device conditions are known.

## Data and licensing

Synthetic data is an offline fixture. TinyStories artifacts retain pinned source/revision and license metadata locally; do not commit downloaded corpus text, prepared arrays, run directories, or checkpoints. Project source is MIT licensed; downloaded datasets retain their own terms.

See [architecture](docs/architecture.md), [model scaling](docs/model-scaling.md), [metrics](docs/metrics.md), [training](docs/training.md), and the [dense-first decision](docs/decisions/0001-dense-first.md).
