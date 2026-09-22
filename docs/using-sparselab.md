# Using Tiny Sparse Lab

SparseLab is a local architecture-learning laboratory. Commands train small causal models, inspect declared model structure, query saved checkpoints, and view recorded telemetry. It does not provide a hosted model, a general-purpose chat API, or distributed training.

## Prepare, inspect, and train

```sh
uv sync --locked --dev
uv run sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run sparselab data prepare configs/smoke_cpu.yaml
uv run sparselab inspect configs/smoke_combined_cpu.yaml --json
uv run sparselab train configs/smoke_combined_cpu.yaml --run-id combined-smoke
uv run sparselab eval combined-smoke
uv run sparselab generate combined-smoke --prompt "Once upon a time" --max-new-tokens 24
```

`inspect` reports the configured architecture before training. `train` writes run-local artifacts and checkpoints. `eval` reports next-token loss and perplexity on the configured validation data. `generate` prints the prompt followed by greedy decoded tokens. A smoke run proves that path works; it is too small and short to promise coherent prose.

## Choose a mechanism deliberately

- `model.ffn: dense|moe` selects dense SwiGLU or local Top-K MoE.
- `model.memory: none|ngram|byte|portable` selects no memory, token n-gram memory, prepared raw-UTF-8 byte memory, or a portable frozen-table adapter.
- `attention.kind: dense|sliding_window|block_sparse|mla` selects full causal attention, a causal window, block-selected keys, or multi-head latent attention.

Each configuration is concrete. Do not infer a result from labels or parameter count alone; compare completed runs only when source, tokenizer, device, sequence length, token budget, optimizer, and seed match. See [architecture](architecture.md), [training](training.md), and [experiments](experiments.md).

## Inspect and verify a checkpoint

```sh
uv run sparselab checkpoint inspect runs/combined-smoke/checkpoints/latest.json --json
uv run sparselab checkpoint verify runs/combined-smoke/checkpoints/latest.json --json
```

A verified PyTorch checkpoint is a local immutable generation. Resume into a child run only with compatible contracts; use promotion for an intentional incompatible change. Optional MLX runs use native checkpoint directories and the same inspect/verify commands.

## Verify withheld-fact fixture evidence

```sh
uv run sparselab facts manifest --seed 0 --output artifacts/withheld-facts-seed-0.json
uv run sparselab facts audit artifacts/withheld-facts-seed-0.json
```

The audit proves the deterministic fixture’s data separation only. It is not a model score or transfer result. Use `sparselab facts evaluate RUN_ID MANIFEST` for retained completions from a real checkpoint, and `facts transfer-evaluate SOURCE_RUN TARGET_RUN MANIFEST` only for the explicit byte-memory adapter-transfer boundary. Read [withheld facts](withheld-facts.md) before drawing conclusions.

## View recorded local runs

```sh
uv run sparselab dashboard --runs-dir runs
```

The read-only, localhost-only viewer shows run status, data source, architecture settings, metrics, events, lineage, and retained evaluation evidence. It never launches or modifies training.
