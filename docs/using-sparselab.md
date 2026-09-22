# Using Tiny Sparse Lab

SparseLab is a local educational laboratory. Commands train small causal models, inspect actual architecture counts, query saved checkpoints, and inspect recorded telemetry. It does not provide a hosted model or a general-purpose chat API.

## Prepare and inspect

```sh
uv sync --locked --dev
uv run sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run sparselab data prepare configs/smoke_cpu.yaml
uv run sparselab inspect configs/smoke_combined_cpu.yaml --json
```

The final command constructs the declared model without downloading data. On the current combined smoke configuration it reports 63,312 total parameters, 44,880 active parameters per token, 5,120 attention parameters, 36,864 local-expert parameters, and 759,744 estimated checkpoint tensor bytes. Counts are architecture facts, not quality metrics.

## Train and query a checkpoint

```sh
uv run sparselab train configs/smoke_combined_cpu.yaml --run-id combined-smoke
uv run sparselab eval combined-smoke
uv run sparselab generate combined-smoke --prompt "Once upon a time" --max-new-tokens 24
```

`generate` prints the prompt followed by greedy newly decoded tokens. A smoke run proves the model/checkpoint/query path works; it is too small and short to promise coherent prose. `eval` reports next-token loss and perplexity for the configured validation data. Compare either only when source, tokenizer, device, sequence length, and token budget match.

## Read model choices

- `model.ffn: dense|moe` selects dense SwiGLU or local top-1 experts.
- `model.memory: none|ngram|byte` selects no memory, tokenizer-ID n-gram memory, or prepared raw-UTF-8 byte memory.
- `attention.kind: dense|sliding_window|mla` selects full causal attention, a local causal window, or multi-head latent attention.

See [architecture](architecture.md), [training](training.md), and [experiments](experiments.md) for boundaries and comparison protocol.

## Verify withheld-fact evidence

```sh
uv run sparselab facts manifest --seed 0 --output artifacts/withheld-facts-seed-0.json
uv run sparselab facts audit artifacts/withheld-facts-seed-0.json
```

The current fixture audit emits:

```json
{
  "format_version": 1,
  "held_out_case_count": 2,
  "held_out_values_absent_from_training": true,
  "seed": 0,
  "sha256": "be21efcc382fb2e920ab109ea6597d2078bfcddfc278f2e872f728a9ae96254e",
  "training_statement_count": 6,
  "valid": true
}
```

Use `sparselab facts evaluate RUN_ID MANIFEST` to save exact decoded held-out completions for a real checkpoint. Use `facts transfer-evaluate SOURCE_RUN TARGET_RUN MANIFEST` only to test the explicit byte-memory adapter transfer boundary. A positive completion is an observation, not a general transfer claim; see [withheld facts](withheld-facts.md).

## View recorded runs

```sh
uv run sparselab dashboard --runs-dir runs
```

The localhost-only viewer lists run status, data source, attention kind, FFN, memory mode, metrics, events, and retained evaluation evidence. Its Learn page points back to the repository documentation; it never launches or modifies training.
