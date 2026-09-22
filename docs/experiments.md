# Scale experiments

Run each configuration with its declared tokenizer, data source, device, sequence length, token budget, and seed. Inspect first; do not infer parameter counts from a filename.

```sh
uv run sparselab inspect configs/micro_dense.yaml --json
uv run sparselab inspect configs/dense_7m.yaml --json
uv run sparselab inspect configs/dense_10m.yaml --json
uv run sparselab inspect configs/dense_25m.yaml --json
uv run sparselab inspect configs/dense_50m.yaml --json
uv run sparselab inspect configs/smoke_sliding_cpu.yaml --json
uv run sparselab inspect configs/smoke_mla_cpu.yaml --json
uv run sparselab inspect configs/smoke_combined_cpu.yaml --json
```


The dense scale presets are inspected at 3,344,064, 6,917,376, 10,244,160, 29,893,120, and 50,274,752 parameters. They share the pinned TinyStories revision, 8192-token tokenizer, sequence length, token budget, optimizer, and seed. Parameter count alone is not a comparison result: report each completed run's observed validation loss, perplexity, throughput, device, and metric coordinates.

## Historical controlled dense runs

These recorded runs used the pinned TinyStories revision, the same 8192-token tokenizer, seed, optimizer, 128-token sequence length, and 204,800 target-token budget on MPS. Validation was standalone evaluation over the retained validation split. They predate the current integrity-bound report format; retain them as historical loss observations, not newly verified capability evidence.

| Run | Parameters | Validation loss | Validation perplexity |
|---|---:|---:|---:|
| `scale-micro-3m` | 3,344,064 | 5.1614 | 174.42 |
| `scale-dense-7m` | 6,917,376 | 4.9942 | 147.55 |
| `scale-dense-10m` | 10,244,160 | 4.8790 | 131.49 |
| `scale-dense-25m` | 29,893,120 | 4.5751 | 97.04 |
| `scale-dense-50m` | 50,274,752 | 4.2770 | 72.03 |

These observations suggest lower validation loss at this fixed budget as dense parameter count increases. They do not establish an optimal scaling law, chat ability, or an Engram benefit. Previously published throughput values are withdrawn: the trainer divided one update's tokens by cumulative run time. New measurements use synchronized update duration; rerun controlled pairs before making a performance claim.
The smoke configurations prove CPU wiring only. For a meaningful comparison, train separate unique run IDs with matching source, tokenizer, device, token budget, sequence length, optimizer, and seed. Store the resulting run directory and SQLite metrics; compare observed loss or throughput only at matching recorded budgets. Do not convert unmatched runs into an aggregate quality score.

Dense attention, sliding-window attention, MLA, MoE, and byte memory alter different resource boundaries. Attribute an observed difference only after a controlled ablation; a combined run is a compatibility check, not evidence that its mechanisms compound beneficially.

## Evidence before comparison

Every serious local run should have verified checkpoint/held-out evidence before it enters a comparison:

```sh
uv run sparselab checkpoint verify runs/RUN_ID/checkpoints/latest.json --json
uv run sparselab evidence RUN_ID --json
```

Training records held-out validation at the initial, configured periodic, and terminal boundaries; each observation is attached to a verified checkpoint generation. This validates the local experiment path, not a general model-quality claim. See [experiment evidence](evidence.md) for evidence levels, controlled-comparison requirements, and the future hardware/reference-harness protocol.

For a named architectural hypothesis rather than aggregate loss alone, use a [capability card](capabilities.md). Cards preserve their own prompt set, scorer, baseline controls, and scope-limited conclusion.
