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

## Observed controlled dense runs

All runs below used the pinned TinyStories revision, the same 8192-token tokenizer, seed, optimizer, 128-token sequence length, and 204,800 target-token budget on MPS. Validation is standalone evaluation over the retained validation split; throughput is the final stored training observation.

| Run | Parameters | Validation loss | Validation perplexity | Tokens/s |
|---|---:|---:|---:|---:|
| `scale-micro-3m` | 3,344,064 | 5.1614 | 174.42 | 17,369.9 |
| `scale-dense-7m` | 6,917,376 | 4.9942 | 147.55 | 15,058.1 |
| `scale-dense-10m` | 10,244,160 | 4.8790 | 131.49 | 12,445.3 |
| `scale-dense-25m` | 29,893,120 | 4.5751 | 97.04 | 5,728.4 |
| `scale-dense-50m` | 50,274,752 | 4.2770 | 72.03 | 3,623.4 |

These five observations show lower held-out loss at this fixed budget as dense parameter count increases, alongside lower observed throughput. They do not establish an optimal scaling law, a compute-optimal allocation, or a sparse-architecture result.
The smoke configurations prove CPU wiring only. For a meaningful comparison, train separate unique run IDs with matching source, tokenizer, device, token budget, sequence length, optimizer, and seed. Store the resulting run directory and SQLite metrics; compare observed loss or throughput only at matching recorded budgets. Do not convert unmatched runs into an aggregate quality score.

Dense attention, sliding-window attention, MLA, MoE, and byte memory alter different resource boundaries. Attribute an observed difference only after a controlled ablation; a combined run is a compatibility check, not evidence that its mechanisms compound beneficially.
