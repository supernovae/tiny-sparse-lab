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
The smoke configurations prove CPU wiring only. For a meaningful comparison, train separate unique run IDs with matching source, tokenizer, device, token budget, sequence length, optimizer, and seed. Store the resulting run directory and SQLite metrics; compare observed loss or throughput only at matching recorded budgets. Do not convert unmatched runs into an aggregate quality score.

Dense attention, sliding-window attention, MLA, MoE, and byte memory alter different resource boundaries. Attribute an observed difference only after a controlled ablation; a combined run is a compatibility check, not evidence that its mechanisms compound beneficially.
