# Scale experiments

Run each configuration with its declared tokenizer, data source, device, sequence length, token budget, and seed. Inspect first; do not infer parameter counts from a filename.

```sh
uv run sparselab inspect configs/micro_dense.yaml --json
uv run sparselab inspect configs/dense_7m.yaml --json
uv run sparselab inspect configs/smoke_sliding_cpu.yaml --json
uv run sparselab inspect configs/smoke_mla_cpu.yaml --json
uv run sparselab inspect configs/smoke_combined_cpu.yaml --json
```

The smoke configurations prove CPU wiring only. For a meaningful comparison, train separate unique run IDs with matching source, tokenizer, device, token budget, sequence length, optimizer, and seed. Store the resulting run directory and SQLite metrics; compare observed loss or throughput only at matching recorded budgets. Do not convert unmatched runs into an aggregate quality score.

Dense attention, sliding-window attention, MLA, MoE, and byte memory alter different resource boundaries. Attribute an observed difference only after a controlled ablation; a combined run is a compatibility check, not evidence that its mechanisms compound beneficially.
