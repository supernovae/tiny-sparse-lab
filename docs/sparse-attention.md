# Block Sparse Attention

`attention.kind: block_sparse` is a readable causal selector. It groups only already-available keys into fixed-size blocks, scores compressed mean-key representations against each query, selects the highest-scoring blocks, gathers their original K/V tokens, and computes regular attention over that gathered subset. It does not allocate a dense `[B, H, T, T]` score matrix.

`block_size` controls compression granularity; `selected_blocks` controls the retrieval budget. Block means use prefix sums instead of constructing a Python list of slices for every query; query-time selection and gathered attention remain a readable Python-loop reference, not a custom sparse kernel. It reports available and selected token counts, selection ratio, selected block IDs, a selected-token attention-work estimate, dense-teacher retained mass, and dense Top-K recall. Sparse-vs-dense quality curves remain a separate controlled experiment.

```sh
uv run sparselab train configs/smoke_sparse_cpu.yaml --run-id sparse-smoke
```
