# Block Sparse Attention

`attention.kind: block_sparse` is a readable causal selector. It groups only already-available keys into fixed-size blocks, scores compressed mean-key representations against each query, selects the highest-scoring blocks, gathers their original K/V tokens, and computes regular attention over that gathered subset. It does not allocate a dense `[B, H, T, T]` score matrix.

`block_size` controls compression granularity; `selected_blocks` controls the retrieval budget. The implementation is intentionally a Python-loop reference, not an optimized kernel. It reports available and selected token counts, selection ratio, selected block IDs, and a simple selected-token attention-work estimate. Future work should add dense-teacher mass/recall and sparse-vs-dense quality curves.

```sh
uv run sparselab train configs/smoke_sparse_cpu.yaml --run-id sparse-smoke
```
