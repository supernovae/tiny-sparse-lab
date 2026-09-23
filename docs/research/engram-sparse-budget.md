# Memory and sparse selection budget

`engram-sparse-budget-v1` asks whether lexical memory changes observed degradation with smaller block-selection budgets. Its default contrasts dense attention with block-sparse attention (block size 16, two selected blocks) both with and without memory. `--design budget-sweep` additionally uses selected-block budgets 1, 2, and 4. The varied fields are `attention.kind`, `attention.selected_blocks`, and `model.memory`.

The recipe holds the usual route-local tokenizer/data packing, matched backbone/seed/optimizer/runtime/budget/evaluator/endpoint policy, and reports LM loss and each capability-card category separately. Available keys, selected keys, union behavior, and estimated attention work are diagnostics, not timing results. Batch/head unions can exceed a per-head selection budget; construction RNG, tokenizer-specific memory addresses, repeated offline data, optimization, scale ranges, and educational reference overhead remain confounders.

```sh
sparselab learn scaffold sparse-attention --scale smoke --data offline --backend cpu --output experiments/learn-sparse
sparselab research scaffold engram-sparse-budget-v1 --scale smoke --data offline --backend cpu --output experiments/sparse-memory
sparselab study plan experiments/sparse-memory/study.yaml
```

A result may show endpoint deltas for its declared contrasts, but it cannot establish an interaction/synergy statistic, speed, or paper-equivalent MLA cache compression. `tiny` is recommended; Python timing is not an architectural verdict. The selector is a local educational design; see [block-sparse attention](../sparse-attention.md).
