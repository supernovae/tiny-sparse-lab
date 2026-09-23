# Lookup and narrower FFNs

`engram-ffn-substitution-v1` asks whether lexical lookup can compensate for a narrower dense FFN at a declared endpoint. It varies only `model.ffn_dim` (4×, 2×, and 1× hidden width) and `model.memory`; the complete design exposes all six raw outcomes rather than treating narrow-plus-memory versus wide-plus-none as a one-variable ablation.

The fixed controls are tokenizer/data packing per route, matched seed, backbone dimensions apart from FFN width, optimizer, runtime, budget, evaluator, and endpoint policy. Held-out LM loss and the three cards—training-format alias retention, held-out-wording alias recall, and context override—remain separate. Recall/completion might survive while composition or override behavior declines. Confounders include changed module inventories/initialization draws, tokenizer-specific addresses, the finite offline curriculum, capacity-dependent optimization, approximate scale ranges, and educational runtime overhead.

```sh
sparselab learn scaffold ffn --scale smoke --data offline --backend cpu --output experiments/learn-ffn
sparselab research scaffold engram-ffn-substitution-v1 --scale smoke --data offline --backend cpu --output experiments/ffn-memory
sparselab study plan experiments/ffn-memory/study.yaml
```

`smoke` is runnable; `tiny` is the catalog recommendation for a larger bounded trial, not a learning-efficiency result. The study can establish declared controlled endpoint deltas, not a training-efficiency, semantic/hybrid-memory, speed, or cross-tokenizer claim. Semantic and hybrid arms are prerequisite-gated and are not generated. [SwiGLU/FFN lesson details](lesson-paths.md) and [Engram diagnostics](../engram.md) explain the mechanisms first.
