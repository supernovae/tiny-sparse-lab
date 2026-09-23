# Routed experts or lookup

`engram-moe-capacity-v1` asks where observed conditional capacity resides: routed experts or lexical lookup. It compares dense 2×/4× FFNs and a local MoE (four experts, FFN width `D`, top-2 routing) with memory absent/present. The independent fields are `model.ffn`, `model.ffn_dim`, `model.num_experts`, `model.experts_per_token`, and `model.memory`; the controlled recipe fixes `model.router_aux_loss_coefficient` at zero.

Matched controls and separate LM/card reporting are the catalog convention. Use router entropy, maximum-expert fraction, and mean top-k probability as mechanism diagnostics; no-memory arms do not manufacture Engram metrics. Module construction RNG, router/optimizer-state changes, tokenizer addresses, finite offline repetition, capacity-dependent optimization, scale ranges, and educational overhead confound interpretation.

```sh
sparselab learn scaffold moe --scale smoke --data offline --backend cpu --output experiments/learn-moe
sparselab research scaffold engram-moe-capacity-v1 --scale smoke --data offline --backend cpu --output experiments/moe-memory
sparselab study plan experiments/moe-memory/study.yaml
```

Dense 2× and MoE approximately share selected FFN matrix capacity; that is not measured equal FLOPs or time. The recipe can show controlled endpoint deltas over its declared dense/routed options, but not iso-compute, speed, semantic-memory, or auxiliary-balancing conclusions. `tiny` is recommended for a larger bounded run. See [the MoE mechanism](../moe.md).
