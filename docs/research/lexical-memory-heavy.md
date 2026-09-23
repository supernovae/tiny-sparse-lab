# Lookup-heavy reduced backbones

`lexical-memory-heavy-v1` asks how useful a lookup-heavy small backbone is across declared lexical-table capacities. Every arm uses a reduced dense backbone (profile hidden width divided by two, at least one layer, and FFN width equal to the profile hidden width); the capacity axis is no memory/small/large/huge lexical tables. The independent fields are `model.hidden_dim`, `model.num_layers`, `model.ffn_dim`, `model.memory_table_size`, and `model.memory_dim`.

The reduced neural backbone is fixed within a selected scale, while data/tokenizer packing, seed matching, optimizer, runtime, budget, evaluator, and endpoint policy are controlled. Report LM loss and the three card categories separately, along with lookup/reuse/collision, gate/norm, and table-byte accounting. Larger tables can alter recall while revealing wording, collision, chat, or override failures. Addressing is tokenizer-specific; finite offline repetition, module/RNG changes, capacity-dependent optimization, approximate scale bands, and reference overhead are confounders.

```sh
sparselab learn scaffold lexical-engram --scale smoke --data offline --backend cpu --output experiments/learn-lexical
sparselab research scaffold lexical-memory-heavy-v1 --scale smoke --data offline --backend cpu --output experiments/lexical-heavy
sparselab study plan experiments/lexical-heavy/study.yaml
```

The catalog recommends `micro` and says to inspect table bytes before `tiny`. This is a fixed-neural table sweep, **not** an iso-total allocation curve. It cannot establish semantic retrieval, code/API/formula/composition behavior, speed, or cross-tokenizer equivalence; the required datasets/cards are unavailable. See [Engram](../engram.md).
