# Engram placement

`engram-placement-v1` asks whether placement changes learned behavior when lexical-memory capacity is fixed. Its hypothesis is that earlier retrieved values may affect later tokens without improving held-out scores; equal capacity therefore does not imply useful retrieval.

The recipe varies `model.memory` and `model.memory_injection`: no memory, final injection, and embedding injection. It fixes tokenizer/data packing within a chosen route, seed-matched width/depth/heads, optimizer, runtime, budget, evaluator, and endpoint policy. It reports held-out language-model loss separately from `chat-alias-retention-v1`, `chat-alias-recall-v1`, and `chat-context-override-v1`; these are respectively training-format acquisition, held-out wording (not unseen facts), and instruction/context stress on offline data. On TinyStories, all three are out-of-domain stress, while held-out cross-entropy is the LM measurement.

```sh
sparselab learn scaffold lexical-engram --scale smoke --data offline --backend cpu --output experiments/learn-lexical
sparselab research scaffold engram-placement-v1 --scale smoke --data offline --backend cpu --output experiments/placement
sparselab study plan experiments/placement/study.yaml
```

Scaffolding writes configurations only. The selected recipe has declared endpoint comparisons, not a prediction of a winner. Construction-RNG differences, tokenizer-specific addresses, finite repeated offline transcripts, optimization differences, approximate scale ranges, and educational-kernel overhead remain confounders. This study covers lexical final and embedding injection only; it does not cover semantic adapter placement. Phase D's direct semantic `DenseLM` API supports embedding, `after_block`, and final sites with multiple named attachments; see [Verified semantic memory](semantic-memory.md). This lexical study cannot establish speed, semantic retrieval, unseen-fact transfer, or cross-tokenizer equivalence. See [Engram](../engram.md).
