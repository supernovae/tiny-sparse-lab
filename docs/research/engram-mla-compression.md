# Memory and MLA latent width

`engram-mla-compression-v1` asks whether lexical memory changes observed degradation as MLA latent width shrinks. The default uses dense versus MLA at latent width `D/2`, with memory absent/present. `--design latent-sweep` adds MLA widths `D` and `D/4`. The declared variables are `attention.kind`, `attention.latent_dim`, and `model.memory`.

Route-local tokenizer/data packing, matched seeds, backbone dimensions unless declared, optimizer, runtime, budget, evaluator, and endpoint policy are fixed. Held-out LM loss remains distinct from the three alias/override cards. Cache bytes must be calculated from this implementation: it stores expanded keys and latent values, and has no fused compressed-cache claim. Other confounders include construction RNG, tokenizer-specific lookup addresses, finite offline repetition, optimization differences, approximate scale ranges, and educational kernel overhead.

```sh
sparselab learn scaffold mla --scale smoke --data offline --backend cpu --output experiments/learn-mla
sparselab research scaffold engram-mla-compression-v1 --scale smoke --data offline --backend cpu --output experiments/mla-memory
sparselab study plan experiments/mla-memory/study.yaml
```

It can establish controlled declared endpoint deltas, not a synergy/interaction statistic, speed claim, or paper-equivalent compression result. `tiny` is recommended for a larger bounded trial. See [MLA](../mla.md) for the actual shapes and cache boundary.
