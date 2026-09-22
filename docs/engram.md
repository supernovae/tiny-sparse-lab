# Engram Memory

SparseLab's token Engram is a causal N-gram lookup table with a latent value width independent of the backbone hidden width. Each position hashes only its current and earlier token IDs, retrieves a latent table vector, projects it to model width, and mixes it with the hidden state through a sigmoid gate:

$$h'_t = h_t + \sigma(g(h_t))\,A\,M[a(x_{\leq t})].$$

Set `model.memory: ngram` with `memory_table_size`, `memory_ngram_size`, and `memory_dim`. The byte-addressed variant (`memory: byte`) receives causal raw UTF-8 suffix addresses prepared alongside tokens; this is an addressing experiment, not evidence of tokenizer-independent knowledge transfer.

Each forward pass records lookup count, unique buckets, collisions, bucket reuse rate, table utilization, largest-bucket fraction, mean gate activation, retrieved-vector norm, and hidden-state norm. High reuse can arise from a small table or repeated corpus structure; it is not by itself a learned-memory success signal. Inspect language loss and withheld-fact evaluation separately.

```sh
uv run sparselab train configs/smoke_memory_cpu.yaml --run-id token-engram-smoke
uv run sparselab train configs/smoke_byte_memory_cpu.yaml --run-id byte-engram-smoke
```

The current implementation has one causal N-gram order and one hash stream per configured memory module. Multi-order/multi-head addressing, a portable package format, and adapter-only transfer training remain future work.
