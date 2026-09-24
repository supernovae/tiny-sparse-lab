# Engram Memory

SparseLab's token Engram is a causal N-gram lookup table with a latent value width independent of the backbone hidden width. Each position hashes only its current and earlier token IDs, retrieves a latent table vector, projects it to model width, and mixes it with the hidden state through a sigmoid gate:

$$h'_t = h_t + \sigma(g(h_t))\,A\,M[a(x_{\leq t})].$$

Set `model.memory: ngram` with `memory_table_size`, `memory_ngram_size`, and `memory_dim`. Optional `memory_ngram_orders` and `memory_hash_heads` create independent causal lookup streams whose projected values are averaged before gating. The byte-addressed variant (`memory: byte`) receives causal raw UTF-8 suffix addresses prepared alongside tokens; this is an addressing experiment, not evidence of tokenizer-independent knowledge transfer.

Each forward pass records lookup count, unique buckets, collisions, bucket reuse rate, table utilization, largest-bucket fraction, mean gate activation, retrieved-vector norm, and hidden-state norm. High reuse can arise from a small table or repeated corpus structure; it is not by itself a learned-memory success signal. Inspect language loss and withheld-fact evaluation separately.

```sh
uv run sparselab train configs/smoke_memory_cpu.yaml --run-id token-engram-smoke
uv run sparselab train configs/smoke_byte_memory_cpu.yaml --run-id byte-engram-smoke
```

Multi-order/multi-head streams preserve causal addressing and allocate one table per stream. Per-run metrics persist aggregate lookup count, unique buckets, collisions, reuse rate, table utilization, and gate mean, plus the same six metrics under `engram/stream_N/*` for each stream. The Architecture dashboard renders those persisted series. Portable packages and frozen adapter training are available through `memory: portable`; the withheld-fact control matrix currently yields no transfer success, so it remains a negative experiment result.

## Learn and compare placement

The [`lexical-engram` lesson](research/lesson-paths.md) provides a standalone final-injection configuration and initialized shape/diagnostic probe. Lexical `memory_injection` supports final and embedding placement; these are the alternatives in the [placement study](research/engram-placement.md), not evidence for lexical middle placement. Phase D separately adds verified semantic retrieval and direct `DenseLM` semantic adapters at embedding, after-block, and final sites, including multiple named/multipack attachments; see [Verified semantic memory](research/semantic-memory.md). Lookup diagnostics remain mechanism observations, not score or transfer proof.


## Adding an Engram variant

Treat a new Engram as an explicit architecture and artifact contract, not as a hidden replacement for an existing mode. Define its causal address inputs, table/adapter ownership, checkpoint compatibility, parameter accounting, and persisted diagnostics before adding it to `model.memory`. Preserve `none`, `ngram`, `byte`, and `portable` semantics unchanged so existing runs remain interpretable. A new addressing scheme must state what equality it preserves and what it does **not** establish; address collisions, lookup reuse, or a successful smoke run are not evidence of retrieval quality or knowledge transfer.
