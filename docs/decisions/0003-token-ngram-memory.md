# ADR 0003: Causal token n-gram memory before byte hashing

## Status

Accepted for Milestone 0.3.

## Context

Milestone 0.2 established local compute sparsity without changing the decoder's attention, token boundary, or data pipeline. The next independent mechanism is an addressed memory adapter. It must not inspect future tokens, reuse MoE dispatch, or claim tokenizer-agnostic transfer before byte-addressing and withheld-fact methodology exist.

## Decision

Add an optional causal token n-gram memory adapter at the decoder's final hidden-state boundary. For input IDs `[B,T]`, address position `t` from its inclusive suffix ending at `t`; the resulting hidden state predicts ID `t+1`, so no target/future ID is read. A fixed polynomial rolling hash maps an n-gram of token IDs into a local learnable table. The table value width is configurable and independent of backbone hidden width. A learned query gate scales an output adapter projection before it is added to the backbone hidden state.

Address generation, value table, latent adapter, gate, and diagnostics are separate components. The table and adapter are ordinary trainable parameters; there is no external retrieval corpus, write cache, distributed shard, approximate nearest-neighbor search, or online mutation. A disabled memory path is an exact no-op.

This milestone deliberately addresses **token IDs**, not bytes. Token-ID addresses are tokenizer-specific. Byte-equivalent span hashing, normalization policy, cross-tokenizer matching, withheld-fact transfer experiments, and claims of knowledge transfer remain future work.

## Consequences

The memory is causal and checkpointed with the model. Dense, MoE, and memory configuration changes remain incompatible for resume. Accounting reports memory storage separately and includes its adapter/gate in active-per-token parameters when enabled. Diagnostics are detached and describe only observed address usage; they do not establish retrieval quality.

## References

- [DeepSeek Engram paper](https://arxiv.org/abs/2601.07372) and [official demo](https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py)
- [Tokenizer-Agnostic Engram](https://arxiv.org/html/2607.29065) and [prototype](https://github.com/jararap/polyhash-engram)
