# ADR 0002: Local Top-K MoE feed-forward routing

## Status

Accepted for Milestone 0.2.

## Context

The dense decoder isolates its feed-forward boundary at `DecoderBlock.ffn`. Attention, tokenization, packing, optimizer policy, and checkpoint format remain useful baseline controls. The next experiment should add compute sparsity without conflating it with sparse attention, external memory, distributed dispatch, or architecture transfer claims.

## Decision

Replace only the dense SwiGLU feed-forward module with a local Top-K routed SwiGLU bank. The router maps each normalized token representation `[B,T,D]` to `num_experts` logits, selects `experts_per_token` probabilities, renormalizes them, and sums the selected expert outputs. An optional shared expert runs for every token.

Selection, normalized weights, dispatch, and diagnostics are distinct implementation boundaries. Dispatch groups flattened token positions by selected expert and scatters outputs back to their original positions. There is no capacity limit, token dropping, all-to-all exchange, or distributed dependency. The configurable auxiliary balance loss is added to the language loss by the trainer.

Diagnostics are detached aggregates: per-expert token counts/fractions, router entropy, maximum fraction, selected-probability mass, and auxiliary loss. Persisted metric series expose the scalar diagnostics for the dashboard.

## Consequences

The dense model remains the default. MoE configuration is explicit and cannot silently select a sparse path. Parameter accounting reports all expert storage and a direct per-token active count; it does not call active parameters FLOPs or imply distributed savings. Existing dense checkpoints are not compatible with an MoE configuration, and resume configuration equality rejects the mismatch.

## References

- [Swiss AI MoE router](https://github.com/swiss-ai/MoE/blob/main/moe.py)
- [Flaxformer routing](https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py)
- [Megatron router](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router.py)
