# ADR 0011: Start attention sparsity with causal sliding windows

## Status

Accepted for the sparse-attention milestone.

## Decision

Add a local causal sliding-window attention mode. Query position `t` attends only keys in `[max(0, t - window_size + 1), t]`; no token may attend future positions. Keep the existing projections, RoPE, values, residual structure, and output shape unchanged.

The first implementation is a reference mask over ordinary attention scores. It demonstrates the information boundary and is not a throughput optimization or a block-sparse kernel claim.

## Consequences

The mode has deterministic semantics and can be compared against dense attention with the same model dimensions. Long-context performance, global tokens, learned routing, kernel efficiency, and distributed attention remain out of scope.
