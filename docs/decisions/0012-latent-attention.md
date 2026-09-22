# ADR 0012: Add a reference multi-head latent attention mode

## Status

Accepted for the MLA milestone.

## Decision

Add a causal multi-head latent-attention mode. It projects each hidden state into a shared latent K/V representation; per-head keys expand from that latent representation for dot products, while values remain partitioned latent vectors and the concatenated attended latent output projects back to model width. Queries retain normal per-head projections and RoPE applies to queries and expanded keys.

This is a compact reference mechanism. It retains ordinary score tensors and does not claim a fused decode cache, optimized latent cache bandwidth, or compatibility with the sliding-window mode.

## Consequences

The model can compare dense and latent attention under a common decoder/trainer interface. The latent width is explicit, positive, and divisible by the number of heads; checkpoints remain architecture-specific.
