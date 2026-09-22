# ADR 0013: Compose independent reference mechanisms explicitly

## Status

Accepted for the combined-architecture milestone.

## Decision

Provide a runnable configuration that combines MLA attention, local Top-K MoE feed-forward routing, and causal byte-address memory in one decoder. Each mechanism retains its own contract: MLA controls attention representation, MoE selects configurable local FFN experts per token, and byte memory receives prepared causal addresses after final normalization.

No mechanism is reinterpreted as another: MLA is not sparse attention selection, MoE is not distributed routing, and byte memory is not external retrieval. The combined reference is a compatibility smoke, not evidence that the mechanisms improve quality or throughput together.

## Consequences

Architecture composition becomes testable with the existing trainer/checkpoint path. Checkpoints remain configuration-specific; individual ablations are required for conclusions.
