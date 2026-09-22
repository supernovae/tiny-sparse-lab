# ADR 0014: Compare scales through recorded, bounded configurations

## Status

Accepted for the scale-experiments milestone.

## Decision

Define scale experiments as explicit, bounded configurations and inspect their actual parameter/storage counts before training. Compare only runs with recorded data source, tokenizer, device, token budget, sequence length, and architecture. Report loss and throughput as observations; do not create a normalized quality score or infer architecture superiority from unmatched runs.

The reference scale matrix covers dense, sliding-window, MLA, and combined configurations. CPU smoke configurations validate wiring; larger configurations are opt-in local runs, not CI work.

## Consequences

Scale claims remain reproducible and condition-aware. This milestone adds experiment protocol and inspectable configurations, not benchmark results.
