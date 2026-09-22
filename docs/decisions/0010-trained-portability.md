# ADR 0010: Measure byte-address portability with real trained runs

## Status

Accepted for the portability-experiment milestone.

## Decision

The portability experiment will treat raw UTF-8 address equivalence as a preflight invariant, then train and evaluate separate byte-memory models with independently trained tokenizer artifacts. Each run must retain its tokenizer hash, verified withheld-fact manifest digest, source documents, checkpoint, and decoded held-out completions.

The experiment will report exact outputs and an explicit-match count for the held-out prompts. It will not infer semantic transfer from fixture integrity alone, pool model weights across tokenizers, or treat a single successful completion as a general portability claim.

## Consequences

This creates an end-to-end, reproducible measurement rather than a simulated score. It requires a dedicated isolated data source and evaluation command before a result can be claimed.
