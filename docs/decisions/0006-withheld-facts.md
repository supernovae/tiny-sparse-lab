# ADR 0006: Synthetic withheld-fact diagnostics are data-separation tests

## Status

Accepted for Milestone 0.6.

## Decision

Provide deterministic synthetic fact records split by fact identity, not random token positions. A held-out fact never appears in the training document iterator. Evaluation prompts contain only public subject/relation text and expect the held-out value. The diagnostic records its exact train/evaluation fact sets and does not supervise a backbone, memory adapter, or router on held-out values.

The first implementation is an offline dataset fixture and verifier, not a training-mode switch. It makes data-leakage boundaries testable before any claim about byte-memory transfer.

## Consequences

A score is meaningful only when its saved fact manifest proves disjointness. The fixture is not TinyStories and does not establish general language-model quality or knowledge transfer.
