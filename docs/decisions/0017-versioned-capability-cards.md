# ADR 0017: Grow claims through versioned capability cards

## Status

Accepted; comparison and chat protocols refined by [ADR 0018](0018-chat-native-experiments.md).

## Context

Small models are useful research instruments when a result can be repeated, falsified, and extended across scales. A smoke run, a chat wrapper, or an aggregate held-out loss does not isolate whether an Engram or another architectural mechanism improved a named capability.

## Decision

Use versioned capability cards for narrow, explicit hypotheses. Each card has fixed cases, a deterministic scorer, a digest, a declared baseline, and an allowed set of variant differences. The initial `engram-recall-v1` surface is now classified as a wiring diagnostic: the prompt contains the answer and changing record IDs does not establish held-out wording. It is retained for historical comparison, not a generalization claim. The newer chat-native cards use disjoint train/validation/test query combinations and full-answer scoring.

Persist a card result with its run. Treat its score delta as evidence for that card only. Reuse the same card across scale configurations to build a cumulative evidence series, and add a new card rather than silently expanding a claim.

## Consequences

Models and experiments remain cumulative rather than disposable: shared tokenization, data, checkpoints, scoring, and evidence contracts persist while scale and mechanism variants evolve. A positive card result can motivate the next scale or task-family experiment; it does not establish general intelligence, broad instruction following, or universal Engram benefit.

## Related decisions

- [ADR 0015](0015-single-host-extension-boundaries.md) keeps extension boundaries explicit.
- [ADR 0016](0016-evidence-before-claims.md) pairs held-out observations with verified checkpoints.
