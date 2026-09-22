# ADR 0017: Grow claims through versioned capability cards

## Status

Accepted.

## Context

Small models are useful research instruments when a result can be repeated, falsified, and extended across scales. A smoke run, a chat wrapper, or an aggregate held-out loss does not isolate whether an Engram or another architectural mechanism improved a named capability.

## Decision

Use versioned capability cards for narrow, explicit hypotheses. Each card has fixed held-out cases, a deterministic scorer, a digest, a declared baseline, and an allowed set of variant differences. The first card, `engram-recall-v1`, evaluates exact associative recall from held-out record wording. The paired comparison command rejects runs whose backbone, attention, dataset, tokenizer, training budget, optimizer, or runtime controls differ; only Engram memory settings may differ.

Persist a card result with its run. Treat its score delta as evidence for that card only. Reuse the same card across scale configurations to build a cumulative evidence series, and add a new card rather than silently expanding a claim.

## Consequences

Models and experiments remain cumulative rather than disposable: shared tokenization, data, checkpoints, scoring, and evidence contracts persist while scale and mechanism variants evolve. A positive card result can motivate the next scale or task-family experiment; it does not establish general intelligence, broad instruction following, or universal Engram benefit.

## Related decisions

- [ADR 0015](0015-single-host-extension-boundaries.md) keeps extension boundaries explicit.
- [ADR 0016](0016-evidence-before-claims.md) pairs held-out observations with verified checkpoints.
