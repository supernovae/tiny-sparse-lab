# ADR 0009: Summarize verified split evidence separately from results

## Status

Accepted for Milestone 0.9.

## Decision

Provide an offline manifest audit that begins with full verification and emits a compact machine-readable summary: fixture format, seed, digest, training-statement count, held-out-case count, and whether held-out values occur in the training-statement text.

The audit emits no model output, score, benchmark comparison, or claim about a training pipeline. It is a report of verified fixture evidence, not an experiment result.

## Consequences

Automation can retain a concise, inspectable record without parsing fixture statements itself. The report remains limited: absence from this manifest's statements does not prove absence from any independently assembled training corpus.
