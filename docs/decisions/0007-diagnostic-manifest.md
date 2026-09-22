# ADR 0007: Withheld-fact diagnostics require an immutable split manifest

## Status

Accepted for Milestone 0.7.

## Decision

Export a canonical JSON manifest before any withheld-fact diagnostic is reported. It contains the fixture format version, seed, ordered training statements, ordered held-out prompt/expected-value cases, and a SHA-256 digest of the canonical payload. Export never includes training weights, tokenized examples, model outputs, or a score.

Existing output is reusable only when its canonical payload matches. A conflicting output fails instead of overwriting recorded split evidence.

## Consequences

Future trained diagnostics can cite an exact split artifact and verify held-out values were not in its training statements. The manifest proves fixture-level separation only; it does not prove a training implementation obeyed that boundary without separately recording its data provenance.
