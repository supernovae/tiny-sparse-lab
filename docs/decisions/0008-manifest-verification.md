# ADR 0008: Verify withheld-fact manifests against the fixture

## Status

Accepted for Milestone 0.8.

## Decision

Provide an offline verifier for a recorded withheld-fact manifest. It checks the required format version and fields, recomputes the canonical SHA-256 digest, and requires the payload to exactly equal the fixture-generated manifest for its recorded seed.

A matching digest alone is insufficient: a self-consistent but altered payload is rejected. The verifier reads evidence only and does not train a model, create data, or report a score.

## Consequences

A future diagnostic can reject corrupted or substituted split evidence before use. Verification proves that an artifact matches this repository's fixture definition; it still cannot prove what data a separate training pipeline consumed.
