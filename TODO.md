# Implementation backlog

This file tracks only repository changes that require code. Scientific questions,
campaign status, and proposed experiments belong in
[`docs/research/roadmap.md`](docs/research/roadmap.md), the versioned research
lifecycle, or a checked-in [`experiments/research/`](experiments/research/)
record. Completed work remains discoverable in Git history and evidence records;
it is not retained here as a second changelog.

When a task moves to GitHub, use the **Code task** issue template and replace the
item below with its issue link. A code item is complete only when its tests and
documentation land. An experiment result, including a negative result, does not
close a code item unless the named software acceptance criteria also pass.

Work from top to bottom within the active priorities. Favor reusable support for
prepare → lock → dispatch → train/resume → evaluate → retain evidence. Campaign
names, snapshot counts, prompts, seeds, and machine-specific measurements belong
in declarations, regression fixtures, or evidence records, not special branches
in shared lab code. Operational changes must preserve scientific settings and
artifact identities.

## Experiment ergonomics

### Rapid iteration

The workload is [TinyStories baseline → exposure extension → one-field
contrast](docs/tinystories-microlab.md) and repeated declared model campaigns.
Use the [existing native route](docs/iteration.md), including the read-only
`iteration check`, direct `experiment bind-inputs` and exact-cell
`experiment export-config` commands. Remaining work below must reuse those
interfaces, not introduce another runner.

- [ ] **P2 — Expose verified declaration/config handoffs.** Where Phase `set`
  and native scaffolds do not suffice, add CLI derivation over the existing
  typed loaders and dotted-path compiler, preserving path anchoring when the
  output moves. Exact resolved-cell export is already available; remaining
  handoffs must avoid Python and reinterpretation of SHA fields. Write new files only,
  record origin/delta, reject unknown fields and scientific incompatibility.
  Existing checkpoint selectors already serve declared chains; add standalone
  immutable selection only if the direct TinyStories route still needs pointer
  parsing. Never select an unverified "latest" directory by sorting filenames.
- [ ] **P2 — Exercise the complete declared TinyStories iteration demo.** Once
  native direct-input binding is available, add a copyable ExperimentPlan/Campaign
  example with one baseline, a checkpoint-bound exposure child, fixed heldout
  suite/descriptive generation panel, a separately labeled fresh one-field
  contrast and optional seed replication. Share the topology with a small offline
  acceptance fixture, including interrupted reconciliation and unchanged parent
  hashes. Publish actual pinned-TinyStories acceptance separately: declaration
  digests, run/ingestion states, checkpoint lineage, counters, evaluation/panel
  references and costs. Keep outputs external and make CLI text/JSON follow the
  same route. No embedded Python, duplicate implicit baseline, automatic promotion
  or claim that offline smoke establishes real-data quality. The current direct
  teaching walkthrough is not this declarative acceptance gate.

### P2 — Make evidence collection and review reusable

- [x] **Lint checked-in research records.** Extend existing identity/schema
  checks to validate protocol/config bindings and evidence references under
  `experiments/research/`, and reject checked-in checkpoints, caches, datasets,
  logs, and mutable run output. Complete with valid, missing/mismatched-reference,
  and forbidden-output fixtures. Validate durable declarations and references
  without requiring live external datasets/checkpoints or adding a new registry.
  Fixed with [offline record lint](experiments/research/README.md#offline-record-lint),
  CI enforcement, and [declaration/reference/output regressions](tests/test_research_lint.py).
- [x] **Preserve a compact metadata explanation of corpus identities.** Extend
  existing archive/provenance machinery with project/declaration identity,
  source-ID→snapshot-SHA mapping, algorithm/file provenance, full build identity
  payload, and release identity. Complete when metadata round trips and survives
  relocation, declared digest payloads can be checked without source bytes, and
  tampering is rejected. Explaining an identity must not claim verification or
  reconstruction of unavailable source contents.
  Fixed with [archive identity metadata](docs/research/lifecycle-recovery.md)
  and [offline round-trip, relocation, and tampering regressions](tests/test_corpus_identity.py).

## P3 — Conditional work; activate for a concrete workload

These remain implementation gaps, but should not displace P1/P2 without a
documented workload need and the required acceptance environment.

- [ ] **Implement native CUDA sparse attention.** Activate when a planned CUDA
  workload needs this path and NVIDIA hardware is available. Preserve reference
  semantics and add hardware-gated correctness tests and component benchmarks.
  CPU or another accelerator cannot close this item; component speed does not
  establish end-to-end throughput or model quality.
- [ ] **Expose verified snapshot inheritance in recovery declarations.**
  Activate when recurring corpus-version workflows need declarative reuse.
  Verified ancestry/reuse already exists in Python APIs; extend the typed
  recovery schema and CLI rather than adding another replay engine. Complete
  when explicit parent evidence and selected source IDs preserve snapshot
  identities without reacquisition, and tampered parents or incompatible source
  declarations are rejected. Use generic ancestry fixtures rather than requiring
  a particular campaign's versions or snapshot counts.
- [ ] **Add an operational spot-safety policy.** Activate for a recurring spot
  executor workload. Existing checkpoint cadence, retention, verification, and
  explicit child resume are foundations, not missing features. Add a separately
  configured policy using observed checkpoint write time, interruption notice,
  workspace capacity, and measured restart cost; record its chosen cadence while
  preserving scientific settings. Cover unavailable measurements, insufficient
  capacity, and the separation of operational cadence from scientific identity.

## Boundaries and acceptance work

Distributed training is deferred under the
[single-host extension decision](docs/decisions/0015-single-host-extension-boundaries.md).
It is not an active compatibility/refactoring task. A future proposal must
define optimizer ownership, data partitioning, checkpoint identity, and MoE
expert placement before implementation.

Real spot/remote recovery validation belongs in operational acceptance records,
following the existing [worker acceptance evidence](docs/workers.md#observed-acceptance--2026-09-23).
Exercise abrupt loss, verified-generation selection, explicit child resume,
remote artifact transfer, and storage retention before claiming spot cost or
recovery benefits. Local loss/resume evidence does not close that gate; add code
tasks here only for missing or defective behavior exposed by acceptance.
