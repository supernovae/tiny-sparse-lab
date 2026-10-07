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

The workload is baseline → exposure extension → one-field contrast and repeated
declared model campaigns. The [TinyStories walkthrough](docs/tinystories-microlab.md)
is one teaching reference for that general workflow.
Use the [existing native route](docs/iteration.md), including the read-only
`iteration check`, direct `experiment bind-inputs` and exact-cell
`experiment export-config` commands. Remaining work below must reuse those
interfaces, not introduce another runner.

Pinned [dataset declarations](docs/datasets.md), resumable snapshots and preparation,
coverage/pass budgets, typed Campaign input binding and existing-input staging
are available through the lab. New Hub workloads use `dataset.source: snapshot`;
historical dataset-specific inputs retain their original verifiers.

### Experiment ledger projection

- [ ] **P2 — Share an evidence-backed experiment ledger between CLI and dashboard.**
  Extend the existing research lifecycle/report readers, rather than adding a
  second registry or runner. Input: lifecycle declarations, verified report roots,
  evidence references and an optional run store. Expose question, declared delta
  and controls, protocol/checkpoint identity, result and limitations, reviewed
  decision, next declared test and current evidence availability. Reuse one typed
  read-only projection for CLI text/JSON and the Research page, with stable links
  to original records. Preserve negative, interrupted, censored, missing and
  unassessed rows; reject tampered evidence without hiding its rejection reason.
  Do not infer promotion, rankings or live execution from historical findings.
  Acceptance: matching CLI/UI rows, duplicate-reference handling, relocated or
  missing evidence, invalid digests and offline browsing without creating a run
  store. The [curated ledger](docs/research/experiment-ledger.md) defines the reader
  need; no ledger command is shipped yet.

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

- [ ] **Add managed Runpod and Vast allocation adapters with explicit spending authorization.**
  Require explicit authorization before any paid action; support price, region,
  VRAM, CPU, RAM, and disk filters; retain quote expiry; use idempotent
  create/adopt tags; reconcile partial creation; surface interruption notices;
  distinguish stop-versus-delete and associated storage billing; and permit
  teardown only after durable-result verification. Acceptance requires provider
  fixtures for expired quotes, duplicate/adopted resources, partial-create
  recovery, interruption, billing-state transitions, and refusal without
  authorization.
- [ ] **Add additional SSH-provider acceptance and an optional lifecycle adapter.**
  Reuse the hosted endpoint-discovery, explicit-bootstrap, byte-transport,
  lifecycle-observation, synchronous relay, and lost-runtime-recovery contract;
  do not introduce another scheduler or change prepared-data identity.
  Acceptance must prove bounded transport, foreground execution,
  cancellation-intent delivery, strict host-key handling where SSH is used, and
  explicit recovery after lost runtime before advertising provider support.
- [ ] **Validate Drive relay across macOS, Linux, Windows and WSL2.** Exercise
  actual rclone Drive API put/readback, interrupted upload, immutable conflict,
  checkpoint collection with the executor unavailable, and full-state child
  resume using private per-host credentials and a dedicated test prefix.
  Cover case/path rules, permissions, filesystem capacity, and process deadlines.
  Test WSL2 with Linux-local roots separately from `/mnt/c` and native Windows;
  retain unavailable lanes. Colab CLI currently advertises Linux/macOS only:
  define a supported native-Windows execution/transport path before claiming
  Windows Colab support. Compare Drive-desktop/FUSE file relays only as weaker
  filesystem-visible copies; do not equate sync completion with API verification.
- [ ] **Implement a real PyTorch/XLA TPU engine/backend.** Cover XLA graph and
  compilation behavior, supported attention and objectives, data feeding,
  optimizer/RNG state codecs, same-backend full resume, cancellation and
  preemption, plus real TPU tests. Do not add an inert TPU enum member or claim
  CPU/CUDA behavior as TPU support.

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
