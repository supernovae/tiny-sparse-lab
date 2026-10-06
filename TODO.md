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

- [ ] **P1 — Stage explicitly bound existing inputs natively.** The Python
  `staging.stage(..., prepared_inputs=...)` path exists, but public `stage` does
  not expose it. Add a typed adapter taking RunConfig, an authenticated existing
  stage-input bundle, selected runtime, through-level and exclusive output;
  emit the normal stage/pilot receipts. Wrong source/config, tampered inventory,
  unsafe links and unavailable runtime must fail before optimizer execution.
  It must not prepare/download/repack, grant runtime/source drift or silently
  fall back to fresh inputs. This is not a shipped CLI option. A continuation
  requiring source authorization still needs that independent review; this
  adapter is not a substitute for that decision.
- [ ] **P2 — Expose verified declaration/config handoffs.** Where Phase `set`
  and native scaffolds do not suffice, add CLI derivation over the existing
  typed loaders and dotted-path compiler, preserving path anchoring when the
  output moves. Exact resolved-cell export is already available; remaining
  handoffs must avoid Python and reinterpretation of SHA fields. Write new files only,
  record origin/delta, reject unknown fields and scientific incompatibility.
  Existing checkpoint selectors already serve declared chains; add standalone
  immutable selection only if a native iteration still needs pointer
  parsing. Never select an unverified "latest" directory by sorting filenames.
- [ ] **P2 — Exercise a complete declared iteration demo.** Using native
  direct-input binding, add a copyable ExperimentPlan/Campaign
  example with one baseline, a checkpoint-bound exposure child, fixed heldout
  suite/descriptive generation panel, a separately labeled fresh one-field
  contrast and optional seed replication. Share the topology with a small offline
  acceptance fixture, including interrupted reconciliation and unchanged parent
  hashes. Use a pinned teaching dataset such as TinyStories and record acceptance separately: declaration
  digests, run/ingestion states, checkpoint lineage, counters, evaluation/panel
  references and costs. Keep outputs external and make CLI text/JSON follow the
  same route. No embedded Python, duplicate implicit baseline, automatic promotion
  or claim that offline smoke establishes real-data quality. The current direct
  teaching walkthrough is not this declarative acceptance gate.

### Native diagnostic interfaces

- [ ] **P3 — Expose supplied-vector semantic probes natively.** Wrap the existing
  verified semantic retriever/adapter with a typed declaration: pack identity,
  canonical query tensors, encoder identity, masks, attachment sites and explicit
  initialized or checkpoint-bound model selection. Emit verified pack/query/model
  identities and per-query retrieval/adapter traces. Reject encoder, shape,
  inventory and checkpoint mismatches before inference; preserve missing,
  conflicting and time-bounded outcomes. No arbitrary Python callbacks or
  implicit text encoder. The [semantic lesson](docs/research/semantic-memory.md)
  currently scaffolds/validates packs but relies on an API demonstration for
  these queries. A natural-language encoder remains a separate research question.
- [ ] **P2 — Expose the bounded preparation benchmark through the lab.** Input:
  an exclusive task workspace, declared generated-corpus sizes, seed, tokenizer
  batch bounds and host thread limits. Reuse the existing preparation path;
  emit phase timing, logical byte/record counts, sampled-memory scope and exact
  prepared identity comparisons. Reject unsafe/reused output, invalid bounds
  and insufficient storage; never download or start model training. Preserve
  null counters and keep performance observations outside scientific identities.
  The historical `benchmarks/preparation_benchmark.py` harness is not a native
  command; the [retained observations](docs/runtime.md#offline-performance-evidence)
  must not be silently relabeled as a new benchmark run.
- [ ] **P2 — Expose continuation/source overlap through the lab.** Wrap the
  existing descriptive diagnostic with typed CLI input: a retained continuation,
  explicit source IDs/passages, n-gram size and bounded edit-distance limit.
  Emit the normal source hashes, normalization identity, raw overlap measures
  and null/unavailable edit similarity. Bind checkpoint/generation identity when
  the input comes from a run. Reject missing sources, duplicate IDs, invalid
  bounds and tampered retained inputs; never infer a copyright threshold,
  eligibility, memorization verdict or publication approval. No Python recipe
  should be required. See [current boundary](docs/memorization.md).
- [ ] **P2 — Expose opt-in phase observations natively.** Add a typed CLI/DSL
  adapter for the existing `BottleneckObserver` on preparation, staging and
  orchestration. Input: the normal declaration/config, selected runtime and
  explicit observation destination. Output: versioned operational phase records
  with measured counters, sampling scope and unavailable reasons, outside
  scientific inventories. Reject unsafe output paths and malformed options;
  missing device probes must remain null, not zero or invented utilization.
  Verify scientific digests remain unchanged and observation failures follow the
  existing non-blocking policy. See [boundaries](docs/capacity-aware-execution.md).

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
