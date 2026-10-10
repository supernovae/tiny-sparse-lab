# Implementation backlog

Only missing or defective code belongs here. Scientific milestones and execution
status live in [the research roadmap](docs/research/roadmap.md) and
[research records](experiments/research/). A passing fixture does not close a
scientific gate. Each item needs code, focused regression coverage and public
documentation; move it to a **Code task** issue when assigned. Reuse native
interfaces, and preserve declaration and artifact identities.

## Active code need

- [ ] **P2 — Shared read-only experiment-ledger projection.** Extend existing
  [lifecycle readers](src/sparselab/research/lifecycle.py) and dashboard reporting
  with one typed projection of questions, deltas/controls, protocol/checkpoint
  identities, results/limitations, reviewed decisions, next tests and evidence
  availability. Inputs are lifecycle declarations, verified report roots,
  evidence references and an optional run store. CLI text/JSON and Research UI
  must agree, preserve negative/interrupted/missing/unassessed rows, handle
  duplicate references and relocation, and expose invalid-digest rejection.
  Offline browsing must not create a run store. The
  [curated ledger](docs/research/experiment-ledger.md) describes the reader need;
  there is no public ledger command yet.

## Conditional code needs

Activate only for a concrete workload; these do not block the current retained
corpus preparation milestone.

- [ ] **Source-bound held-out-item CLI adapter.** Expose
  [`freeze_card03_items`](src/sparselab/evaluation/kml_card03_items.py) through the
  existing evaluation CLI. Bind a cold-verifiable release, frozen family
  inventory, reviewed item draft and fresh output directory; report immutable
  item/chunk/denominator digests. Reject changed release, train-family leakage,
  unbound offsets/digests, unreviewed or missing items, incomplete denominators,
  malformed controls and existing output. Reusing already frozen items needs no
  new authoring engine or publication step.
- [ ] **Signed warm source-snapshot proof reuse.** Review the semantic import
  closure and pinned exclusions: the source-snapshot authority scan currently
  reaches the dynamic import in `recovery.implementation_replay` and declines a
  signed proof. Preserve path-security and proof invalidation checks. Acceptance
  requires authenticated warm hits plus cold fallback for changed authority or
  inputs. Existing cold verification remains the accepted route.
- [ ] **Read-only tokenizer-artifact CLI verification.** Add a native adapter
  that binds the configured path to its manifest, origin and vocabulary and
  reports digest/special-token IDs without fitting. Reject missing/tampered
  outputs, underfilled vocabulary and changed config/source. Existing prepared
  bundle verification already provides a cold path; transport budget reporting
  is implemented and is not part of this gap.
- [ ] **Acquisition receipt resource counters.** Extend bounded acquisition
  receipts with measured per-shard expanded bytes and peak task-owned staging
  bytes/inodes. Preserve legacy hashes when fields are absent. Fail closed for
  required unavailable readings; retain interrupted-transfer counters. Test exact
  caps, overlong lines and resume with mocked HTTP. Never infer or backfill
  historical readings or reacquire only to fill optional fields.
- [ ] **Declarative verified snapshot inheritance.** Expose existing Python
  ancestry/reuse support in recovery declarations and CLI. Bind parent evidence
  and selected source IDs; preserve snapshot identities without reacquisition.
  Reject tampered parents and incompatible source declarations using generic
  ancestry fixtures.
- [ ] **Operational spot-safety policy.** Add a separately configured policy
  based on observed checkpoint-write time, interruption notice, capacity and
  measured restart cost. Record chosen operational cadence without changing
  scientific settings. Cover unavailable observations and insufficient capacity;
  existing checkpoint verification, retention and child resume are foundations.
- [ ] **Managed Runpod/Vast allocation adapters.** Add explicit spending
  authorization, resource/region/price filters, expiring quotes, idempotent
  create/adopt tags and partial-create reconciliation. Distinguish stop/delete
  and storage billing; gate teardown on durable-result verification. Use provider
  fixtures for expired quotes, duplicate resources, interruptions, billing
  transitions and refusal without authorization.
- [ ] **Additional SSH-provider lifecycle adapter, when needed.** Reuse the
  existing endpoint discovery, explicit bootstrap, bounded transport, lifecycle
  observation, synchronous relay and lost-runtime recovery contract. Cover
  foreground execution, cancellation delivery, strict host keys and recovery
  after lost runtime before advertising support; do not add another scheduler.
- [ ] **Native CUDA sparse attention.** Implement when a planned NVIDIA workload
  needs it. Preserve reference semantics; require hardware-gated correctness and
  component benchmarks. Component speed is not end-to-end or quality evidence.
- [ ] **Real PyTorch/XLA TPU backend.** Implement graph/compilation behavior,
  supported attention/objectives, feeding, optimizer/RNG state codecs, same-backend
  full resume, cancellation and preemption with real TPU tests. An enum or CPU
  fixture does not establish TPU support.

## Execution and qualification are tracked elsewhere

Completed native Kernel Memory Lab work is linked from
[STATUS](experiments/research/kernel-memory-lab/STATUS.md), including C05-T1–T11,
C05-C1–C4 and C05-Q2. Existing snapshot/preparation, typed Campaign binding,
`iteration check`, `experiment bind-inputs`, `experiment export-config`, corpus
budget reporting and the shared rclone relay are shipped capabilities, not open
implementation tasks. Their current public usage belongs in
[iteration](docs/iteration.md), [datasets](docs/datasets.md) and
[workers](docs/workers.md).

Actual S3-compatible relay transfers, Drive OAuth enrollment/credential lifecycle,
Drive cross-platform transfers, spot-loss recovery, live ROCm and hosted-platform
acceptance are optional qualification work. Keep results and unavailable lanes in
operational/research acceptance records; add a code item here only when a concrete
missing behavior is exposed. Preserve narrow credentials, verified content
identity, immutable conflicts and explicit paid-resource authorization during any
separately authorized qualification. Local fixture success does not qualify a
provider, platform or model run.

The retained-only corpus milestone still needs real reviewed admission, protected
lineage, measured supply, exact mixture and cold bundle evidence under a fresh
bounded allocation. All stopped B11–B16 evidence remains historical, including
B16's unverified outer receipt-based shutdown. B17 has no ledger and consumed no
production allocation; its pinned authority does not transfer to a refactored
head. No runtime work is authorized by this backlog.

Distributed training remains deferred under the
[single-host decision](docs/decisions/0015-single-host-extension-boundaries.md).
A future proposal needs optimizer ownership, partitioning, checkpoint identity
and expert placement before implementation.
