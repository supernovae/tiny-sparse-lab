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

## Completed conditional code

The first six conditional needs from merged PR #56 are implemented with
[offline evidence and explicit qualification limits](docs/refactors/2026-10-conditional-lifecycle-code.md).
These checkmarks close code scope, not production admission or runtime qualification.

- [x] **Source-bound held-out-item CLI adapter.** Native `evaluation freeze-items`.
- [x] **Signed warm source-snapshot proof reuse.** Authenticated hits and cold invalidation.
- [x] **Read-only tokenizer-artifact CLI verification.** Native `tokenizer verify`.
- [x] **Acquisition receipt resource counters.** Measured bounded-transfer receipts and retained interruption readings.
- [x] **Declarative verified snapshot inheritance.** Recovery declarations, verified native imports and generic ancestry checks.
- [x] **Operational spot-safety policy.** Separate direct-training policy and read-only planning; live interruption/resume qualification remains open.

The next research target is **100M training targets**, subject to the
[offline readiness assessment](docs/research/100m-readiness.md), retained-corpus
verification and a new exact reviewed allocation. Historical 50M packets below
retain their original identities and grant no 100M authority.

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
