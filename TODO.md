# Implementation backlog

Only open work belongs here: priorities first, then experiments, then parked
items. Done items are deleted, not checked off. Scientific milestones and
execution status live in [the research roadmap](docs/research/roadmap.md) and
[research records](experiments/research/). A passing fixture does not close a
scientific gate. Each code item needs code, focused regression coverage and
public documentation; move it to a **Code task** issue when assigned. Reuse
native interfaces, and preserve declaration and artifact identities.

## Priorities

Land in this order, one small PR each.

1. **PR2 — Test health.**
   - Known failures on clean `main`: `test_evaluation_suite.py` signed
     replay/reuse (2), `test_attempt_contract.py::test_outer_receipts_crossing_cap_cannot_return_success`,
     `test_workers_cli.py::test_composed_run_observes_reconnection_and_artifact_completion`
     and `test_operational_monitor_safety.py` (4).
   - About 65 more failures in a full local run (verifier-authority exclusion
     signatures such as `('data.packing', 'resource_envelope')`, snapshot
     authority, verification reuse, story-artifact reuse, campaign,
     operational monitor, and workspace hygiene on
     `docs/research/sample-report.md` size). CI's PR jobs run only an explicit
     node list, so these went unnoticed.
   - The full CPU suite takes over 25 minutes. Split it into a fast tier
     (every PR) and a full tier (nightly or manual).
   - The "inadequate measured RAM after reserve" preflight trips under
     parallel pytest workers (`-n 6` on a 16 GB box). Scale the reserve per
     worker or mark those tests serial.
2. **PR3 — Evaluation integrity.**
   - A held-back final evaluation set that agents never select on.
   - A text-level needle (fixed character lengths) so needle also runs on
     references.
   - Fact recall: every tiny lab model picks the first candidate for every
     question (`red`, `panda`); find the bias and pin it with a test.
   - Fold `_reference_point` into `resolve_point`.
   - The probe verdict prints "Promising: escalate" on a try whose
     comparison is `NOT_COMPARABLE`; a non-comparable try must not escalate.
   - The per-try held-out-loss error bar (window-clustered SE) is about 10x
     smaller than seed-to-seed spread at tiny budgets. Report a seed floor or
     require paired seeds before a verdict claims a win.
3. **PR4 — References expansion.** Time the references on one fixed device so
   they appear on the latency axis; add more Pythia trajectory checkpoints;
   run all references at `--lm-eval-limit 500` plus winogrande and
   arc_challenge (the mechanism exists, the expensive run does not).
4. **PR5 — MLX and semantic packs in probe/explorer.** Both refuse MLX
   checkpoints and attached semantic packs with a reason today.
5. **PR6 — Cleanups.**
   - Matrix-ify configs: replace hand-expanded seed/budget variants (e.g.
     `context_study_*`) with one matrix declaration each.
   - Archive the kernel-memory-lab paper trail: move CARD proposals and
     per-step result notes under `experiments/research/history/`, leaving one
     STATUS page with current decisions and links.
   - Dead-code and legacy sweep with coverage plus vulture; remove
     experiment-specific paths the fast loop does not use.

## Experiments

Run through lab mode ([first model](docs/first-model.md),
[lab mode](docs/lab-mode.md)). Code needs they expose get filed above.

- **Real 100M run on an existing dataset.** Train the 100M target on an
  already-available corpus (e.g. a FineWeb-Edu sample) in lab mode, without
  waiting on retained-corpus admission; retained-corpus work stays a
  release-mode track. The next research target is **100M training targets**,
  subject to the [offline readiness assessment](docs/research/100m-readiness.md).
- **Memory-offload showcase.** At fixed compute, dense vs. dense +
  Engram/memory; then swap the facts held in memory and check that recall
  follows the swap (portability). Re-run the hash-affected baselines first
  ([first model](docs/first-model.md#re-baseline-the-hash-affected-configs)).
- **Agent iteration loop.** The probe verdict (`action`, `next_tier`) and
  held-out guard exist. An agent reads `evidence`/`triage` output, proposes
  the next single-variable delta and re-runs through lab mode. Keep a held-out
  probe split the agent never optimizes against, and persist a memory of prior
  attempts to avoid repeats.

## Parked (low priority)

- **GPU instrumentation.** WSL2 on the 5900x desktop, self-hosted
  CUDA/ROCm/MLX CI runners, Colab.
- **Served-identity export.** Emit a samesies-style signed manifest for
  exported checkpoints so lab provenance carries through to serving.
- The P2 experiment-ledger projection and the conditional code needs below.

## Parked: experiment-ledger projection

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

## Parked: conditional code needs

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
