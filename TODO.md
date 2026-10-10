# Implementation backlog

Only missing or defective code belongs here, plus the lab-velocity pillars below
(items marked **Directional** are experiment direction rather than single code tasks). Scientific milestones and execution
status live in [the research roadmap](docs/research/roadmap.md) and
[research records](experiments/research/). A passing fixture does not close a
scientific gate. Each item needs code, focused regression coverage and public
documentation; move it to a **Code task** issue when assigned. Reuse native
interfaces, and preserve declaration and artifact identities.

## Lab velocity: break the ceremony loop

Goal: an idea expressed as a YAML delta reaches a scored comparison against a
known baseline the same day, with full provenance reserved for release runs.
These items take priority over new lifecycle, admission or orchestration code.

### Pillars

- [x] **P0 — Fast loop (`lab mode`).** *Done: `sparselab try` / `report`,
  [lab mode](docs/lab-mode.md); a CPU smoke delta reaches its report in ~15 s.* One command takes a YAML change and a
  baseline and returns a scored comparison (e.g. `sparselab try <delta.yaml>
  --vs <baseline>`). Lab mode skips plan locks, approvals, admission reviews and
  campaign reconciliation; it still records config, seed, code revision and
  data identity in one compact run record. Release mode keeps today's full
  provenance. Target: tiny-model smoke delta to report in under 15 minutes on CPU.
- [x] **P0 — Fast-fail probe battery.** *Done: `sparselab probe` and the
  fast tier inside every `try`, [probe battery](docs/probe-battery.md);
  versioned suite digest, dev/held-out item splits with an overfit guard,
  tiered fast-fail, verdict + next action, optional lm-eval backend.* A fixed, versioned suite that runs in
  minutes on any checkpoint: held-out perplexity delta vs. baseline, top-token
  agreement with baseline, fact recall with reworded (held-out) prompts, simple
  needle-in-context retrieval. Cheap metrics filter most ideas before any longer
  run. Probes are screening signals: keep a separate, untouched final evaluation
  because repeated selection on held-out probes turns them into development data.
- [ ] **Probe battery follow-ups.** More dashboard charts (per-probe trends
  over tries, needle accuracy by length across checkpoints, calibration
  curves); broader benchmarks beyond the four lm-eval tasks at `limit=50`
  (larger limits, more task families, a held-back final evaluation set that
  agents never select on). Reference follow-ups: score references on the
  tokenizer-independent standard-tier probes too (needs a text-level fact
  recall/needle path), time references on one fixed device so they appear on
  the latency axis, more checkpoints along the Pythia trajectory, and runs
  outside `WORK_DIR/lab` (plain `sparselab train` runs) on the Pareto view.
- [x] **P1 — Known reference points.** *Done: `sparselab probe ref:NAME --tier
  full` scores pinned Pythia-70M/160M-deduped and SmolLM2-135M/360M (immutable
  commits, safe snapshots, weights digest) through the same lm-eval adapter;
  their sealed results ship in `probes/reference_results/` so CI and the
  dashboard never download a model; `sparselab compare RESULT --references`
  places any result on that curve with paired SEs, only within one benchmark
  group (otherwise NOT COMPARABLE / MISSING EVIDENCE),
  [docs](docs/probe-battery.md#reference-models-and-sparselab-compare).*
  Import SmolLM2 and Pythia checkpoints in
  the 70M–360M range (and reuse lm-evaluation-harness tasks where possible) so
  every result sits on a known curve instead of only comparing to our own runs.
- [x] **P1 — Pareto view in the dashboard.** *Done: the Probes page plots
  held-out loss or lm-eval accuracy against resident or active parameters,
  resident or active weight bytes (memory), training tokens and scoring
  latency, for every scored try arm (with or without probes), probed
  checkpoints and the reference models, per comparison group, with the
  frontier and a learner explainer.* Plot quality against tokens,
  memory, latency and parameter count (resident vs. active), so trade-offs are
  visible instead of a single number.
- [ ] **Directional — Memory-offload showcase experiment.** At fixed compute,
  dense vs. dense + Engram/memory; then swap the facts held in memory and check
  that recall follows the swap (portability). This is an experiment program
  rather than a single code task; code needs it exposes get filed here.
- [ ] **Directional — Agent iteration loop.** *The probe verdict
  (`action`, `next_tier`) and held-out guard now exist for it.* An agent reads `evidence`/`triage`
  output, proposes the next single-variable delta and re-runs through lab mode.
  Keep a held-out probe split the agent never optimizes against, and persist a
  memory of prior attempts to avoid repeats.

### Tactical cleanups

- [x] **Rewrite AGENTS.md for lab mode by default.** Agents currently generate
  much of the ceremony (proposal, binding and stop documents). Default agent
  runs to lab mode; require proposals only for release runs or paid compute
  above an explicit budget.
- [x] **Shrink the CLI surface.** *Done for help output: `try`/`probe`/`report`/
  `compare` and the fast path first, release commands grouped last. No commands
  were removed.* ~100 subcommands today. Put the fast path
  (`try`, `probe`, `compare`, `report`) up front and move lifecycle/campaign
  commands under an `advanced`/`release` group in help output.
- [ ] **Matrix-ify configs.** Replace hand-expanded seed/budget variants in
  `configs/` (e.g. the `context_study_*` files) with one matrix declaration each.
- [ ] **Archive the kernel-memory-lab paper trail.** Move CARD proposals and
  per-step result notes under `experiments/research/history/`, leaving one
  STATUS page with current decisions and links.
- [ ] **Unblock a real 100M run on an existing dataset.** Train the 100M target
  on an already-available corpus (e.g. a FineWeb-Edu sample) in lab mode rather
  than waiting on retained-corpus admission; treat retained-corpus work as a
  release-mode track.
- [ ] **Dead-code and legacy sweep.** Run coverage plus a dead-code scan
  (e.g. vulture) after the recent cleanup and remove experiment-specific code
  paths that the fast loop doesn't use.
- [x] **Loop-time CI check.** *Done: the `lab-loop` CI job runs
  `tests/test_lab_mode.py::test_cpu_smoke_loop_yaml_to_report_within_budget`
  on CPU for every PR and main (900 s budget).* Add a CI job that times the CPU
  smoke path from YAML to report and fails if it regresses past the target.
- [x] **Borrow before building.** *Now a rule in AGENTS.md §2.* For new training, eval or quantization needs,
  check nanoGPT/modded-nanogpt, litgpt, lm-evaluation-harness and llm-compressor
  (GPTQ/AWQ/SmoothQuant baselines) first and wrap them rather than reimplementing.
- [ ] **Optional — served-identity export.** Emit a samesies-style signed
  manifest for exported checkpoints so lab provenance carries through to serving.

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
