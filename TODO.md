# Capability status and open verification backlog

This is the living checklist for what SparseLab can execute, what has been smoke-tested, and what still needs evidence before stronger capability claims. A passing test or smoke run proves a code path, not useful model behavior. Completed acceptance records remain below as a historical evidence ledger; the current open work is organized by capabilities, not release phases. See the [capability roadmap](docs/research/roadmap.md).

## Current capability snapshot

| Capability | Implemented and exercised | What remains unproven |
|---|---|---|
| Training and runtime | CPU, MPS, and MLX/Metal training, evaluation, checkpoint recovery, inference, and independent-worker paths have acceptance evidence. | Fit, throughput, or reliability at larger workloads and on untested vendor hardware. |
| Learning and experiment workbench | Packaged lessons/catalog, explicit scaffolds, paired and factorial comparisons, static reports, and read-only dashboard paths exist. CPU/offline smoke, nano, and MPS/FineWeb-Edu micro studies are recorded. | A smoke is not a quality result; time-to-target, complete cost accounting, and broader independent task results remain open. |
| Token/byte Engram | Trainable tables are integrated with the model/training/checkpoint paths; byte-address smoke and controlled lexical-memory studies exist. | No consistent held-out benefit or general knowledge-transfer result; results vary by task and budget. |
| Portable byte Engram | Exported table identity can be verified and reused with a frozen table plus trainable recipient adapter. A two-case comparison is recorded. | That small comparison was negative; broader multi-seed and multi-recipient transfer remains unverified. |
| Semantic EngramPack | **implemented** verified exact retrieval from caller-supplied vectors, direct PyTorch `DenseLM` adapters, structured synthetic-world controls, and verified semantic query/mask allocation sidecars consumed by `PyTorchEngine`. | **still unproven** useful recipient behavior, natural-language query production, replacement-world transfer, and cross-width portability. No standard text-query pipeline. |
| Other architectures | PyTorch attention, MLA, local MoE, and memory paths have tests and integration runs; native MLX sparse attention has component measurements. | Most evidence is mechanism correctness or small, task-specific studies—not general quality or speed superiority. |
| Useful local models | A measured synthetic instruction starter and narrow capability cards provide honest failure examples. | No independently evaluated, useful real-world task model has been established. |

The latest offline workstation regression run was `uv run pytest -m 'not cuda and not rocm and not xpu and not network'` (**524 passed**). This validates software paths; it does not change any model-capability status above.

## Engram portability evidence ledger

Evidence labels are deliberately separate: **implemented** means a code path exists; **smoke-tested only** means the path ran without establishing useful behavior; **experimentally measured** means bounded behavioral measurements exist; **still unproven** means the claim lacks adequate evidence; **hardware-blocked** means the required target is unavailable; **deferred** means excluded until a named prerequisite is met. A mechanism can be implemented or smoke-tested while its behavioral claim remains still unproven.

| Question | Current evidence |
|---|---|
| Token-address portability | **implemented** address identity is tokenizer-dependent; a tokenizer match is an address invariant, not hidden-coordinate alignment. Cross-tokenizer behavior is **still unproven**. |
| Raw-byte address portability | **implemented** byte-address mechanisms; **experimentally measured** existing negative two-case portable-byte comparison. Broader address/behavior evidence is **still unproven**. |
| Immutable-artifact portability | Portable byte export/load and semantic-pack verification are **implemented**; cross-recipient frozen-artifact behavior is **still unproven**. |
| Dimensional/interface portability | Structured semantic K=32, V=8 differs from recipient width by design; interface compatibility is **implemented**, useful transfer is **still unproven**. |
| Behavioral portability | **still unproven** beyond the bounded local MiniLM pilot below; that pilot does not establish frozen-recipient adaptation. |
| Zero-shot portability | **still unproven** for independently prepared recipients. |
| Adapter-tuned portability | **still unproven** under frozen-backbone, update-isolated recipient adaptation. |
| Cross-hidden-width portability | **still unproven**; changing width/depth in DenseLM is not cross-architecture evidence. |
| Cross-scale portability | **still unproven**. |
| Cross-architecture portability | **still unproven**; initial DenseLM recipients all use dense attention. |
| Knowledge-swap portability | **still unproven** on neural recipients. Exact structured traversal is retrieval-only and cannot count as learned recipient behavior. |

**Bounded local MiniLM pilot — experimentally measured, limited scope:** widths 32/64, seeds 17/41/73, six rows; correct-pack answer accuracy 1.0, baseline/random/disabled 0.125, incomplete 0.5625, conflicting 0.0. Pack ID `2eae0ee0fe90b06bf06c31683b9e41e735ee80c73df49f7e1987c42310e7dc51`. Joint training ran 200 updates on 16 facts with precomputed MiniLM query vectors; held-out wording referred to training facts. This is not frozen-backbone adaptation, unseen-world swapping, micro-scale evidence, or ordinary language generation.

**Portability runner full smoke matrix — experimentally executed; useful behavior remains unproven:** campaign `/tmp/engram-portability-v1-final-matrix`, report `portability_evidence-a39168d1a7d8b69c.json`. All 120 declared arms ran at smoke scale for two updates across seeds 17/41/73: 42 token, 42 byte, 36 semantic, widths 32/64, all declared controls. The report contains 324 immutable checkpoint observations; 102 training audits passed and the 18 `frozen-only` arms used step-zero recipient checkpoints. No arm failed; all 120 were right-censored at the predeclared 0.95 development threshold. Development and final exact answer/path accuracy were 0.0 for every arm. The 468 saved A/B/A probes reproduced exact A answers and paths; a post-fix semantic-arm smoke also verified exact-comparison reporting. A semantic adapter arm recorded 520 retrieval hits, four conflicts, four temporal misses, and 16 unknowns while answer accuracy stayed 0.0. The protocol uses direct structured token/byte compiles and supplied structured semantic vectors, not source-model-trained artifacts or a natural-language query producer. This is workflow/retrieval evidence only, not useful transfer.

**Measured campaign resource envelope:** analytical maximum single-arm RAM estimate 69,222,400 bytes; total checkpoint-storage estimate 185,317,632 bytes (177,061,248 training, 8,256,384 preparation). Across the 120 saved arm receipts, observed median train-plus-checkpoint-observation wall time was 6.992303667 seconds per arm.

**Other measured negative/mixed evidence:** the two-case portable-byte comparison was negative. FineWeb-Edu micro has 18 MPS endpoints at 1,024 updates/262,144 targets; lexical-minus-none mean validation-loss deltas were −0.003625/ +0.006116/ −0.013402 for FFN widths 5120/2560/1280, with mixed per-seed signs and all stress cards zero. Neither result licenses a general portability or scaling claim.

**Separate unresolved mechanisms:** a reproducible text-query producer is **still unproven as a supported reproducible integration**. The local MiniLM producer is a prototype, not an absent mechanism; integration must bind encoder name, immutable revision, artifact digest, output dimension, normalization, representation-space ID, producer time/compute, and input provenance. Teacher-derived hidden-state compilation is **deferred** until ordinary semantic-pack portability has evidence.

Portability checklist (each item requires its own evidence; preparation or implementation does not close the behavioral question):

- [x] **implemented, smoke-tested only** shared-fact provenance, split/role permissions, deterministic worlds, verified structured packs, and lexical/token/byte representation manifests. The smoke-scale generator and replay ran; useful recipient behavior remains **still unproven**.
- [x] **implemented, smoke-tested only** independently initialized width-32/64 recipients with per-seed preparation checkpoints and no source-backbone tensor transfer. The full matrix preserves separate seed/width identities; adaptation quality remains **still unproven**.
- [x] **implemented, smoke-tested only** frozen-pack controls, exact optimizer membership, and byte-identical update isolation; all 102 trained-arm audits passed. The 18 `frozen-only` arms performed no updates.
- [x] **smoke-tested only** structured replacement-world trajectories and A→B→A attachment replay; all 468 saved probes reproduced the exact A answer/path. Recipient correctness remains **still unproven**.
- [x] **smoke-tested only** immutable step-zero/intermediate/final observations and censored threshold reporting; 324 observations cover the 120 arms, all right-censored at two updates.
- [x] **smoke-tested only** three-seed evidence across token, byte, semantic representations, and declared controls: 120 arms executed, with 0.0 final development and exact-answer/path accuracy. Useful behavior remains **still unproven**.
- [x] **implemented, smoke-tested only** reference-small configuration basis exists in `configs/instruction_100m.yaml`; analytical resource estimates and measured arm wall time are available, but no useful model claim is established.
- [x] **implemented** restored lifecycle guide with explicit A/B/C/D/E/F/Z model acceptance checkpoints.
- [ ] **still unproven** smallest supported reproducible text-query producer integration.
- [ ] **deferred** teacher-derived hidden-state compilation pending ordinary semantic portability evidence.


## Open capability work

- [ ] **still unproven** matched multi-seed lexical Engram behavior on a non-alias task with a training-only corpus, independent held-out cases, dense/no-memory controls, and collision/address diagnostics. Token and byte addressing remain separate.
- [ ] **still unproven** portable byte Engram beyond the recorded negative two-case comparison; test multiple facts/phrasings, recipients, disabled/random/frozen-only/trained-adapter controls, and immutable exported values.
- [ ] **still unproven** semantic EngramPack useful recipient behavior. Exact encoder-space compatibility requires key/value encoder identities, dimensions, normalization, and representation-space ID—not a shared tokenizer.
- [ ] **Allocation curve:** complete or explicitly bound the current 75-coordinate design; the recorded CPU smoke executed only one coordinate. Preserve per-task outcomes, all declared ownership/weight combinations, actual targets, and nonmonotonic results.
- [ ] **Architecture evidence:** choose one task and compare one mechanism at a time—MLA, sparse attention, MoE, placement, or FFN width—with matched data, tokenizer, seed, endpoint, and backend. Keep task scores, parameter/cache estimates, synchronized update timing, and end-to-end wall time separate.
- [ ] **Useful narrow model:** select one low-risk job, use permissioned non-synthetic examples, freeze an independent test set, and compare against a simple non-neural baseline. Predeclare acceptance criteria; include ambiguous/unknown cases, per-case errors, multiple seeds, and human review before describing the result as useful.
- [ ] **Learning and cost curves:** define a versioned observation policy for held-out task scores at committed token/checkpoint boundaries, thresholds, censored runs, wall/device time, and measured memory. Preserve update time separately from setup, evaluation, and checkpoint overhead; do not label a smoke timing as time-to-quality.

CUDA/HIP/ROCm/XPU, actual cross-host hardware, and distributed training retain their separate constraints below. They are not closed by CPU or Apple Silicon smoke results.

## Earlier acceptance evidence

## Local runtime

- [x] Execute isolated smoke and warmup staging pilots. **Astra:** real isolated subprocess pilots, fresh-run RNG/state parity, portable-file staging, offline frozen assets, and failure/proposal regressions passed in the 91-test integration sweep.
- [x] Implement capability-tested mixed precision and overflow recovery. **Astra:** real CPU BF16 full 4/128 versus part 2/64 plus resumed 4/128 matched model, optimizer, schedule, scaler, cursor and RNG bitwise with FP32 master weights (`/tmp/sparselab-mixed-supervision-cST7Ou/acceptance.json`); both overflow/retry regressions passed. Accelerator modes remain gated by actual disposable probes, not declarations.
- [x] Complete memory monitoring, calibration, and synchronized timing. **Astra:** 79 focused runtime/memory/staging/store/conversation regressions passed; actual CPU RSS and MLX active/peak/cache records preserve missing-counter distinctions. Sampled MPS peaks never lower estimates or certify fit. [Evidence](artifacts/acceptance/memory_optimizer_2026_09_22.json).
- [x] Complete explicit memory-policy search and proposal output. **Astra:** actual CLI low-memory/balanced proposals preserve effective batch and source bytes; loadable YAML/report hashes and overwrite refusal verified. Fault regressions preserve uncertainty margins and roll back failed publication. [Evidence](artifacts/acceptance/resource_policy_2026_09_22.json).
- [x] Make recomputation diagnostics safe and memory-bounded. **Astra:** dense/MoE/combined gradient and update regressions, nonempty-chunk accounting, detached masked diagnostics, native MLX recomputation, and actual MPS offload composition passed. [Paired measurements](artifacts/acceptance/offload_2026_09_22.json).
- [x] Complete lineage-best recovery and interruption acceptance. **Astra:** actual SIGTERM commits step 4,642/148,544 targets; corrupt explicit resume creates no child; recovery selects verified step 4,600 and commits child 4,601 without changing selected parent files. Older-parent lineage excludes future generations. [Recovery](artifacts/acceptance/signal_recovery_2026_09_22.json), [lineage](artifacts/acceptance/lineage_bound_2026_09_22.json).

Checkpoint/recomputation contracts precede runtime integration; measured staging precedes worker scheduling. Existing canonical identities, immutable data verification, PyTorch-native v2 / MLX-native v1 checkpoint checks, writer leases, shape-only inspection, CPU/byte-data continuation, Adafactor continuation, and local MPS resume remain regression requirements rather than work to replace.

## Persistence and interface

- [x] Implement transactional metrics migration and durable event records. **Astra:** migration/history preservation, transactional rollback, contiguous/idempotent replication, strict 64-KiB envelopes, exact batch framing, and schema-corruption regressions passed; real staged training persisted the records.
- [x] Complete the Runtime, Checkpoints, Memory, and Stages dashboard. **Astra:** populated CPU/MPS/MLX browser checks, mixed-optimizer comparison, live updates, persistent selection, full parent identity, 31 valid/1 corrupt generation, unchanged SQLite data version, stale-read recovery, UTC age, and independent metric scales. [Evidence/screenshots](artifacts/acceptance/dashboard_2026_09_22.json).
- [x] Complete educational contracts and installed-wheel acceptance. **Astra:** core-only wheel outside checkout, 13 packaged guides, browser Learn, offline native verification without MLX, explicit SDK-required execution failure, and CPU inference/evaluation after owned external inputs were removed. [Wheel evidence](artifacts/acceptance/core_wheel_2026_09_22.json), [final frozen wheel](artifacts/acceptance/single_host_gate_2026_09_22.json).

## Training and inference

- [x] Implement verified KV-cached autoregressive generation. **Astra:** focused generation regressions cover cached/reference parity and bounded-window behavior; actual API/CLI CPU generation agrees. Native MLX remains explicitly full-prefix. [CLI evidence](artifacts/acceptance/host_cli_2026_09_22.json).
- [x] Implement validated legacy-checkpoint and pretrained-weight import. **Astra:** 35 focused import/checkpoint/generation/recomputation tests pass, including independent Llama reference parity. Actual historical 50M import preserves exact logits and original files, rejects full resume, and commits a promoted update. Legacy inspect/verify reports weights-only scope. [Evidence](artifacts/acceptance/weight_import_2026_09_22.json).
- [x] Curate licensed domain conversations and audit semantic leakage. **Astra:** CPython 3.14.7 independently confirms all 44 labels; normalized path/operation/argument identities have zero cross-split overlap; all 26 card cases match their source envelopes and all 28 frozen domain file identities verify. Acquisition is six selected training cases, not the entire split. [Evidence](artifacts/acceptance/domain_curation_2026_09_22.json).
- [x] Implement assistant-only loss and versioned tool conversations. **Astra:** actual CPU and MLX commands commit exactly 13 supervised targets and evaluate 33 targets after excluding 19 masked blocks; held-out evidence binds the supervision digest. Mask-tamper regression and conversation/packing suites pass. [Evidence](artifacts/acceptance/assistant_only_2026_09_22.json).

## Capability experiments

- [x] Train context overrides and evaluate untouched assignments. **Astra:** all 24 preregistered endpoints and 72 fixed-order cards verify; all untouched outcomes remain 0/8. Independently rescored the archived responses without regenerating them. [Acceptance](artifacts/acceptance/scientific_studies_2026_09_22.json).
- [x] Run preregistered multi-seed, multi-budget architecture comparisons. **Astra:** seeds 17/41/73, exact 24,576/49,152 targets and 199/398 updates; all 18 paired deltas independently checked, no endpoint/seed selection. [Results](docs/context-engram-study.md#execution-results--2026-09-22).
- [x] Measure Engram collisions, addressing, and matched-parameter alternatives. **Astra:** four actual address streams recomputed over all 27 blocks; repeated-key reuse, excess-key aliases and unordered collision pairs distinguished. Dense-total is 160 parameters larger; diagnostic variants remain confounded. [Acceptance](artifacts/acceptance/scientific_studies_2026_09_22.json).
- [x] Measure domain reliability and retention after adaptation. **Astra:** six exact 30,720-target endpoints, 18 cards, verified promotion lineage and six independently recomputed 30-block/2,880-target retention measures. Acquisition improves; development stays 0/8, adapted frozen total is 1/36, and retention worsens sharply. Original pre-execution infrastructure retries remain documented. [Results](docs/path-domain-corpus.md#2026-09-22-execution-record).

Keep the previous failed override controls and negative Engram comparison. Do not select favorable seeds or silently turn development cases into an untouched test set.

## Backend and performance

- [x] Complete MLX checkpoint, inference, evidence, and recomputation parity. **Astra:** real Metal full versus interrupted/resumed states match bitwise; canonical PyTorch logits agree within 5.97e-7, generation/chat/evaluation preserve RNG, and cross-engine promotion commits a fresh update. Core-only offline verification is separate from SDK-required execution. [Evidence](artifacts/acceptance/host_cli_2026_09_22.json).
- [x] Complete activation-offload probes and measured transfer comparisons. **Astra:** actual paired MPS baseline/offload/recomputation CLI runs, synchronized transfer measurements, parameter parity, host-budget rejection before run creation, and saved-storage lifetime/version regressions pass. Unified-memory capacity gain remains zero; discrete hardware claims remain blocked. [Evidence](artifacts/acceptance/offload_2026_09_22.json).
- [x] Complete Adafactor accounting, education, and integration coverage. **Astra:** actual state is 3,436 bytes versus AdamW 86,444 bytes for the same tiny architecture, exactly matching factor/alias-aware estimates; Adafactor full/resumed states match bitwise. Browser Learn identifies relative learning-rate cap semantics. [Evidence](artifacts/acceptance/memory_optimizer_2026_09_22.json).
- [x] Correct configured optimizer inventory across inference reports. **Astra:** reproduced missing AdamW step scalars and Adafactor inference mislabeled as AdamW-sized state; unified configured reports with canonical accounting. Real consumer values now 86,444/3,436 bytes; 35 accounting/inference/planner regressions pass. [Evidence](artifacts/acceptance/configured_accounting_2026_09_22.json).
- [ ] Implement and benchmark native CUDA sparse attention. **Blocked:** no reachable CUDA target.
- [ ] Implement and benchmark native HIP sparse attention. **Blocked:** no reachable ROCm target.
- [x] Implement and benchmark native MLX sparse attention. Astra: custom Metal forward/dQ/dK-dV kernels; 10 native regressions pass, including sparse recomputation and wider-head gradients; matched Metal measurements at 32/128/512 tokens. [Evidence and limits](docs/sparse-attention.md#measured-metal-attention-component).
- [ ] Validate ROCm acceptance on actual target hardware. **Blocked:** no provisioned AMD host.
- [ ] Validate XPU acceptance on actual target hardware. **Blocked:** no provisioned Intel host.

The current host is Apple Silicon. No remote hosts are registered with the harness. CUDA/HIP/XPU execution and kernel-performance claims require their actual machines and vendor-provisioned environments; CPU or Apple execution cannot close those gates.

## Independent orchestration

- [x] Implement versioned local and SSH worker contracts. **Astra:** strict framing/version/identity and SSH quoting regressions pass; installed local CPU and actual MLX/Metal workers execute and ingest. Genuine source-mismatch worker stays queued without an executor. Actual SSH hardware execution remains unclaimed. [Gate evidence](artifacts/acceptance/independent_workers_2026_09_23.json).
- [x] Implement physical-device leases and durable launch receipts. **Astra:** inherited physical-device exclusion regressions pass; three concurrent real logical CPU workers have distinct process/start identities, and replaying launch twice preserves the original PID/token/attempt. [Gate evidence](artifacts/acceptance/independent_workers_2026_09_23.json).
- [x] Implement safe cancellation and verified artifact ingestion. **Astra:** real cancellation commits step 26,943; explicit child reaches 32,768 updates / 2,097,152 targets and matches uninterrupted model, optimizer, RNG, cursor, scaler, schedule and counters bitwise. Offline promotion verifies fresh optimizer/RNG and unchanged parent files; native MLX verifies without its SDK. [Gate evidence](artifacts/acceptance/independent_workers_2026_09_23.json).
- [x] Implement the independent-experiment queue and explicit matrices. **Astra:** dry-run has no store/input side effects; actual installed CLI matrix enqueues three unique coordinates and all finish with verified ingestion at exactly 2 updates / 64 targets. Unsupported engine/capacity and a genuinely different source remain queued without execution. [Gate evidence](artifacts/acceptance/independent_workers_2026_09_23.json).
- [x] Verify independent-worker recovery and concurrent execution. **Astra:** all three CPU workers advance while the stopped controller's counters remain fixed; reconnect yields contiguous deduplicated records and zero echo outbox. Identity-checked SIGKILL leaves the old attempt UNKNOWN; only an explicit new child resumes full state to 32,768 updates / 1,048,576 targets. [Gate evidence](artifacts/acceptance/independent_workers_2026_09_23.json).
- [ ] Verify actual Mac, AMD, and Intel concurrency. **Blocked:** requires real overlapping runs and disconnect/recovery evidence from all three target hosts.

Workers execute whole independent experiments. No shared optimizer, distributed backward, or cross-host SQLite WAL. Local logical workers must truthfully report CPU rather than pretend to be AMD/Intel/Apple accelerators.

## Astra acceptance

- [x] Verify every completed slice with Astra review. **Astra:** independently verified subagent reports, real artifact identities/counters and recovery behavior; reproduced and fixed final lifecycle/codec findings. At that acceptance revision, Ruff checks and 335 tests passed. Later regression results are summarized above. Earlier scientific and single-host gate evidence remains intact; hardware-dependent claims stay blocked.
- [x] Complete the integrated single-host runtime acceptance gate. **Astra:** accepted after actual CPU FP32/BF16/Adafactor and native MLX continuation, MPS 20/640 full/resumed comparison and CPU-weight promotion, signal/corruption recovery, isolated warmup, dashboard, core-only wheel, and integrity regressions. The immutable execution wheel and hashed evidence index are retained. [Gate record](artifacts/acceptance/single_host_gate_2026_09_22.json).
- [x] Complete end-to-end independent-worker acceptance. **Astra:** final installed core-only wheel, concurrent CPU runs, actual MLX worker, controller disconnect/replay, cancellation, explicit crash recovery, offline promotion/evaluation/inference, real CLI matrix, admission rejection and read-only dashboard checks passed. Actual browser WebGL curve pixels were captured and visually inspected; standard screenshots/runtime-table rasterization stalled, so runtime table values were verified with Streamlit's real app harness. [Gate record](artifacts/acceptance/independent_workers_2026_09_23.json), [rendered curves](artifacts/acceptance/workers_training_2026_09_23.svg).
- [x] Publish verified changes and accurately record blocked gates. **Astra:** implementation, curated data, study results, acceptance records, rendered UI evidence and both frozen execution wheels were published to `origin/main` in [e8a60d8](https://github.com/supernovae/tiny-sparse-lab/commit/e8a60d8). At that acceptance point, 31 locally actionable tasks were complete and five actual-hardware gates remained blocked. The capability work now tracked at the top of this file extends beyond that historical acceptance.

Evidence must include actual CLI scenarios, checkpoint/optimizer/RNG comparisons, negative integrity and recovery cases, populated browser verification, installed-wheel behavior, and real hardware measurements where claimed. Shape-only inspection is not a large-model training result. Sampled MPS peaks are lower bounds, not native allocator high-water measurements.

## Excluded from this execution

- [ ] Distributed MoE expert sharding, all-to-all dispatch, and multi-rank GPU training.
- [ ] Homogeneous distributed data-parallel training.

These remain future research items, explicitly outside the user's current non-distributed scope.
