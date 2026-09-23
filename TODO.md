# Completion backlog — independent experiments, no distributed training

This is the active execution checklist for the approved runtime/staging/workers plan. Subagents implement isolated slices; Main (Astra) reviews their changes and runs the acceptance scenarios. A subagent completion report is not acceptance. Checkboxes close only with recorded behavioral evidence, and hardware-dependent gates stay open when their hardware is unavailable.

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

- [x] Verify every completed slice with Astra review. **Astra:** independently verified subagent reports, real artifact identities/counters and recovery behavior; reproduced and fixed final lifecycle/codec findings. Final Ruff checks and all **335 tests** pass. Earlier scientific and single-host gate evidence remains intact; hardware-dependent claims stay blocked.
- [x] Complete the integrated single-host runtime acceptance gate. **Astra:** accepted after actual CPU FP32/BF16/Adafactor and native MLX continuation, MPS 20/640 full/resumed comparison and CPU-weight promotion, signal/corruption recovery, isolated warmup, dashboard, core-only wheel, and integrity regressions. The immutable execution wheel and hashed evidence index are retained. [Gate record](artifacts/acceptance/single_host_gate_2026_09_22.json).
- [x] Complete end-to-end independent-worker acceptance. **Astra:** final installed core-only wheel, concurrent CPU runs, actual MLX worker, controller disconnect/replay, cancellation, explicit crash recovery, offline promotion/evaluation/inference, real CLI matrix, admission rejection and read-only dashboard checks passed. Actual browser WebGL curve pixels were captured and visually inspected; standard screenshots/runtime-table rasterization stalled, so runtime table values were verified with Streamlit's real app harness. [Gate record](artifacts/acceptance/independent_workers_2026_09_23.json), [rendered curves](artifacts/acceptance/workers_training_2026_09_23.svg).
- [x] Publish verified changes and accurately record blocked gates. **Astra:** implementation, curated data, study results, acceptance records, rendered UI evidence and both frozen execution wheels published to `origin/main` in [e8a60d8](https://github.com/supernovae/tiny-sparse-lab/commit/e8a60d8). All 31 locally actionable tasks are complete; five actual-hardware gates remain explicitly blocked. Owned acceptance services and throwaway drivers were removed or stopped; canonical run/checkpoint/proof directories were retained.

Evidence must include actual CLI scenarios, checkpoint/optimizer/RNG comparisons, negative integrity and recovery cases, populated browser verification, installed-wheel behavior, and real hardware measurements where claimed. Shape-only inspection is not a large-model training result. Sampled MPS peaks are lower bounds, not native allocator high-water measurements.

## Excluded from this execution

- [ ] Distributed MoE expert sharding, all-to-all dispatch, and multi-rank GPU training.
- [ ] Homogeneous distributed data-parallel training.

These remain future research items, explicitly outside the user's current non-distributed scope.
