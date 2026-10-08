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

- [ ] **Expose the Card 03 held-out item freeze through a typed native CLI.**
  The new `sparselab.evaluation.kml_card03_items.freeze_card03_items` Python
  API accepts a cold-verifiable corpus release, frozen family-inventory JSONL,
  reviewed draft JSON and fresh output directory, and returns immutable item,
  chunk and denominator digests. Add a small `sparselab evaluation
  freeze-card03 ITEMS --release RELEASE --families INVENTORY --output OUTPUT
  --json` adapter that reports those identities and fails on changed release,
  train-family leakage, unbound chunk offsets/digests, unreviewed or missing
  items, incomplete 20-by-10/40-by-10 denominators, malformed controls or an
  existing output. Do not turn the adapter into a second authoring engine.

- [ ] **Expose prepared-input bundle publication and cold verification through a typed CLI.**
  The native `materialize_prepared_inputs(config, destination)` and
  `verify_prepared_inputs(root, config)` Python APIs can seal and authenticate a
  data-only bundle, but `sparselab data prepare` publishes only the cache.
  Add a small CLI adapter taking a validated run config and exclusive output
  path, returning the bundle manifest digest and supervised-target counters.
  Verification must reject altered assets, config/source-identity drift, a
  missing supervision mask, or a target count different from the verified
  mixture receipt. Card 03 used the existing APIs directly; this command is
  not shipped yet.

- [ ] **Expose native read-only verification for bounded project budgets and tokenizer artifacts.** Kernel Memory Lab Cards 03/04 currently need Python APIs to inspect a live `TransportBudget` ledger and to call `verify_tokenizer_artifact`. Add small typed adapters, not another orchestrator: `sparselab corpus budget status PROJECT --json` should return the project/attempt binding, deadline, charged/actual source and metadata body bytes, transfer statuses and preserved failures; it must fail on missing, corrupt, expired-clock or identity-mismatched ledgers without mutating them. `sparselab tokenizer verify CONFIG --json` should bind the configured tokenizer path to its native manifest and declared source/revision/vocabulary, then report digest, vocabulary and special-token IDs with exact provenance; it must reject missing/tampered outputs, underfilled vocabulary or a changed config/source rather than silently accepting a same-sized file. Keep optional round-trip probes separately declared and report their denominators. These would replace ad hoc API invocations in project handoffs; they are not shipped commands today.

- [ ] **Extend the native operational monitor for accelerator memory and added output caps.** Kernel Memory Lab Card 04 P3's external `amd-smi` sampler failed while the native `sparselab monitor` still tracked RSS and free storage. Add optional typed `max_device_memory_bytes`, `max_added_workspace_bytes` and `max_added_workspace_inodes` policy fields with a registered ROCm sampler and a fail-closed unavailable-reading path. Record every sample, cap event and owned-process termination in the native completion receipt; baseline only the declared task workspace, retain process identity, and avoid interpreting sampled VRAM as an exact peak. Mock a sampler that fails after launch, a cap crossing, absent device, and interrupted monitor; no real GPU test is needed for the adapter. Until then, a corrected external sampler is needed for an approved Card 04 replacement profile.

- [ ] **Retain bounded corpus decompression and temporary-storage high-water counters.** Kernel Memory Lab Card 03 P1 enforced the distinct expanded-stream and projected disk caps, but the native receipt does not report actual decompressed bytes or peak staging occupancy. Extend the existing bounded acquisition receipt, without changing legacy hashes when no new fields are present, to report per-shard bytes consumed from the expanded stream and peak task-owned staging bytes/inodes. Fail closed if a required reading is unavailable and preserve raw counters on interrupted transfers; test exact caps, overlong lines and resumed attempts with mocked HTTP. Do not backfill P1 with inferred measurements or require reacquisition only to fill historical optional fields.

- [x] **Materialize and verify a deterministic multi-source training mixture.**
  The native `corpus materialize-mixture` and `corpus verify-mixture` operations
  bind a verified release, tokenizer, family inventory and per-stratum quotas;
  they reject rights or split leakage, quota shortfall and changed inputs. Card
  03's 5,000,000-position output and cold replay are recorded in its offline
  continuation result. Requested metadata remains separate from realized tokens.
- [x] **Bound pinned Git blob acquisition without an unbounded fetch.** Native
  `corpus acquire` now supports optional exact GitHub commit/tree/blob declarations
  with the shared persistent response-body ledger, pre-transfer identity/size
  checks, streamed source caps, content SHA-256 and Git object SHA-1 receipts,
  and conservative retry/resume charges (code commit `ec91e9f`). The legacy
  pattern-based `git fetch --depth=1` mode remains unbounded and is **not**
  suitable for the PagerDuty pilot. The new mode never fetches the unselected
  PDF or builds a Git cache. Added inode and temporary-storage high-water
  receipts remain the separate monitor/receipt task above; no real PagerDuty
  content acquisition has been approved or performed.
- [x] **Admit exact Hub shards with a shared transfer budget.** The native
  bounded-HF declaration and `corpus budget-init` path now accept exact pinned
  shard files with declared config/split metadata, preflight pinned file
  size/SHA-256 and the sole reported config/split, and enforce a durable,
  attempt-bound source/metadata response-body ledger across retries and resumes.
  Redirect and HTTP-error bodies count; source reads never request an extra byte
  beyond the declared cap. Local/mock tests cover invalid selections, changed
  size/checksum, partial transfer, exhausted/corrupt budgets and legacy hash/
  receipt reuse. See `23db56c`/`c77074e` and Kernel Memory Lab [S4](experiments/research/kernel-memory-lab/results/2026-10-07-card03-transport-fix.md).
  This code completion does not approve acquisition or authenticate live source
  rights; Card 03 retains those separate gates.
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
- [ ] **Prioritize live acceptance of provider-neutral S3-compatible relays.**
  The shared rclone relay and S3 profile example already exist; validate and
  close any exposed compatibility gaps rather than adding an AWS-only SDK or
  another storage/orchestration engine. Use S3-compatible storage for the next
  hosted TinyStories/T4 acceptance before tackling Drive OAuth.
  Cover Cloudflare R2 first, then representative alternatives such as Backblaze
  B2's S3 API, Wasabi, DigitalOcean Spaces, and Scaleway Object Storage; retain
  AWS S3 compatibility without making Amazon an infrastructure requirement.
  Follow the [rclone S3 provider guidance](https://rclone.org/s3/) for explicit
  provider, endpoint, region, bucket/prefix, signing and addressing settings.
  Supply separate private controller/worker credentials scoped as narrowly as
  the provider supports; document permission limitations, rotation and expiry.
  No credentials in profiles, bundles, receipts, command logs or Git.
  Acceptance must exercise native relay preflight, actual object put and cold
  SHA-256 readback, immutable conflicts, partial/multipart failures, bounded
  timeouts, authenticated commit publication, collection with the executor
  unavailable, and full-state child resume from the last verified checkpoint.
  Preserve unavailable quota and permission observations; do not trust ETags,
  sizes or provider success text as content identity. Prove the same workflow
  across macOS, Linux, Windows and WSL2 where supported, recording skipped lanes
  without claiming native-Windows Colab support.
  Compare current storage, request, egress, minimum-retention/minimum-billing
  costs, region and measured upload/readback latency and throughput for the
  actual workload; do not hardcode a cheapest/fastest provider. Paid resources
  require explicit authorization. A local rclone filesystem round-trip does
  not close any cloud-provider or T4 training gate.
- [ ] **Defer Drive OAuth setup and unattended-credential lifecycle validation.**
  Keep Drive as an optional relay using the same verified-copy contract, not a
  required dependency of the S3/T4 path. Complete the dedicated Desktop OAuth
  client and `drive.file` consent workflow with private, separate per-host
  configs using the same client; verify app-created folder visibility and
  actionable failures for missing/revoked consent, expired credentials, and
  Testing-mode grants that expire after a week. Document safe headless worker
  enrollment and refresh without interactive Colab login or token logging.
  Colab ADC is not Drive authorization. The live Drive gate remains open:
  no authorized client JSON or actual Drive API transfer has been supplied.
  Never automatically publish an OAuth app, fall back to whole-Drive access,
  or rely on rclone's retiring shared client to bypass this prerequisite.
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
