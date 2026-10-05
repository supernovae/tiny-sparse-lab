# Capacity-aware deterministic execution v1

Operational code work only. MODEL-0 science and retained artifacts are immutable;
no MODEL-1, training or distributed execution is part of this work.

## Historical baseline

[Compact baseline](../artifacts/benchmarks/capacity-aware-execution-v1-baseline.json)
binds exact JSON pointers into versioned
[execution records](../experiments/research/devmind-pretrain-v5/model0-execution.md),
[preparation](../experiments/research/devmind-pretrain-v5/model0-preparation-verification.json),
[lock acceptance](../experiments/research/devmind-pretrain-v5/model0-plan-lock-acceptance.json),
[current-source readiness](../experiments/research/devmind-pretrain-v5/model0-current-source-full-shape-readiness.json)
and [completed result](../experiments/research/devmind-pretrain-v5/model0-result.json).
These are retrospective operational evidence, not new scientific results.

| Boundary | Observed wall seconds | Interpretation |
| --- | ---: | --- |
| Source denominator | 29.325819 | Native encoding/scan; 1,469,599,744 B peak tree RSS |
| Preparation | 2,274.680755 | Inclusive; 8,834,408,448 B peak tree RSS |
| Lock / independent cold explain | 1,604.321869 / 1,538.799164 | Repeated full input closure authentication |
| Pilot input validation | 621.631 / 628.283 | Smoke/warmup inclusive authentication |
| Pilot materialization | 638.220 / 609.643 | Smoke/warmup inclusive materialization |
| Optimizer updates | 1,979.606588 | 5,525 updates; 22,863.533 targets/s |
| Training operation | 2,459.213342 | Excludes worker staging and Campaign reconciliation |
| Native heldout | 7.823494 | Not a demonstrated host bottleneck |
| Four training checkpoints | 13.698850 | Nested in training operation |
| Reconciliation censored windows | 25,550.210 / 21,601.291 | Separate interrupted windows, not successful elapsed |
| Final closure | 8,309.562 | Successful same-attempt closure |

Preparation records 4,081,965,445 logical input B, 4,031,007,060 logical
output B and 812,974 records. Native tokenizer encoding is 598.125071 s;
final fsync is 17.091629 s. These are nested counters, not disjoint phases.
The retained prepared root has four files (two arrays, manifest and ownership
marker), totaling 4,031,011,511 B. Per-file sizes and frozen array SHA-256
are in the baseline. CPU and physical I/O counters absent from the records
are null. Disk free delta is not physical write traffic. Historical preparation
swap is host-wide used swap; later monitored swap is owned process-tree swap.

## Call graph and data-flow boundaries

| Entry | Repeated downstream checks/copies | Identity authority | Proposed reuse boundary |
| --- | --- | --- | --- |
| `resolve_plan` / `open_lock` | `verify_artifact` for each input and corpus variant | Typed kind/version/identifier/domain SHA plus package source | Explicit trusted store; direct calls remain cold |
| Tokenizer domain verifier | Export, release, build, snapshots via provenance | Tokenizer bytes and exact export/release closure | Unchanged typed ancestor closure |
| Prepared domain verifier | `load_prepared_data`, array SHA and header validation | Manifest/settings/tokenizer/source SHA | Host-local signed proof; new process-local seal |
| `stage` / pilot `input_validation` | Tokenizer closure, stage inventory, prepared input checks | Stage inventories and prepared seals | Explicit mode passed into pilot child |
| Pilot `input_materialization` | Source arrays into private prepared and assets trees | Canonical SHA, source fingerprint, destination verification | Private copies/clones; private stage-only links |
| Dispatch install/materialize | Asset SHA cache install, cache rescan, private copy | Existing dispatch asset/bundle CAS SHA | TRANSFER cold; same-store signed reuse |
| Campaign / Controller reconciliation | Artifact checks, RPC status, receipt ingestion and collection | Durable attempt state and typed output identities | Unchanged local evidence only; terminal ingestion gating retained |
| Archive / recovery / family | Independent byte and semantic verification | Final durable evidence authority | Always COLD |

## Ranked measured opportunity, before tuning

1. Repeated typed closure authentication: 1,539–1,604 s lock boundaries and
   622–628 s pilot gate. Highest eligible operational reuse opportunity; byte
   hashing versus semantic parsing share is not yet isolated.
2. Private materialization: 610–638 s inclusive pilot phase, roughly 4 GB
   canonical arrays. Clone/copy mechanisms require measured destination safety.
3. Preparation native encoding: 598 s nested component. Already Rayon-backed;
   no Python parallel producer or larger thread default is justified yet.
4. Reconciliation: largest observed windows but censored and unattributed.
   Profile before claiming improvement; 0.1 s sleep does not explain hours.
5. Independent multi-file cold SHA: real prepared deep check 15.172 s; eligible
   only when comparable medium timings demonstrate benefit. Single-file SHA
   remains native sequential SHA-256.

Heldout (7.823 s), checkpoint writes (13.699 s), ISA-specific hashing, GPU
parallelism, source parsing parallelism and new package dependencies are
unmeasured/ineligible optimizations, not claimed gains.

## Trust modes

- **COLD**: independent full byte and domain-semantic verification. Default for
  direct Python APIs, external paths and final archival.
- **VERIFIED_REUSE**: unchanged bytes and typed upstream closure in the same
  authenticated, owner-only registered persistent store, using signed local
  receipts. Not protection against a compromised same-UID owner.
- **STRUCTURAL**: same-operation, process-owned sealed `VerifiedFile` or
  `_VerifiedArtifact`; fingerprints, inventory and headers still checked.
- **TRANSFER**: full SHA and semantic authentication once upon entry to a new
  worker/cache trust domain, before publication.

Operational hardware, selected workers, proof/key paths and trust modes never
enter scientific, prepared, artifact or plan digests. Relocation, missing or
foreign receipts, changed fingerprints/closure or implementation, unsafe paths
and signature failure require COLD fallback; failed cold verification is fatal.

## Bounded host planner

`host_capacity.plan_host_workers` bounds native independent tasks by affinity,
physical CPUs (logical fallback), measured available RAM after explicit reserve,
per-worker memory and operator cap. Unknown RAM permits only one <=8 MiB SHA
worker. Callers reserve max(1 GiB, total RAM/10); no memory-heavy unknown-RAM work.
The SHA bound includes the 1 MiB hasher buffer and thread overhead; observed
medium process RSS differs by only a few MiB across 1/2/4 workers.
`run_ordered` keeps at most the selected worker count outstanding, reduces in
input ordinal order and cancels/joins on failure or iterator close.

[Before measurements](../artifacts/benchmarks/capacity-aware-execution-v1-before.json)
use independent processes and uncontrolled OS cache, not evicted-cache claims.
Their elapsed times include package imports. Threaded SHA did not demonstrate a
material end-to-end benefit; production cold SHA defaults to one worker.
The bounded host probe reports x86 SSE4.1/4.2, AVX/AVX2, SHA-NI and AES here;
AVX512F is absent. Native SHA is `_hashlib`, linked OpenSSL 3.5.5. Capability
presence is not an acceleration result; no custom hashing or dependency added.

## Host-local proof protocol

`verification_proofs.ProofStore(work_root)` stores versioned HMAC-SHA256 receipts
under `cache/verification-v1/`. The separate 32-byte secret is
`$XDG_CONFIG_HOME/sparselab/verification-key-v1` (absolute XDG only), otherwise
`~/.config/sparselab/verification-key-v1`. Receipt/key files are 0600 and their
private directories 0700. No proof is written inside an inventoried artifact.

Bindings include typed kind/version/identifier/SHA, canonical absolute path,
manifest/inventory digest, exact upstream identities, full member/dependency
device/inode/mode/size/mtime-ns/ctime-ns fingerprints and verification success.
The signed envelope binds verifier schema 1 and actual package source SHA.
Only currently minted, process-owned cold verifier seals can publish receipts.
Warm validation checks the complete signature and binding, trusted ownership
and fingerprints before and after lookup, then mints a new process-local seal.
Prepared inventory, manifest digest, dtype and shape checks still run.

Trust requires the selected root and paths below it to be service-UID-owned and
not group/other writable. Symlinks, multiply linked regular files, unsafe stores,
foreign/missing keys, unsigned/stale receipts and changed bindings miss to cold.
Cold corruption fails; there is no mtime/size-only authority or same-UID attacker
protection. Direct Python verification remains cold unless explicitly opted in.

Experiment `inspect`, `diff`, `lock`, `bind`, `run`, `collect`, `explain` and
`reconstruct`, and `stage`, accept `--cold-verify`. Normal commands select the
registered persistent root; only inputs beneath its trusted paths can reuse.
Cold mode constructs no proof store and reads/writes no proof. Operational
CLI proof counters are outside plan/scientific digests. Stage/pilot/worker
parameters forward mode explicitly; portable corpus attachments retain their
independent binding verification. Archive, recovery and family remain cold.

## Prepared array verification

Cold independent array hashes use `run_ordered` with canonical filenames and
native full-file SHA-256. The measured default remains one worker, including for
the single large `train.npy`; no tree/chunk digest substitution. All 1/2/N choices
preserve exact array SHA and manifest bytes in regression checks.

Per-array signed receipts bind SHA, full fingerprint, shape/dtype metadata and
cache identity. `_receipt_from_proofs` independently rechecks the current
manifest digest, exact inventory, every new process-owned file seal and each
NPY header. A changed array node is cold-checked; unchanged array-node proofs
remain eligible while the prepared/plan closure is invalidated.
Same-size mutation with restored mtime, inode replacement, truncation, altered
manifest/dependency, relocation and forged/stale/missing receipts are covered.
Independent cold loads remain reliable; unsigned metadata cannot mint seals.

## Owned stage and worker materialization

`owned_copy` authenticates the source seal/fingerprint and destination SHA before
exclusive publication and fsync. Linux FICLONE and macOS clone are capability
attempts, not assumed available. Partial native copies reset before bounded
buffered fallback. Published destinations are private inodes (0600); source
permissions and bytes are unchanged. Directory layout and scientific inventories
still contain the same names and exact bytes.

Only a freshly materialized stage-private `prepared/assets` tree may link to the
same stage's `assets`. Both historical paths remain in the inventory. Link
creation changes ctime, so that intentional transition receives an independent
SHA check before refreshing process-owned seals. Canonical inputs, supplied
external stage/prepared roots and worker CAS never link to mutable worker paths.
Persistent receipt trust rejects multiply-linked files; stage-private links use
operation-local seals and cold verification, not a relaxed persistent policy.

Existing worker `.dispatch-cache` is unchanged: TRANSFER authenticates missing
content and a semantic/portable/continuation closure before signing cache
receipts. Same trusted closure hits avoid CAS SHA rescans; a changed/corrupt
existing CAS asset is fatal, never silently repaired. Destination copies remain
independently authenticated. Public `verify_dispatch_bundle` remains cold.
Copy/cache observations are optional return-side collections, never inventory or
scientific digest members; unavailable physical I/O remains null.

Three alternating fresh-process copies of the same prehashed 67,108,992-byte
medium array on this host used `copy_file_range` (FICLONE unavailable):
0.123528 s median, 0.122716–0.123896 s range, versus bounded buffered copies
0.130750 s median, 0.129062–0.138465 s range. Exact SHA matched all six outputs.
CPU medians were 0.076741/0.075849 s and process high-water RSS maxima
377,782,272/377,884,672 bytes respectively. OS cache was uncontrolled, source
prehash preceded each timed copy; this is a copy-plus-destination-SHA/fsync
comparison, not end-to-end preparation or a physical-I/O claim.

Materialization gate: 74 focused copy/proof/stage/worker tests passed, including
late competing-directory no-replace publication. Actual disposable CPU runtime
exercised stage validate → dispatch prepare → TRANSFER → warm install/private
materialize: unchanged worker manifest digest, zero warm CAS SHA reads, isolated
inodes, private corruption rejected and canonical array SHA unchanged. Config
key creation preflights symlink ancestry before creating directories.

## Corpus ancestor reuse and bounded parsing

Mutation commands forward the same host-local proof mode through snapshot,
build, release, export and tokenizer verification. Defaults on direct APIs and
recovery remain cold. Mutable local acquisition inputs are independently read
and hashed: a published snapshot receipt never blesses its original raw pathname.
Full inventory/dependency fingerprints, not a manifest-only memo, bind release
reuse. Rendering, split statistics and tokenizer document iteration still read
the bytes their semantics require. Config/evaluation changes reuse unchanged
ancestors; changed tokenizer/export nodes receive their own canonical bindings.

The measured parser bottleneck included research-exclusion normalization
(3.082 s of 4.519 s profiled record parsing), not just JSON decoding. Independent
HF/Wiki JSONL shards at least 32 MiB, with rows bounded to 1 MiB, can use two
processes. Selection uses the shared CPU/affinity/RAM planner, a 512 MiB bound per
worker and its reserve; unknown/inadequate capacity, long rows, nested metadata,
other formats and already prepared shards retain serial processing. Submission
is bounded and results/counters reduce in canonical source order. SQLite
deduplication, release identity, provenance, emitted bytes and error ordering
are unchanged. No arbitrary regex/ISA rewrite or blanket source parallelism.

Three alternating fresh-process observations of two 32 MiB shards include row
preflight, process startup and shard receipt/output hashing. Serial median:
5.076521 s (5.059783–5.076647); two-process median: 3.415149 s
(3.383993–3.447172). Exact output SHA matched all six observations. Process-tree
RSS maxima were 270,241,792/908,754,944 bytes, observed swap zero; sampled CPU
lower-bound medians 5.23/7.98 s, physical I/O unavailable. This trades CPU/RAM for
shard preparation wall time, not a measured end-to-end corpus speedup.
Separate warmed multi-file SHA observations were not comparable end-to-end
release measurements; release hashing remains serial.

Raw measurement: task-owned `corpus-process-final.json`,
SHA-256 `7222e089bf001452b96f0a38aaac82fae39456601c07ade4547627726b775ce9`;
package implementation `3d55097f77cee97595f50030ca57e9669c2077943187bbfa5d56a3e336fc7557`.
The 138-test focused gate covers snapshot/ancestor reuse, same-size restored-mtime
mutation, canonical tokenizer/export binding, config/evaluation-only changes,
and serial/process full-build bytes plus failure counters. Actual CLI acquire,
repeated offline build and freeze completed in the disposable sample workspace.

## Native encoding and evaluation capacity

Medium profiling retained the existing 256-document/1 MiB tokenizer batches and
fresh native Rayon child. Three alternating fresh cases per 1/2/4 choice encoded
66,272,100 source UTF-8 bytes into 19,011,624 tokens. Before planner integration,
wall medians were 24.871841/15.423598/10.363416 s; native encoding medians
20.999457/11.467340/6.838632 s; process-tree RSS maxima
477,204,480/479,256,576/481,943,552 bytes. Full arrays, manifest and cache identity
matched across worker choices within that revision.

`PreparationEncoder` now selects CPU/affinity/RAM-bounded native threads with the
shared planner before creating its child. Operator `max_workers` is an upper
bound, not permission to exceed measured four-thread capacity. The conservative
per-thread bound is 512 MiB, reserve max(1 GiB, total RAM/10); unknown/inadequate
RAM rejects memory-heavy work. The plan is operational receipt telemetry only.
Same-input post-change medians were 24.585059/15.442836/10.392618 s, tree RSS maxima
476,958,720/479,350,784/481,722,368 bytes. Four-thread before/after ranges
10.312176–10.419641 / 10.375050–10.414132 s overlap; the 0.29% median difference
does not demonstrate a throughput gain. Retained the existing four-thread native
choice with capacity guards, not extra producer processes or larger batches.
Arrays match across the cutover; scientific identities match among worker
choices within each revision, not across changed package identities.

Three warm-process, line-traced CPU-only evaluations of a two-update synthetic
checkpoint preserve exact metrics and canonical index bytes on replay. Median
host preprocessing/reporting were about 1.060 ms/0.012 ms versus native model
7.422 ms, authenticated model load 10.501 ms and suite wall 28.523 ms. CPU
pre/postprocessing did not dominate: no host worker pool added, GPU execution
unchanged and unmeasured. Process-lifetime RSS (497,868,800 bytes) includes
fixture creation and is not an evaluation-specific peak. Physical I/O and
accelerator utilization were unavailable, not inferred from wall time.

Raw task-owned evidence SHA-256:
- `encoding-profile.json`: `418cea59da8d1211697378066b8d1780f6a475be6e2eb5ce7ecbe4510c394c24`
- `encoding-profile-after.json`: `d6d58da9c27d4b6631721a3d6876caabaa9b864a71a78c1c9f4e99990e96c690`
- `evaluation-profile.json`: `324b9c074c90656f809d90e83573a9f6b30a2f6b97d1a9ee910d7c9aea980380`

The 92-test gate covers host bounds, fresh Rayon exact-output parity, streaming
packing, resource envelope enforcement and checkpoint-bound evaluation suites.

## Durable reconciliation and measured observations

Campaign run reconciliation re-queries durable attempt state immediately after
each controller tick. COMPLETE requires ingestion COMPLETE; a changed durable
row is re-evaluated without sleeping. Only an unchanged row before the shared
deadline consumes bounded idle sleep. Replay IDs, terminal/ingestion distinction,
pending-on-deadline and crash recovery remain authoritative.

Run inventories and checkpoint members use the existing signed file proof index
with authoritative run inventory/checkpoint/manifest bindings; semantic checks,
prepared headers and evaluation/checkpoint correspondence still execute.
`experiment_evidence`, collection, retrospective reads and Campaign evaluation
consumers forward explicit proof mode. Native inference still reads model weights
for loading and evaluation inputs for computation; avoiding their redundant SHA
scan is not avoiding the required computation/input read. Independent direct
APIs, recovery and archival remain cold.

CLI `experiment run --cold-verify` now reaches controller preparation, worker
TRANSFER/cache installation, private materialization and pilot/training gates.
RPC `cold_verify` is operational, outside ExperimentSpec and bundle digests.
Missing direct/external launch policy defaults cold; cold requests are sticky
across PREPARED same-ID replay. Running/terminal replay is not mutated or retrained.
The worker's private `verification.json` is outside scientific inventories.

`BottleneckObserver` is opt-in on preparation, staging, Controller and Campaign.
Phase records measure matched surviving-process user/system CPU and available
process I/O, endpoint-sampled tree RSS/swap and an optional real accelerator
utilization probe. Missing counters/probes are null. CPU/I/O exclude children not
alive at both endpoints; memory maxima are sampled endpoints, not lifetime peaks.
Records never enter prepared/stage inventories, scientific/plan hashes or metric
schemas. Probe failures report unavailable observations without blocking work.
Labels are diagnostic heuristics, not scientific findings or acceptance gates:
accelerator, input_host, memory_pressure, unknown; host cpu_bound, io_bound,
serialization_bound, verification_bound, copy_bound, cache_hit, cache_miss,
unknown. Phase and cache labels require measured activity/counters plus caller
provenance; low CPU/device utilization alone is not a failure.

Before this change, bounded read-only fixtures observed expired controller tick
0.000378 s and cold evidence read of an existing two-update synthetic run
0.010273 s. They exercised no live RPC, transfer, ingestion, collection,
evaluation or idle wait; those fields remain null, not a control-plane speedup.
Actual normal CPU integration completed one two-update worker run and ingestion,
preserving canonical input arrays and cold/warm evidence. Its 39 records observed
cache_hit/cache_miss/copy_bound/unknown, endpoint tree RSS at most 400,863,232
bytes and observed swap zero; accelerator and physical I/O unavailable.
Inclusive totals included discovery RPC 2.588331 s, bundle transfer 4.267456 s,
status RPC 13.692612 s and active polling 56.961635 s. They overlap and must not
be summed. A final 31.992090 s tick included ingestion: the 0.1 s idle sleep does
not explain long reconciliation windows.

The independent cold-worker runtime also reached COMPLETE + ingestion COMPLETE
with `cold_verify=true` persisted outside the spec, unchanged canonical arrays
and equivalent cold/warm evidence. Its 51 records separately observed record
ingestion 13.578576 s, artifact transfer/verification 28.839650 s and terminal
receipt ingestion 30.144051 s (inclusive/nested); endpoint tree RSS at most
400,814,080 bytes. No physical-I/O or accelerator counters were manufactured.
Source package implementation:
`18d5ac531bf89cf60333f63d2d57be015e17edcb44bfe1d0b6b9082a0467356d`.

The broad 231-case gate initially passed 229 cases; two new counters incorrectly
counted package-source hashing or verifier entry calls as payload SHA. Corrected
their measurement scope, then all seven affected payload/observation cases
passed. Twelve follow-on controller deadline, durable replay and immediate
Campaign progress cases also passed. Different-process run evidence skips
unchanged payload SHA; restored-mtime corruption still rejects; suite replay
preserves canonical index bytes; observation failure preserves real preparation
identity. The optional-observation TODO is closed; worker-provenance TODO stays
open.

Raw task-owned evidence SHA-256:
- `control-evidence-before.json`: `b7e8ccaf32add93abf334f02a90960b7a91f8034d3693eff19f599960a5b50b0`
- `step9-smoke.json`: `ae1c5a16e9ba8ee0fdd6177b7defffb2f2f2614bebad8cc629e83cbc46a7e6e7`
- `step9-cold-smoke.json`: `2feb7c21fd1ed2be98eb99e71ece06541709ca7e6ebfe2ab8111eb9aabc35e67`

## Final comparable harness

The original cold benchmark operations and their import-inclusive timing
boundaries remain intact. Additional cases prime signed proofs before fresh
processes measure prepared/plan/lock warm reads, worker TRANSFER/cache reuse and
owned materialization. Planned cold arrays alternate requested 1/2/4 workers.
Logical payload-file SHA and private-copy authentication SHA bytes are separate;
physical I/O remains unavailable. Independent output-array SHA validation is
reported outside the additional operation timer, not silently excluded from the
original baseline materialization operation.

Copy-method equivalence compares the **same frozen stage closure** using native
and forced buffered fallback, and dispatch manifests created from the same
frozen stage. Independent `stage(..., through="validate")` invocations legitimately
have different runtime timestamps/RAM observations and therefore different stage
receipt digests; those receipts are never stripped or re-pinned for equality.
The frozen copy smoke observed `copy_file_range` and buffered transfer with
exactly equal stage digest, array bytes and SHA. Reflink is unavailable on this
filesystem; simulated clone/reflink and cross-device failures remain safety
regressions, not measured native acceleration.

An authentic HMAC receipt from an older implementation now has a dedicated
consumer regression: a current verifier falls back to full array SHA rather
than accepting stale signed authority. Config/evaluation changes resolve a new
protocol identity while retaining eligible unchanged ancestors; this does not
authorize in-place rewriting of an accepted Campaign declaration or migration
of its immutable stage receipts.

## Final observations and remaining ROI

Measured clean revision `63cc36ae73bc202603e399261b0c43d829842677`,
package implementation
`18d5ac531bf89cf60333f63d2d57be015e17edcb44bfe1d0b6b9082a0467356d`.
The final harness passed 28 small and 84 medium cases; every medium case has
three repetitions. Full median/range, user/system CPU, sampled peak tree RSS,
logical SHA/copy authentication bytes and identity assertions are in
`artifacts/benchmarks/capacity-aware-execution-v1-results.json`.

| Workload | Before wall s | After wall s | Avoided SHA bytes | Workers | Peak tree RSS B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Medium prepared cold | 1.403549 | 1.446355 | 0 | 1 | 384479232 |
| Medium resolve cold | 1.486785 | 1.501696 | 0 | 1 | 387932160 |
| Medium open cold | 1.455063 | 1.469207 | 0 | 1 | 387940352 |
| Medium materialize cold | 1.626155 | 1.680747 | 0 | 1 | 389292032 |
| Medium prepared warm | not measured | 1.437198 | 68157696 | 1 | 383152128 |
| Medium worker install warm | not measured | 1.464224 | 68157696 | 1 | 384413696 |
| Original MODEL-0 cold, single/cache uncontrolled | 3.363676 | 8.793402 | 0 | 1 | 383934464 |
| Task-owned MODEL-0 warm, three repetitions | not measured | 1.414212 | 4031007060 | 1 | 382341120 |
| Independent two 32 MiB JSONL shard preparation | 5.076521 | 3.415149 | 0 | 2 | 908754944 |
| Isolated 64 MiB copy, buffered → native | 0.130750 | 0.123528 | 0 | 1 | 377782272 |

Warm real verification range: 1.405802–1.416089 s, zero payload SHA calls in
each fresh process. Frozen original train/validation SHA and manifest identity
remain exact. The task-owned real copy exercised actual cross-filesystem
buffered fallback (`st_dev` 2096 → 2112), independent destination authentication
and cold semantic/index priming: 13.440605 s, 385253376 B peak tree RSS.
This transfer/prime boundary includes an extra destination cold scan to prime
the standalone harness; it is not the sealed structural materialization path.

No cold end-to-end speedup is established. Medium import-inclusive medians
overlap or modestly regress; original real cold observations have uncontrolled
OS-cache/storage conditions and must not be treated as a code comparison.
Physical read/write counters remain null/unavailable. The native isolated copy
gain does not imply end-to-end gain: frozen-stage native/buffered medians
1.970161/1.961420 s have overlapping ranges; dispatch-create native/buffered
1.992360/1.989157 s also overlap. Keep serial cold SHA and serial evaluation;
retain measured bounded two-process shard preparation and the isolated native
copy mechanism with safe buffered fallback. Tokenizer throughput remains
statistically overlapping, not a claimed acceleration. ISA capability presence
and OpenSSL `_hashlib` use are operational observations, not benchmarked SIMD.

Remaining measured ranking: native CLI/worker import and serialized RPC startup;
artifact transfer plus receipt/checkpoint ingestion; required destination SHA,
semantic/tensor validation and mandatory private-stage hardlink cold checks;
then native tokenizer work. Durable state idle sleep is not the demonstrated
reconciliation bottleneck. No new DAG, CAS, scientific defaults, metrics,
acceptance criteria or worker provenance authority was introduced.

## Full-gate recovery correction

The first full suite completed 1602 passes, 12 skips and eight failures; the
300-case focused gate and independent lock/explain CLI proof passed first.
Published-preparation recovery still compared its historical four-field raw
chunk fingerprint to the new six-field `VerifiedFile` seal. Recovery now uses
the verifier fingerprint (including mode and ctime) for published arrays; raw
chunk receipt schema and length indexing remain unchanged. Existing publication
and cleanup crash-window regressions pass without re-encoding.

A real external-root smoke injected a crash immediately after atomic
publication, then another interpreter cold-verified all four ID/byte-address
arrays exactly once, completed cleanup and recovered without encoding.
Implementation identity:
`466b154e5c7e0b2d7d8250929cdf3a515723b4169f0896a6e092f0232bde5f7b`.
The corpus corruption fixture now owns a copied snapshot tree rather than
passing an unsafe symlink. Runtime registry-relocation coverage uses a real
resolved/published tokenizer+prepared lock, not a mocked `open_lock` signature
or incomplete fabricated availability. Obsolete exception-wording/private
inventory tests and the unsafe persistent-hardlink warm-skip expectation were
removed rather than re-pinned. Consumer-level mutation, cold verification,
zero-hash trusted reuse and copy isolation coverage remain.

The affected 70 cases passed across the 69-pass broad subset and the corrected
runtime-relocation case. Corrected-source small/medium/real benchmark and CLI
gates are required before final publication. The earlier measurement table is
revision-labelled evidence, not silently relabelled.
