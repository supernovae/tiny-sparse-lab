# DevMind v5 MODEL-0 execution boundary

Status: **INCOMPLETE — full-shape smoke censored by the 900-second pilot deadline**.
The authenticated lock and independent cold readback passed. Fresh registered
ROCm doctor is READY; exact locked 69M inspection and staging validation passed.
Smoke timed out before a successful pilot report, and warmup was not reached.
Campaign training, collected checkpoint, evaluation and model closure remain unexercised.

## Accepted results

- Tested implementation: `e45fe9428ca7b68240d6051ba613eb8a14aa1913`;
  [implementation verification](source-token-implementation-verification.json).
- One guarded authenticated denominator: **30,175,366** raw developer source
  tokens from **9,035** distinct kept train documents, **101,034,268** source
  UTF-8 bytes; **818,630** release rows scanned through authenticated EOF.
  [Measurement verification](source-token-measurement-verification.json) binds
  canonical receipt SHA `24097253544f31c7c5b22b838f635a0869c7dade8bada03c5c77c8b8c1294d4e`.
  Encoding/scanning wall time was 29.325819 seconds; the monitor observed
  1,469,599,744-byte process-tree peak RSS, with no guard violation.
- Frozen budget: `B=8192`, `floor(3*D/2)=45263049`, **5,525** updates,
  **45,260,800** supervised targets, remainder **2,249**. Requested developer
  passes are 1.5; this is not actual developer packed-target attribution.
  [Science declaration](model0-science.json) records the exact formula,
  descriptive mixture, unchanged controls and **69,317,760** parameter inventory.
- One immutable full LM export:
  `621f407e5a1e506435c4c5c92d580d8b8735ed7ca89cbcf4f402ca8365d3c6c6`.
  Generated dataset fields and recorded train/validation counts and byte ceilings
  match; only the selected tokenizer path changed in [model0-run.yaml](model0-run.yaml).
  [Export verification](model0-export-verification.json) binds the input closure.

## Historical first blocker and original stop decision

The single guarded `data prepare` attempt exited 1 before tokenization, in
`verify_tokenizer_artifact → verify_release_export`, with
`ValueError: corpus export request or license mismatch`.
The producer placed the export under the requested global persistent work root,
`/srv/sparselab/state/corpora/devmind-v5/exports/...`.
The consumer's `src/sparselab/corpus/export.py:269–270` instead requires the
export beside the release, under the primary replay workspace's
`.../corpora/devmind-v5/exports/...`. The request digest matches and the generated
dataset is unchanged; the location predicate fails.

[Negative evidence](model0-preparation-stop.json) preserves the exact command,
PID/create times, source commit, immutable input hashes, failure logs and monitor
completion. The process-tree peak was 1,152,475,136 bytes; no resource guard fired.
There was no sealed prepared directory. The approved fail-closed contingency
stops this execution here: no retry, relocation, changed generated dataset,
corpus rebuild, tokenizer refit, CPU training fallback or scientific relaxation.

The pre-worker export-location contract gap was subsequently repaired and its
satisfied TODO closed; the separate worker-sealing provenance item remains
unchanged. Original crash/corpus-recovery records and negative attempts remain
intact. Generation-stage DSL limitations have not been exercised or newly asserted.

At the original stop, unexercised gates were: sealed real preparation; four-artifact ExperimentPlan lock;
fresh ROCm doctor and full-shape inspect/validate/smoke/warmup; Campaign approval,
dispatch/reconciliation/ingestion/collection; exact final optimizer/target totals;
checkpoint-bound heldout evaluation and greedy panel; model readiness; family,
model-continuation recovery inspection and thin archive. No weights were
published and no model was promoted.

## Authorized post-repair preparation

[Implementation verification](export-location-contract-verification.json) binds
repair commit `18fa32f9a7fc3459fc8dce73ec18006eab5cf0a8`, the cross-root CLI
smoke, **1,448 passed / 3 skipped** full CPU suite, and Ruff checks.
[Input authentication](model0-input-authentication-after-repair.json) verifies the
existing export in place without relocation, regeneration, tokenizer refitting or
recomputation of the accepted denominator/budget.

The single newly authorized preparation completed with all **796,563 train** and
**16,411 validation** documents retained and zero skipped/truncated documents.
The separate cold, deep load verified both array hashes, source/config/tokenizer
bindings, `contiguous-eos-v6`, all-token supervision, and absence of masks or
memory sidecars. [Preparation verification](model0-preparation-verification.json)
binds the manifest, logs, source revision, timings and unchanged resource guards:

- Prepared settings: `15986e13517461f080e18bc19cdd6c5ac4008155d53dcc4ce21bdaff62487b84`.
- Logical manifest: `6d05088bb9552c1ec5527c374c41ccd08d8dca00bdae67c7a0354fded70429a3`.
- Train: **984,316,775 packed IDs including EOS**, **984,316,774 adjacent
  next-token targets**, **984,315,904 full 1,024-target block slots**.
- Validation: **23,434,926 packed IDs including EOS**, **23,434,925 adjacent
  next-token targets**, **23,434,240 full 1,024-target block slots**.
- Monitor end-to-end elapsed: **2,274.681 seconds**; sampled process-tree peak:
  **8,834,408,448 bytes**; no guard violation.

These materialization counts are not the distinct raw developer denominator,
actual developer packed-target attribution, or evidence of model quality.
ExperimentPlan declaration validation and inspection subsequently passed; lock
readback and all later execution gates did not complete.

## Historical blocker: guarded lock readback

The complete [ExperimentPlan declaration](model0-plan.yaml) is committed and
binds all four required input kinds: release, export, tokenizer and prepared data.
Its typed artifact digests use their actual schema domains; file hashes are
recorded separately. The heldout suite and all retention flags are bound.
`experiment validate` and `experiment inspect` passed with the unchanged
5,525-update / 45,260,800-target controls.

The guarded `experiment lock` command was terminated on **SWAP_LIMIT** after
**2,021.690 seconds**: host-wide swap reached **1,076,547,584 bytes**, exceeding
the unchanged **1,073,741,824-byte** ceiling. Sampled process-tree peak was
**8,724,733,952 bytes**; available RAM at the violating sample was
**24,465,362,944 bytes**. The process tree exited and final inventory is empty.
These counters do not identify the cause of all host swap usage.

Lock and availability files were published before interruption:
`275e0975d0d000555f8570f0a7ba12051c2311a859da497ed1c67ef15267dc45`.
Their recorded scientific digest is
`975295d546eadd6d23b44a0fcb8f84cc08aeffb8f2982985f1ecc2675b03b8af`.
They are preserved, hash-referenced, **not accepted as a completed gate**:
the command emitted no success receipt and publication calls `open_lock` before
returning. No independent readback, explain, retry, swap reset or guard relaxation
was performed. [Negative lock evidence](model0-plan-lock-stop.json) binds the
exact command, source, PID/create times, samples and partial files.

Unexercised dependent gates: completed lock readback/explain; fresh ROCm doctor;
full-shape inspect/validate/smoke/warmup; Campaign authorization, dispatch,
reconciliation, ingestion and collection; exact training budget and verified
checkpoint; heldout FP32 evaluation and exact greedy panel; readiness; family;
model-continuation recovery inspection and thin archive. No fallback training,
scientific change, SFT, weight publication or promotion occurred.

[Procedural audit](model0-procedural-audit.json) distinguishes scientific choices,
operator policy, implementation limits, unexercised DSL questions and the ordinary
resource-censored execution. Only the demonstrated overbroad checkpoint storage
preview received a new code TODO; worker provenance remains independently open.

## Authorized operational repair and lock acceptance

[Lock acceptance](model0-plan-lock-acceptance.json) binds implementation commit
`b2160c19018d89c633e75a734dff72cec53e4a44`, the explicitly authorized
[source compatibility record](model0-operational-source-compatibility.json),
its published authority commit, the new operational policy and actual receipts.
The execution package retains its actual new source identity; the compatibility
record authenticates the historical lock/prepared source, not a regenerated cache
or a spoofed runtime identity.

The first-class typed monitor now attributes RSS and swap to owned PID/create-time
identities. Its independent host swap emergency uses **free** swap, not absolute
host-wide used swap. Guards remain 12 GiB owned RSS, 1 GiB owned swap, 8 GiB
available RAM and 1 GiB free host swap, with quarter-filesystem projected headroom.
The historical failed lock and its negative evidence remain unchanged.

- Real lock: **COMPLETE**, child exit 0, no violations, 1,604.322 seconds,
  peak owned RSS **8,858,251,264 bytes**, owned swap **0**.
- Separate cold explain: **COMPLETE**, child exit 0, no violations,
  1,538.799 seconds, peak owned RSS **8,790,904,832 bytes**, owned swap **0**.
- Scientific digest remains `975295d546eadd6d23b44a0fcb8f84cc08aeffb8f2982985f1ecc2675b03b8af`;
  plan digest remains `275e0975d0d000555f8570f0a7ba12051c2311a859da497ed1c67ef15267dc45`.
  The preserved lock file remains byte-identical, SHA
  `b3b28b5b0ed6b1e8d6a18acbb1f96e17dee78dba394d5623c450d6eb0daf9855`.
- Actual bound storage inspection: **4** checkpoint writes upper bound,
  **4,031,011,511** existing prepared bytes, **0** future packed-cache bytes,
  **4,031,011,511** future run-copy bytes; storage reported adequate.
- Implementation verification: **44** focused tests passed; full CPU suite
  **1,491 passed / 3 skipped**; Ruff check and format check passed. Guarded real
  CLI smoke covered resolve/publication, separate cold open/explain and nonzero
  child status preservation. Disposable smoke data is not scientific evidence.

The checkpoint-preview TODO is closed; worker-sealing provenance remains open.
The [Campaign declaration](model0-campaign.yaml) binds the retained release,
export, tokenizer, prepared data, original token-only readiness receipt, accepted
plan, registered ROCm runtime and all-upstream model approval. Static validation
passed; this does not claim Campaign execution or acceptance of a model.

## Historical blocker: full-shape smoke deadline

[Full-shape stop evidence](model0-full-shape-stop.json) binds the fresh native
`rocm-7900xtx` doctor, exact config agreement, passive inspection, preserved failed
stage bundle and guarded command. Execution used published commit
`7e380777b79d359b96572c7fe542ecf4041157a1`; actual package SHA remains
`8d8268b961aa73c91651e7e3a18ae80d0764d5a35e7624c3e0988fd5b711d3c6`.
The native runtime probe uses a distinct digest scope, recorded separately.
Vendor interpreter/framework packages were preserved with locked `--no-sync`.

The effective stage config equals the requested locked config: **69,317,760**
parameters, sequence **1,024**, micro-batch **2**, accumulation **4**, BF16,
transformer-block recomputation, no optimizer/activation offload. The passive
unknown-capacity 1×8 execution proposal was not adopted.

Configured, inspected and validated stages completed. The smoke subprocess raised
`subprocess.TimeoutExpired` at `_run_pilot`'s hard-coded **900 seconds**.
Its execution log is empty, no pilot report was produced, and no finalized
checkpoint exists. The preserved bundle records `SMOKE_TEST` failed; warmup
was not reached. No successful optimizer-update evidence is claimed.

The outer command exited **1** after **1,484.600 seconds**, with no guard
violations: peak owned RSS **9,975,336,960 bytes**, owned swap **0**, minimum
available RAM **21,533,728,768 bytes**, minimum free host swap
**6,152,577,024 bytes**. Recorded owned process identities were confirmed gone.
This is a pilot wall-time censoring event, not proof of OOM, bad scientific inputs,
full-shape fit, throughput or model quality. The time-consuming phase remains
unknown; elapsed time alone is not a diagnosis.

No retry, larger deadline, smaller shape, CPU training fallback, source/data
regeneration or scientific relaxation was applied. The authored Campaign remains
statically validated but unapplied: **zero Campaign attempts**. Approval,
dispatch/reconciliation, the 5,525-update / 45,260,800-target training budget,
checkpoint ingestion/collection, heldout FP32 evaluation, greedy panel, model
readiness, family, continuation recovery and thin archive remain unexercised.
No SFT, promotion or weight publication occurred.

`TODO.md` now records the demonstrated code gap: explicit typed operational pilot
limits, phase progress and sealed timeout evidence, retaining cold authentication
and failed bundles. The checkpoint-preview item remains closed, worker provenance
remains open, and no unexercised generation-stage gap was added.

## Actual full-shape deadline repair acceptance

[Deadline audit](model0-pilot-deadline-audit.json) preserves the opaque failed
bundle. [Operational CPU readiness](model0-pilot-deadline-readiness.json) records
actual CLI smoke/warmup, checkpoint reload and a sealed controlled timeout.
Implementation `f2a856e81d47d2f5c347533abf454df0796dacdf` and its
[renewed source authority](model0-pilot-deadline-source-compatibility.json) were
pushed before the actual native retry.

[Full-shape readiness](model0-full-shape-deadline-readiness.json) binds the
immutable new bundle and a separate cold byte/checkpoint verification. The
requested scientific config, prepared identity, D, budget and original plan
remain unchanged; execution package SHA is
`5a011e4ccad4c123bba10239c8bf89a4e8c28b226ed84f46c00d53f8bccd392f`.
The explicit operational allowances are 1,800 seconds initialization, 1,200
seconds no progress, 7,200 seconds absolute and 5 seconds termination grace.

- Actual smoke: **2** updates, **16,384** committed supervised targets,
  full-state checkpoint verification and finite reloaded forward.
- Actual warmup: **5** updates, **40,960** committed supervised targets,
  full-state checkpoint verification and finite reloaded forward.
- Both use the frozen 69,317,760-parameter, 1,024-sequence, 2×4 BF16
  transformer-block ROCm device-0 shape. All recorded gradient norms are finite.
- Warmup updates 2–5: **0.348336–0.354823 seconds**, median **0.352458 seconds**,
  **23,271.487 targets/second**. This excludes initialization and other phases;
  the optimizer-only full-budget forecast is not end-to-end runtime.
- Native peak device reservation: **2,797,600,768 bytes**. Outer guard:
  **COMPLETE**, no violations, **3,175.315 seconds**, peak owned RSS
  **9,902,919,680 bytes**, owned swap **0**. Pilot wall times were
  **1,258.175** and **1,303.425 seconds**—both exceed the old opaque 900-second cap.
- Independent cold bundle/deep-checkpoint verification: **COMPLETE**, no
  violations, **40.416 seconds**.

The trusted ledger measures input materialization at **609.170/628.318 seconds**.
It also exposes **599.622/624.354 seconds** between initial input validation and
stage-bundle verification. Code inspection localizes that boundary to the
pre-stage Corpus Forge/export/tokenizer authentication gate; no finer timing or
retroactive diagnosis of the original failed attempt is claimed.
[Corpus input-phase readiness](model0-corpus-input-phase-readiness.json) records
the subsequent operational-only enclosure of this gate in `input_validation`,
including portable worker bindings. Actual sample Forge CLI smoke/warmup
verified its phase attribution, 2/5 updates and finite checkpoint reload;
**150 tests passed, 1 skipped**. Cold semantic checks are retained, not bypassed.
Full-shape evidence above stays bound to its actual earlier execution source.

These pilots establish execution of the measured shape and updates, not
full-budget stability, model quality or promotion. Campaign dispatch, exact
training-budget completion, evaluation and lineage still require their gates.

## Current-source full-shape acceptance

[Current-source readiness](model0-current-source-full-shape-readiness.json)
binds the subsequently tested and pushed execution source
`2f911e6d2a833e748971bd3621f95307b05ac8e536e23ebcc447368a84a88b4f`.
The original frozen run, plan, corpus, tokenizer and prepared-data identities
remain unchanged.

- Actual ROCm smoke/warmup: **2/5** updates and **16,384/40,960**
  committed targets, finite gradient norms, verified full-state checkpoint
  reload and finite forward.
- The formerly unlabelled authentication gate is now measured within
  `input_validation`: **621.631/628.283 seconds**. Input materialization
  remains **638.220/609.643 seconds**, retaining cold semantic verification.
  Durations are inclusive; counters do not represent unique bytes.
- Outer resource guard: **COMPLETE**, no violations, **3,235.980 seconds**;
  peak owned RSS **9,985,232,896 bytes**, owned swap **579,973,120 bytes**,
  minimum host available RAM **22,319,841,280 bytes**.
- Independent cold inventory and full-state checkpoint verification:
  **COMPLETE**, no violations, **23.226 seconds**. Its initial operator
  assertion incorrectly assumed uppercase pilot status; that failed
  verification log is preserved, and only the assertion was corrected.

This accepts current-source execution readiness, not full-budget stability or
model quality. The earlier read-only Campaign status command was censored by
its 180-second outer limit; it dispatched nothing.

## Completed frozen MODEL-0 run and negative observations

[MODEL-0 result](model0-result.json) binds the one approved Campaign attempt,
completed controller ingestion and collection, and all twelve completed stages.
Scientific settings remained frozen: **69,317,760 parameters**, measured
**D=30,175,366**, **5,525 updates**, **45,260,800 committed supervised targets**,
sequence **1,024**, microbatch **2**, accumulation **4**, BF16 transformer-block
checkpointing on ROCm device 0, without offload.

- Experiment: `70f48824-aa0a-4672-a1b0-70813515c9bb`.
- Sole training attempt: `71b0df59-9edb-4946-b57b-33163e387cbb`.
- Run: `762c4104-3a3b-4226-afa6-402cc350356e`.
- Unique collected terminal generation: `step_00005525_gen_000004`,
  checkpoint SHA
  `52aba43c269204ada5ce4101414f71aa7a8bbe116d5d1d58780eb2e2717d2709`.
- Independent full-state verification/load confirms exact counters; all
  **5,525** ordered update/gradient rows are finite and each commits **8,192**
  targets, with no overflow retries. Retained generations at steps 0, 2,048,
  4,096 and 5,525 remain intact.
- Optimizer updates total **1,979.607 seconds**: median **0.355749**, p95
  **0.381953**, range **0.346642–0.536276 seconds**, aggregate
  **22,863.533 targets/second**. Recorded training-operation end-to-end time is
  **2,459.213 seconds**; this does not include worker staging or Campaign
  verification/reconciliation. Those boundaries are preserved separately.
- Exact-generation FP32 ROCm heldout: **64 batches**, **131,072 valid targets**,
  loss **3.208992707**, perplexity **24.754138982**. Evaluation index SHA
  `1bd90e0301a8b1f19408c5505d8be13717b40972157fa48fe2ee492b83161816`.
- Preregistered greedy panel ran once on that same checkpoint/index:
  temperature 0, top-k 0, 64 new tokens, seed 42. Raw text/token IDs are retained.
  The Python-prefix continuation repeats a C/JavaScript-like block, SQL repeats
  `--echo #`, and shell repeats comment markers. These are negative descriptive
  observations, not a quality threshold added after seeing the outputs.
- Readiness SHA
  `46b6f8ff39ee57fb312db7aac8fe9d8ba702f01ddc20f18701a34a42764c81b8`
  reports **READY_FOR_NEXT_STAGE** under the original evidence policy, with one
  completed heldout gate and no missing gates. It does not establish usefulness.

The one-dispatch runner exhausted its bounded reconciliation lifetime after
training completed; repeated 120-second foreground windows left ingestion
`PENDING`, not `ERROR`. A bounded 3,600-second same-attempt window completed
ingestion, collection and evaluation; its outer cap preserved readiness
`RUNNING`. A final same-attempt reconciliation completed readiness. All censored
logs remain, no resource guard reported violations, and no fresh attempt,
science change, CPU fallback, SFT or promotion occurred.

[Family](model0-family.yaml) pins the unpromoted parent-null node.
[Continuation recovery](model0-recovery.yaml) deliberately declares retained
corpus/export/tokenizer/prepared/lock/checkpoint bytes as external requirements.
Exact physical availability is recorded separately: opaque `external_required`
steps cannot resolve those locations and do not claim deterministic replay.
The earlier corpus recovery and crash-recovery records are unchanged.

The descriptive panel remains caller-created procedural glue; its demonstrated
non-gating Campaign-stage gap is recorded in `TODO.md`. The independent sealed
Forge `run.yaml` provenance TODO remains: this completed prepared-input route
does not close its broader worker/relocation acceptance criteria.

## Verified lineage and external recovery boundary

[Lineage verification](model0-lineage-verification.json) records successful
`family show`, `family graph`, `family verify` and `recovery inspect`.
Parent-null `model-0` node SHA is
`8707d174f5215c085ba540cb00fd2ce0cfd5ae72857115d4099847fc33cb71f1`.
Checkpoint, evaluation index, readiness result and plan are **PRESENT**.
Recovery intentionally reports **BLOCKED / MISSING_EXTERNAL** for opaque
external requirements, rather than claiming those bytes can be reconstructed.
Physical retained-byte availability is authenticated and located in
`model0-result.json`; the inspection schema does not resolve those locations.

The recovery metadata revision is
`e7f6f19dbfcb169781c0353c199eb4861965db44`; the actual run remains bound to
tested execution code and run source `5fcefc19d6b021e9f2e356cc2a29bdadca444823`.
Native [evaluation](model0-evaluation-reference.json) and
[checkpoint](model0-checkpoint-reference.json) references were emitted by
`research evidence export`, not by assigning a supported marker to a custom
summary. An initial unsupported-reference draft and an invalid raw-index copy
were rejected and corrected; the failure evidence remains. No scientific
outputs, source bytes or acceptance criteria changed.
