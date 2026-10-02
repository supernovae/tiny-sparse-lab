# DevMind v5 MODEL-0 execution boundary

Status: **INCOMPLETE — ExperimentPlan lock readback resource-censored**.
Real preparation and its separate cold, deep load passed. The subsequent guarded
lock command crossed the unchanged host-wide 1 GiB swap limit and was terminated.
No completed lock/readback gate, ROCm full-shape proof, training, checkpoint,
evaluation, model readiness, ModelFamily or model-continuation archive is claimed.

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

## Current first blocker: guarded lock readback

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
