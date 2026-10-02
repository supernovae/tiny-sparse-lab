# DevMind v5 MODEL-0 execution boundary

Status: **INCOMPLETE — preparation input-integrity gate blocked**. No real v5
prepared arrays, ROCm full-shape proof, training run, checkpoint, evaluation,
model readiness, ModelFamily or model-continuation archive is claimed.

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

## First blocker and stop decision

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

The distinct pre-worker export-location contract gap is recorded in
[TODO.md](../../../TODO.md#experiment-ergonomics); the existing worker-sealing
provenance item remains unchanged. Original crash/corpus-recovery records and
negative attempts remain intact. Generation-stage DSL limitations were not
exercised or newly asserted.

Unexercised gates: sealed real preparation; four-artifact ExperimentPlan lock;
fresh ROCm doctor and full-shape inspect/validate/smoke/warmup; Campaign approval,
dispatch/reconciliation/ingestion/collection; exact final optimizer/target totals;
checkpoint-bound heldout evaluation and greedy panel; model readiness; family,
model-continuation recovery inspection and thin archive. No weights were
published and no model was promoted.
