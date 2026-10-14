# Current Workshop acceptance plan (not executed)

This is a proposed bounded CPU engineering check, not permission to launch it or
scientific evidence. The implementation PR runs only focused fixtures and ordinary
CI. Parent/operator must approve this complete envelope before executing the plan.
No GPU, external corpus/model downloads, user-host work, paid allocation, runner
hold changes, historical record migration, release or website publication is included.

## Pins and prerequisites

The inspected Workshop main is `aaa80f91302abf413a7b0856600574f420f0fd55`.
The lab base for this consolidation is
`b066e18b940618a87744681341d270f9d6a5dac8` (PR132 merge).
**The execution lab pin is deliberately not assigned yet:** it must be the clean,
reviewed, merged commit containing this consolidation, with exact-head CI recorded.
Record its full SHA, native installed-source identity, `uv.lock` byte digest,
Python executable/version and package versions. Use the already provisioned locked
CPU environment; installation or environment changes require separate approval.
All commands below use that interpreter as absolute `$P`; no ambient `sparselab`
launcher or implicit environment synchronization.

The Workshop source is an authoring starting point, not a runnable current proof.
Before launch, author and review a new local recipe revision from that exact pin:

- Preserve `run.json`, `plan.json`, `suite.json`, tokenizer config and monitor policy.
  `run.json` byte SHA-256 is
  `f91515509c99017eaa47bb2ad61cc89c956a5047094617dba661183a8ab872bf`.
- Add a terminal Campaign `experiment_collect` stage `final-evidence`, with
  `plan: lock-plan`, `run: train-once`, `evaluations: [evaluate-heldout]`, and
  `requires: [lock-plan, train-once, evaluate-heldout]`. This is the current native
  graph requirement; do not run the old eight-stage Campaign.
- Regenerate preparation bindings/admissions using the current schemas. No
  `preparation_bindings_version` or `preparation_admission_version` fields.
  Preparation descriptor version 1 is unchanged.
- After fresh preparation, copy only verified native output identities into the
  workflow via Workshop's pinned `scripts/seal-inputs.py`. It updates Campaign
  artifact references and inventory member byte digests, not scientific hashes.
  Native inspect/resolve still must verify these inputs. Commit the authored
  inventory and record the new full SHA before creating workflow admission.
- Current workflow admission has `limits`, not `workflow_admission_version` or
  `pair_limits`. Bind reviewed commit and the actual inspect digests. Keep the
  normal `approve-model` gate and exact pinned runtime/interpreter checks.

Do not modify Workshop main or reuse its historical admissions, summaries, locks,
receipts or prepared IDs. Use isolated local recipe checkouts with committed
inventories and two fresh absolute work roots outside Git, called A and B below.
Store bindings, admissions, logs and outputs outside those recipe checkouts.
Check free disk/inodes, no symlink/overlap and no existing operational stores.
Both repetitions must use the same reviewed lab/environment and identical declared
science. Each gets independently prepared payloads and separately authored bindings.
Record all resulting recipe/binding/plan/scientific/source/suite/content identities.

## Exact scientific and operational envelope

| Work per fresh root | Declared amount |
| --- | --- |
| Preparation | synthetic seed 7; 8 train/4 validation documents; at most 256/128 packed tokens; tokenizer 260 vocabulary, 8 documents |
| Model | dense CPU PyTorch fp32, seed 7, hidden 16, FFN 32, 1 layer, 2 heads, max sequence 32, no memory |
| Main attempt | exactly one cell `main:single`; microbatch 1, accumulation 1, sequence 16; 1 update/16 target positions |
| Pilots | 2 smoke + 5 warmup updates, each 16 targets: 112 additional target positions |
| Training total | 8 updates/128 targets including pilots; two fresh roots total 16/256 |
| Validation | initial plus each update: smoke 3, warmup 6, main 2 one-batch passes; 176 targets total |
| Diagnostic | one checkpoint-bound heldout suite, one batch/16 targets; total evaluation 12 batches/192 targets per root |
| Runtime qualification | fixed tiny tensor probes are additional optimizer work, not training targets; expected worker + stage validation + smoke/warmup/main + diagnostic engine validation (6 probes per root) |
| Replay | zero new model attempts, training updates, diagnostic executions or runtime probes |

The probe uses a fixed 4×4 parameter and one optimizer update in the dense CPU
profile (`runtime_probe.py`). Record actual probe invocations separately. A changed
call path, extra attempt, retry, missing stage evidence or accounting discrepancy
stops this acceptance; do not silently extend the envelope or retry a failed root.
Native stage/attempt records establish actual pilot work; final main checkpoints
alone do not account for pilots. Preparation work is separate from model updates.

Preparation admission ceilings per root: `max_documents: 12`, `max_tokens: 384`,
`max_vocab_size: 260`, `min_free_bytes: 1073741824`, `min_free_inodes: 128`, no
snapshot caps or acquisition grants. Workflow `limits` per root:
`{max_training_updates: 8, max_target_positions: 128, max_eval_batches: 1}`.
The evaluation limit is **per pass**, not a cumulative ledger. The table accounts
for cumulative work. These admissions are checks, not resource allocation quotas.

Proposed native monitor policy, copied unchanged from the pinned Workshop example:
600 seconds per command, 4 GiB process-tree RSS, 1 GiB added workspace bytes,
1-second sampling, 5-second termination grace. Use one persistent storage baseline
per root so repeated commands do not reset growth accounting. Native monitor limits
are observed stop policies, not kernel hard quotas. Do not parallelize the two roots.
Retain every failed/cancelled command and its monitor completion. The parent must
accept these time/storage caps explicitly along with the model work before launch.

## Public command sequence

The following is a runbook, not an automatic launcher. Set `$R` to the committed
recipe's `examples/synthetic-cpu`, `$W` to the selected fresh external root,
`$P` to the pinned CPU interpreter, and `$LAB` to the clean lab checkout.
Set CPU thread counts to 1, hide accelerators, and enable HF/Transformers/UV offline
flags. Set `SPARSELAB_WORKER_LEASE_DIR=$W/worker-leases` in each command environment
so leases stay in this isolated external root; do not change global settings. Use unique output/log paths; never overwrite earlier evidence.

```sh
"$P" -m sparselab experiment recipe preparation inspect "$R/preparation.json" --bindings "$W/preparation-bindings.json" --json
"$P" -m sparselab monitor-baseline "$W" --output "$W/storage-baseline.json" --json
"$P" -m sparselab monitor --policy "$R/monitor.json" --workspace "$W" --baseline "$W/storage-baseline.json" --log-dir "$W/logs/preparation" --json -- "$P" -m sparselab experiment recipe preparation apply "$R/preparation.json" --bindings "$W/preparation-bindings.json" --admission "$W/preparation-admission.json" --work-dir "$W" --json
```

Preparation bindings explicitly map `tokenizer-config.json` output/cache and
`run.json` tokenizer/cache/logging fields below `$W/payload`. Admission is authored
from inspect output, not guessed. Capture the inner command's native Campaign
JSON as `$W/preparation-result.json` (from monitor stdout log, not its completion
summary). Require both preparation stages COMPLETE. Then author input identities:

```sh
"$P" "$WORKSHOP/scripts/seal-inputs.py" --receipt "$W/preparation-result.json" --recipe "$R" --bindings "$W/workflow-bindings.json" --logging-root "$W/payload/runs"
"$P" -m sparselab experiment recipe workflow inspect "$R/workflow.json" --bindings "$W/workflow-bindings.json" --json
"$P" -m sparselab experiment recipe workflow resolve "$R/workflow.json" --bindings "$W/workflow-bindings.json" --max-runs 1 --json
```

Commit/review the new authored files. Write fresh current workflow admission using
that commit, inspect digests and the limits above. Run each command through the
same `monitor` prefix, replacing the unique log directory each time:

```sh
"$P" -m sparselab experiment recipe workflow apply "$R/workflow.json" --bindings "$W/workflow-bindings.json" --admission "$W/workflow-admission.json" --work-dir "$W" --json
"$P" -m sparselab experiment recipe workflow approve "$R/workflow.json" approve-model --bindings "$W/workflow-bindings.json" --admission "$W/workflow-admission.json" --work-dir "$W" --note "Approved bounded current Workshop CPU acceptance" --json
"$P" -m sparselab experiment recipe workflow resume "$R/workflow.json" --bindings "$W/workflow-bindings.json" --admission "$W/workflow-admission.json" --work-dir "$W" --execute-runs --max-wait-seconds 600 --json
```

The first apply must stop at the approval gate. Approval follows operator review
of that exact plan/runtime evidence; the written runbook does not grant it.
Require all nine stages COMPLETE and one ingested completed attempt. Select the
lock/workspace from `lock-plan` availability and the terminal index from
`final-evidence` availability, never a guessed `latest`. Write reports with:

```sh
"$P" -m sparselab experiment recipe workflow report "$LOCK" --workspace "$EXPERIMENT" --index "$INDEX" --output "$W/report-before.json" --json
"$P" -m sparselab experiment recipe workflow report "$LOCK" --workspace "$EXPERIMENT" --index "$INDEX" --output "$W/report-before.md" --format markdown --json
```

Before replay retain the native controller attempt rows, all Campaign receipts,
selected index, pilot/main manifests and monitor records. Repeat the exact completed
`workflow resume` command once under a fresh monitor log directory. Fetch the
returned terminal index again and write new report paths. Check all controller
attempt rows (including failures) and receipt bytes unchanged; no new run/pilot
folders, runtime probes or evaluation records. Do not feed the before index twice
as a substitute for this observation. Then:

```sh
"$P" "$LAB/tools/check_workshop_repeat.py" replay --left "$LOCK" "$EXPERIMENT" "$INDEX_BEFORE" --right "$LOCK_AFTER" "$EXPERIMENT_AFTER" "$INDEX_AFTER"
```

Repeat the entire preparation/authoring/run sequence independently in B, with no
copied payload/cache/controller state. Native cold verification must succeed in
both roots. Compare the selected verified terminal snapshots:

```sh
"$P" "$LAB/tools/check_workshop_repeat.py" repeat --left "$LOCK_A" "$EXPERIMENT_A" "$INDEX_A" --right "$LOCK_B" "$EXPERIMENT_B" "$INDEX_B"
```

The read-only tool reopens native locks and evidence; it never launches or collects
missing work. It requires complete single-attempt coverage, matching scientific,
source/config/data/suite identities and declared budgets, distinct fresh attempt/run
IDs, and finite diagnostic loss/perplexity within **absolute or relative 1e-6**.
This tolerance is declared before execution. Replay requires identical selected
attempt/evidence/provenance projections. Native selected snapshots cannot prove
absence of unselected attempts; the isolated-root controller inventory check above
is mandatory. Tests use synthetic report projections and a substituted verifier
refusal; they are not evidence of an actual current-baseline run.

## Evidence and cleanup

Deliver both full pin sets, clean Git status, exact command argv/environment,
monitor completions (wall/RSS/growth/descendants/violations), admissions, approval,
preparation identities, all attempts including pilots, final checkpoint-bound
results, terminal native indices, JSON/Markdown reports, replay inventories and
checker outputs. State verification scope (e.g. weights-only) truthfully. Different
run IDs, timestamps and locations are expected across fresh roots; no history is
rehashable to force equality. A repeated synthetic single-seed result proves
engineering repeatability only, not model quality or a controlled improvement.

Use native cancellation/SIGTERM if needed, wait for descendants to exit and retain
interrupted evidence. Do not delete or overwrite either root during acceptance.
After parent verifies the retained evidence and approves cleanup, remove only the
new disposable recipe checkouts/work roots named in this plan. Old experiments,
RUN12 and runner holds remain untouched. TinyStories, paired real-data CPU/GPU
portability, current recipe publication and a tagged release remain separate work.
