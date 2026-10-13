# Portable CPU proof: execution boundary

`tests/test_portable_cpu_proof.py` characterizes how far the current portable
preparation and native model interfaces connect under a **one-update total CPU
test budget**. It stops before model dispatch. This slice does not establish a
complete dispatch → ingestion → evaluation → publication-report proof.

## What the regression executes

An external synthetic preparation recipe runs through `PreparationExecution` with
explicit reviewed commit, inventory/binding digests, current Python and bounded
document/token/storage limits. Its real output identities and locations feed a
**separately authored ordinary native Campaign fixture** in another temporary Git
repository. This is test plumbing, not a production conversion of a portable
workflow into a native declaration.

The native Campaign reaches runtime acceptance and a fixture approval, then stops
at `experiment_run` without `execute_runs`. Replaying preparation forbids adapter
redispatch. Replaying the downstream Campaign retains completed outputs and no
controller attempts. The report API renders an honest planned-only report:
`pending`, no attempts, scientific result `not_assessed`.

The fixture dimensions are the existing synthetic bounds: 8/4 document caps,
129/81 train/validation token caps, vocabulary 264, one layer, hidden width 16,
FFN width 32, sequence length 16, batch one, CPU/PyTorch/fp32. The proposed main
training config requests one update and evaluation one batch. The checked-in test
never starts either. The preparation test is selected in `lab-loop` and marked
slow; existing job limits and CPU/offline settings remain unchanged.

## Discovered workload blocker

`workers/execution.py` calls native `stage(..., through="warmup")` before main
training. `StagingConfig` requires positive `smoke_steps` and `warmup_steps`.
The existing fixture requests two smoke plus five warmup updates, independently
of the one-update main budget. Even reducing both pilots to one would require
three optimizer updates in total, in addition to preceding runtime validation.
Preparation approval and `training.max_steps=1` do not cap this transitive work.

During local development, one attempted integration reached those stages before
this mismatch was recognized: native reports recorded smoke step 2, warmup step
5 and main step 1. The test driver was interrupted; that attempt is **not** a
passing end-to-end proof. The native terminal receipt was `COMPLETE`, but
controller ingestion was still `PENDING`; suite evaluation, collection and the
completed report were not reached. No further model executions were launched. This
characterization replaces the attempted smoke in automatic CI; it does not lower
pilot requirements, synthesize runtime evidence or bypass native safety policy.

The next runtime choice is explicit: approve a bounded pilot-inclusive CPU proof,
or separately review a native admission change that can honor a one-update total
budget. No TinyStories, GPU, download, provider or release authority follows from
these fixtures.

## Change-impact and remaining code gaps

| Fence | What it establishes |
| --- | --- |
| External preparation → native declaration → ready run | Actual preparation identities bind native plan/runtime gates |
| Preparation and native Campaign replay with dispatch forbidden | No duplicate preparation and no model attempt inferred from approval |
| Changed recipe, bindings or inventoried config | Refusal before dispatch or store creation |
| Portable lock request refusal | Inventory integrity cannot grant model execution |
| Portable evaluation-suite refusal | The missing context/admission connection remains visible |
| Planned-only report | Missing execution is pending, without fabricated observations |
| Positive pilot schema bounds | One main update is not the total native workload |

Portable workflow v1 still needs an admitted evaluation-suite subset in
`workflow_recipe.load_workflow` and context-aware suite resolution in
`lock.resolve_plan`. Committed inventory provenance must then flow through native
`declaration_preflight`, `locked_cell_request` and iteration gates. Removing the
profile guard or publishing an unmarked native lock is not that implementation.
The shared model-execution admission contract and its binding to native runtime
and attempt identity remain design work. A heldout-only synthetic subset is the
proposed first scope; broader routes remain unavailable.

No production schema, hash algorithm, legacy seed acceptance, evidence reader,
pilot policy, runtime capability or runner hold changes here. The report CLI,
Workshop, website and release remain separate work. A later successful native
smoke would still be engineering evidence, not TinyStories qualification or a
scientific comparison.
