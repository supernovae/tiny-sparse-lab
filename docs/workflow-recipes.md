# External prepared-input workflow locks

This bounded LAB-73/LAB-74 PR B slice connects an inventoried external recipe to
native artifact verification and lock publication. It starts **after native
preparation has produced a tokenizer and prepared-data artifact**. It does not
run a preparation graph or submit training. `portable_workflow_version: 1` is a
separate profile from the unchanged `portable_recipe_version: 1` matrix preview.

The [CPU investigation](portable-cpu-investigation.md) records partial LAB-73/74
verification and the unresolved execution-budget decision.

Use three explicit operations:

```sh
sparselab experiment recipe workflow inspect recipe/workflow.yaml --bindings local.yaml --json
sparselab experiment recipe workflow resolve recipe/workflow.yaml --bindings local.yaml --json
sparselab experiment recipe workflow publish recipe/workflow.yaml --bindings local.yaml --workspace /absolute/work/experiment --json
```

`inspect` reads bounded metadata only, validates native config expansion and
reports artifacts as unverified and runtime capability as unknown. It does not
open payloads, load optional computation, probe a runtime, create a work root or
launch a process. `resolve` cold-verifies the already available artifacts and
calls native `resolve_plan`; it does not publish. `publish` resolves and uses
native `publish_lock`, including no-clobber availability sidecars and artifact
reverification. All three report `executable: false`. Global work/token-file
options are ignored by this recipe route; publication requires its own explicit
`--workspace`.

## Portable descriptor and local bindings

The descriptor directory is the explicit recipe root, regardless of an enclosing
Git checkout or current directory. Example (replace every digest and identifier
with the real native identity):

```yaml
portable_workflow_version: 1
id: prepared-width-study
entrypoint: plans/plan.yaml
contracts: {experiment_plan: 1, run_config: 2}
files:
  - {path: plans/plan.yaml, role: experiment_plan, sha256: <exact file SHA-256>}
  - {path: configs/run.json, role: run_config, sha256: <exact file SHA-256>}
artifacts:
  token:
    kind: tokenizer
    version: 1
    producer: sparselab tokenizer train
    identifier: tokenizer
    sha256: <native tokenizer SHA-256>
  packed:
    kind: prepared_data
    version: 1
    producer: sparselab data prepare
    identifier: <native settings SHA-256>
    sha256: <native manifest SHA-256>
```

`plans/plan.yaml` is an ordinary `plan_version: 1` declaration with
`base_run: configs/run.json` and
`inputs: {tokenizer: token, prepared_data: packed}`. References are recipe-root
relative and must be inventoried. The profile supplies native Artifact values
in memory from the identity and binding documents; it writes no intermediate
plan YAML and duplicates no preparation or verifier logic.

The separate local document can live outside the recipe:

```yaml
workflow_bindings_version: 1
artifacts:
  token: /data/tokenizer/tokenizer.json
  packed: /data/cache/native-prepared-id
logging_root: /work/runs
config_paths: {}
```

Local relative paths anchor to the **bindings document's directory**; absolute
paths are permitted. Traversal, ambiguous spellings and symlinks are refused.
The inventory retains the shared limits of 64 members, 1 MiB per document and
8 MiB total recipe metadata; the separate bindings document is limited to 1 MiB.
Locations need not exist for inspection. Native verification requires them at
resolution. Tokenizer path comes from its artifact binding, cache directory from
the prepared artifact's parent, and logging root from its explicit binding.
`config_paths` must supply exactly any other present native typed input paths,
such as snapshot train/validation/manifest locations. Unknown fields and
capability assertions are rejected. Native verifiers still decide whether these
locations and artifacts agree with the scientific config; bindings cannot
substitute for verification.

Config documents retain canonical relative path spellings. Only supported
model-owned Path fields are bound; arbitrary metadata/plugin `path` keys remain
opaque. Path-changing axes (including whole-section patches) and mixture paths
are unavailable. The first profile requires exactly one tokenizer and one
prepared-data identity. Additional artifact kinds, corpus variants, authored
phases, inline evaluations, attempt contracts and worker/workspace declarations
are explicitly refused. An inventoried diagnostic suite is supported only with
the explicit Campaign contract below. Native APIs retain their existing capabilities.

## Identity, context and admission

The new recipe fingerprint is SHA-256 of `sparselab-portable-workflow-v1\0` plus
canonical validated descriptor JSON, with inventory sorted by path. It includes
member byte hashes and typed artifact identities. Descriptor formatting has a
separate byte digest. Bindings have their own byte digest and are excluded from
the recipe fingerprint. This does not define new scientific equivalence: native
scientific/config/lock hashes and backend/precision distinctions remain intact.
Legacy seed coercion (`true`, `"7"`, `7.0`) is preserved. Version fields in the new
contracts reject booleans. No historical hashes or records are rewritten.

An immutable RecipeContext carries inventoried bytes and explicit local config
bindings into the native planner through `recipe_run_config` and `resolve_plan`. The
pinned legacy `base_run_config` function stays byte-for-byte unchanged; calls
without a context retain historical declaration lookup and serialization. Contextual
resolution records recipe-root inventory digests in availability, without Git or
CWD fallback. The workflow annotation records descriptor/binding identity and
locations in the native availability sidecar. Adding that annotation forfeits
the resolver's object-local proof reuse: publication deliberately reverifies.
No verifier exclusions change.

**A published workflow lock is not an execution grant.** Both the experiment run
route and the shared CLI/Campaign worker-request constructor refuse this profile
with `WORKFLOW_EXECUTION_UNAVAILABLE` unless the shared request constructor receives
the authenticated CPU execution context described below, before selecting/submitting work. Unknown
or stale capability information cannot make it ready; there is no capability
input or override. Existing legacy admission and recovery paths are unchanged.

## Change-impact map and remaining work

| Boundary | Tests |
| --- | --- |
| Descriptor/inventory/local bindings | Exact digests, inventory order, strict versions, changed content, traversal/symlinks, opaque metadata, unknown binding/capability fields |
| Explicit context → native planner | Snapshot versus changed disk, decoy source directory/CWD, native identity parity, legacy recipe/native-plan regressions |
| Metadata versus artifact operations | Fresh-process CLI/help import/write/network/process/payload fences; unsupported features refused before optional imports |
| Native verification → publication | Tiny synthetic tokenizer/packing fixtures, changed-artifact refusal before publication, native reopen, immutable availability collision |
| Lock → execution boundary | CLI selection spy and shared Campaign/CLI request refusal; no worker submission or runtime probe |
| Compatibility | Legacy seeds, native science under changed logging binding, verifier authority, declaration preflight, normal safe/fast/tiny CPU lab-loop CI |

Remaining work includes recipe context through preparation graphs and evaluation
closures, broader artifact/operation contracts, committed recipe provenance at
admission, existing runtime-binding/capability evidence, execution/recovery and
run-evidence linkage. Those must connect to CampaignEngine/Controller and their
current gates; this profile is not a replacement executor. Actual TinyStories
CPU/GPU portability proof, publication to Workshop and any release remain
separately admitted later work. Full LAB-73/LAB-74 acceptance is still partial.

## Refusal and guidance follow-up

The focused guards also cover binding documents overlapping the descriptor or
inventory, artifact kinds at input ports, execution workspace/worker fields,
base-config inventory role, undeclared native path bindings, context bindings for
uninventoried configs, and direct native contextual resolution of unsupported
variants/suites/attempt contracts. Exact errors and read/verifier spies distinguish
these guards from incidental failures. Overlap is rejected before interpreting a
file as local bindings. Member-role lookup has an explicit `None` default;
uninventoried references still fail earlier in `RecipeContext.reference`.

The nested `workflow_profile.execution: unavailable` value is explanatory. The
existing execution authority tests the **presence of `workflow_profile`** in the
authenticated availability sidecar. Removing the explanatory value does not grant
execution; no test treats that informational mutation as a security failure.

A published-profile iteration check reconstructs descriptor-supplied declarations
from the authenticated sidecar, verifies descriptor/binding/inventory identities,
and returns the explicit `WORKFLOW_EXECUTION_UNAVAILABLE` gate. The former
`DECLARATION_OR_LOCK_INVALID` regression was replaced with this stronger boundary;
malformed or changed declarations still retain their own errors. Inventory
integrity does not establish committed provenance. CLI/shared Campaign execution
refusal remains unchanged. See [preparation planning](preparation-recipes.md) for
the new separate inspect-only contract and remaining admission/context work.


## Prepared-input CPU execution

The narrow execution contract adds optional `campaign: campaign.json` to the
workflow descriptor. Inventory that native Campaign and its evaluation suite as
`metadata` members. The Campaign owns the connections: two artifact references,
one experiment-plan lock, runtime acceptance, explicit approval, one
`main:single` run, collection and evaluation. Stage IDs are authored, not fixed.
Artifact-reference IDs name the descriptor artifacts; their `artifact` objects
contain exactly the descriptor identity fields, with no path. Locations come
only from local bindings. The experiment-plan stage references the inventoried
entrypoint; its suite references one inventoried diagnostic `heldout_lm` suite.
The existing typed Campaign validator checks dependencies and producer kinds.

`WorkflowExecution` accepts the descriptor, bindings, absolute work root outside
Git and a `WorkflowAdmission` containing version 1, the reviewed recipe commit,
recipe and binding digests, and current CPU Python. These are explicit caller
assertions, not remote author attestation. It requires unchanged clean committed
inventory, one unexpanded synthetic CPU PyTorch cell, and local native runtime
acceptance. Profile/worker selection, accelerator evaluation, axes, comparisons,
continuations, acquisition and alternate evaluation kinds remain unavailable.

```python
from sparselab.experiments.workflow_execution import (
    WorkflowAdmission,
    WorkflowExecution,
)

context = WorkflowExecution(descriptor, bindings, work_root, admission)
engine = context.engine()
planned = engine.apply()  # verifies inputs, publishes native lock, awaits approval
engine.approve(gate_id, "approve", "explicitly authorized bounded CPU experiment")
completed = engine.apply(execute_runs=True, max_wait_seconds=540)
replayed = engine.apply(resume=True, execute_runs=False)
```

Use native operational monitoring around the complete process tree. A one-update
main config does not bound pilots or runtime probes. The measured fixture uses
600-second, 4-GiB RSS and 1-GiB workspace-growth stop ceilings; sampled monitoring
is not an instantaneous kernel quota. No new report CLI or scheduler is added.
The existing collection and publication-report APIs read the resulting native
lock, Controller evidence and checkpoint-bound evaluation. Rendering conveys
integrity and coverage, not scientific effectiveness or automatic promotion.

Context is refreshed at admission, approval, every dispatch and receipt boundary,
and the shared locked-cell request. The native availability sidecar authenticates
recipe, inventory and binding identity; Campaign receipts and submitted request
provenance retain the explicit admission. A changed committed recipe or local
binding cannot be admitted by replaying old approval receipts. Ordinary
`experiment run` and iteration remain unable to grant this execution context.
Existing workflow descriptors without `campaign` retain their original recipe
fingerprint; historical config, scientific, lock and receipt hash algorithms and
legacy seed coercion are unchanged.

| Change / boundary | Regression evidence |
| --- | --- |
| Changed descriptor, inventory, binding, suite or reviewed commit | Refusal before dispatch and work-store creation |
| Authored Campaign connections | Native typed edges plus required explicit approval |
| Native lock/request authority | Real plan-to-approval path; lock without context refuses; stale binding refuses shared request |
| Metadata inspection | Fresh-process import, filesystem, subprocess and payload fence |
| Evaluation declaration separation | Existing schema/runtime readers share the unchanged data-only models |

The bounded actual CPU proof is additional wiring evidence. TinyStories,
accelerator portability, broader contracts and scientific comparison qualification
remain separately scoped work.
