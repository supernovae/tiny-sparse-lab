# External prepared-input workflow locks

This bounded LAB-73/LAB-74 PR B slice connects an inventoried external recipe to
native artifact verification and lock publication. It starts **after native
preparation has produced a tokenizer and prepared-data artifact**. The metadata
operations do not run a preparation graph or submit training; explicit CPU
execution is described below. `portable_workflow_version: 1` is a separate profile from the unchanged `portable_recipe_version: 1` matrix preview.

The [CPU investigation](portable-cpu-investigation.md) records the earlier
partial LAB-73/74 verification. The execution and CLI sections below describe
the current bounded contract.

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

Remaining work includes broader artifact/operation contracts, independent
Workshop authoring and measured portability at its final pins. The preparation
and authenticated synthetic CPU contracts below connect to the existing
CampaignEngine/Controller and their gates. Actual TinyStories CPU/GPU portability,
scientific comparison qualification and any release remain separately admitted.

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
is not an instantaneous kernel quota. The CLI below exposes the native executor
and report APIs without adding a scheduler.
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


## Authenticated workflow CLI

Run from the provisioned CPU interpreter selected in admission. These commands
require a separately admitted workload. **`apply` without `--execute-runs` may
still run native runtime acceptance probes.** Omitting the flag prevents model
run dispatch; it does not make `apply` an inspection-only command. Include these
probes in the admitted resource envelope and monitoring. Help and metadata
`inspect` remain non-executing. Preparation remains a separate
`recipe preparation apply/resume` operation; workflow execution consumes already
verified native artifacts.

```sh
sparselab experiment recipe workflow apply recipe/workflow.yaml --bindings local.yaml --admission admission.json --work-dir /absolute/work --json
sparselab experiment recipe workflow approve recipe/workflow.yaml gate --bindings local.yaml --admission admission.json --work-dir /absolute/work --note "authorized bounded CPU experiment" --json
sparselab experiment recipe workflow apply recipe/workflow.yaml --bindings local.yaml --admission admission.json --work-dir /absolute/work --execute-runs --max-wait-seconds 540 --json
sparselab experiment recipe workflow resume recipe/workflow.yaml --bindings local.yaml --admission admission.json --work-dir /absolute/work --json
```

Replace `gate` with the authored approval-stage ID. `approve --decision reject`
records a native rejection. No command implicitly approves. Each command reloads
and validates the descriptor, local bindings and strict admission document:

```json
{
  "workflow_admission_version": 1,
  "reviewed_commit": "<full committed recipe SHA>",
  "recipe_sha256": "<recipe_sha256 from workflow inspect>",
  "binding_sha256": "<binding_sha256 from workflow inspect>",
  "cpu_python": "/absolute/provisioned/environment/bin/python"
}
```

Use the exact running interpreter path (`sys.executable`), not a guessed alias.
The reviewed commit is the **recipe repository** commit, distinct from the lab
software pin. Admission is a caller assertion, not automatic approval, compute
credit or remote author attestation. Keep admission and bindings outside the
committed recipe; require an absolute work root outside Git. Global prefix
work/token options do not select execution resources on this route.

`apply` and `resume` default to `execute_runs=false` and a 120-second native
controller wait. `--max-wait-seconds` accepts finite nonnegative seconds; zero
can queue without ticking a worker. It is not a whole-process resource limit.
Use the existing `sparselab monitor --policy POLICY --log-dir LOGS --workspace
WORK -- COMMAND ...` around admitted work. Pilots and probes remain additional
work beyond the main run's update budget. `resume` delegates native Campaign
replay/recovery; it does not assert full-state training resume was verified.

Execution and report commands emit one JSON object to stdout (with or without
`--json`); adapter progress goes to stderr. The envelope is
`sparselab-workflow-command-v1`, with `command` and native result fields.
Validation/I/O errors and native FAILED stages exit 1. Argument errors exit 2
with argparse diagnostics on stderr. As with native Campaign commands, waiting,
blocked, rejected or running stages can exit 0: inspect stage states; exit 0
does not mean complete or scientifically successful. Existing metadata command
output is unchanged.

### Publication report

Select the exact native plan-stage `availability.path` (lock) and
`availability.workspace`. Select the collect-stage `availability.path` (evidence
index), **not** the evaluation-stage index. Report rendering never recollects or
selects a newer snapshot automatically.

```sh
sparselab experiment recipe workflow report /work/plan.lock.json --workspace /work/experiment --index /work/evidence/index.json --output /work/report.json --json
sparselab experiment recipe workflow report /work/plan.lock.json --workspace /work/experiment --index /work/evidence/index.json --output /work/report.md --format markdown
```

The example paths are placeholders: use paths from native results. Supply both
`--workspace` and `--index`, or neither for a planned-only report. The report
adapter accepts native locks; it needs no execution admission and grants none.
It verifies the selected snapshot, preserves incomplete/missing evidence, writes
exclusively, permits identical-byte replay and refuses a different existing
output. No training, evaluation, acquisition or controller tick occurs. A later
evidence snapshot requires explicit native collection; this command does not
refresh the Campaign's earlier collection receipt. Keep the evaluation index
alongside the report when recording completed evaluation evidence.

### Independently authored Workshop example

The subsequent Workshop change must use the final reviewed lab pin and its
locked, provisioned CPU environment. It must not import `tests`, copy fixture
builders as a runtime dependency, or introduce another executor. Commit:

- A native synthetic RunConfig, one-cell ExperimentPlan, diagnostic heldout suite,
  native Campaign and workflow descriptor with complete member hashes and ports.
- Small synthetic preparation declarations and documented native preparation
  commands. Extract tokenizer/prepared identities from native receipts; bind their
  locations locally. Commit and review the workflow only after its identities
  are known. Changed preparation/code identities require fresh verification and
  a new recipe identity, never rewriting historical receipts.
- Exact lab/runtime pins, licenses, intended workload including pilots/probes,
  monitor policy, setup/approval/run/report instructions and expected evidence
  paths. Machine-local bindings, admission and generated payloads stay outside Git.

Freeze the final installed lab source identity **before fresh preparation**.
Commit preparation declarations at A and run them with admission for A. COMPLETE
preparation JSON stages expose `outputs` (native `kind`, `identifier`, `sha256`)
and `availability.path` (local location). A standard-library authoring helper can
select the expected stage IDs/kinds, require COMPLETE, copy those native identity
fields into the workflow's version-1 artifact records with an explicit producer,
and place paths in local bindings. It must not invent identities for missing or
failed stages. Commit that independently authored workflow at B and admit B.
Retain the A checkout and exact local binding/admission bytes for preparation
replay: advancing its checkout to B invalidates A's admission. Artifacts measured
under an earlier installed lab source identity are not reusable evidence for a
new installed source identity. Preserve them unchanged.

Acceptance needs a clean committed Workshop checkout, public CLI preparation and
workflow execution, explicit approval, one native completed/ingested attempt,
checkpoint-bound evaluation, verified report and replay without a new attempt.
Record recipe/lab/runtime pins and measured wall/RSS/workspace growth separately
from quality claims. That measured run and Workshop publication are subsequent
work, not established by CLI unit tests. Snapshot TinyStories, multiple cells,
continuations, generation panels and accelerators remain outside this profile.

| CLI boundary | Evidence in `tests/test_workflow_cli.py` |
| --- | --- |
| Public entry → authenticated native Campaign | Real tiny preparation, approval gate, zero-wait queue, no-tick replay |
| Admission → store/dispatch | Recipe/binding digest, interpreter, outside-Git, clean committed inventory and stale context refusals |
| New admission → existing approval | Rebound binding with updated admission rejects old receipt before worker tick |
| Native result → stdout/exit | Single JSON result, stderr progress, failed-stage exit, finite wait parsing |
| Native snapshot → report | Planned/evidence output, identical replay, no-clobber, corrupt-index and unpaired-input refusal |
| Help → optional imports | Fresh-process fences for every new subcommand |

These tests address the admission coverage correction in
[PR127 review 5512227409](https://github.com/supernovae/tiny-sparse-lab/pull/127#pullrequestreview-5512227409).
They preserve the existing guards rather than changing runtime contracts, legacy
seed acceptance or historical hashes. Normal safe/fast/lab-loop CI remains the
broader compatibility gate; no full measured training is added to CLI tests.
