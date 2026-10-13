# External prepared-input workflow locks

This bounded LAB-73/LAB-74 PR B slice connects an inventoried external recipe to
native artifact verification and lock publication. It starts **after native
preparation has produced a tokenizer and prepared-data artifact**. It does not
run a preparation graph or submit training. `portable_workflow_version: 1` is a
separate profile from the unchanged `portable_recipe_version: 1` matrix preview.

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
phases, evaluations/suites, attempt contracts and worker/workspace declarations
are explicitly refused. Native APIs retain their existing capabilities.

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
with `WORKFLOW_EXECUTION_UNAVAILABLE`, before selecting/submitting work. Unknown
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

An actual published-profile iteration check currently returns `BLOCKED` and an
`experiment explain` command, not `experiment run`. Its native declaration walker
cannot interpret the workflow entrypoint's descriptor-supplied artifacts, so it
reports `DECLARATION_OR_LOCK_INVALID` and `DELTA_UNRESOLVED`, alongside missing
pilot/runtime evidence. A read-only regression covers this outcome. No iteration
guidance code is changed here, and ordinary-lock guidance is unchanged. Dedicated
workflow-context guidance remains part of the future context/admission work; that
work must retain the profile execution authority before permitting readiness.
