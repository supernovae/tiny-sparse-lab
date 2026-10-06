# Capacity-aware execution and verified reuse

Keep execution efficient without changing the declared experiment. The lab can
reuse authenticated inputs, bound preparation work to available host resources,
and materialize private worker/stage inputs while preserving scientific identity.
Use these native paths instead of custom hashing, copying or orchestration scripts.

## When to use it

Repeated iterations often share a corpus, tokenizer and prepared arrays. Rebuilding
those inputs or repeatedly hashing their full unchanged dependency closure can
cost more than the changed training phase. Supported native verification boundaries
reuse signed local proofs when eligible, while retaining input and source checks.
Reuse is an execution optimization, not permission to change data, precision,
effective batch, architecture or the training budget.

| Situation | Verification behavior |
| --- | --- |
| New input, changed identity or unsupported reuse boundary | Cold byte and domain verification. |
| Unchanged input in an authenticated owner-only registered store | Eligible native commands can reuse signed host-local proofs. |
| Transfer to another worker/store trust domain | Authenticate bytes and semantics before publishing the destination. |
| Archive, recovery or model-family verification | Independent cold verification. |

Changed fingerprints, dependencies, implementation identity, unsafe paths,
missing/foreign proofs or invalid signatures require cold fallback. A failed
cold check is an error. Same-user compromise is outside the protection offered
by local proofs. File size, remembered hashes and directory presence are not
substitutes for verification.

## Inspect through the lab

Use [iteration checks](iteration.md#read-only-iteration-check) for the declared
delta, input bindings, parent compatibility, storage and runtime prerequisites.
Use `experiment explain` for a resolved lock and `campaign status`, `next` and
`explain` for orchestration state. These read paths do not launch training.

Supported ExperimentPlan operations and `stage` expose `--cold-verify`; use it
at a new trust boundary or when independently checking retained bytes. There is
no Campaign-wide `--cold-verify` switch. Reuse and cold-fallback diagnostics belong
to operational evidence, outside scientific/plan identities.

## Bound preparation and materialization

The [runtime resource envelope](runtime.md#operational-resource-envelopes)
controls native host work. Worker counts are bounded by available CPU capacity,
RAM, per-worker requirements and explicit limits. Results are reduced in input
order so concurrency does not redefine prepared data. Unknown capacity is
reported conservatively rather than treated as unlimited.

Stage and worker materialization verify source and destination identities and
publish private copies. Native clone/copy support depends on the destination;
unsupported operations fall back to bounded copying. Physical storage placement
and worker choice remain operational settings. Follow [workspace policy](workspaces.md)
for capacity, ownership, relocation and retention.

## Interpret measurements

Keep preparation, verification, copying, optimizer updates, evaluation and
checkpointing timings separate. Record logical bytes independently of physical
I/O, sampled memory independently of native peaks, and unavailable counters as
unavailable. Avoid treating an isolated copy or hash improvement as a full-run
speedup or model-quality result.

Detailed phase observation currently has an internal API but no general native
switch; the [diagnostic backlog](../TODO.md#native-diagnostic-interfaces) describes
that adapter. The [retained development evidence](research/development-evidence.md#capacity-execution-measurements-and-implementation-notes)
contains implementation details, measured comparisons and corrections from prior
workloads. Those workloads do not define the reuse interface or your experiment.
