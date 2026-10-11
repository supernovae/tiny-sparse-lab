# Launch contracts

Use `try DELTA.yaml --vs BASELINE.yaml` for a local comparison. Use `train`
for explicit checkpoint transitions, and `experiment submit` for independent
queue dispatch. Waiting is a submission option: `experiment submit CONFIG
--wait`. The queue, worker protocol and receipt ingestion are shared with
non-waiting submissions; [workers](workers.md) describes their operation.

## Contract and caller audit

The audit covers the eight requested executable command paths at revision
`c6246c5211aac9bad6388c16b6bd22be04de1e69`. This change removes top-level `run`
from the parser, help groups and dispatch; no alias remains. The eight paths
become seven (nine become eight when counting `try`). Generic queue launch
choices decrease from two to one. These counts describe public command paths,
not training engines, worker transports or all executable commands in the CLI.

| Path | Inputs, outputs and active callers | Decision and remaining gap |
| --- | --- | --- |
| `train` | RunConfig plus explicit resume, extend-budget, promote or recover checkpoint; stage bundle, runtime authorization, spot policy and resource envelope. Native trainer emits checkpoint/manifest/evidence. Used by training/checkpoint guides, readiness subprocesses and supervised attempt commands. | Retain. Generic submission does not express all direct checkpoint transitions, run-ID selection, stop-after-step and spot-policy obligations. |
| `run` | RunConfig, optional registered worker or authorized local runtime, stage/promotion binding and resource limits; submission plus terminal/ingestion result. Active public examples were workers and hosted-environments guides; runtime-selection and retired-dataset tests exercised its parser/preflight. | Remove. All those inputs and waiting semantics are handled by `experiment submit`; active examples and tests use that route. |
| `stage` | RunConfig, explicit output, inspect/validate/smoke/warmup gate, optional already-prepared bundle, cold verification, pilot deadline and observation destination. Emits a reusable stage bundle without advancing eventual training state. Used by readiness, iteration/runtime/corpus guides and preparation phase commands. | Retain. Queue staging is part of eventual execution, not an independently requested pilot receipt with these output and verification controls. |
| `experiment submit` | Single config or matrix, worker requirements, stage/promotion binding; native queue submission identities. Matrix expansion preserves coordinates and atomic submission. Used by worker/matrix guides and now hosted wait examples. | Retain as the generic queue route. Single-config waiting and runtime/resource controls share one config submission adapter. Matrix waiting/runtime overrides are rejected before execution preflight. |
| `experiment run` | Immutable ExperimentPlan lock, cell/phase, runtime binding or worker; authenticates declaration provenance, evaluation declaration, artifacts and parent completion. Emits plan SHA and submissions carrying locked-cell/runtime identities. Used by experiment-program guides and iteration next-action recommendations. Campaign uses the underlying locked-cell request and controller APIs. | Retain. Generic submission cannot consume a lock, select dependency-ready cells, authenticate bound runtime receipts or emit the same release evidence. Removing it requires those exact inputs and provenance checks in the surviving route. |
| `study submit` | Architecture study, lifecycle diagnostics, optional receipt destination and worker/store; writes an exclusive study receipt containing study/matrix/card digests and coordinate-to-submission identities. Used by architecture guides and generated research scaffold instructions; collect/report consume that receipt. | Retain. Raw matrix submission lacks study validation, card identity and the authenticated receipt required by collect/report. A replacement must preserve exclusive receipt publication and expose already-submitted IDs on publication failure. |
| `attempt run` | Immutable ledger, content identity, monitor policy, workspace baseline, explicit update/target/generation reservations, command and native/Campaign receipt bindings. Reserves persistently, supervises the owned process tree and reconciles charged work. Used by release attempt packets, preparation compiler output and contract tests. | Retain. Queue dispatch has no corresponding cumulative ledger, arbitrary native phase, receipt accounting or zero-survivor completion contract. It cannot replace this supervisor. |
| `attempt run-phase` | Canonical preparation plan, phase label, bindings and owned attempt root, plus ledger/policies/baseline. Compiles authenticated phase arguments and delegates through the native attempt parser/handler with zero model reservations. Used by preparation guides, phase-command tests and release preparation packets. | Retain. `attempt run` alone does not accept the canonical plan/phase inputs or enforce their compilation; replacing this adapter requires a typed phase mode with the same authenticated argument and supervision obligations. |

Historical research records, source pins, receipts and evidence remain unchanged.
The native controller still owns scheduling, cancellation, resume and ingestion;
this command consolidation grants no spending, execution or release approval.
`try` remains the beginner front door. Remaining consolidation must close the
specific contracts above before deleting additional commands. Everyday options
and result-identifier design are separate work.

## Verification boundary

Parser and mocked adapter checks cover removal, runtime conflicts, early rejection,
option forwarding, local versus registered worker selection, terminal failure,
controller lock ownership, reconnection/ingestion, Colab UNKNOWN handoff and
passive Ctrl-C. Focused native queue, attempt-ledger and preparation-command
checks cover durable cancellation, atomic submission, retained accounting and
owned-tree shutdown without model execution. They do not qualify a runtime,
train a model or establish scientific results.
