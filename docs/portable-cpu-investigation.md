# Portable CPU investigation: partial LAB-73/74 verification

This PR resolves no backlog card and proves no end-to-end portable execution.
Its retained regressions are small extensions of the existing contract tests:

| Card / boundary | Retained verification | Still missing |
| --- | --- | --- |
| LAB-73 inventory and bindings | Existing workflow test exercises a syntactically valid unsupported suite reference before optional imports or work | Admitted evaluation references through the same explicit recipe context |
| LAB-74 preparation admission | Changed descriptor, bindings or inventoried config after context creation refuse before dispatch or store creation | Portable model execution, native attempt/replay identity and real dispatch-to-report proof |
| LAB-75 | No change; the stopped test's planned-only report check was removed | This investigation supplies no completed execution evidence |

The existing native workflow execution-refusal tests remain in place. This adds
no new execution route, public schema, hash/seed policy or CI workload selection.
The large stopped integration and numerical pilot assertions were removed:
current missing capabilities and pilot defaults are not permanent requirements.

## Investigation finding and retained evidence

`workers/execution.py` calls `stage(..., through="warmup")` before training.
`staging.pilot_config` derives each pilot's budget from its staging setting,
independently of the main training budget. Runtime validation also performs toy
optimizer probes. A one-update main request therefore does not bound total work.

One development attempt exceeded the assigned envelope because this transitive
work was missed before dispatch: smoke 2, warmup 5, main 1 model-training updates.
The driver was interrupted. Worker receipt was COMPLETE, controller ingestion
PENDING; suite evaluation, collection and completed reporting were not reached.
No subsequent model proof was run. Exact run IDs, timings, resource readings,
retained paths and cleanup state are in the [PR126 evidence addendum](https://github.com/supernovae/tiny-sparse-lab/pull/126#issuecomment-6157874827).
These are development observations, not acceptance thresholds or scientific claims.

The next owner decision is a separately bounded pilot-inclusive CPU proof, or a
reviewed native total-budget admission change that refuses before dispatch when
all requested work cannot fit. Neither is implemented here. Portable execution
also needs shared committed-inventory provenance and bounded evaluation/runtime
admission in the existing planner and request path. No second executor is needed.
TinyStories qualification, LAB-75 CLI, Workshop, website and release remain outside
this slice; historical hashes, seed acceptance and runner holds are unchanged.
