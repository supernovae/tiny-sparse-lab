# Implementation ladder

Each row is a separate bounded change with its own focused tests. Do not build
all rows before obtaining the preceding stage's evidence.

| Order | Component | Input → output | Validation |
|---|---|---|---|
| 0 (implemented) | `substrate.py compile/query/replay` | Local Git Python source → integrity-bound AST bank → typed observations | Tamper/error/real-source/replay tests |
| 1 | `task_bundle.py` | Human-authored cases + independent scorer labels → sealed train/dev/test manifests | Module/composition ownership, duplicate/leakage checks; no labels in model view |
| 2 | `controller.py` | Explicit local model + user task + ABI → bounded action/result loop and final evidence claims | Malformed action, absent/ambiguous records, timeout, abstention; checkpoint identity |
| 3 | `evaluate.py` | Frozen manifest + arm traces → per-case scores and paired report | Independent scoring, failure denominators, token/time accounting, deterministic coverage |
| 4 | `export_traces.py` | Verified train-owned trajectories → v2 local-chat JSONL + provenance manifest | Actual repo schema validation; tools/IDs/order/masks; no dev/test targets |
| 5 | `train_policy.py` or native recipe | Explicit supported model path + train traces → tuned policy | Resource smoke, trainable ownership, frozen endpoint, matched tuning control |
| 6 | `compare_snapshot.py` | Two verified source banks → added/removed/changed signatures | A→B→A and identical-snapshot checks; no retraining |
| 7 | `orbit_package.py` | Unit-bearing records + simulator parameters → trajectories/diagnostics | Analytic circular orbit, convergence, conservation, unit errors |
| 8 | `diagnose.py` | Symptoms + hypotheses + interventions → observation ledger and resolution | Held-out faults/combinations, unknown causes, no hidden labels exposed |
| 9 (optional) | structure encoder/adapter | Graph records → encoded queries/residuals | Serialized/latent, random/permuted, cost and recipient-compatibility controls |

## Connecting existing lab components

- `docs/instruction-training.md`: local-chat v2 trace format and assistant-only
  masking. Tool messages are inert until an explicit executor is added.
- `docs/research/compiled-knowledge-engram-v1.md`: keep compiled record use
  distinct from acquisition through SGD. This ABI pilot is a new branch of that
  question, not a change to the documentation-only Engram experiment.
- `docs/research/learned-engram-portability.md`: preserve Experiment A and its
  calibrated recipients. Reuse identity/control discipline, not its synthetic
  results as evidence for semantic composition.
- `src/sparselab/engram/semantic.py` and `docs/research/semantic-memory.md`:
  pre-encoded semantic interfaces are potential Stage 5 comparators, not a ready
  natural-language or AST encoder. Equal dimensions do not prove alignment.
- Corpus Forge: reuse provenance/split/release semantics when exporting larger
  licensed sources. The Stage 0 bank intentionally uses only local `src/**/*.py`.
- Research campaign/readiness: integrate only once there is a real sealed model
  protocol. Do not add a catalog entry implying this pilot trains a controller.

## Continuity loop

Keep one durable record per stage: `question → intervention → controls → sealed
gate → evidence → human decision → next test`. A blocked stage remains blocked;
new ideas enter a parking list or new protocol, not the running experiment.

The first remaining coding task is rows 1–3. Before any training, produce a
human-readable side-by-side report of actual questions, actions, cited source,
expected/scored answers and failures. If exact retrieval already solves a task,
say so; the neural policy earns its place through language interpretation or
composition, not through hiding a lookup inside a fluent answer.
