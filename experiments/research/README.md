# Research experiments

Each subdirectory represents one real campaign or experiment family. Check in
only the material needed to understand and reproduce the design:

- question, competing explanations, and falsifiable acceptance gates;
- immutable or content-addressed config/protocol references;
- declared data ownership, splits, seeds, controls, and budgets;
- iteration log explaining why the next run changed;
- status and links to verified evidence or reports;
- explicit negative, failed, censored, and unresolved outcomes.

Use a matching ignored workspace for execution:

```sh
WORK="$PWD/sparselab-work/experiments/<campaign>"
export SPARSELAB_WORK_DIR="$WORK"
mkdir -p "$WORK"
df -h "$WORK"
df -i "$WORK"
```

Pass `--store "$WORK/runs"` and `--receipt "$WORK/receipt.json"` to study submission, and `--runs-dir "$WORK/runs"` to collection. All seed, architecture, and budget coordinates and resumed children share that store. Keep staging, exercises, captures, and local reports under the same `WORK`. When relying on automatic study/scaffold workspace naming instead, set the global work-directory base to `sparselab-work` (or an external equivalent), not an already named experiment root.

Do not check in datasets, caches, checkpoints, run databases, raw logs, or
anonymous temporary paths. The durable record should reference verified evidence
without pretending that a hash establishes scientific validity.

Current long-form protocols and findings remain indexed from
[`docs/research/README.md`](../../docs/research/README.md). New campaigns can
adopt this directory incrementally; moving an established identity-bound config
is not required merely to satisfy the layout.
