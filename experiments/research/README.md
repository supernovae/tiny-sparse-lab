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
export SPARSELAB_WORK_DIR="$PWD/sparselab-work/experiments/<campaign>"
mkdir -p "$SPARSELAB_WORK_DIR"
df -h "$SPARSELAB_WORK_DIR"
df -i "$SPARSELAB_WORK_DIR"
```

Do not check in datasets, caches, checkpoints, run databases, raw logs, or
anonymous temporary paths. The durable record should reference verified evidence
without pretending that a hash establishes scientific validity.

Current long-form protocols and findings remain indexed from
[`docs/research/README.md`](../../docs/research/README.md). New campaigns can
adopt this directory incrementally; moving an established identity-bound config
is not required merely to satisfy the layout.
