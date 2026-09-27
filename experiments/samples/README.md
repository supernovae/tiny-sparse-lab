# Sample experiments

Samples teach the workflow and are safe to copy. They are not claims that a
mechanism improves quality or performance.

Start with a packaged scaffold:

```sh
WORK="$PWD/sparselab-work/experiments/engram-ffn-substitution-v1"
uv run --locked sparselab --work-dir "$PWD/sparselab-work" research scaffold engram-ffn-substitution-v1 \
  --scale smoke --data offline --backend cpu \
  --output "$WORK/scaffold"
uv run --locked sparselab study plan "$WORK/scaffold/study.yaml"
```

The generated README identifies the explicit preparation and execution steps, sets one `WORK`, and keeps all seed coordinates in `$WORK/runs` with one `$WORK/receipt.json`. Its scratch setting is exported after scaffolding; setting the workspace itself as the global base before scaffolding would append another `experiments/<study-name>` level. Explicit generated store and receipt paths keep later commands inside the selected workspace.
For a checked-in reference layout, see
[`configs/references/dense-small-v1`](../../configs/references/dense-small-v1/).
Copy editable inputs; do not edit a known-good reference or treat smoke output as
research evidence.

A sample should say what it demonstrates, what it deliberately does not prove,
the expected cost class, and which paths can be deleted safely.
