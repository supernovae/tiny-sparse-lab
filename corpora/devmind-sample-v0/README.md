# DevMind corpus-layer teaching sample

This is an original MIT-licensed, tiny offline fixture, not a DevMind research campaign or model-training evidence. Three authored source families provide local Markdown, code and YAML; a fourth declaration names the deterministic `pathlib.PurePosixPath` suffix oracle. `technical_docs`, `code`, `configs`, and `systems_scenarios` are illustrative domains. The requested mixture is descriptive, not an optimized sampling policy. All generated text and snapshots live beneath the ignored, task-owned `sparselab-work/corpora/devmind-sample-v0/` directory and may be deleted when no run depends on them. No public corpus is downloaded.

Run from the repository root (no model training):

```sh
WORK="$PWD/sparselab-work"
uv run --locked sparselab --work-dir "$WORK" corpus acquire corpora/devmind-sample-v0/corpus.yaml
BUILD=$(uv run --locked sparselab --work-dir "$WORK" corpus build corpora/devmind-sample-v0/corpus.yaml --offline)
RELEASE=$(uv run --locked sparselab --work-dir "$WORK" corpus freeze "$BUILD")
uv run --locked sparselab --work-dir "$WORK" corpus describe "$RELEASE"
uv run --locked sparselab --work-dir "$WORK" corpus sources "$RELEASE"
uv run --locked sparselab --work-dir "$WORK" corpus audit "$RELEASE"
SAMPLE_ID=$(uv run --locked sparselab --work-dir "$WORK" corpus sample "$RELEASE" --domain technical_docs | uv run --locked python -c 'import json,sys; print(json.load(sys.stdin)["records"][0]["record_id"])')
uv run --locked sparselab --work-dir "$WORK" corpus lineage "$RELEASE" "$SAMPLE_ID"
uv run --locked sparselab --work-dir "$WORK" corpus review "$RELEASE" --kind chat
EXPORT=$(uv run --locked sparselab --work-dir "$WORK" corpus export "$RELEASE" --view lm --base-run-config configs/runtime_smoke_cpu.yaml --vocab-size 300)
uv run --locked sparselab --work-dir "$WORK" tokenizer train "$EXPORT/tokenizer.yaml"
uv run --locked sparselab --work-dir "$WORK" data prepare "$EXPORT/run.yaml"
uv run --locked sparselab --work-dir "$WORK" corpus describe "$RELEASE" --tokenizer "$EXPORT/tokenizer/tokenizer.json"
uv run --locked sparselab --work-dir "$WORK" corpus export "$RELEASE" --view chat --base-run-config configs/runtime_smoke_cpu.yaml --vocab-size 300
```

Frozen test rows are held out of tokenizer fitting and training exports. Assistant/tool messages are inert transcript examples; corpus derivation does not execute tool calls. A successful smoke proves local execution and provenance wiring, not source rights outside this MIT fixture, model usefulness or scientific performance.
