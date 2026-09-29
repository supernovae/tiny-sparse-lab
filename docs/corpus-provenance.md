# Corpus Forge provenance and shape experiments

Corpus Forge stores acquisition receipts, normalized source documents, derived records, and frozen releases separately. A release's lineage ledger records three **independent**, versioned classifications per record:

- **Origin:** where the record came from (`primary_source`, `human_authored`, `deterministic_synthetic`, `source_transformed_synthetic`, `multi_source_synthetic`, `free_generated_synthetic`, or `model_self_generated`). The source snapshot and parent document IDs remain separate from this label. A prompt backed by a source is not the source itself.
  `sources/*.yaml` may explicitly declare `origin: human_authored` for independently authored input; unspecified source origin defaults to `primary_source`, not an inference of authorship from a local file path.
- **Verification:** what supports a particular answer or claim (`unverified`, `schema_validated`, `source_entailed`, `oracle_verified`, `cross_source_verified`, `human_reviewed`, or `rejected`). A verification receipt retains source passage and verifier rule, oracle inputs and comparison, or review protocol and decision, as appropriate. Malformed or rejected generated outputs remain inspectable; a model's own assertion is not independent verification.
- **Training shape:** how the example was constructed (`raw_document`, `direct_qa`, `troubleshooting_scenario`, `decision_record`, `tool_trace`, and other registered shapes). Normalized interaction, grounding, answer style, supervision, reasoning depth, task family, and source domains describe the rendering rather than inferred internal reasoning mechanisms. Rendered-text SHA-256 and shared parent document IDs let different shapes of one concept be queried together.

**Origin is not quality. Verification is not usefulness. Training shape is not truth.** Source-entailed QA may be poor training for tool use. An artificial but oracle-verified bounded scenario can be useful for a causal task. Fluent free-generated prose can remain unsupported. No scalar trust score, universal synthetic cap, or automatic recommendation follows from these fields.

Source declarations, normalized documents and lineage preserve `modality: text` / `modalities: [text]` explicitly. The current ingestion and rendering contracts are text-only; non-text data cannot be relabeled as text to pass validation. A future version must define a real acquisition, content identity and renderer before adding image, audio or multimodal records. This dimension is provenance for future assimilation comparisons, not evidence that such models have been trained.

For prospective acquisitions, [the v2 rights protocol](rights-policy.md) adds
file-level SPDX, source notice and explicit training-restriction evidence.
Rights, verification, origin and training shape remain distinct dimensions;
metadata-only publication does not republish the training export. Historical
DevMind fail-gates and v1 release identities are not reinterpreted.

## Compare releases without reacquiring sources

One project may point to multiple checked-in release declarations while retaining the same `id`, source declarations, snapshots and acquisition lock. For example, keep `sources/*.yaml`, `splits.yaml`, and `transforms/*.yaml` fixed, then make project variants that differ only in the `release:` YAML reference. Declare `include_shapes` and/or `include_origins` in each release YAML. `corpus build ... --offline` verifies the shared acquisition lock, builds the selected view without mutating a frozen release, and `corpus freeze BUILD` publishes a new release ID. Inspect its `manifest.json` snapshots, `lineage.jsonl`, `audit.json`, and `report.json`; only then export the selected training view. A variant whose selected training split is empty fails freeze rather than silently changing a run.

A fraction-controlled variant may declare `fraction: {generated_share: 0.25, train_tokens: 2000, tokenizer_path: tokenizers/fixed.json, tokenizer_sha256: <full SHA-256>}`. The tokenizer must be a checked-in, regular project file matching its digest. Selection counts the **actual encoded training examples** (including chat rendering) and requires complete examples, the exact total and generated token counts, and representation of every candidate source family. An impossible fraction or budget fails; no bytes-per-token estimates or hidden replacement are used. Keep tokenizer, source families, splits, and total budget fixed between variants to isolate the attempted treatment. Equal token counts do **not** establish equal semantic information.

`corpus describe RELEASE` reports immutable record counts; token totals are `null` unless the release declares a pinned selection tokenizer. `corpus describe RELEASE --tokenizer PATH` verifies the release and recounts actual encoded view records by origin, verification, shape, generator and template, without changing the frozen report. The `training_mixture` denominator counts only selected **train** views; validation/test and unselected inventory remain separately visible. Watch both **record and token** proportions, duplicate and prefix concentration, lexical diversity, and shared source/world/template lineage across splits. Descriptive warnings are not corpus-quality judgments; holdout gates must be declared in the split policy.

After training, attach each capability observation to its exact release ID, model/checkpoint, measured token budget and evaluation suite. A shape-by-capability matrix and transformation usefulness state (`HELPFUL_UNDER_TESTED_TASK`, `NO_MEASURED_EFFECT`, `MIXED`, `REGRESSION`, `UNKNOWN`) are meaningful only for those bound observations; do not add a single aggregate score or a global label to a shape. This release ledger also preserves lexical/semantic candidate parentage so future Engram-only, neural-only and combined assimilation studies can compare the same source knowledge without interpreting a candidate inventory as model training.

After a run has a verified checkpoint and a content-addressed capability card,
`sparselab corpus compare RELEASE --run-dir RUN` reports each card separately
alongside the selected training-shape mixture. Repeat `--with RELEASE RUN_DIR`
to compare at least two releases; the command requires identical source snapshot
IDs, model architecture, card protocols, and checkpoint token budgets. It
reports `UNKNOWN` usefulness per transform until paired evidence isolates the
variable. Its integrity checks bind results to a run/checkpoint and registered
card/scorer; they cannot establish an independent test population or causal
effect. A release with no capability results has no capability matrix.
