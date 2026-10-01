# Independent experiments and controlled comparisons

This guide explains the execution machinery. Copyable walkthroughs are indexed
under [`experiments/samples/`](../experiments/samples/); identity-bound research
campaigns and iteration notes belong under
[`experiments/research/`](../experiments/research/). Keep downloads, prepared
data, run stores, checkpoints, logs, and generated reports in a named ignored
`sparselab-work/experiments/<campaign>/` directory rather than anonymous `/tmp`.
Code gaps discovered during a campaign go to [`TODO.md`](../TODO.md); scientific
next steps remain in the research roadmap/lifecycle.

Start with [a complete training program](experiment-programs.md) for authored
plans and checkpoint phases, or [the TinyStories microlab](tinystories-microlab.md)
for a first model and a runnable matrix comparison.

Run each concrete configuration with its declared tokenizer, data source, engine/backend, precision, optimizer, sequence length, effective batch, token budget, and seed. Inspect first; do not infer parameter counts or scientific equivalence from a filename.

```sh
uv run --locked sparselab inspect configs/micro_dense.yaml --json
uv run --locked sparselab inspect configs/dense_7m.yaml --json
uv run --locked sparselab inspect configs/dense_10m.yaml --json
uv run --locked sparselab inspect configs/dense_25m.yaml --json
uv run --locked sparselab inspect configs/dense_50m.yaml --json
uv run --locked sparselab inspect configs/smoke_sliding_cpu.yaml --json
uv run --locked sparselab inspect configs/smoke_mla_cpu.yaml --json
uv run --locked sparselab inspect configs/smoke_combined_cpu.yaml --json
```


The dense scale presets are inspected at 3,344,064, 6,917,376, 10,244,160, 29,893,120, and 50,274,752 parameters. They share the pinned TinyStories revision, 8192-token tokenizer, sequence length, token budget, optimizer, and seed. Parameter count alone is not a comparison result: report each completed run's observed validation loss, perplexity, throughput, device, and metric coordinates.

## Authored experiment plans and resolved locks

[Campaign v1](campaigns.md) orchestrates Corpus Forge releases and these plans
with typed dependencies, declared readiness policies, input-bound authorization
and recoverable runtime-bound execution. It does not replace ExperimentPlan's
scientific configuration or infer data/architecture/quality thresholds.

`experiment plan_version: 1` declarations describe a concrete base `RunConfig`,
bounded labeled axes, typed source/artifact inputs, optional Corpus Forge release
recipes, exact-field comparison contracts and ordered checkpoint phases. The
versioned [JSON Schema](../schemas/experiment-plan-v1.schema.json) describes the
authoring format. A declaration is not executable: `validate`/`inspect` expand
configuration values without asserting that referenced files, worker runtime or
future checkpoints exist. For declared `corpus_variants`, `prepare` explicitly
acquires pinned local/offline sources, freezes releases, exports data, trains
tokenizers and verifies prepared arrays without starting training. External
sources must already be acquired. A plain `base_run` does not get tokenizer or
data preparation from this command; prepare those assets explicitly.
`diff` requires those inputs when a comparison
selects corpus variants, and rejects every undeclared resolved-field difference.

Corpus variants choose exactly one tokenizer strategy: `vocab_size: 300` trains
from that release, or `tokenizer_artifact: shared_tokenizer` references a named
external `artifacts.shared_tokenizer` with `kind: tokenizer`, version, producer,
identifier, path to `tokenizer.json`, and SHA-256. Reuse verifies the
tokenizer's provenance manifest and actual vocabulary before acquiring or
building anything. Corpus Forge's immutable export `run.yaml` remains a generic
tokenizer-training template; the prepared and locked cell instead binds the
verified external tokenizer path.
Comparisons using one tokenizer declare the resulting release, export, prepared
data and dataset-revision identity differences explicitly and keep
`artifacts.tokenizer.sha256` invariant; a changed tokenizer SHA is independent
tokenizer drift, not an incidental corpus difference. Fractional releases still
require a project-relative pinned selector file and SHA. In reuse mode its SHA
must match the reused artifact unless `fraction_tokenizer` explicitly names a
second verified external tokenizer artifact with that selector's SHA.
`release_set.accepted_generation_statuses` accepts Corpus Forge's verification
labels as unordered filter classes, not a quality ranking or score.

```sh
uv run --locked sparselab experiment validate experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked sparselab experiment prepare experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked sparselab experiment diff experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked sparselab experiment lock experiments/samples/corpus-shape-fraction.yaml --json
```

`lock` re-verifies artifact digests and hardware-free runtime requirements,
then publishes an immutable content-addressed scientific plan. Its separate
availability sidecar records local input paths, not executable capability.
Accelerator runs require per-cell operational receipts from `experiment bind
LOCK --runtime-profile PROFILE` or `--worker NAME`; `experiment run` accepts
those sources directly or a previously published `--binding RECEIPT`, always
with fresh revalidation. Only explicit PyTorch CPU supports ambient local
registration. Runtime binding does not change either lock digest.
`experiment run LOCK --cell CELL --json` submits the selected cell;
`sparselab controller run --store "$SPARSELAB_WORK_DIR/experiments/<plan-id>/controller"`
executes and ingests it. A dependent phase must be submitted separately with
`--phase` after the parent has an ingested, verified generation. Terminal and
best-validation selectors bind the exact parent generation before dispatch.
The parent `execution_binding_sha256` remains distinct from the operational
`runtime_binding_sha256`; mutable aliases are never checkpoint identities.
Resume, guarded budget extension, and weight promotion preserve their distinct
trainer semantics.

`experiment collect LOCK --json` creates a hash-addressed evidence index from
matching worker specs/receipts and ingested manifests. Missing, failed,
interrupted, and invalid cells remain visible. `experiment reconstruct LOCK
--index INDEX --json` re-verifies it and returns identity/provenance, contrasts,
phase lineage, evidence coverage, and readiness/decision views. Planned-only
views omit `--index`. These are software and provenance reports, not automatic
scientific findings or baseline promotion; preregistered held-out evaluations
and human review remain separate. See the
[four-arm CPU example](../experiments/samples/README.md#corpus-shape-and-exact-generated-fraction)
and [retrospective limits](research/experiment-dsl-retrospectives.md).

Existing `matrix_version: 1` and `study_version: 1` callers remain usable:
their axis expansion and exact-difference checks now share the plan compiler,
but their receipts are not relabeled as new locks. To migrate, copy the base
config and axis choices into a new `plan_version: 1` document, declare artifact
identities/release recipes and exact interventions, run `prepare` when needed,
and create a new lock before dispatch. Preserve old receipts and research
reports with their original protocol identities. A lock cannot be retroactively
assigned to a prior result solely because its config fields happen to match.

## Explicit matrices and worker execution

The checked-in matrix expands the CPU runtime smoke configuration over seeds 7, 17, and 41:

```sh
WORK=sparselab-work/experiments/runtime-matrix
export SPARSELAB_WORK_DIR="$WORK"
uv run --locked sparselab experiment submit --matrix tests/fixtures/runtime-matrix.yaml \
  --dry-run --store "$WORK/runs"
```

Dry-run prints all resolved coordinates and hashes without downloading/preparing data, creating the store, or enqueueing. For execution, prepare the tokenizer, register an eligible worker, remove `--dry-run`, and run `sparselab controller run --store "$WORK/runs"`. Every coordinate gets distinct experiment/attempt/run IDs. Preparation must succeed for all coordinates before the enqueue transaction.

Matrix v1 uses an explicit base config and insertion-ordered axes with labeled dotted-path patches. Duplicate labels, conflicting patches, invalid configs, and excessive expansion are rejected. Labels such as “50M” or “MoE” never infer a model. Workers execute independent optimizers, not distributed gradients; unsupported requirements remain queued with a reason. See [worker contracts and matrix format](workers.md#explicit-matrices).

## Architecture study campaigns

An architecture study is an optional campaign wrapper over the existing explicit matrix format. It does not add a closed architecture registry or change `RunConfig`: every matrix coordinate still resolves to a normal concrete config, with the same dotted-path validation as `experiment submit --matrix`. Use direct training when that is simpler:

```sh
uv run --locked sparselab train --runs-dir sparselab-work/runs configs/my_architecture.yaml
uv run --locked sparselab inspect configs/my_architecture.yaml --json
```

If an architecture needs model-code or config-schema changes, make those normally and continue to train its concrete config directly. The campaign layer does not make arbitrary Python model code auto-configurable; it does not restrict existing direct builds to named presets.

For a controlled FFN-width experiment, save these two files as `experiments/ffn-matrix.yaml` and `experiments/ffn-study.yaml`:

```yaml
# experiments/ffn-matrix.yaml
matrix_version: 1
base_config: ../configs/runtime_smoke_cpu.yaml
axes:
  architecture:
    - label: baseline
      set: {}
    - label: wider-ffn
      set:
        model.ffn_dim: 48
```

```yaml
# experiments/ffn-study.yaml
study_version: 1
name: ffn-width-smoke
matrix: ffn-matrix.yaml
cards:
  - engram-recall-v1
comparisons:
  - id: wider-ffn
    vary: custom
    vary_fields:
      - model.ffn_dim
    baseline:
      architecture: baseline
    variant:
      architecture: wider-ffn
```

The smoke config demonstrates wiring, not model quality; use a task-appropriate trained config and capability card for useful evidence. Built-in cards are named in [capabilities](capabilities.md); custom cards are referenced by JSON path. A card is a fixed post-training probe, not a training reward.

Plan before allocating workers:

```sh
uv run --locked sparselab study plan experiments/ffn-study.yaml
```

The plan expands every config, hashes the inputs, reports parameter inventory and estimated weight/optimizer/checkpoint bytes, and requires each declared comparison to match one-to-one over all unselected axes. It rejects undeclared config changes before submission. For multi-seed ablations, add a `seed` matrix axis and leave it out of both selectors; the same seed then pairs baseline and variant. Seeds cannot be the varied field. `vary_fields` must list exactly the changed dotted config fields. The built-in `memory`, `attention`, `ffn`, and `scale` modes enforce their corresponding model field families.

Submit the immutable run inventory to the existing independent-worker controller, then run that controller in a separate terminal:

```sh
WORK=sparselab-work/experiments/ffn-width-smoke
export SPARSELAB_WORK_DIR="$WORK"
mkdir -p "$WORK"
uv run --locked sparselab worker register cpu-one --backend cpu --store "$WORK/runs"
uv run --locked sparselab study submit experiments/ffn-study.yaml \
  --worker cpu-one --store "$WORK/runs" \
  --receipt "$WORK/receipt.json"
uv run --locked sparselab controller run --store "$WORK/runs"
```

After all coordinates finish, collect held-out validation loss/perplexity and each card against the run checkpoint:

```sh
uv run --locked sparselab study collect experiments/ffn-study.yaml \
  "$WORK/receipt.json" --runs-dir "$WORK/runs"
```

Collection verifies the receipt against the current study and each run's resolved config, records checkpoint identities, and emits a content-addressed report beside the receipt. Missing runs or invalid evaluations stay explicitly inconclusive; they are not dropped from the denominator silently. `--checkpoint NAME` selects the same named checkpoint for every run; the default is each run's latest checkpoint. Paired deltas are descriptive, not statistical significance or a universal architecture ranking.

A study submitted without path overrides uses `sparselab-work/experiments/<study-name>/runs` and one sibling `receipt.json`. Collection defaults to the receipt’s sibling `runs/`. Pass the same explicit store as `--runs-dir` for custom layouts. Seeds, architecture cells, budget coordinates, and resumed children share this store; they do not create peer workspaces. See [workspace policy](workspaces.md) for external-disk overrides and evidence retention.

The study layer compares only configurations and evidence supported by the current training/evaluation stack. It does not add RL reward training or wire record-based Engram packs into model execution; those require separate model/trainer work. Keep data, tokenizer, source revision, runtime, and actual step/token budgets matched within each architecture comparison.

## Completed multi-seed studies

- [Context/Engram study](context-engram-study.md#execution-results--2026-09-22): 24 endpoints spanning seeds 17/41/73, two exact target budgets, dense/backbone and dense-total comparisons, and collision/address-order diagnostics. Every untouched override endpoint remained 0/8. Added memory capacity and observed bucket collisions are not evidence of a generalization advantage.
- [Domain adaptation](path-domain-corpus.md#2026-09-22-execution-record): three pretraining/adaptation pairs with frozen supervision, provenance, semantic leakage checks, per-case results, and static-retention measurements. Training-case acquisition improved, but held-out reliability remained poor and retention worsened sharply.

[Independent acceptance](../artifacts/acceptance/scientific_studies_2026_09_22.json) binds the input inventories, endpoint identities, stored responses, paired deltas, and retention observations. These studies retain all declared seeds/endpoints and their negative outcomes; they are not a general model-selection or significance framework.

## Historical controlled dense runs

These recorded runs used the pinned TinyStories revision, the same 8192-token tokenizer, seed, optimizer, 128-token sequence length, and 204,800 target-token budget on MPS. Validation was standalone evaluation over the retained validation split. They predate the current integrity-bound report format; retain them as historical loss observations, not newly verified capability evidence.

| Run | Parameters | Validation loss | Validation perplexity |
|---|---:|---:|---:|
| `scale-micro-3m` | 3,344,064 | 5.1614 | 174.42 |
| `scale-dense-7m` | 6,917,376 | 4.9942 | 147.55 |
| `scale-dense-10m` | 10,244,160 | 4.8790 | 131.49 |
| `scale-dense-25m` | 29,893,120 | 4.5751 | 97.04 |
| `scale-dense-50m` | 50,274,752 | 4.2770 | 72.03 |

These observations suggest lower validation loss at this fixed budget as dense parameter count increases. They do not establish an optimal scaling law, chat ability, or an Engram benefit. Previously published throughput values are withdrawn: the trainer divided one update's tokens by cumulative run time. New measurements use synchronized update duration; rerun controlled pairs before making a performance claim.
The smoke configurations prove CPU wiring only. For a meaningful comparison, train separate unique run IDs with matching source, tokenizer, device, token budget, sequence length, optimizer, and seed. Store the resulting run directory and SQLite metrics; compare observed loss or throughput only at matching recorded budgets. Do not convert unmatched runs into an aggregate quality score.

Dense attention, sliding-window attention, MLA, MoE, and byte memory alter different resource boundaries. Attribute an observed difference only after a controlled ablation; a combined run is a compatibility check, not evidence that its mechanisms compound beneficially.

## Evidence before comparison

Every serious local run should have verified checkpoint/held-out evidence before it enters a comparison:

```sh
uv run --locked sparselab checkpoint verify sparselab-work/runs/RUN_ID/checkpoints/latest.json --json
uv run --locked sparselab evidence RUN_ID --json
```

Training records held-out validation at initial, configured periodic, and terminal boundaries. A validation metric is not by itself a saved model: checkpoint cadence, a new best loss, or termination triggers a verified generation. Use checkpoint-bound evaluation reports when claiming results for exact weights. See [experiment evidence](evidence.md) for evidence levels, controlled-comparison requirements, and the separately unimplemented hardware/reference-harness protocol.

For a named architectural hypothesis rather than aggregate loss alone, use a [capability card](capabilities.md). Cards preserve their own prompt set, scorer, baseline controls, and scope-limited conclusion.
