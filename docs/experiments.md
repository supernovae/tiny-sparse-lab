# Independent experiments and controlled comparisons

Run each concrete configuration with its declared tokenizer, data source, engine/backend, precision, optimizer, sequence length, effective batch, token budget, and seed. Inspect first; do not infer parameter counts or scientific equivalence from a filename.

```sh
uv run sparselab inspect configs/micro_dense.yaml --json
uv run sparselab inspect configs/dense_7m.yaml --json
uv run sparselab inspect configs/dense_10m.yaml --json
uv run sparselab inspect configs/dense_25m.yaml --json
uv run sparselab inspect configs/dense_50m.yaml --json
uv run sparselab inspect configs/smoke_sliding_cpu.yaml --json
uv run sparselab inspect configs/smoke_mla_cpu.yaml --json
uv run sparselab inspect configs/smoke_combined_cpu.yaml --json
```


The dense scale presets are inspected at 3,344,064, 6,917,376, 10,244,160, 29,893,120, and 50,274,752 parameters. They share the pinned TinyStories revision, 8192-token tokenizer, sequence length, token budget, optimizer, and seed. Parameter count alone is not a comparison result: report each completed run's observed validation loss, perplexity, throughput, device, and metric coordinates.

## Explicit matrices and worker execution

The checked-in matrix expands the CPU runtime smoke configuration over seeds 7, 17, and 41:

```sh
uv run sparselab experiment submit --matrix tests/fixtures/runtime-matrix.yaml \
  --dry-run --store /tmp/sparselab-controller
```

Dry-run prints all resolved coordinates and hashes without downloading/preparing data, creating the store, or enqueueing. For execution, prepare the tokenizer, register an eligible worker, remove `--dry-run`, and run `sparselab controller run --store /tmp/sparselab-controller`. Every coordinate gets distinct experiment/attempt/run IDs. Preparation must succeed for all coordinates before the enqueue transaction.

Matrix v1 uses an explicit base config and insertion-ordered axes with labeled dotted-path patches. Duplicate labels, conflicting patches, invalid configs, and excessive expansion are rejected. Labels such as “50M” or “MoE” never infer a model. Workers execute independent optimizers, not distributed gradients; unsupported requirements remain queued with a reason. See [worker contracts and matrix format](workers.md#explicit-matrices).

## Architecture study campaigns

An architecture study is an optional campaign wrapper over the existing explicit matrix format. It does not add a closed architecture registry or change `RunConfig`: every matrix coordinate still resolves to a normal concrete config, with the same dotted-path validation as `experiment submit --matrix`. Use direct training when that is simpler:

```sh
uv run sparselab train configs/my_architecture.yaml
uv run sparselab inspect configs/my_architecture.yaml --json
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
uv run sparselab study plan experiments/ffn-study.yaml
```

The plan expands every config, hashes the inputs, reports parameter inventory and estimated weight/optimizer/checkpoint bytes, and requires each declared comparison to match one-to-one over all unselected axes. It rejects undeclared config changes before submission. For multi-seed ablations, add a `seed` matrix axis and leave it out of both selectors; the same seed then pairs baseline and variant. Seeds cannot be the varied field. `vary_fields` must list exactly the changed dotted config fields. The built-in `memory`, `attention`, `ffn`, and `scale` modes enforce their corresponding model field families.

Submit the immutable run inventory to the existing independent-worker controller, then run that controller in a separate terminal:

```sh
uv run sparselab worker register cpu-one --backend cpu --store /tmp/sparselab-study
uv run sparselab study submit experiments/ffn-study.yaml \
  --worker cpu-one --store /tmp/sparselab-study \
  --receipt /tmp/sparselab-study/ffn-receipt.json
uv run sparselab controller run --store /tmp/sparselab-study
```

After all coordinates finish, collect held-out validation loss/perplexity and each card against the run checkpoint:

```sh
uv run sparselab study collect experiments/ffn-study.yaml \
  /tmp/sparselab-study/ffn-receipt.json --runs-dir /tmp/sparselab-study
```

Collection verifies the receipt against the current study and each run's resolved config, records checkpoint identities, and emits a content-addressed report beside the receipt. Missing runs or invalid evaluations stay explicitly inconclusive; they are not dropped from the denominator silently. `--checkpoint NAME` selects the same named checkpoint for every run; the default is each run's latest checkpoint. Paired deltas are descriptive, not statistical significance or a universal architecture ranking.

The default controller store and collection run directory are both `runs/`; if `--store` is changed, pass that same directory to `study collect --runs-dir`.

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
uv run sparselab checkpoint verify runs/RUN_ID/checkpoints/latest.json --json
uv run sparselab evidence RUN_ID --json
```

Training records held-out validation at initial, configured periodic, and terminal boundaries. A validation metric is not by itself a saved model: checkpoint cadence, a new best loss, or termination triggers a verified generation. Use checkpoint-bound evaluation reports when claiming results for exact weights. See [experiment evidence](evidence.md) for evidence levels, controlled-comparison requirements, and the separately unimplemented hardware/reference-harness protocol.

For a named architectural hypothesis rather than aggregate loss alone, use a [capability card](capabilities.md). Cards preserve their own prompt set, scorer, baseline controls, and scope-limited conclusion.
