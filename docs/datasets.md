# Declare, pin and reuse datasets

A dataset is an input to an experiment, not a training function. A source
declaration describes where text comes from and how it is selected; a lock pins
the source; a snapshot retains the selected records and exclusion accounting.
RunConfig references that immutable snapshot through `dataset.source: snapshot`.
The [TinyStories walkthrough](tinystories-microlab.md) is one reference declaration
of this general pattern.

## Source declaration and lock

Use the sample [source.yaml](../experiments/samples/tinystories-microlab/source.yaml)
as a starting point. Change repository, exact commit revision, configuration,
logical split mapping and text field for another standard Hub text dataset.
Keep attribution and terms explicit. No callback or repository Python loader is
part of the interface. Unsupported formats fail rather than executing code.

`selection.mode: bounded` names per-split retained-document targets.
`selection.mode: exhaustion` consumes the declared source splits to their end.
Both have resource limits. Reaching a safety limit is not evidence of complete
source coverage. Source revisions must be immutable; moving names such as `main`
are rejected. The sample exact-text dedup policy processes validation first,
removes within-split repeats and excludes training overlap. Selection and policy
are part of scientific identity, not invisible cleaning defaults.

```sh
uv run --locked --extra cpu sparselab data lock source.yaml --output source.lock.json --json
uv run --locked --extra cpu sparselab data snapshot source.lock.json --output snapshot --json
```

Use new output paths. A snapshot retains its source lock, selected records,
source positions, exclusion inventory, stop reasons and content digests.
Do not edit it to add data. Increasing bounds, changing the source revision or
selecting another split creates a new lock/snapshot and new downstream identities.
Model-width, seed or training-budget comparisons can reuse the unchanged inputs.

For an interrupted acquisition, repeat the snapshot command with `--resume`.
Its adjacent work journal is authenticated before reuse; immutable sources can
be replayed to verify the completed prefix when seeking is unavailable. Atomic
publication prevents a partial snapshot from masquerading as complete input.
Disk-backed indexes bound deduplication memory; resource limits and storage
headroom still need admission on the actual filesystems. Hub caches may consume
more than the selected text, so small document bounds are not a download-size
guarantee.
The record-byte ceiling is checked after the upstream loader decodes a record;
it does not cap that loader's transient allocations. Pilot the actual format and
resource policy before attempting a full split.

Corpus Forge's explicit files and checksummed shards remain available. Use
`kind: forge_files` to select verified Forge snapshot members with declared
per-split file bindings. This route serves sources needing explicit file control;
it does not remove release/export requirements from a Forge training contract.

## Preparation and exposure

Run/tokenizer configurations name snapshot `train_path`, `validation_path`,
`source_manifest_path`, revision and license. Native verification matches those
paths and identities before fitting or preparing. Tokenizer fitting remains
train-only; its input byte/document bounds are separate from training exposure.
Whole-document packing fails on insufficient token bounds rather than silently
truncating selected text. Preparation uses disk-backed resumable chunks.

```sh
uv run --locked --extra cpu sparselab tokenizer train tokenizer.yaml
PREPARED=$(uv run --locked --extra cpu sparselab data prepare run.yaml)
uv run --locked --extra cpu sparselab data coverage --prepared-root "$PREPARED" --config run.yaml --json
uv run --locked --extra cpu sparselab data budget --prepared-root "$PREPARED" --config run.yaml \
  --passes 1 --no-require-full --output one-pass.yaml --json
```

Coverage distinguishes source exhaustion, exclusions, retained documents,
truncation, packed tokens, complete usable blocks, tails and supervised targets.
Passes are over usable prepared blocks; they are not raw-document epochs.
`--require-full` is the budget command's default and rejects incomplete or
unproven full-source coverage. `--no-require-full` explicitly permits a bounded
selection. Objective masks are counted; unsupported objectives fail.

Budget authoring writes a new config with absolute input paths and proposed
step/target limits. It preserves optimizer settings, so review the fresh schedule
before training. It does not derive cumulative continuation budgets or approve
a run. Check measured memory fit and disk growth with native inspection,
preflight and staging.

For another pilot with authenticated materialized inputs, use
`stage CONFIG --prepared-inputs PRIOR_STAGE/prepared --through warmup --output NEW_STAGE`.
This is the sealed prepared-input bundle, not the array cache returned by
`data prepare`. The adapter verifies the existing inventory and requested config;
missing or mismatched inputs fail without downloading or repacking. Select the
registered runtime explicitly when using an accelerator.

The Python `stage(..., prepared_inputs=...)` API enforces the same input checks,
including for `through="inspect"`. Existing-input staging requires a new output;
the destination is checked again under the stage lock and published without
replacing a competing writer. Generic `allow_runtime_drift` is not accepted in
this mode. Ordinary staging without supplied inputs retains verified output reuse.
Validated stage receipts record the input bundle's producer, the prepared data's
production source, and the current verification source separately. These fields
describe provenance; they do not grant source compatibility or change the sealed
input bytes. Older input bundles remain readable without these additional fields.

## Campaign and existing-input reuse

Campaign `dataset_snapshot`, `tokenizer_train` and `data_prepare` stages call
these native operations. `experiment_plan` with `mode: bind` authenticates their
outputs and binds a phase template before locking. Runtime, run, collection,
evaluation and generation stages consume the resulting identities.

Already prepared inputs can also enter through typed artifact references and
standalone `experiment bind-inputs`. Direct snapshots do not need a fabricated
Corpus Forge release. Forge-backed inputs retain their release/export gates.
Missing inputs, inconsistent tokenizer/snapshot provenance and un-ingested
parents block dependent work. See [Campaigns](campaigns.md).

## Legacy input migration

The old `tinystories` / `local_stories` authoring forms and implicit TinyStories
snapshot command are retired for new execution. Historical configurations,
snapshots, locks and receipts remain available to read and verify. Migration
always writes new artifacts; it never changes a historical identity in place.

For a direct-prefix migration, save an explicit resource policy as
`migration-resources.yaml` (adjust these bounds after admission):

```yaml
max_source_records: 20000
max_text_bytes: 16777216
max_record_bytes: 1048576
max_work_bytes: 268435456
min_free_bytes: 1073741824
```

```sh
uv run --locked --extra cpu sparselab data migrate old-run.yaml \
  --output migrated --resources migration-resources.yaml --accept-policy-change --json
```

Review the migration receipt and emitted instructions before acquisition. The
acknowledgement is explicit because old direct prefixes could truncate records,
while new snapshot selection/deduplication and whole-document packing differ.
Do not pool old/new results as if the inputs were unchanged. Snapshot imports
retain the original evidence and produce a separately identified generic input.
