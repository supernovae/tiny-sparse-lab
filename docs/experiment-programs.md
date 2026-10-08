# Chain a training program

An authored YAML program can describe inputs, bounded axes, controlled changes
and a sequence of checkpoints. SparseLab resolves it into an immutable lock,
executes each cell through a worker, and collects evidence with the selected
parent generation attached to each child. This guide runs a small offline
synthetic fixture through fresh training followed by a full-state AdamW budget
extension. For a story model first, see the [TinyStories microlab](tinystories-microlab.md).

The example uses a small offline fixture and explicit runtime settings so you
can learn the whole workflow without downloading a corpus. Runtime choices
belong to the concrete config and execution settings; see [runtime support](runtime.md).
Corpus Forge programs seal portable provenance with their prepared inputs; see
[Forge worker provenance](#forge-worker-provenance) for relocation and continuation.

## Read the program

The runnable [offline-chain.yaml](../experiments/samples/training-program/offline-chain.yaml)
is a phase template paired with [run.yaml](../experiments/samples/training-program/run.yaml)
and [tokenizer.yaml](../experiments/samples/training-program/tokenizer.yaml).
Prepare the offline fixture and bind its actual hashes below; training then
proceeds through these phases:

```yaml
phases:
  - id: pretrain
    transition: fresh
  - id: continue
    transition: extend_budget
    parent: pretrain
    selector: terminal
    at_step: 20
    set:
      training.max_steps: 40
      training.max_tokens: 1280
      optimizer.decay_steps: 20
```

The base config has sequence length 16, microbatch 1 and accumulation 2: 32
supervised targets per full optimizer update. The parent ends at 20 updates /
640 targets; the child ends at 40 cumulative updates / 1,280 targets. The child
uses the parent's full training state and original decay horizon, rather than
starting a fresh optimizer. This demonstrates preparation and checkpoint
lineage, not useful technical knowledge or an architecture improvement.

| Transition | What the child carries forward |
|---|---|
| `fresh` | New weights and training state. |
| `resume` | Full compatible training state and unchanged scientific settings/budget. |
| `extend_budget` | Full terminal AdamW state with explicit larger whole-update budget and original decay horizon; currently PyTorch-only. |
| `promote` | Compatible weights and tokenizer; fresh optimizer, schedule, RNG and cursor. |

Phases reference an earlier parent with the same axis coordinate. A terminal
selector must name the parent's declared final update through `at_step`.
`best_validation` instead selects one verified lowest-validation-loss
generation and cannot also specify `at_step`. Once selected, the actual
checkpoint digest is recorded in the child execution binding.

## Prepare the inputs and bind the program

From the repository root, use the locked Python 3.14 environment and an external
persistent root. The plan ID supplies the experiment subdirectory. The sample
run/tokenizer configurations retain their explicit in-checkout input destinations;
this root setting does not relocate those paths. Author new configurations if
you need external input storage, without editing an active or frozen sample:

```sh
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
SAMPLE=experiments/samples/training-program
WORK="$SPARSELAB_WORK_DIR/experiments/offline-training-chain-v1"
mkdir -p "$WORK/inputs"
df -h "$WORK"
df -i "$WORK"
uv run --locked --extra cpu sparselab tokenizer train "$SAMPLE/tokenizer.yaml"
PREPARED=$(uv run --locked --extra cpu sparselab data prepare "$SAMPLE/run.yaml")
```

Bind the already-prepared tokenizer and packed root directly. The adapter
authenticates the real tokenizer, prepared arrays and their relation to the run
config before exclusively writing a complete plan; it never retrains, downloads
or prepares inputs. Authoring a plan from an authenticated historical cache does
not authorize a new execution source: `experiment lock` separately enforces
source identity or reviewed operational compatibility.
The root is the directory containing `manifest.json`, not
its parent `cache_dir`. The run's relative paths are anchored to its own file.
The phase template's evaluation suite must resolve to the same declaration
from the output plan's location.
The template may retain separately authenticated checkpoint artifacts for an
external `extend_budget` parent. Input names `tokenizer`, `packed`, `release`
and `export` are reserved; existing input or corpus-variant bindings are rejected.
Relative external checkpoint locations are anchored to the template before
export, so moving the authored output does not select another parent.

```sh
uv run --locked --extra cpu sparselab experiment bind-inputs \
  "$SAMPLE/run.yaml" "$SAMPLE/offline-chain.yaml" \
  --prepared-root "$PREPARED" \
  --output "$WORK/inputs/plan.yaml" --json
PLAN="$WORK/inputs/plan.yaml"
uv run --locked --extra cpu sparselab experiment validate "$PLAN" --json
uv run --locked --extra cpu sparselab experiment inspect "$PLAN" --json
LOCK=$(uv run --locked --extra cpu sparselab experiment lock "$PLAN" --json | jq -r '.lock')
uv run --locked --extra cpu sparselab experiment explain "$LOCK" --json
```

`PREPARED` is the actual directory printed by `sparselab data prepare`,
under the run's configured `dataset.cache_dir`; do not infer its name.
`jq` extracts the emitted lock path without reimplementing lock parsing.
Each `--output` destination must not exist.

Validation checks declaration/config shape. Inspection authenticates
an unambiguously bound existing prepared artifact before reporting its measured
storage footprint; neither operation proves device fit. Preparation launches
no training. `experiment prepare` materializes declared `corpus_variants`;
it does not automatically prepare an arbitrary `base_run`. Externally prepared
inputs need typed artifacts and digests, as above.
Locking verifies actual inputs, concrete configs, runtime selection
and continuation constraints, then publishes a content-addressed plan.
During `experiment lock`, one resolver operation verifies every external artifact,
then carries a private, nonserialized proof through publication and canonical
lock/availability readback. This avoids repeating full prepared-array and corpus
scans within that operation, but rejects an artifact whose path or file metadata
changed in the meantime. The proof is bound to its issuer process and exact
resolved object; it cannot be supplied as plan metadata, copied to another lock
or inherited across a fork as reusable verification. A later
`experiment explain`, Campaign reopen, or other fresh-process `open_lock`
independently runs the full artifact verifiers again (with explicit cold
verification available through `--cold-verify`); neither the availability
sidecar nor a remembered digest substitutes for verified bytes.


The printed lock path is authoritative. Keep one code revision, the declaration
and its inputs fixed across execution. Reopen checks reject changed source or
artifact identities. For edits/repetitions, copy the declaration, choose a new
plan ID and adjust its file references to their new location before preparing;
never rewrite a running plan or its artifacts.

An optional `execution.attempt_contract: {path, sha256}` pins a source-relative
operational allocation. Locking authenticates its exact bytes and requires its
content identity to match the scientific lock; opening the lock checks both
again. The contract digest changes the operational plan identity, which Campaign
approval binds through the existing locked-plan output. Omitting this field
preserves legacy declaration and lock identities. A declaration alone creates
no ledger, grants no runtime approval, and does not replace the native run's
actual step and target counters.

### Direct dataset inputs

Generic `dataset.source: snapshot` runs use the same two external artifacts
shown above: `tokenizer` and `prepared_data`. They do not require a Corpus Forge
release or export. The [dataset interface](datasets.md) declares and locks the
source, acquires an immutable snapshot and authenticates its selection policy.
Pass the run config, phase template, existing prepared root and new output to
`experiment bind-inputs`. This operation verifies existing inputs; it never
refits, downloads or prepares them.

Keep the snapshot manifest, split records and accounting inventory available
on the controller. Locking verifies tokenizer training provenance, snapshot
identity and the prepared dataset/tokenizer/packing/source binding. A tokenizer
from another snapshot is not silently substituted. Worker dispatch carries
verified prepared arrays and tokenizer bytes so execution does not need to fetch
the Hub source again.

The [TinyStories walkthrough](tinystories-microlab.md) shows standalone phases
and a Campaign using this same contract. Campaign bind mode derives the run
config from the preparation stage and rebases the evaluation-suite reference
without changing its content identity. Ordinary standalone binding still requires
suite paths to resolve consistently between template and output locations.

Historical `tinystories` and `local_stories` artifacts remain verifiable, but
new authoring/execution requires [write-new migration](datasets.md#legacy-input-migration).
Do not rewrite old manifests, source paths or receipts to claim a new identity.

### Forge worker provenance

Preparing immutable inputs for a Forge cell verifies its release/export and
tokenizer origin while the controller sources are available. The sealed assets
include the tokenizer and its manifest, packed arrays, release manifest, corpus
report, license report, audit, export metadata and portable `corpus/binding.json`.
Selected bakeoff tokenizers also retain their selection report. A reused frozen
tokenizer keeps its original export identity even when the training corpus changes.

Local and worker smoke/warmup pilots verify this portable evidence without
reopening the original export's `run.yaml`. The original config paths remain
recorded provenance; relocation does not rewrite scientific identities. Both the
sealed file inventory and the internal release/export/tokenizer bindings must
verify. Missing or changed evidence fails closed, including when prepared inputs
are reused for dispatch.

Training copies the evidence into the native run's authenticated artifact
inventory. Controller ingestion preserves those bytes, and a dispatched child
resume carries the parent's evidence and selected checkpoint. The bounded offline
regressions in `tests/test_forge_dispatch.py` cover authored prepare/lock, relocated
smoke and warmup, parent ingestion, child resume, and missing/tampered provenance.
They establish execution and integrity only; no accelerator or model-quality
claim follows. Older sealed inputs without portable provenance must be rebuilt
from available verified sources before reuse.

### Explicit operational source compatibility

An operational-only code repair still changes the package source digest. Reusing
historical lock and prepared bytes across that change requires an explicitly
reviewed, versioned `OperationalSourceCompatibility` record, not a source-identity
override or a verifier bypass. The record binds full baseline/execution commits,
their independently derived package inventories, the exact changed-file digests
and the operator's non-scientific authorization. Only the listed operational
integration paths are eligible; eligibility is not proof of semantic equivalence.
Review the concrete diff before publishing the record.

The operator pins all three environment variables:
`SPARSELAB_SOURCE_COMPATIBILITY` (absolute committed record path),
`SPARSELAB_SOURCE_COMPATIBILITY_SHA256` (reviewed record byte digest), and
`SPARSELAB_SOURCE_COMPATIBILITY_COMMIT` (trusted commit containing those exact
bytes). Partial, changed, symlinked, uncommitted or incorrectly inventoried records
fail closed. Git authentication and actual installed-package identity remain
mandatory. Local runtime children inherit these pins; a remote worker must
independently receive and authenticate them.

The compatibility binding preserves historical lock identity and permits deep
authentication of the already-existing historical cache. It never prepares new
data under a legacy digest. Fresh runtime probes, dispatch specifications, run
manifests and checkpoints record the actual execution source digest. Without the
explicit binding, ordinary exact-source rejection remains unchanged. Public
`open_lock` still performs cold artifact verification with the authenticated
binding; publication's private proof remains limited to its original process.



## Derive a verified declaration

Use `experiment derive` to publish a new declaration after a typed, cold
resolution of the complete candidate. It is for declaration-level changes that
phase `set` cannot express, such as retention, comparisons, axes, or a base-run
setting. It does not train, enqueue work, publish a lock, or make a runtime-fit
claim.

Values are strict JSON and each changed field needs its own `--set`. Replace
phases, axes, or comparisons as complete JSON arrays; dotted sequence indexes
are not supported. The output parent must already exist without symlinked
components. Neither output can be overwritten.

```sh
uv run --locked --extra cpu sparselab experiment derive "$PLAN" \
  --set 'id="offline-training-chain-v2"' \
  --set retention.keep_periodic=false \
  --output "$WORK/inputs/plan-v2.yaml" --json
```

The output is a YAML declaration and
`plan-v2.yaml.derivation.json`. The receipt binds source and output byte hashes,
records the requested assignments, normalized observed declaration delta, path
rebindings, and a `resolved_experiment` validation boundary. It is an
operational provenance record, not a lock or a scientific artifact identity.
Pass `--prepared PREPARATION.json` only when the candidate requires an existing
prepared corpus-variant record; the supplied record is authenticated and not
rewritten. Its plan ID must match the derived declaration: an ID change cannot
reuse preparation bound to the old ID.

Derivation validates the same declared inputs, artifact identities, comparisons,
and continuation rules as a cold `experiment lock` resolution. It preserves
declaration references only when the new output location can resolve to exactly
the same targets. It rejects an unsafe relocation rather than copying or
rebasing corpus, suite, capability, or prompt declarations. Then create the
immutable runnable identity through the ordinary `experiment lock` command:

```sh
LOCK=$(uv run --locked --extra cpu sparselab experiment lock \
  "$WORK/inputs/plan-v2.yaml" --json | jq -r '.lock')
```

Use `experiment bind-inputs` before derivation for templates that lack
authenticated direct inputs. A derivation cannot make an unbound template
verified or infer replacement artifacts, checkpoint selections, or continuation
compatibility.

## Pilot the resolved configuration

The locked cells contain the effective run configs after input binding and
phase overrides. Extract them for explicit local inspection and pilots:

```sh
uv run --locked --extra cpu sparselab experiment export-config "$LOCK" \
  --cell pretrain:single --output "$WORK/pretrain-effective.yaml" --json
uv run --locked --extra cpu sparselab experiment export-config "$LOCK" \
  --cell continue:single --output "$WORK/continue-effective.yaml" --json
uv run --locked --extra cpu sparselab inspect "$WORK/pretrain-effective.yaml" --json
uv run --locked --extra cpu sparselab stage "$WORK/pretrain-effective.yaml" \
  --through warmup --output "$WORK/stages/pretrain"
uv run --locked --extra cpu sparselab inspect "$WORK/continue-effective.yaml" --json
uv run --locked --extra cpu sparselab stage "$WORK/continue-effective.yaml" \
  --through warmup --output "$WORK/stages/continue"
```

Pilots demonstrate fit and execution on the selected runtime. They do not run
the checkpoint transition or establish model quality.

## Dispatch the parent, then its child

Submit the root phase and start the controller:

```sh
uv run --locked --extra cpu sparselab experiment run "$LOCK" --phase pretrain --json
uv run --locked --extra cpu sparselab controller run --store "$WORK/controller"
```

The first command registers a local worker when none is specified and enqueues
the selected cell. The controller stays in the foreground. In a second terminal
with the same `WORK` and `LOCK`, monitor the parent:

```sh
uv run --locked --extra cpu sparselab experiment list --store "$WORK/controller"
```

Wait for the parent's `status: COMPLETE` and `ingestion_status: COMPLETE`.
Then explicitly submit the dependent phase; the same running controller will
dispatch it:

```sh
uv run --locked --extra cpu sparselab experiment run "$LOCK" --phase continue --json
uv run --locked --extra cpu sparselab experiment list --store "$WORK/controller"
```

`experiment run` enqueues work; it does not wait or automatically schedule
dependent phases. Without `--phase`, it selects root phases only. A dependent
submission requires exactly one matching completed and ingested parent, verifies
the selected generation, and records its digest before dispatch. Failed,
interrupted or pending parents do not satisfy this dependency. Use `--cell
pretrain:single` to select this example's single root cell; larger matrices
include their axis labels in the returned cell IDs.

## Collect and read the outcomes

After both phases complete and ingest:

```sh
uv run --locked --extra cpu sparselab experiment collect "$LOCK" --json > "$WORK/collection-result.json"
INDEX=$(jq -r '.index_path' "$WORK/collection-result.json")
uv run --locked --extra cpu sparselab experiment reconstruct "$LOCK" --index "$INDEX" --json
uv run --locked --extra cpu sparselab dashboard --runs-dir "$WORK/controller"
```

The index retains pending, failed, interrupted and invalid cells as distinct
states. Reconstruction verifies the index and reports five views: input
identity, declared contrasts, observed phase lineage, evidence coverage and
decision readiness. Omitting `--index` gives planned-only views. These reports
do not score arbitrary declared evaluations, establish effectiveness or promote
a baseline automatically.

Use the run IDs returned by submission/list for checkpoint verification,
held-out evaluation and generation, with `--runs-dir "$WORK/controller"`:

```sh
uv run --locked --extra cpu sparselab eval RETURNED_RUN_ID --runs-dir "$WORK/controller"
uv run --locked --extra cpu sparselab evidence RETURNED_RUN_ID --runs-dir "$WORK/controller" --json
uv run --locked --extra cpu sparselab generate RETURNED_RUN_ID --runs-dir "$WORK/controller" \
  --prompt "A path" --max-new-tokens 8
```

Keep the declaration and small protocols under version control; keep mutable
sources, workers, stores, stages and reports under this ignored workspace.
Preserve failed and interrupted runs, and retain verified previous/best/terminal
checkpoints under the declared policy. Only remove task-owned output after
confirming no worker is active. For exact comparison contracts, typed external
artifacts and research lifecycle decisions, see [experiments](experiments.md)
and [evidence](evidence.md).
