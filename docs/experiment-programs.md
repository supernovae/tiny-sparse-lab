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

From the repository root, use the locked Python 3.14 environment. Leave the
global work base at `sparselab-work`; the plan ID supplies the experiment
subdirectory:

```sh
export SPARSELAB_WORK_DIR="$PWD/sparselab-work"
SAMPLE=experiments/samples/training-program
WORK="$SPARSELAB_WORK_DIR/experiments/offline-training-chain-v1"
mkdir -p "$WORK/inputs"
df -h "$WORK"
df -i "$WORK"
uv run --locked --extra cpu sparselab tokenizer train "$SAMPLE/tokenizer.yaml"
uv run --locked --extra cpu sparselab data prepare "$SAMPLE/run.yaml"
```

Turn the phase template into a complete authored plan with typed external
artifacts. This script re-verifies the prepared cache and uses actual domain
identities rather than placeholder hashes:

```sh
uv run --locked --extra cpu python - <<'PY'
import json
import os
from pathlib import Path
import yaml
from sparselab.config import load_config
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.manifest import sha256_file

sample = Path("experiments/samples/training-program")
work = Path(os.environ["SPARSELAB_WORK_DIR"]) / "experiments/offline-training-chain-v1"
config = load_config(sample / "run.yaml")
prepared = prepare_data(config, load_tokenizer(config.tokenizer.path))
manifest = json.loads((prepared.root / "manifest.json").read_text())
plan = yaml.safe_load((sample / "offline-chain.yaml").read_text())
run_path = work / "inputs/run.yaml"
run_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
plan["base_run"] = str(run_path)
plan["artifacts"] = {
    "tokenizer": {
        "kind": "tokenizer", "version": 1, "producer": "sparselab tokenizer train",
        "identifier": config.tokenizer.path.parent.name,
        "path": str(config.tokenizer.path), "sha256": sha256_file(config.tokenizer.path),
    },
    "packed": {
        "kind": "prepared_data", "version": 1, "producer": "sparselab data prepare",
        "identifier": manifest["settings_sha256"], "path": str(prepared.root),
        "sha256": manifest["manifest_sha256"],
    },
}
plan["inputs"] = {"tokenizer": "tokenizer", "prepared_data": "packed"}
(work / "inputs/plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False))
PY
PLAN="$WORK/inputs/plan.yaml"
uv run --locked --extra cpu sparselab experiment validate "$PLAN" --json
uv run --locked --extra cpu sparselab experiment inspect "$PLAN" --json
uv run --locked --extra cpu sparselab experiment lock "$PLAN" --json > "$WORK/lock-result.json"
LOCK=$(uv run --locked --extra cpu python -c 'import json,sys; print(json.load(open(sys.argv[1]))["lock"])' "$WORK/lock-result.json")
uv run --locked --extra cpu sparselab experiment explain "$LOCK" --json
```

Validation checks declaration/config shape. Inspection also cold-authenticates
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
independently runs the full artifact verifiers again; neither the availability
sidecar nor a remembered digest substitutes for cold verification.


The printed lock path is authoritative. Keep one code revision, the declaration
and its inputs fixed across execution. Reopen checks reject changed source or
artifact identities. For edits/repetitions, copy the declaration, choose a new
plan ID and adjust its file references to their new location before preparing;
never rewrite a running plan or its artifacts.

### Direct story inputs

Direct `tinystories` and manifest-backed `local_stories` runs use the same two
external artifacts shown above: `tokenizer` and `prepared_data`. They do not
require a Corpus Forge release or export. Use the story run as `base_run` and
prepare its tokenizer and data explicitly before constructing the artifact
bindings; `experiment prepare` still only materializes `corpus_variants`.
The [TinyStories microlab](tinystories-microlab.md) supplies matching run and
tokenizer preparation examples. Use absolute paths in generated declarations.

For direct TinyStories, pin `dataset.revision` to the full 40-character Hub commit
SHA (for example `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`), with
`dataset_config` omitted or `default`. The loader uses the `text` field and the
separate `train` / `validation` splits. A moving revision such as `main` cannot
be locked. Locking verifies the tokenizer's train-only source/revision provenance
and the prepared arrays' dataset, tokenizer, packing and implementation identity;
it does not download the corpus or repeat tokenizer training.

For `local_stories`, retain the snapshot's `manifest.json`, `train.jsonl`,
`validation.jsonl` and excluded-ordinal inventory on the controller. Set the
run's `train_path`, `validation_path` and `source_manifest_path` to that snapshot,
with its pinned revision and license. Train the tokenizer against this same
snapshot. Lock verification checks all snapshot bytes and split disjointness,
the tokenizer's snapshot-manifest digest, and the prepared data's canonical
snapshot identity. A valid tokenizer from another snapshot is insufficient.
Local story token caps must accommodate every selected whole story; preparation
rejects a cap that would truncate one.

Publication and later lock reopening reject changed source files or artifacts,
including when verification reuse is enabled. The availability sidecar retains
the original snapshot paths used to verify each tokenizer; phase-level source
paths are checked independently. Keep those controller snapshots available for
publication and reopening. Worker dispatch carries the verified
prepared arrays and tokenizer in its sealed bundle; it can execute offline after
relocation without reacquiring or copying the raw story source. These checks
establish input integrity and execution, not model quality or remote corpus
availability. Corpus Forge and other dataset routes retain their existing input
requirements.

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


## Pilot the resolved configuration

The locked cells contain the effective run configs after input binding and
phase overrides. Extract them for explicit local inspection and pilots:

```sh
uv run --locked --extra cpu python - "$LOCK" "$WORK" <<'PY'
import sys
from pathlib import Path
import yaml
from sparselab.experiments.lock import open_lock

locked = open_lock(Path(sys.argv[1]))
work = Path(sys.argv[2])
for cell in locked.cells:
    path = work / f"{cell.phase}-effective.yaml"
    path.write_text(yaml.safe_dump(cell.config.model_dump(mode="json"), sort_keys=False))
    print(cell.id, path)
PY
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
INDEX=$(uv run --locked --extra cpu python -c 'import json,sys; print(json.load(open(sys.argv[1]))["index_path"])' "$WORK/collection-result.json")
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
