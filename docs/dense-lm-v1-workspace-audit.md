# dense-lm-v1 workspace audit (2026-09-27)

This is a local execution-state migration after the accepted `dense-lm-v1`
promotion. It does not change a scientific result, acceptance gate, published
artifact, run identity, or checkpoint. No training, optimizer update, evaluation
inference, or new generation was performed.

## Origin of the three stores

At the audited `main` revision `fcf63db`, `configs/dense_lm_v1.yaml`,
`configs/dense_lm_v1_seed17.yaml`, and `configs/dense_lm_v1_seed73.yaml` explicitly
selected three `runs-dense-lm-v1-seed*` logging roots. The reference matrix repeated
those root overrides for each seed. The execution instructions in
`docs/research/dense-lm-v1.md` constructed a seed-specific `RUN_ROOT`, trained each
config separately, and told readers to continue each seed in its configured
root. Thus both the supplied configs and the manual procedure encoded the wrong
filesystem topology. It was not an unavoidable CLI default or controller limit.

The three original databases each contained six runs (one parent and five
milestone children), 86,076 metric rows, 173 events, 114 stage rows, 30 checkpoint
records, and 4,467 replication envelopes. Their controller worker, experiment,
and attempt tables were empty. There was no historical study submission receipt
to merge. All runs were terminal: each seed had five interrupted ancestors and
one completed terminal child. No active experiment was relocated.

## Path classification

| Reference | Meaning and treatment |
| --- | --- |
| Run IDs, parent IDs, manifest/config hashes, checkpoint hashes, tokenizer/data/source identities | Scientific identity; unchanged. |
| `logging.root_dir` and staged tokenizer paths inside resolved configs, manifests, observations, stage records, and replication envelopes | Historical execution location; preserved byte-for-byte, including old repository-root and `/tmp` paths. |
| SQLite `runs.latest_checkpoint` | Current local locator; rebased in the new working projection only. The original database and original replication envelopes preserve its historical value. |
| Checkpoint member paths, manifest artifact paths, learning-observation checkpoint paths | Relative to the explicitly selected run tree; ordinary relocation is supported. |
| Lifecycle registry, promotion, acceptance, static report inputs | Deliberately retained evidence under `artifacts/`; unchanged. Frozen report inputs keep their original paths/configs. |
| Prepared-data and tokenizer caches under `artifacts/` | Reusable input identities and existing cache policy; not relocated into the experiment. |

`load_run` verifies a run through the explicit `runs_dir`, then resolves tokenizer,
data, manifest members, and checkpoint generations relative to that run. It does
not need the old logging root or staging tokenizer pathname. Checkpoint lineage
uses parent IDs and digests, not a new filesystem root per continuation.

## Consolidation

The current local layout is:

```text
sparselab-work/experiments/dense-lm-v1/
  receipt.json
  runs/
    experiments.sqlite3
    dense-lm-v1-seed17/                 # interrupted parent
    dense-lm-v1-seed17-step400/         # child; other milestones are peers
    ...                               # all 18 run IDs across all three seeds
  staging/
  exercises/
  captures/
  local-reports/relocation/
    journal.json
    inventory-before.json
    verification.json
    staging-relocation.json
    original-databases/                # provenance archive, not execution stores
```

There is one working store. Original databases are retained as forensic evidence
inside the ignored experiment workspace; they are not per-seed stores for future
submission, resume, collection, or dashboards. The single `receipt.json` lists
all seed coordinates and their complete lineages, explicitly labels itself as a
historical consolidation record, and does not pretend the manually executed
experiment was a controller study submission.

Ordinary directory renaming alone could not merge three SQLite files: each had a
distinct replication origin, colliding local sequence/event numbers, and absolute
current checkpoint pointers. The narrow internal
`consolidate_standalone_stores` helper reuses existing checked replication import,
preserves timestamps, rebases only current checkpoint pointers, and compares all
scientific projection tables against the original databases. It rejects active
runs, overlapping run IDs, duplicate origins, controller queues, replicated
sources, unsafe paths, and existing destinations. It does not train, rewrite
source databases, move files, or create a new public CLI/resolver. Callers must
own stopped sources exclusively. Failed destination projections remain available
for inspection and cannot be silently reused.

On this machine the run trees and target shared a filesystem with approximately
908 GiB and 66.9 million free inodes before relocation. All 18 complete run
directories were moved by same-filesystem rename. Every one of their 633 files
retained its device, inode, size, and modification timestamp. Original SQLite
files, including sidecars, were archived without rewriting them. The three old
top-level seed directories no longer exist. `runs-rocm/` was left untouched.

Eleven completed seed staging bundles and earlier dense-LM readiness pilots/probes
were also found in `/tmp`, on a different filesystem. All eleven were relocated
by verified copy followed by removal of the original: all 798 files matched
SHA-256 inventories and sealed-bundle verification passed at both locations.
The read-only reference smoke exercise JSON was likewise moved into `exercises/`
after its bytes matched. Failed/negative observations within the readiness
artifacts were retained. Original path fields remain
historical. Source-version drift is explicitly allowed only for this read-only
verification of old bundles; it is not permission to execute a stale stage.

## Verification and limits

- All 90 retained checkpoint generations passed full native-state verification.
- All 18 runs loaded their existing integrity-bound evaluation evidence.
- All 21 learning observations retained valid file and embedded identity hashes.
- The three terminal runs loaded for CPU inference using the new store, without
  evaluating or generating any tokens. All three loaded again after the original
  `/tmp` staging paths were removed, confirming those historical paths are not
  needed for inference loading.
- All eight retained static report bundles passed `load_report_bundle`.
- Lifecycle baseline description remains `known_good` and `verified` for
  `dense-lm-v1`. Its compact published evidence intentionally does not promise
  availability of full checkpoint bytes on every machine.
- The focused relocation regression checks three seed origins, six parent/child
  trees, one external destination, unchanged file bytes/inodes and historical
  config/event paths, preserved parent IDs, rebased current locators, and rejection
  of active and overlapping runs. Both tests pass.
- Ruff lint and formatting pass for the helper and its tests. `git diff --check`
  passes; no tracked file under `artifacts/` or the lifecycle registry was changed.

Global lifecycle validation still reports an unrelated missing report bundle for
`learned-engram-portability-v1` (`maturity_measured_evidence_missing`), plus five
missing-evidence warnings. This migration does not repair or reinterpret those
records. Three byte-bound historical instructional documents
(`context-engram-study.md`, `path-domain-corpus.md`, and `portable-engram.md`) remain
exactly as registered, even where their old examples use root-level run paths.
The current [workspace guide](workspaces.md) supersedes their local path advice;
their registered hashes were not rewritten.

The integrated focused suite passed 80 tests across `test_workspace_hygiene`,
`test_workspace_relocation`, `test_study`, `test_research_workbench`,
`test_study_reporting`, `test_reference_exercise`, `test_runtime_cli`,
`test_workers_cli`, `test_worker_queue`, `test_config`, `test_workdir`, and
`test_research_lifecycle` (all under `tests/`, using `uv run --locked pytest`).
The subsequently added historical-document identity regression also passed;
rerunning all six workspace-hygiene tests brings the verified total to 81 distinct
cases. Repository-wide Ruff lint passed, and all nine changed Python files passed
format checks. Repository-wide formatting still reports pre-existing differences
in unchanged `experiments/reporting.py`, `research/lifecycle.py`, and
`tests/test_research_lifecycle.py`.

Local verification details and the migration scripts are retained under
`local-reports/relocation/`, not promoted into scientific evidence. Hardware
execution, new training, new generation, and long-run resume are deliberately
unexercised. Old explicit external paths remain supported; after this local move,
commands addressing the former seed-store locations must instead select the
new shared `runs` directory. Historical evidence must not be edited to hide that
location change.
