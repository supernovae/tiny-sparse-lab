# DevMind v4 MODEL-0 Try 2: acquisition identity blocker

## Verdict and stop boundary

`DEVMIND_MODEL0_INCOMPLETE`. Reconstruction was explicitly stopped during
acquisition after demonstrating implementation-identity divergence. No tokenizer
was fitted, no MODEL-0 run was submitted, and no Developer SFT was performed.
The historical corpus expectation and all prior research records are unchanged.

The checkout started clean on updated `main` at
`d47b2cdd725b444ad5d2297ee9096673fa7733c7`. Work uses the normal branch
`research/devmind-model0-try2`, without a worktree. Commit `f12506d` independently
removed only the delivered issue #3 backlog section; `651bb35` committed the
operational preflight before acquisition. The commit containing this report is
the publication commit, not a scientific-input or training commit.

## Demonstrated earliest divergence

The current project acquisition digest equals the historical landmark:
`ee95707a159d4fbc62e8f22cce6301010cbf86a500db6c010a2d282e1166e6b6`.
The pinned source commit remains
`e260ea866ded8ec5beb23cbacec1fb3c120c1d16`.

`src/sparselab/corpus/acquisition.py:_adapter` binds the complete implementation
file SHA in each newly acquired snapshot. Both acquisition and `verify_snapshot`
include that adapter record in the snapshot hash. The implementation at the
pinned source commit hashes to
`15e891576f81a8c2096e9559c07bef436cb6ed617b131f08af6d9c0e9be40974`;
the execution checkout hashes to
`acde12f8c10e13c1c8bf3dd535fa19cc0f27287f9a6096b8617acb8fd5d76bb4`.
The sole intervening acquisition-file commit is `c82993b`, “Format existing
corpus code and tests with Ruff”. Parsing both files and comparing Python ASTs
without location attributes returned equality.

A real network acquisition of `v4_db_clickhouse_sql`, pinned to
`5ac4352fc81602c8c3824c9ebfad10a4b0e900ab`, produced a snapshot independently
reopened with `verify_snapshot`:

- Declaration SHA: `f34535d58b0a4bfe4daf5998db06c42f3ddd4d06934c570884cbc37da19ae32e`.
- Actual verified snapshot SHA: `d2809715da66e7c7df5fdbd68cd5fe3cd1ac214260053cc64702dc4b17744a22`.
- Selected files: 11,713.
- Holding that verified file inventory and declaration fixed, substituting only
  the pinned implementation's adapter hash produces
  `3e5576cfcb4e39dc5b73cc9aded06f563ae27a3e1f637c2a72514daeed248bfe`.

The last value is a **counterfactual calculation**, not an observed historical
snapshot receipt. No historical snapshot receipt was recovered. This proves
fresh acquisition identity is sensitive to the intervening formatting change;
it does not manufacture an actual reconstructed build or release SHA. The
expected release remains
`0f06976e1c90309c7920cd23b1079179ba3cd026fdde55e934c1320fe6dc5fd4`,
and actual release SHA is unavailable. Recovery verifies source declarations
against its source pin but executes acquisition from the current interpreter;
it does not replay the pinned implementation automatically.

Following the approved earliest-divergence stop rule, the reconstruction process
was cancelled and its partial snapshots/downloads were preserved. Process
inventory afterward showed no surviving acquisition process. No build, release,
or acquisition lock was present in the corpus root. No digest was aliased and
no receipt was rewritten. Repairing this provenance contract is outside the
approved schema-3/tokenizer-export compatibility fix.

## Verification and retained evidence

Commands exercised:

- `git switch main`, `git pull --ff-only`, clean status, actual starting SHA,
  and creation of `research/devmind-model0-try2`.
- Locked CPU control-plane `runtime env list`, `show rocm-7900xtx`, and
  `doctor rocm-7900xtx`. Doctor returned `READY` with a real BF16
  forward/backward/AdamW update on device 0, AMD Radeon RX 7900 XTX.
  Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`; no vendor package changes.
- Read-only `recovery inspect` and `recovery plan`; both reported missing
  external corpus bytes and four external scientific decisions. These calls
  did not create the corpus or task namespace.
- `recovery reconstruct experiments/research/devmind-pretrain-v4/recovery.yaml
  --allow-network --json` with `SPARSELAB_WORK_DIR=/srv/sparselab/state`.
  It acquired real pinned sources before the explicit identity stop. It did
  not return a successful reconstruction result; empty stdout/stderr files
  are not success receipts.
- Independent `verify_snapshot` on the ClickHouse snapshot; historical/current
  module hashing; Python AST equality; fixed-file identity sensitivity check.
- Focused existing regression set: **29 passed** across runtime forecasting,
  runtime CLI, training throughput, data progress, dashboard queries, and
  completed-training forecasting reuse. No Python implementation was changed;
  the broader suite and Ruff were not rerun.

Checked-in compact observations:
[`model0-try2-preflight.json`](model0-try2-preflight.json) and
[`model0-try2-blocker.json`](model0-try2-blocker.json).
Full operational receipts remain at
`/srv/sparselab/state/experiments/devmind-pretrain-v4-model0-try2/`, including
runtime list/show/doctor, initial recovery inspect/plan, reconstruction logs,
and acquisition-divergence JSON. Partial payloads remain under
`/srv/sparselab/state/corpora/devmind-v4/`; no source text is published.

A thin archive of the unchanged historical recovery declaration was created at
`/srv/sparselab/state/experiments/devmind-pretrain-v4-model0-try2/blocked-recovery-thin.tar`
and independently verified: **128 verified members**, `family_lineage:
NOT_DECLARED`. The five unresolved references are `corpus`, `tokenizer-choice`,
`model-zero-intent`, `runtime-requirement`, and `evaluation-declaration`, all
`MISSING_EXTERNAL`. It is a declaration archive, not a MODEL-0 archive, and
contains no source/weight payload or the newly authored blocker report.
Create/verify JSON receipts are retained beside it. No portable archive was
attempted under the metadata-only publication policy.

## Acceptance inventory

| Category | Observed outcome |
| --- | --- |
| Beginning/ending revision | Beginning `d47b2cd`; pre-acquisition `651bb35`; publication is the commit containing this report, reported with its remote ref at handoff |
| TODO audit/gaps | Removed closed issue #3 only; added demonstrated pinned-implementation recovery/preflight gap with tests/docs acceptance; other backlog retained |
| Root/storage/headroom | External ext4 `/dev/sde`, `/srv/sparselab/state`; after stop 1,018,651,287,552 bytes and 67,080,718 inodes available; required byte margin 270,275,294,208 |
| Capacity projection | Preliminary 568 GB growth reservation, not a guarantee; remote Git cache unbounded; exact model/export preflight unavailable before those declarations |
| Runtime | Registered ROCm device 0 READY, BF16 optimizer pilot passed; separate CPU control plane |
| Corpus | Historical expected release SHA unchanged; no actual release verified; acquisition identity blocker above |
| Bakeoff candidates | 16,384 / 24,576 / 32,768 candidates not fitted; scores and per-group measurements unavailable |
| Selected tokenizer | Unavailable; no winning SHA inferred |
| Architecture/parameter count | Planned architecture not materialized; actual inventory unavailable |
| Prepared data | Not prepared; cache ID, manifest SHA, train/validation document and token totals unavailable |
| Developer tokens/budget | D, steps, supervised targets and actual exposure unavailable; historical counts not substituted |
| Scientific/plan identities | No MODEL-0 plan locked; scientific and plan SHA unavailable |
| Runtime binding | No MODEL-0 operational binding SHA; doctor is not a binding |
| Full-shape smoke/warmup | Not run; no MODEL-0 VRAM fit or throughput claim |
| Campaign | No declaration materialized or attempt submitted |
| Run/experiment/attempt IDs | None; training attempt count zero |
| Steps/targets | MODEL-0 not started; completed training measurements unavailable |
| Checkpoint | No generation or checkpoint SHA |
| Heldout diagnostics | Unavailable; no loss, perplexity or valid-target count |
| Generation diagnostics | Unavailable; no checkpoint-bound prompt panel executed |
| Evaluation index | Unavailable; no suite index SHA |
| Model readiness | Not evaluated; neither READY_FOR_NEXT_STAGE nor quality readiness claimed |
| Family | Not authored without verified model lineage |
| Recovery | Historical declaration/source pin/expected release preserved unchanged; partial acquisition retained |
| Archive/publication | Thin historical declaration archive verified, 128 members and five unresolved references; compact blocker/preflight/report published separately; no model/source/weight payload |
| Code/hardware verification | 29 focused CPU tests; real registered BF16 ROCm doctor; actual network acquisition and snapshot verification; full MODEL-0 hardware and long-run gates not exercised |

## Required resolution

Restore a verified original release/snapshot closure, or explicitly review a
pinned-implementation replay that preserves the original immutable identities.
Do not repair this by changing the historical expected release, spoofing adapter
metadata, or accepting a newly hashed release as the old one. Tokenizer fitting
and all model stages remain blocked on exact corpus verification.
