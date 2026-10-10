# DevMind MODEL-0 Try 2 ancestry replay

**Verdict: `CORPUS_LINEAGE_REPLAY_BLOCKED`.** The first real corpus gate failed at
v2's expected build identity. No release was frozen, no snapshots were imported
into v3, and no later corpus or MODEL-0 stage was attempted.

## Execution binding

- Starting clean `main`: `2241cc7bdabbc373dd234a95c2e43a54db1e36e0` (PR #26).
  SSH pull failed authentication; authenticated HTTPS `git pull --ff-only`
  reported main already current. One normal branch in the same checkout:
  `research/devmind-model0-try2-ancestry`; no worktree.
- Lineage and the two future code items committed before acquisition:
  `7d0f6df`. Tested replay implementation, coordinator, regressions and lifecycle
  documentation committed/pushed before acquisition:
  `906435203c58a7b0af9cc8d908ef2f0f78c58532`. The launch checkout was clean.
- Operational roots: `ROOT=/srv/sparselab/state`,
  `TASK=$ROOT/experiments/devmind-model0-try2-ancestry`, `V4_WORK=$TASK/v4`.
  This was a fresh task namespace, not the previous clean-v4 replay workspace.
  The original 19 pinned sources and reviewed private-research rights policy
  were retained; no publication authorization or Developer SFT was added.
- Exact four-commit gates are registered in
  [`ancestry-replay-lineage.json`](ancestry-replay-lineage.json).
  All four commit objects and project blobs were present; final CMake/Metro IDs
  were checked against their pinned Git YAML. Read-only `--validate-only`
  reported `LINEAGE_VALID` without acquisition/environment creation.

## 1. v2 project, acquisition, build and release

The real command was `uv run --locked --extra cpu python
experiments/research/devmind-pretrain-v4/replay_ancestry.py --state-root "$TASK"
--through v2 --allow-network`, executed with a task-owned external live storage
monitor. Historical producer: `078b8f5ed533c1dd915394ea57033de646ae5d15`;
Git tree: `090b5d8f20ff583e7136fc3973ada5a52bfcb113`.

| Check | Observation |
| --- | --- |
| Historical project SHA | `e58e647981f4a879439ff1785045023e652255ef5f8778769d09ac4c0932f54d` |
| Expected project SHA | Not documented; recorded from authenticated Git declarations, not guessed |
| Acquisition | Complete 19-source lock; all snapshots independently verified |
| Recorded adapter SHA, all 19 | `32f94cae274e6c98057608e37fb59473cf8ba417cfa0aa85d7ab164abbd8c56b` |
| Expected build | `36abf24160bf55d3e76dae5e67766dca7ef61d73e217f007f215bd76b92652bf` |
| Actual independently verified build | `21b85b0f3e1f9a78fdd24dc276d5c254a2b94c2c1ce6b879580874bc92bd2de2` |
| Build gate | `EXPECTED_BUILD_MISMATCH`; stopped before freeze |
| Expected release | `50ee7ae6355ae97aeae44544d9f1f66362249767f315b1e98346f3b8df216b6d` |
| Actual release | None; the release gate was not executed |

The current verifier authenticated the historical Git source, archived lock,
complete snapshot inventories, acquisition receipts, operational logs and actual
immutable build when recording and reopening the failed receipt. This proves
artifact integrity, **not** equality with the historical expected build.

### Preserved references

Paths below are relative to `$TASK` unless absolute:

- Replay receipt: `replay/receipts/2e508fcf7a9c46e091ad6dbfd5f1e47d.json`.
  File SHA-256: `850c6d2a78671283f6729b7c0a63eae7854d99c32dc9d52dea1df2e7f85bb65d`.
  Canonical record SHA-256:
  `399b85e2cd71534892b087938535e820960521d3cd4591f764af01f683c0a895`.
- Unaltered archived acquisition lock:
  `replay/acquisition-closures/2e508fcf7a9c46e091ad6dbfd5f1e47d.json`.
  SHA-256: `61c02674331f8c8a9f6cd4f8dfaa46a18c44461a0e4cf9288bea4f6b8cda36ef`.
- Typed attempt: `replay/ancestry/attempts/v2-4aeae59236d84c6b8c88b522ae833b38.json`;
  stage record: `replay/ancestry/stages/v2.json`.
- Actual build: `replay/work/078b8f5ed533c1dd915394ea57033de646ae5d15/corpora/devmind-v2/builds/21b85b0f3e1f9a78fdd24dc276d5c254a2b94c2c1ce6b879580874bc92bd2de2/`.
  Its `build.json` file SHA-256:
  `84c1350bcddff9573d642df070b88d36bca73a91deffc537b9dc2acc223e2559`.
- Launch/stdout/stderr/storage records:
  `replay/logs/launch-monitor/v2-1790890769727692987.*`.
  Prerequisite record: `replay/logs/v2-launch-verification.json`.
- Separate current-code reauthentication: `VERIFIED_FAILURE`, 341.46 seconds;
  `replay/logs/independent-v2-failure-verification.json`.
  SHA-256: `e109fdbded7b20dde14d5226cf4b4ab0cdb9e84182707794c2a710e42d8c4fe3`.
  It confirmed no release directory and no later phase stage records.
- Checked-in compact reference:
  [`ancestry-replay-blocker.json`](ancestry-replay-blocker.json).

All 19 observed source-ID→snapshot-ID mappings and recorded adapter hashes are
retained in the authenticated receipt/stage record and compact blocker reference.
Full locks, fetched bytes, caches, snapshots, normalization receipts, build data,
source export and dedicated uv environment remain external and preserved.

### Divergence limit and secondary diagnostics

The authenticated digest gate is the provable identity divergence. The original
historical v2 `build.json`/acquisition closure and expected 19-snapshot map were
not located in the checked-in v2 records or inspected persistent stores; the
protocol's legacy `sparselab-work/experiments/devmind-pretrain-v2/` path is absent.
A broad persistent-tree glob timed out; it is **not** proof of global absence.
Without original metadata, `identity.lock.<source_id>.snapshot_sha256` and the
remaining original build-identity leaves cannot be compared. No leaf-level hash
root cause is claimed, no alternate producer was tried, and no SHA was injected.

The fresh build agrees with historical aggregate counts: 710,034 normalized,
709,326 retained, LM train/validation/test 698,643/10,571/112. Smaller reported
measurements differ from the versioned v2 protocol:

| Diagnostic field | Historical protocol | Fresh build |
| --- | ---: | ---: |
| Retained FineWeb train documents | 360,871 | 360,868 |
| Retained OpenWebMath train documents | 45,089 | 45,092 |
| Wikipedia train UTF-8 bytes | 104,291,290 | 104,276,300 |
| All train UTF-8 bytes | 3,313,224,913 | 3,313,252,138 |

These are observed summary discrepancies, not original identity-payload leaf
comparisons or proof of a cause. Equal aggregate counts cannot override a digest
mismatch. The missing original metadata is needed to explain the identity.

## 2. Nineteen exact v2→v3 imports

**Not attempted: 0 imported.** A failed v2 receipt is not a successful parent
closure. No v3 target lock or imported source identities were manufactured.

## 3. v3 project/build/release checks

**Not executed.** Producer `b87134335efb8eaeccd91a246fadad4242fd9635` and exact
expected build/release remain registered, not observed outputs of this attempt.

## 4. Fifty-eight exact v3→v4 imports

**Not attempted: 0 imported.** No successful v3 parent exists in this attempt.

## 5. Intermediate v4 123-source build-only gate

**Not executed.** Producer `4a01123beacd4f6a5191ae845bd976568d5d06f5`, project
`2a5c988d5e1eccca8b52f78b391ee4ab50358813c1982195d111733f1c394ae8`, and build
`dba7359677569d8ed905d3f5fe7552d323b240e969354b9b19c551b899f4d985` remain
expectations. No intermediate freeze or build-only success is claimed.

## 6. Final v4 121 retained/two changed identities

**Not executed.** The lineage correctly distinguishes 123 previously present
sources from 121 reused immutable snapshots and two changed declarations:
`v4_iac_cmake_build` and `v4_runtime_metro_js`. No final-v4 acquisition map,
retention count or changed snapshot identity was observed.

## 7. Final build-before-freeze, exact release and independent verification

**Not executed.** The final source pin `e260ea866ded8ec5beb23cbacec1fb3c120c1d16`,
expected build `79b568629fc6682c33bf1212eef0653cdb05d26a772654019458f44af2ceaee5`
and release `0f06976e1c90309c7920cd23b1079179ba3cd026fdde55e934c1320fe6dc5fd4`
were not substituted or changed. No `corpus describe` of that release ran.
The historical 818,630/816,117 normalized/retained documents, LM
796,563/16,411/3,143 and 735,281,099 developer train bytes are untested landmarks
for this attempt, not recovered evidence.

## 8. Code backlog and bounded replay verification

`TODO.md` retains the blocked-recovery receipt-path item and adds exactly two
future code items: typed parent snapshot inheritance without reacquisition
(19→58 IDs with tamper rejection), and a compact durable identity metadata closure
with provenance, complete build identity, tamper and round-trip verification.
Neither broad DSL feature was implemented as a campaign prerequisite.

The bounded Python API/worker/coordinator supports historical Git declarations,
parent-bound immutable reuse, mixed authentic adapter provenance, explicit safe
work roots, build-only phases, stop-before-freeze gates, and archived-lock
verification independent of a later active lock. Default version-1 replay and
unflagged recovery reconstruction retain their original path. Documentation:
[`lifecycle-recovery.md`](../../../docs/research/lifecycle-recovery.md#bounded-ancestry-replay-and-archived-acquisition-locks).

Verification on the published replay code:

- Focused locked CPU regressions: **44 passed** in 291.77 seconds across
  `test_implementation_replay.py`, `test_replay_ancestry.py`,
  `test_recovery_manifest.py`, and `test_lifecycle_recovery.py`.
- Broader locked CPU suite, excluding CUDA/ROCm/XPU/network markers:
  **1,339 passed, 10 skipped, 2 deselected** in 1,166.94 seconds.
- Changed-file Ruff: **all checks passed**.
- Real local-Git coordinator smoke: `MATCH`, genuine historical build/release,
  archived lock and current-verified receipt. Four-stage regression exercises
  mixed adapters, unequal source sets, build-only no-freeze, two final declaration
  changes, archived intermediate verification after active-lock replacement,
  and tamper refusal. This is software evidence, not a DevMind release match.
- Test temporary directories and uv environments used external named scratch.
  Fixture-authoring errors were corrected before the successful verification;
  package files were held stable for the successful focused and broader runs.

Launch ran 21:39:29–22:21:36 UTC on 2026-10-01 (about 42 minutes end-to-end,
including acquisition/build and current verification). Live 30-second storage
samples remained above the projected-use plus 25% floor: first sample
980,797,878,272 free bytes / 66,291,166 free inodes; last sample
947,614,498,816 bytes / 66,239,389 inodes. No storage violation was reported.
RAM available immediately before launch was 31,598,682,112 bytes; no accelerator
job was active. These are execution observations, not a portable resource bound.

## 9. Tokenizer selection, export and contiguous-eos-v6 preparation

**Not attempted.** No bakeoff declaration/fit, selected vocabulary, measured
`developer_systems` token denominator, derived MODEL-0 budget, release export or
v6 prepared-data manifest exists from this attempt. The conditional generic
schema-3 bakeoff/selected-tokenizer compatibility fix was not unlocked or applied.
No tokenizer was re-fitted and no lost token estimate was reused.

## 10. MODEL-0 runtime, Campaign, checkpoint and evidence closure

**Not attempted.** No MODEL-0 input/config, plan/lock/binding, fresh ROCm doctor,
full-shape stage, Campaign approval/run, checkpoint, evaluation, generation panel,
readiness, family or continuation recovery declaration was produced. All hardware
and long-run MODEL-0 gates remain unexercised. No CPU fallback or Developer SFT.

Historical `recovery.yaml` retains its exact `e260ea8` pin and expected v4 release,
with existing `external_required` steps unchanged. No new thin archive is claimed
for an absent verified release/model closure. Prior Try-2 and pinned-replay
reports, blocker JSON, receipts and acquisition roots were not edited or repaired.
This operational negative result neither changes the original MODEL-0 scientific
settings nor establishes model quality, historical recovery or weight rights.
