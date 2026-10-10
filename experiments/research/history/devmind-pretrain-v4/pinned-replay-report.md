# DevMind v4 pinned implementation replay: build gate blocked

## Verdict and immutable gates

**PINNED_REPLAY_BLOCKED.** The implementation is verified, but actual historical
execution did not reproduce the required DevMind v4 build identity. The worker
stopped before freeze. No v4 release, tokenizer selection, MODEL-0 declaration,
preparation, model training, checkpoint, evaluation, or readiness result was
produced by this continuation.

| Gate | Expected | Observed |
| --- | --- | --- |
| Project/acquisition declaration | `ee95707a159d4fbc62e8f22cce6301010cbf86a500db6c010a2d282e1166e6b6` | Exact match; current acquisition verifier passed |
| Build | `79b568629fc6682c33bf1212eef0653cdb05d26a772654019458f44af2ceaee5` | `2c27d12fb9f80c63f0e228750093c0ec200a1793edc55908e5839480ecc58637` |
| Release | `0f06976e1c90309c7920cd23b1079179ba3cd026fdde55e934c1320fe6dc5fd4` | Not produced; no v4 replay release directory exists |

No expected identity was changed. No current-produced manifest received injected
historical hashes. The prior Try-2 report, blocker and preflight records remain
unchanged as the record of that earlier attempt.

## Source and execution boundary

- Starting merged main: `d47b2cdd725b444ad5d2297ee9096673fa7733c7`.
- Normal branch: `fix/pinned-corpus-implementation-replay`; no worktree.
- Tested, committed and pushed implementation/execution revision:
  `75c0eab9e794558795a82c113133d18ea6f797c9`; launch recorded a clean tree.
- Historical producer commit: `e260ea866ded8ec5beb23cbacec1fb3c120c1d16`.
- Exported source: `/srv/sparselab/state/replay/source/e260ea866ded8ec5beb23cbacec1fb3c120c1d16`.
- Git tree: `24e31bd6db063f572193cf72c2dd36c9d24d839f`.
- Verified source inventory SHA-256:
  `4591c18eb4fa2a75a3a848f6b22039224963f13065b89d91d0647b218ba1a702`.

The export comes from verified Git object bytes, rejects replacement-ref
redirection and conflicting materializations, and is read-only after publication.
The dedicated subprocess imports that source directly rather than requiring old
CLI packaging. Acquisition, build and freeze are historical producer operations;
current SparseLab orchestrates and independently verifies. The failed real build
never reached freeze. An earlier bounded sample smoke exercised all three real
historical operations and current verification successfully; it is not evidence
that the v4 release reproduced.

## Identity-coupled implementation inventory

The full machine-readable audit and before-network CLI preflight are
[pinned-replay-implementation-inventory.json](pinned-replay-implementation-inventory.json)
and [pinned-replay-preflight.json](pinned-replay-preflight.json).

| Component | Historical SHA-256 | Current SHA-256 at execution | Comparison |
| --- | --- | --- | --- |
| `corpus/acquisition.py` | `15e891576f81a8c2096e9559c07bef436cb6ed617b131f08af6d9c0e9be40974` | `acde12f8c10e13c1c8bf3dd535fa19cc0f27287f9a6096b8617acb8fd5d76bb4` | different |
| `corpus/pipeline.py` | `d51b5f2cf108054d7b58d664c17b2dc9c356e5b27e25b1e04ee4d1e9830fdd8d` | `90c2447190e9b54acee904ad54c0b7c0749672f83b4235d7e864d3758e883805` | different |
| `corpus/project.py` | `7f19a933d74d6e5e78d03e9abe7c8c33b1d9a185c9bed171bae3dee8a0e79f12` | `e2eef1ca75101828faf11448a0af3368d3d8355a40b91c1bcdd2844ae5e4df47` | different |
| `corpus/provenance.py` | `a73e9cdf6d0705988f50833e856261dabb26ba3bfb33df5f802b54f137cfb589` | `a73e9cdf6d0705988f50833e856261dabb26ba3bfb33df5f802b54f137cfb589` | same |
| `corpus/rights.py` | `ff3486999e00007da93aea661299be42b8a4f86682e1d0308a7c63a3a9cddf24` | `df5e2d7951e892e82b35f845b2a8c502ce15815782d7118845ca49f611613083` | different |
| `corpus/large_build.py` | `5df4e0965a91e38d0a76304f675b6e2d00fb190804dad8e0849f79285af690e5` | `525e2e1004c6b90a0ffe4047111c0ff28bdbc5fa412516de488079578d5c115d` | different |

Acquisition binds the snapshot adapter hash. Build binds pipeline, schema and
provenance hashes, plus rights for schema 2/3 and the large builder on this v4
schema-3 LM path. Freeze inherits build identity and verified file/stage/snapshot
inventories; it does not add its own source-file hash. Export binds release,
configuration and split bytes, with no additional own-file implementation hash.
The preflight also covers release/export helpers, canonical hashing, configuration
models, pyproject and uv.lock. Historical schema-v1/v2/v3 semantics remain intact;
the separate future semantic-version design note is in the recovery documentation.

The real retained snapshots all record historical acquisition SHA
`15e891576f81a8c2096e9559c07bef436cb6ed617b131f08af6d9c0e9be40974`.
Current read-only verification also confirmed that all five recorded build
implementation hashes equal the verified historical source bytes.

## uv-only environment

- uv: `uv 0.12.19 (x86_64-unknown-linux-gnu)`.
- Python: `3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0]`.
- Dedicated data/build environment:
  `/srv/sparselab/state/replay/env/e260ea866ded8ec5beb23cbacec1fb3c120c1d16`.
- Historical dependency lock SHA-256:
  `dbad957a010eef6f29679e6ae29239eec060f7d53cb140f1fef909edf50dafbc`.
- Locked replay Torch: `2.14.0+cpu`; not a model accelerator runtime.

Installation uses uv with the historical lock and no project/CLI installation;
execution uses uv and isolated Python imports. No pyenv/conda/Poetry environment
or interpreter search was used. The recorded package inventories and pyvenv.cfg
hashes for both the checkout environment and registered `rocm-7900xtx` matched
before/after real replay. Registered vendor Torch stayed
`2.13.0+rocm10.0.0`; no ROCm provisioning or synchronization occurred.

## Observed artifacts and current verification

The real historical worker produced **123** snapshots. Their complete observed
IDs, declarations, adapter hashes and selected byte/file counts are preserved in
[pinned-replay-blocker.json](pinned-replay-blocker.json). These are actual outputs,
not counterfactual digests. For example, `v4_db_clickhouse_sql` is now actually
observed and verified as
`3e5576cfcb4e39dc5b73cc9aded06f563ae27a3e1f637c2a72514daeed248bfe`.
The earlier Try-2 record's value of the same digest was only a counterfactual;
that original record was not rewritten.

After the stop, current code performed **read-only** verification:

- historical source inventory: MATCH;
- versioned failed receipt and its logs: valid;
- acquisition lock, all 123 snapshot inventories and selected file bytes: MATCH;
- build inventory/provenance: internally consistent under its actual different ID;
- original expected build identity: MISMATCH;
- release verification: not possible because freeze was not executed.

All 123 source selected-file counts and byte totals match the checked-in historical
source-quality measurements. Aggregate corpus counts also match the historical
protocol: **818,630** normalized documents, **816,117** retained, and LM
train/validation/test **796,563 / 16,411 / 3,143**. Counts and sizes do not establish
immutable identity or model quality.

The **first provable divergent component is the build identity**. A finer
historical leaf comparison is unavailable: checked-in evidence has the expected
build digest but not its original identity payload or complete per-source
snapshot-ID baseline. An exact-digest repository search found the protocol,
recovery report and new gate declaration; historical evidence commit `26f00a8`
published measurements/protocol, not the original build/acquisition manifests.
Therefore this record does not claim all original snapshot IDs reproduced or
invent a root cause from matching aggregate measurements. Localizing the differing
leaf requires the original build identity/acquisition closure from the earlier
workspace or another authenticated copy. No further producer was run after the
mismatch, and the actual build was not aliased to the expected one.

## Receipts, preservation and storage

- Exact failed receipt: [pinned-replay-receipt.json](pinned-replay-receipt.json),
  externally `/srv/sparselab/state/replay/receipts/6f388615b12745ada0220dd1d325dbda.json`.
- Receipt record digest:
  `c3a35ef88e40a18e34ebea9d8dd30571612efb4e17d0b50173ca7dfbd86ddec5`.
- Launch and preservation evidence: [pinned-replay-launch.json](pinned-replay-launch.json).
- Historical artifacts remain under
  `/srv/sparselab/state/replay/work/e260ea866ded8ec5beb23cbacec1fb3c120c1d16/corpora/devmind-v4`.
- New operational logs remain under
  `/srv/sparselab/state/experiments/devmind-pretrain-v4-pinned-replay` and
  `/srv/sparselab/state/replay/logs`.

The original current-code partial acquisition was deliberately not reused as
historical snapshots. Its **27,643-entry**, **7,366,654,779-byte** content inventory
matched before/after, digest
`6f2c4024dc5af3cacfa8e06054222bca1e9042b5852ae00ba66e87caa4d9f3dd`.
Nothing from it or previous blocker evidence was deleted.

The launcher made **132** storage checks, enforcing
25% free-byte/inode floors. Minimum observed free space was
**972829573120 bytes**, above the
**270,275,294,208-byte** floor. Final free space/inodes were
**982545477632 bytes / 66426420 inodes**.
Task-local launch/verification scripts were removed after execution; all actual
artifacts, typed receipts and diagnostic logs remain.

## Tests and smoke

[pinned-replay-verification.json](pinned-replay-verification.json) records commands,
results and the corrected test-scratch incident.

- Focused replay/recovery/lifecycle/archive suite: **33 passed**.
- Broader locked CPU suite: **1,323 passed, 10 skipped, 2 deselected**.
- Added uv-only real subprocess policy check: **1 passed**.
- Ruff on changed Python files: passed.
- Actual CLI inspect/plan: reported replay requirement before network acquisition;
  no replay state existed after those read-only commands.
- Actual unflagged reconstruct with network permission: refused implicit replay.
- Actual `e260ea8` sample acquire/build/freeze and independent verifier: MATCH;
  historical producer `__file__` paths recorded in
  [pinned-replay-smoke.json](pinned-replay-smoke.json).

An earlier verification attempt filled the 16 GiB /tmp tmpfs with disposable uv
environments. Its failure log is retained. Tests were moved to persistent scratch,
fixtures now remove their own environments, and the subsequent focused/broader
suites passed. Only this task's disposable test environments were removed to
restore host RAM before real replay. Hardware/model/long-training gates were not
exercised by this continuation.

## TODO, remaining gap and downstream status

The completed full implementation-identity preflight/replay TODO was removed.
A distinct demonstrated ergonomics gap was added: blocked recovery JSON must
expose the exact retained failed-replay receipt path/digest. This attempt returned
the correct build mismatch but required manual receipt discovery. The TODO asks
for a real mismatch regression without producer retries or identity changes.
The missing original identity payload is an evidence prerequisite, not a guessed
algorithm defect.

MODEL-0 continuation is **blocked before tokenizer**. No Developer SFT was begun.
The release expectation remains unchanged, and no claim of
`DEVMIND_MODEL0_COMPLETE` is made.

**PINNED_REPLAY_BLOCKED**
