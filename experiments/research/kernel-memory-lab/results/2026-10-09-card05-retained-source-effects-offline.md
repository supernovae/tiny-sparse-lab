# C05-T5 — retained-source effects and live acquisition qualification

**Status: READY FOR REVIEW, offline only.** This repair follows the stopped
[C05-B12](2026-10-09-card05-50m-attempt2-source-reuse-stop.md) at `b4000fb`.
B11/B12 ledgers, result files, original v1 declaration/template bytes and
partial artifacts were not changed. No production ledger, source request,
tokenizer, model or GPU operation was run.

The new optional complete `source_effects` declaration pins each source ID and
declaration SHA-256. PagerDuty, Gutenberg and Scoutflo additionally pin their
approved `kernel-memory-lab-card05-base-v1` origin and exact snapshot SHA-256;
only the additional Wikimedia selection has `acquire` permission. The native
online `corpus acquire` verifies every retained origin through the existing
artifact path-security and real `ProofStore` path before opening the transport
ledger or making any metadata request. There is no retained-source fallback to
network. The verified bytes are copied locally under a separate v2 project
root, because the existing build/release verifier binds that layout. Both the
origin and copy are checked before acquisition-lock publication and again on
cold/live lock readback. These local copies count against the unchanged
preparation disk cap, but consume no transport allowance. The retained
snapshots measured 336,158, 129,962,212 and 2,331,780 apparent bytes at
their old paths (132,630,150 bytes combined), before prospective copying.
An additional read-only cold check of those actual origins passed the new
generic path-security and source-binding path: PagerDuty 38 files, Gutenberg
one file and Scoutflo 433 files, each at its declared snapshot and source
declaration digest. It ran under a 120-second timeout and 2 GiB address-space
limit, completing in 2.11 seconds. It made no destination copy or lock.

The v2 prospective acquisition project is
[`project-acquire-reuse-v2.yaml`](../card05-base-50m/project-acquire-reuse-v2.yaml),
raw SHA-256 `74e4326e261589bd0c29915911572074a71183fe7039b1f641faef069bcbc43c`,
native acquisition identity
`12604ffb8ccfe0351f6379ea0f781b3f9b45f6fb4ad9c8cfa80dd0b744f8b83c`.
The revised [launch packet](../CARD05_BASE_50M_LAUNCH_PACKET.md) references
separately named v2 build, application, contract and run templates. The old
v1 project remains at raw SHA-256
`6326d66589fa787fd3e85d834417d968b21eb6f808f1b6c85635ac255714aa51`.
The source pins, rights policy, scientific settings and numerical attempt caps
did not change; the v2 identity is prospective and has no acquisition lock or
admitted release.

**Bounded offline check.** The actual public CLI parser and `corpus acquire`
handler ran in `verified_reuse` mode with a real `ProofStore` on tiny local
fixtures. Only the declared Wikimedia leaf used fake transport. An independent
tripwire blocked sockets, urllib metadata/openers and all subprocess transport;
the CLI's exact local `git rev-parse --show-toplevel` storage check was allowed.
Tests covered empty/repeated proof-store use, re-verification after
mutation, corrupt/missing/mismatched retained data, alias retargeting, unsafe
and symlinked members, incomplete permission declarations, a corrupt local
copy, altered lock origin, and an origin mutated during permitted transport.
On failure there was no successful acquisition lock or retained-source request
or charge. Existing local Git, declaration/symlink and legacy alias security
regressions remained selected and passed.

Final exact selection: `tests/test_corpus_source_effects.py`,
`tests/test_kml_50m_launch_packet.py`, and
`tests/test_corpus_acquisition.py::{test_git_pinned_revision_and_symlink,
test_invalid_declarations_and_symlink,
test_verified_snapshot_alias_reused_across_project_ids_without_transfer}`.
It ran under the native operational monitor with a 180-second wall cap, 4 GiB
process-tree RSS, 512 MiB added bytes and 5,000 added entries at
`/srv/sparselab/state/experiments/kernel-memory-lab/card05-source-effects-offline-qualification/`.
The final monitor completion, command output and owned-process receipt are
retained there. Ruff rule/format, research lint and `git diff --check` are
separate zero-model checks.
The reviewed selection passed **18/18** in 16.35 seconds, with zero monitor
violations and zero surviving owned descendants. Peak process-tree RSS was
488,759,296 bytes; peak added workspace was 2,275,699 bytes and 4,256 entries
against the original local baseline. The final monitor completion SHA-256 is
`37285f8739c4236ebc490b3508ad9e568fe1a0865ee71c7d711044435a74f1c3`.

**Limit:** The current `source_snapshot` verifier authority scan reaches the
unrelated dynamic import in `recovery.implementation_replay`, so a signed
source-snapshot proof is not recorded. Repeated `verified_reuse` calls therefore
fall back to full cold verification; a warm proof **hit** is not qualified.
The tests exercised the real store and verified this fail-closed behavior,
without changing proof invalidation or path-security rules. This is a
performance/authority-closure gap, not permission to skip byte verification.
No live external Wikimedia transfer, production storage copy, release build,
model or hardware behavior was qualified.

The next 50M production attempt requires a separate approval and fresh
identity/ledger. C05-B12's charged allocation cannot be retried or resumed.
