# C05-B14 — retained-snapshot 50M attempt stopped at baseline binding

**Gate: FAILED before the first native preparation leaf.** The owner approved
one fresh conditional local attempt at clean, pushed
`7b9fe21f27f74d215e761c0da2146657e6aaa611` under the retained-snapshot
proposal and corrected packet. The attempt root is
`/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-50m-attempt4-7b9fe21/`.
Its single ledger started `2026-10-09T22:38:58.899459Z`, deadline
`2026-10-10T08:38:58.899459Z`, with the unchanged 36,000-second and all other
declared ceilings. B11–B13's ledger hashes remain, respectively,
`09a5b2a4348417bb2b90d2d56603e5a02208c07c7ea46863795c0f0ae0ea3d2d`,
`19a88098595d69613e50bb707ac39c211d88a831125832995825ac915c3f7358`
and `89fb0cb6191c208e177708717089725cbd0e6c96533ead411ebe8a8`.

Preflight verified the clean exact head, no active experiment process, the
original C03-C1 release manifest `718a5f1d…`, tokenizer `308b33a6…`, family
inventory `f3b506be…`, frozen evaluation profile `bb4c1b71…` and suite file
`7b28f37d…`. Native fixed-slice verification reported 24 slices, 24 utility
pairs and eight continuations, without a model forward. The registered ROCm
environment reported PyTorch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, one
visible device; AMD SMI reported the declared GPU UUID
`cdff744c-0000-1000-8055-1c9de49f245d`. Before ledger creation, the
common-root baseline reported 127,224,067,780 apparent bytes, 50,615 entries,
897,579,286,528 free bytes and 66,987,430 free inodes. The baseline receipt's
**embedded identity** is `b43ae20ed74a30c0661853807854afeb832cab14991e53ebf0fdb3f689edde45`;
the baseline **file hash** is
`a354b2417191b24d566620ae6126af026c803b09a1a39e5a8357726534384073`.
The acquisition project raw hash, existing acquisition-lock file hash, proposal
and packet matched their recorded identities before ledger creation. The four
retained snapshots were present, but this attempt did **not** reach their cold
verification or the native offline-acquisition leaf; their validity is not a
B14 gate result.

An initial pre-ledger contract rendering used a mistaken content identity;
that unbound file remains at `attempt-contract.json` (SHA-256
`21aa94b69d75fab51c662e5ae2832f03f5061aff1bd6e2bd07bbb92324fa3fac`).
The operator rendered a separate corrected file before initialization and
bound the ledger to **only** `attempt-contract-corrected.json` (SHA-256
`0077b0b42fe0eaf85ad47d2a10b38dde749da198d9b3901eca69ba6a1e5fc433`).
Its content identity correctly matched `preflight-identity.json` SHA-256
`ec89b5c9ca709ba5f386cff047ac6f41c5ac39399b37af0167fe433d893af38e`.
However, `workspace_baseline_sha256` in that contract was filled with the
**file hash**, `a354b241…`, instead of the native baseline's **embedded
identity**, `b43ae20e…`. The native `attempt run` rejected this mismatch before
reservation, monitor launch or `corpus acquire --offline` dispatch. This is an
operator declaration-binding error, not an observed source or model failure.
The ambiguous `${COMMON_STORAGE_BASELINE_SHA256}` packet slot requires an
explicit pre-ledger identity check in any future plan; this spent ledger must
not be edited, restarted or resumed.

The attempted command was the reviewed Linux owned supervisor, enclosing
`sparselab attempt run --label acquire-offline --activity inspect` with the
one whole policy, common baseline and zero model reservations, then the bound
preparation monitor and the exact leaf `sparselab corpus acquire
experiments/research/kernel-memory-lab/card05-base-50m/project-acquire-reuse-v2.yaml
--offline`. Native dispatch returned
`AttemptBudgetError: workspace baseline identity differs from contract`.
The only phase receipt is `phases/acquire-offline.supervisor.json`, SHA-256
`2f8ac88a569b96b050d11fa8240830b7ea6016af427e54a19d6d558a3b434f82`:
return code 1, `living_descendants: 0`. A subsequent process inventory found
no owned worker. No native monitor or phase-storage completion exists because
dispatch failed before launch.

The ledger SHA-256 is
`55240d0268f438bc8b18a05d5f9469a03007b63a20e697dbc4924efb58e512da`.
Native status shows **no reservations** and zero charged updates, target
positions, fixed/aggregate forward positions, operational validations,
generation calls and requested tokens. The retained acquisition lock remains
`cf9fc14df9c756c5507e26db6b3c5f1f53880b790aa50c756d626c1a5cc4621d`
and its transport ledger remains
`5f4323c8f2559ba12ebbb8d52faf8a4dfefed7996d82c0811f3a9ed0ce8dbd76`:
**zero new network/metadata requests or transfer bytes**. No draft admission,
v3 inventory, family freeze, release, mixture, prepared bundle, staging,
training, fixed score or completion was produced. There is no learning curve
or scientific outcome from this attempt. The model remained untouched.

The allocation is spent by the first failed gate. Preserve this root and all
earlier attempts. The smallest next task is an offline packet correction that
names the embedded native baseline identity, verifies contract and baseline
through the actual public preflight before a ledger, and tests the wrong-file-
hash rejection. It needs separate review and a fresh runtime allocation;
there is no authority to retry B14. C05-B8 remains the latest completed model
run, C05-N1 remains 0/200, and Card 06 remains blocked.
