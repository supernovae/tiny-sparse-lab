# 50M offline launch packet — prospective source-effects revision after stopped C05-B12

This packet supersedes the **planning status**, not the evidence, of
[C05-T2](CARD05_BASE_50M_EXECUTION_BINDING.md). The scientific and numerical
contract remains [the approved proposal](CARD05_BASE_50M_PROPOSAL.md). No new
source, release, mixture, run, checkpoint or admission identity is claimed.
The B7/B9 releases and all previous attempt records remain unchanged.
[C05-B11](results/2026-10-09-card05-50m-attempt1-dispatch-stop.md) and
[C05-B12](results/2026-10-09-card05-50m-attempt2-source-reuse-stop.md) remain
stopped, with their ledgers and evidence unchanged. This packet does not
authorize a replacement attempt.

## Frozen inputs and prospective declarations

`card05-base-50m/project-acquire-reuse-v2.yaml` is a valid native acquisition-only
project, ID `kernel-memory-lab-card05-base-50m-v2`, project acquisition digest
`12604ffb8ccfe0351f6379ea0f781b3f9b45f6fb4ad9c8cfa80dd0b744f8b83c`.
The separate v2 project and transport identity leave B11/B12's shared v1
source root and spent transport ledger untouched. Its complete `source_effects`
binds the three retained sources to the exact prior project, snapshot and
declaration; only the additional Wikimedia selection permits acquisition.
The v1 acquisition, build, application, contract and run-template files remain
byte-for-byte historical. This prospective packet uses the separately named
`project-build-reuse-v2.template.yaml`,
`application-template-reuse-v2.template.json`,
`attempt-contract-reuse-v2.template.json` and
`RUN_CONFIG_TEMPLATE_REUSE_V2.md`.
Its PagerDuty, Gutenberg and Scoutflo source-declaration digests remain,
respectively, `8b4293a3f3cd434555ff95c45b3e40af4f5e445026f1b31efc3f6f022029b9d1`,
`e6a5841624c0c86a4e80d67914a91b7eaf40c3391c58c3d3ff6a8b0dc54ecc32`,
and `2fc7879758c96d4913b721fc474d6b5420ebad15366b781e7fcbe04d594527b4`.
The only source-selection change is Wikimedia declaration digest
`46743b8c67a1374c9c59c7caf13842970f81a57889a5937f9753440b09e33d1f`:
the same pinned complete 480,242,750-byte shard and revision, hash modulus 16,
remainders `[0,1,2]`, ≤384 MiB selected and ≤4 GiB expanded. The acquisition
project's placeholder all-train split and quarantined release must **never**
be used for model input.

The three existing snapshot identities are `e040f260758cb00dd18e2a8a7b66a2e5d74455cbd26e1527304a52a81d984830`
(PagerDuty), `db2ce3b1af503bb9444cc83f8e8fd1a72e603463ff0724ae41385f5d19627a1e`
(Gutenberg), and `574e06a4a327a3e25c04782c829b7015b8ba541da1d8e8f66ae3e4f1f7dfda0a`
(Scoutflo). They are present under the old project snapshot root; before reuse,
the native `corpus acquire` preflight verifies all three exact approved origins
through the generic artifact-security and `verified_reuse` path **before** any
network request, including metadata. It publishes independently verified local
snapshot copies under the v2 project root because existing build/release
verification requires that layout. The copies and their origins are both
authenticated on cold and live acquisition-lock readback; copied bytes count
against the unchanged preparation/storage ceilings, while retained sources
consume zero transport bytes. A missing, unsafe or mismatched origin stops the
operation without fallback transfer. No `alias-snapshot` command is used. The
new Wikimedia snapshot and acquisition-lock digests can only be recorded after
the permitted transfer.
The three retained origins currently total 132,630,150 apparent bytes, so
their local copies need at least that much of the unchanged preparation cap.
Current `verified_reuse` source-snapshot proof authority declines to sign a
warm proof; native verification falls back to cold byte checking. This does
not relax the required origin or local-copy authentication.

The accepted source and scale policies remain unchanged. The prospective
[application policy](card05-base-50m/APPLICATION_POLICY_PROSPECTIVE.md) and
`application-template-reuse-v2.template.json` preserve B7's four rights objects. The
template deliberately has unresolved lock and Wikimedia-snapshot identities:
bind those to the cold-verified new acquisition lock before admission-draft.
Use the same automated source/record exception screen, 8/12/24/24 clear-row
spot sample by source, and manual review only of flagged exceptions proposed
for qualification. Unknown/material row exceptions remain quarantined;
source-wide rights conflicts stop that source. No template is an admission.

`project-build-reuse-v2.template.yaml`, `release-reviewed.template.yaml`, and
`mixture.template.yaml` specify the native operations after review. The build
release uses `normalizer-structure-v3`, reviewed admission hash and frozen
splits; the original C03-C1 tokenizer config and origin release ID remain
fixed. `token-floors.yaml` requires 20.25M/4.50M/0.25M post-cleaning,
post-admission distinct training positions. The mixture requires exactly
40.5M/9M/0.5M targets, seed 17 and ≤2 exposures per source position. The
[the run-config template](card05-base-50m/RUN_CONFIG_TEMPLATE_REUSE_V2.md) retains the 341,885,952-parameter architecture, BF16/ROCm,
context 1,024, effective batch one, seed 17, AdamW 3e-4 peak/3e-5 floor,
500-step warmup/cosine through update 48,829, exact 50M target cap,
every-5,000-update checkpoints/one-batch validation, and fresh paths.
Unresolved `${...}` values are intentionally invalid until the corresponding
native receipt is cold-verified. Render into new, exclusive files, validate
with the native loaders and record their byte digests; do not rewrite templates.
The rendered build project lives in a new attempt-owned directory containing
byte-identical copies of the four checked-in source YAMLs and `transforms/lm.yaml`,
plus only the new reviewed admission and split files. Recheck each copy's
digest before `corpus build`; there is no independent cache-path override.

The original held-out release is
`e893cb2e3c65f7f6360f73e4daa159dcec5da1b4c2c6f911ff5d9d549b815818`
with manifest SHA-256 `718a5f1df007e59a32717bc660708ab05e4c5c8b90da963e1e8a960208c4f3ca`.
The original frozen family inventory SHA-256 is
`f3b506be3dcd9b99401c5dae72f6ef16b51441394e9a485913200e7eeaad1af5`;
the unchanged tokenizer SHA-256 is
`308b33a6edbed613f3caf5232b124a7c6105a7a3dc273c999725be51c3ffeaa9`.
The fixed profile SHA-256 is
`bb4c1b719b29fa96c4429516022a5038f0e3595b9d8ef63394bf0a17a5c6851d`;
the frozen 600-item content digest remains
`49576570d5f490055d2c378a17cfb6094368bdd0877b0687d1d4f8b38f275907`.
The suite file is
`/srv/sparselab/state/experiments/kernel-memory-lab/card03-scale-operations/continuation2/eval-work/frozen-card03-evaluation-v1.json`,
SHA-256 `7b28f37de2c22121b27626462aafaa65921185fdabce606bb3016ac2a250bd55`.
The new training release is never substituted for this evaluation release.

## Exact native operation map and stop points

Here `S` means `uv run --locked --no-sync sparselab --work-dir
/srv/sparselab/state/experiments/kernel-memory-lab`; `P` is the checked-in
acquisition project, `A` a new exclusive attempt directory under that work
root, `R` the newly verified/admitted release, `F` its finalized family
inventory, `M` its rendered mixture declaration, `B` its verified prepared
bundle, `C` its rendered run config, `Q` the exact frozen profile above,
`H` the original C03-C1 held-out release, `TC` that release's tokenizer.yaml,
and `N` a new run ID. All paths passed to attempt/selector APIs are absolute.
Every command below is a command **shape**, not an instruction to run it now.

| Phase | Reviewed command/operation | Mandatory verification before dependent work |
| --- | --- | --- |
| Pre-ledger | Check `git rev-parse HEAD`, clean checkout, proposal and packet hashes, active-run absence, free bytes/inodes, measured retained-space headroom and device UUID. Compute the static content identity from the pinned proposal/packet/head without writing to A; run `S monitor-baseline <common-root> --output A/workspace-baseline.json --seconds 4 --json` and cold-load it. Render one `attempt-contract.json` bound to that baseline and `monitor-whole.yaml`. | The common root and baseline never change; ≥2 GiB projected free margin and enough room for full checkpoint retention are required. The baseline is taken before *any* attempt files other than the exclusive root itself. No ledger is created during this offline packet task. |
| One attempt/transport | `S attempt init --ledger A/attempt-ledger.sqlite --contract A/attempt-contract.json --contract-sha256 <hash>`; `S corpus budget-init P`; run `S corpus acquire P`, then `S corpus acquire P --offline`, all phase commands under the same owned supervision. | One persistent 36,000-second ledger, one independent 7,200-second transport ledger, 960,485,500 source and 4,194,304 metadata response-body bytes; at most one charged retry of the single Wikimedia transfer, no restart. Complete retained-origin preflight precedes any live request; cold acquisition-lock readback rechecks the origins and local copies. The contract pins raw acquisition-project SHA-256 `74e4326e261589bd0c29915911572074a71183fe7039b1f641faef069bcbc43c`; online acquire requires its initialized bound transport ledger. Stop on any source/lock mismatch. |
| Rights/families | `S corpus admission-draft P --template <bound-template> --policy-document <prospective-policy> --output A/prep/admission-draft.json`; review and freeze decisions. `S corpus split-inventory P --output A/prep/split-inventory.jsonl --json`; review exact-content components against the accepted prior inventory/cluster mapping, author a new reviewed cluster declaration, then `S corpus freeze-splits P --inventory <inventory> --clusters <reviewed-clusters> --output <reviewed-splits> --json`. | Preserve accepted family IDs, splits and strata. Incompatible connected accepted families stop; new hints may map only to a compatible prior identity. The split receipt must bind admission, inventory, prior inventory and reviewed clusters. Reviewers record decisions within eight aggregate preparation reviewer-hours. The B7 hardcoded Python is not a command here. |
| V3 release/lineage | Render reviewed `project-build.yaml` with that admission and split declaration. `S corpus build <project-build> --offline`; `S corpus freeze <verified-build-root>`; `S corpus audit R`; `S corpus near-duplicates R`; `S corpus finalize-family-inventory R --splits <reviewed-splits> --output F --json`; `S corpus audit-protected-lineage --prior-release H --candidate-release R --prior-inventory <original-family-inventory> --candidate-inventory F --profile Q --suite <frozen-600-item-suite> --output A/prep/protected-lineage.json --json`. | Cold release verification, recorded v3 source-span/cleaning and rights review, no exact/near cross-split leakage, all protected content/families excluded from candidate train and unchanged retained assignments. A BLOCKED lineage receipt stops the attempt. Explicitly admit the exact release after review; a CLI success alone is not admission. |
| Supply/mixture | `S corpus measure-tokens R --tokenizer <original-tokenizer> --tokenizer-origin-release H --family-inventory F --policy <token-floors.yaml> --output A/prep/unique-train.json --batch-source-bytes 8388608 --json`; cold-read its receipt. Render `M` with `R`, `F`, `TC`; `S corpus materialize-mixture M --output A/prep/mixture --json`; `S corpus verify-mixture M --output A/prep/mixture --json`. | All three distinct-position floors, exact 81/18/1 50M targets and ≤2 exposures must pass. The new release is the measured/training release; H is solely the tokenizer's fitting and evaluation origin. Stop on a shortfall, without changing source/mix/exposure. |
| Run data/preflight | Render `C` from verified counts and paths, validate native RunConfig, cold-read its bytes, then **once** `S attempt bind-artifact --ledger A/attempt-ledger.sqlite --kind train_config --path C --sha256 <actual-C-sha256> --content-identity-sha256 <contract-content-sha256> --workspace <common-root>`; then `S corpus verify-export C`; `S data prepared-inputs publish C --output B --resource-envelope <resource-envelope.yaml> --tokenizer-batch-source-bytes 8388608`; `S data prepared-inputs verify C B`; `S inspect C --json`; `S workspace preflight C`. | The late-resolved C digest is sealed in the same ledger before stage/train. The attempt's static packet/content identity remains unchanged; the direct train receipt requires the bound C digest. Verify tokenizer bytes/origin, new release, exact mixture mask, bundle identity, config/architecture/schedule and 50M target count. Stage is not a pre-ledger preflight. |
| Runtime | `S stage C --through validate --prepared-inputs B --cold-verify --output A/stage-validate --runtime rocm-7900xtx --resource-envelope <resource-envelope.yaml>`; then **one** `S train C --stage-bundle A/stage-validate --run-id N --runs-dir A/runs --runtime rocm-7900xtx --resource-envelope <resource-envelope.yaml>` under native `attempt run`. | Stage reserves 0 updates/targets; train pre-reserves 48,829 updates/50M targets and binds its native train receipt. Require fresh seed-17 weights and empty optimizer/scheduler; no smoke/warmup training, resume or second invocation. Success means 48,828 full 1,024-target batches plus a 128-target final batch with 896 masked targets. |
| Fixed validation/selection | Before the first fixed score, capture a supplemental common-root baseline with `S monitor-baseline <common-root> --output A/evaluation-output-baseline.json --seconds 4 --json`, then bind its actual digest once with `S attempt bind-artifact --ledger A/attempt-ledger.sqlite --kind evaluation_baseline --path A/evaluation-output-baseline.json --sha256 <actual-baseline-file-sha256> --content-identity-sha256 <contract-content-sha256> --workspace <common-root>`. For each declared step `0,5000,...,45000,48829`, use `S evaluation fixed-slices score Q N --release H --tokenizer-config TC --family-inventory <original-family-inventory> --expected-profile-sha256 <Q-hash> --expected-family-sha256 <original-family-hash> --checkpoint <verified-generation> --runs-dir A/runs --backend rocm --runtime rocm-7900xtx --mode validation --max-forward-positions 3084 --json`. Render `selection.template.json` with sealed `C`, `Q`, `N`, `A`; run `S evaluation fixed-slices select <declaration> --output A/selected-fixed-validation.json --json`, then `verify-selection`. | Wrap each fixed score and subsequent evaluation-output command in the bound `S monitor --policy A/policies/monitor-eval-output.yaml --log-dir <fresh-log-dir> --workspace <common-root> --baseline A/evaluation-output-baseline.json -- <exact-native-command>` inside `attempt run`. The original cumulative baseline remains in the outer whole monitor. All eleven complete verified checkpoints, one-batch native validations, 12 exact validation windows/3,072 targets each, finite token-weighted losses, matching run/config/checkpoint/profile/tokenizer identities; earliest-step tie break. Seal selection **before** test/prose. |
| Test/report | After selection, use the same fixed-slice command with selected checkpoint in `--mode test --max-forward-positions 3084` and `--mode utility --max-forward-positions 4608`; run `evaluation fixed-slices continuations` with the same Q/H/TC/family/selected-checkpoint binding. Repeat the declared v2 validation/test/utility and eight prose comparisons only. Cold `verify-selection`, `verify-mixture`, prepared-input `verify`, native `evidence`, selected `checkpoint verify`, and `attempt status`; preserve monitor/owned receipts and write the result/index. | Fixed scoring is bounded by 55,000 inputs; aggregate nontraining forwards by 170,000; operational validation 11 batches/11,264 positions; generation 16 calls/1,024 requested new tokens, evaluation output ≤64 MiB, evaluation reviewer time ≤2 hours. Report negative results without Card 06 promotion. |

Every phase command runs through `attempt run` with the **same** ledger,
content identity, baseline, common-root workspace and `monitor-whole.yaml`,
inside the existing Linux `run-owned-phase-command.py` with one absolute
whole-attempt deadline and bounded TERM-to-KILL shutdown. The preparation
phase additionally runs through `S monitor --policy <monitor-preparation.yaml>
--log-dir <new-log-dir> --workspace <common-root> --baseline
A/workspace-baseline.json -- <phase-command>` to enforce its
stricter 8 GiB added-byte/14,400-second ceilings; the contract's whole-policy
binding remains the 72 GiB policy and an optional nested-policy hash pins the
8 GiB policy separately. The only ledger administrative operations outside
`attempt run` are `attempt init`, the one-time resolved-config binding, the
one-time supplemental evaluation-baseline binding, and read-only status.
`attempt run` accepts only the reviewed native `uv run --locked --no-sync
sparselab --work-dir <common-root> <operation>` shape (or its exact direct
module/CLI equivalent), one checked native monitor wrapper with the same work
root and declared policy/baseline identity, and an explicitly enumerated native
leaf. It rejects shell/Python dispatch, extra or abbreviated options and
hidden model verbs. Native online `corpus acquire` still requires the pinned
project and its initialized persistent transport ledger. The 21,600-second
runtime subcap, 18,000-second stage/train/selection subcap and 2,400-second
evaluation subcap are computed from the same persisted start time, not fresh
per-command clocks. Check phase remaining time before each operation and
apply bounded `timeout` inside the owned supervisor; exhaustion, sensor loss,
numerical failure, resource breach, invalid receipt or incomplete owned
shutdown stops the attempt. The original common-root baseline covers preparation,
checkpoints, logs, scores and receipts. A supplemental baseline on that **same**
root, captured before the first fixed score, is used only by
`monitor-eval-output.yaml` to enforce 64 MiB of new evaluation output during
every fixed score, selection, comparison and reporting command. It never
replaces or resets the original baseline. The whole policy enforces 20 GiB
whole-device VRAM, 24 GiB tree
RSS, 72 GiB new apparent bytes, 10,000 entries and ≥2 GiB projected free
space. The preparation monitor enforces 8 GiB added apparent bytes; these are
not separate additive disk allocations.

The native `attempt run` reservations are zero updates/targets/generation
for all preparation, stage, selector and read-only reporting phases; the
single training call reserves 48,829 updates and 50M targets with
`--receipt-kind train --receipt-path A/runs/N`; validation/score/generation
phases charge the native forward/generation counters before model work. Each
generation reserves its requested 64 tokens, including failures. Fixed-slice
score and continuations are classified as metered model work. Before loading
a checkpoint, the native fixed-slice CLI checks the owned phase and remaining
fixed/total forward, generation-call and requested-token allocations. Every
score forward and 64-token generation request charges the same ledger before
the model call; successful phase completion reconciles their reservations.
Missing or exhausted accounting fails before model load/forward. Direct legacy
evaluation outside an attempt retains its earlier behavior. Keep score
and generation commands inside their declared per-call forward caps and the
aggregate ledger. `attempt status` reconciles reserved and actual model work;
no retry/resume follows an exhausted or failed ledger.

For each `attempt run`, supply `--ledger A/attempt-ledger.sqlite --label
<unique-phase> --activity <inspect|validate|train|evaluate>
--content-identity-sha256 <frozen-content-sha> --policy <monitor-whole.yaml>
--baseline A/workspace-baseline.json --workspace <common-root> --completion
<new-owned-receipt> --updates <reservation> --target-positions <reservation>
--generation-calls 0 --generated-tokens 0 --receipt-kind <none|train>
[--receipt-path A/runs/N for train] -- <exact native phase command>`.
The only nonzero pre-reservation is the single train phase's 48,829/50M;
fixed-score forwards and the 16 separately requested 64-token generations
are charged durably by their native calls before model work. Do not nest a
second optimizer invocation in a phase or use `receipt-kind none` for train.
The direct monitor and owned supervisor retain their distinct completion
receipts, while the attempt ledger retains the aggregate counters. The
output monitor covers comparisons written under both the new and historical
run roots because both reside under the same common root.

The declared fixed-score reservation is 52,392 forward-input positions
(11 new validation scores, new test/utility and v2 validation/test/utility),
below the 55,000 fixed-profile cap. Operational validation adds 11,264;
worst-case full-prefix generation adds at most 16 × 6,112 = 97,792,
for 161,448 planned positions under the 170,000 aggregate ceiling.
Actual counts and cache behavior remain recorded by native receipts; this
arithmetic does not weaken pre-forward charging or any hard cap.

The prospective values frozen now are source pins, policies, tokenizer and
evaluation origins, profile, family inventory hash, scientific settings,
selector steps and all caps. The new acquisition lock/snapshot, reviewed
admission and clusters, split, v3 release, finalized inventory, token receipt,
mixture, prepared bundle, run config, checkpoints and fixed scores exist only
after their upstream phases. Each must be cold-verified, recorded by digest,
and inserted into the **next** declaration before that phase starts. Never
predeclare a guessed release, checkpoint or receipt hash. The native contract
template pins the acquisition-project, original fixed-profile/family and
nested-monitor hashes, and requires
one-time cold bindings of the actual resolved run config and supplemental
evaluation baseline. Older contracts omit these optional fields without
changing their declaration bytes or hash. The 50M candidate
remains unadmitted at this offline checkpoint.
