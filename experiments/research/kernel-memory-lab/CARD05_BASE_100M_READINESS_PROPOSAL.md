# Prospective 100M base-training readiness — approval required

Assessment at merged main `1978ebeefed1a2ea7b9acf1c5050a0efddfb1016`, whose
PR #57 final reviewed head was `0de752b322e357d35bc64a6afdf03382528c58e8`.
This is retained-only inventory and a prospective protocol, not corpus admission,
a production ledger, a run configuration or runtime approval. The current 50M
preparation declaration remains canonical for its historical scope. C05-B7/B8,
B11–B17, C05-N1 and all prior artifacts remain unchanged. Card 06 is blocked.

## Fresh host verification and its limits

The assessment root is
`/srv/sparselab/state/experiments/kernel-memory-lab/assessment-100m-readiness-1978ebe-20261010`.
Its native common-root baseline identity is
`8327fe5a4d2f1d24db6dc47cc754ee141891cdeb0aab92fbba66247da2389f30`.
Native monitor policy limited each inspection to 1,800 seconds,
24 GiB process-tree RSS, 256 MiB added apparent storage and 1,000
added entries. Offline mode was set; no production attempt ledger, acquisition
transport, tokenizer fit or model work was used. Inspect the retained monitor
completion files for exact resource samples and the single failed command below.
The largest sampled process-tree RSS was 2,082,127,872 bytes; the assessment
directory ended at 18,437,251 apparent bytes and 67 path entries. No native
monitor reported a resource violation. The assessment ended within its
1,800-second inspection window.

| Input | Fresh read-only check | Identity |
| --- | --- | --- |
| Four source copies and three immutable reuse origins | `corpus acquire ... --offline` completed, including source declaration, copy, origin and lock authentication | Project `kernel-memory-lab-card05-base-50m-v2`, acquisition lock SHA-256 `cf9fc14df9c756c5507e26db6b3c5f1f53880b790aa50c756d626c1a5cc4621d` |
| PagerDuty | Same cold lock verification | Snapshot `e040f260758cb00dd18e2a8a7b66a2e5d74455cbd26e1527304a52a81d984830`; declaration `8b4293a3f3cd434555ff95c45b3e40af4f5e445026f1b31efc3f6f022029b9d1` |
| Gutenberg | Same | Snapshot `db2ce3b1af503bb9444cc83f8e8fd1a72e603463ff0724ae41385f5d19627a1e`; declaration `e6a5841624c0c86a4e80d67914a91b7eaf40c3391c58c3d3ff6a8b0dc54ecc32` |
| Scoutflo | Same | Snapshot `574e06a4a327a3e25c04782c829b7015b8ba541da1d8e8f66ae3e4f1f7dfda0a`; declaration `2fc7879758c96d4913b721fc474d6b5420ebad15366b781e7fcbe04d594527b4` |
| Wikimedia | Same | Snapshot `8164f132bc1b6d13c09cff4de9e45bb9c637fbeca1f9ea13c3d04a0fda899ac5`; declaration `46743b8c67a1374c9c59c7caf13842970f81a57889a5937f9753440b09e33d1f` |
| Accepted B7 release | Native `corpus describe` cold verification completed | `67d727a073e59283e76e81965ff7dce33ceba41f0d6a9ee2161705d725f8d798` |
| Original tokenizer-fit / held-out release | Native `corpus describe` cold verification completed | C03-C1 `e893cb2e3c65f7f6360f73e4daa159dcec5da1b4c2c6f911ff5d9d549b815818` |
| Original tokenizer | Native `tokenizer verify` cold verification completed against B7 run config and origin | Tokenizer SHA-256 `308b33a6edbed613f3caf5232b124a7c6105a7a3dc273c999725be51c3ffeaa9`, vocabulary 32,768 |
| Frozen fixed-evaluation profile | Native `evaluation fixed-slices verify` completed with original C03-C1 export tokenizer config and family inventory | Profile `bb4c1b719b29fa96c4429516022a5038f0e3595b9d8ef63394bf0a17a5c6851d`; 24 slices, 24 pairs, 8 continuations |
| Protected family/suite binding | Native `corpus audit-protected-lineage` completed, no changed protected assignments or training collisions | Original family inventory `f3b506be3dcd9b99401c5dae72f6ef16b51441394e9a485913200e7eeaad1af5`; B7 inventory `5f30437998b01694a7f146aff54103c4a63388e7a33870d23e23dd71a9e53d95`; frozen suite content `49576570d5f490055d2c378a17cfb6094368bdd0877b0687d1d4f8b38f275907` |

The first fixed-profile verification passed a run config where the command
requires a tokenizer-fit config, and failed with exit 1. Its `monitor-fixed-profile`
receipt is preserved. The corrected call used the original C03-C1 export's
`tokenizer.yaml` and completed; this is an invocation error, not evidence of a
bad profile. The current host has not admitted a `normalizer-structure-v3`
release from the four snapshots. Cold authentication proves retained input
identity, not review acceptance, train eligibility or unique supply.
The retained project snapshot trees occupy approximately 0.34 MB PagerDuty,
130 MB Gutenberg, 2.33 MB Scoutflo and 315 MB Wikimedia apparent bytes.
Raw snapshot bytes do not predict eligible unique positions after cleaning,
family separation and admission.

Baseline apparent common-root usage was 127,381,504,262 bytes and 50,753
entries, with 897,421,447,168 free bytes and 66,987,292 free inodes. The B8
checkpoint directories measure 1,367,622,268 bytes at step zero and
4,102,897,231–4,102,897,237 bytes at each of the ten later checkpoints
(apparent sizes; `du -sb`, read-only). These numbers are host inventory, not a
new hardware-fit test. The B8 training report recorded sampled peak VRAM
10,953,015,296 bytes, RSS 4,019,204,096 bytes and 46,863,235,311 cumulative
added bytes; those are historical measurements only.

## Supply and next preparation decision

The B7 receipt `card05-base-preparation-v4/prep/unique-train-v6.json` (file
SHA-256 `c2974bc5af9f66479ef1c46eebecd14f65820e6da374138e3aadaddbc95a0bc0`)
historically measured *distinct train content positions excluding EOS* with
the original tokenizer. Its release and tokenizer identities agree with the
freshly verified artifacts, but this assessment did not rerun token measurement
or claim the receipt's old implementation hash as current code evidence.

| Stratum | B7 historical unique positions | 100M floor at two exposures | Minimum additional positions versus B7 |
| --- | ---: | ---: | ---: |
| General | 20,802,818 | 40,500,000 | 19,697,182 |
| Explanatory | 3,155,976 | 9,000,000 | 5,844,024 |
| Incident | 343,703 | 500,000 | 156,297 |
| Total | 24,302,497 | 50,000,000 | 25,697,503 |

This is a lower-bound comparison, not a forecast for the four retained snapshots.
The retained Wikimedia shard may add explanatory supply; the retained sources
do not establish the general and incident floors. At 81/18/1, B7 explanatory
supply alone permits only about 35.07M content-position exposures. EOS, packing
and final-batch masking affect exact supervised-target accounting separately.

**Single next requested allocation: retained-only 100M supply preparation.**
Use one new preparation identity, common-root baseline and persistent native
attempt ledger. Reuse and cold-verify all four snapshots; admit through native
inspection-backed item review, `normalizer-structure-v3` pre-freeze inventory,
family decisions, split freeze, build, leakage/protected-lineage audit and
explicit release acceptance. Then measure post-cleaning, post-admission unique
train positions with the unchanged C03-C1 tokenizer-origin binding. Stop with a
verified supply receipt and per-source/stratum/family yield; **do not materialize
a 100M mixture or prepared bundle unless all three floors pass**. If they pass,
publish and cold-verify the exact 81/18/1 bundle, with at most two exposures per
content position and exactly 100M supervised targets. Keep protected held-out
families out of training. Do not fit a tokenizer or run a model.

Proposed hard limits for this *preparation-only* allocation: 14,400 seconds
whole attempt including review, tests and shutdown; 24 GiB process-tree RSS;
8 GiB added apparent bytes, 10,000 added entries and at least 2 GiB projected
free-space margin, all from one cumulative root baseline; at most 8 aggregate
agent-assisted reviewer-hours tracked separately. Zero network/metadata
requests, transferred bytes, acquisition retries, GPU, model forwards, updates,
generation, cloud use or spend; no attempt restart or resume. An unavailable or
invalid retained input, review conflict, lineage leak, cap failure or measured
shortfall stops with receipts. These limits are a **request**, not authority.

If a floor fails, quantify the post-cleaning shortfall before selecting new
sources. Favor additional independent, rights-reviewed general-language works
or broad prose collections for general coverage; another independently sampled
Wikimedia component for explanatory coverage; and a distinct rights-reviewed
incident-response documentation family for incident coverage. Avoid counting
more pages from one family as independent supply, repeating positions above two
exposures, or stripping useful structure merely to boost token counts. Any
new-source pins, rights/use terms, selection size, filtering yield, transport
budget and revised mix require a separate proposal. 81/18/1 is the comparison
baseline; a mix change would alter per-stratum floors and evaluation meaning and
needs explicit review rather than an implicit fallback.

## Prospective training and evaluation declaration — not executable yet

Use a fresh seed-17 341,885,952-parameter dense model with empty optimizer and
scheduler state, BF16/ROCm only after device preflight, context 1,024,
microbatch 1 and accumulation 1. Reuse the authenticated C03-C1 32,768-token
tokenizer byte-for-byte. Bind the future admitted v3 release, family inventory,
measured supply receipt, mixture and cold prepared bundle at their actual hashes;
none is invented here. Keep the fixed profile and held-out inputs bound to the
original C03-C1 release. Never resume B8's completed 25M schedule or any 50M
stopped attempt.

The draft run config changes the existing 50M template's `max_tokens` and
`dataset.train_max_tokens` to **100,000,000**, `max_steps` and
`optimizer.decay_steps` to **97,657**, and checkpoint and one-batch operational
validation `every_steps` to **10,000**. Keep 500 warmup updates, AdamW peak/floor
`3e-4`/`3e-5`, batch/architecture/precision and seed as a prospective combined
recipe. Native final-batch masking must yield 97,656 full 1,024-target updates
and one 256-target update with 768 masked positions, for exactly 100M
nonmasked supervised targets. Preserve full checkpoints and one-batch
operational validation at steps `0,10000,...,90000,97657` (11 events).
The [prospective run template](card05-base-100m/prospective-run.template.yaml)
records these values. Its release, bundle, validation counts, storage paths and
admitted-license summary are deliberately late-bound, so it is not a runnable
configuration and cannot substitute for reviewed admission or native inspect.
Select the earliest checkpoint with minimum finite token-weighted fixed
validation loss across all 12 windows/3,072 targets at each event, seal the
native selection receipt, then score fixed test/utility/prose. Do not use test
outcomes for selection.

The proposed evaluation retains the original 24 true/decoy pairs, 12 fixed
test windows, eight greedy 64-new-token prose continuations on the selected
model and matched v2 comparison, for at most 16 generation calls/1,024
requested tokens. Budget at most 11 operational validation batches/11,264
input positions, 55,000 fixed-profile scoring positions and 170,000 aggregate
nontraining forward-input positions (including generation context processing),
64 MiB output and 2 aggregate agent-assisted review hours. Those are proposed
caps based on B8's 11-checkpoint workload, not measured costs of a 100M run.
Keep the existing prospective base criteria: at least 10% lower fixed test loss
than v2, at least 18/24 correct preferences and four more than v2, and at least
6/8 coherent sustained prose continuations. Report negative results. The
unchanged C05-N1 0/200 instruction score and Card 06 reader gate remain
separate.

B8 measured 6,348 seconds for the 25M training monitor and about 7,960 seconds
end to end. `4 × 6,348 + (7,960 - 6,348) = 27,004` seconds (~7.5 hours) is
only a historical 11-checkpoint extrapolation, not a measured 100M runtime or
confidence bound. For a later *separate* training approval, propose a
43,200-second whole-runtime safety ceiling (39,600 stage/train/selection and
3,600 evaluation), 20 GiB whole-device VRAM, 24 GiB process-tree RSS,
80 GiB cumulative added apparent bytes and 15,000 entries from a fixed baseline,
and at least 4 GiB projected free-space margin. B8's measured 11 checkpoint
directories total about 42.4 GB; allow one additional ~4.1 GB transient
checkpoint plus prepared inputs, logs and evaluation before finalizing that
ceiling against the actual new release. Use one ledger, fail-closed monitoring,
one training invocation, zero retries/resumes and zero cloud spend. This future
allocation is **not requested for execution now**; the preparation result may
require changing its storage/time projection before a runnable declaration.

## Branch and CI disposition

PR #57 was merged at the exact main SHA above, and `main` was fast-forwarded.
The readiness branch was created from that commit. Prior to branch deletion,
every removed tip was an ancestor of `origin/main`, no branch had an open PR or
active worktree, and remote branch protection was checked. Deleted local tips:
`codex/card02-budget-confirmation` `3186a776351b438fa53b2b73c949dbae4b97326d`,
`codex/kernel-base-50m-proposal` `fb5b1d826428a8a7dc0a44c388de919db78bd7ec`,
`codex/kernel-base-pretraining` `ca0ce581c2aeea33ad3340179b1e68a0531f91fc`,
`codex/kernel-memory-lab` `cfd1d5cd50921958e8eb6875cda5f6df70f0ada7`, and
`codex/shared-corpus-cleaning` `fdd4af9660c94bfebecf90393145355e09dd4f30`.
Deleted remote tips: those five plus
`codex/corpus-interface-consolidation` `2b981988dc454820d296bf6c950683be3123c975`,
`codex/kernel-memory-lab-cloud-review` `c6f2bd44a366d1e3d7677ab388cc1ed159adc48d`,
`codex/kernel-memory-lab-log-failure` `76ba63af7396fc5f9ac5d6c20ba6562b54c4e94d`,
`docs/kernel-memory-lab-guides` `d1ae5867c938eae77c5cb7e65711aa9cb589f7a9`, and
`feat/conditional-lifecycle-code` `0de752b322e357d35bc64a6afdf03382528c58e8`.
Retained `main`, `origin/main` and the active readiness branch. No run data or
unmerged history was removed. `.github/workflows/ci.yml` runs only its explicit
zero-model safe suite on pull requests; optimizer/generation and full CPU jobs
are manual `workflow_dispatch` lanes. A draft PR therefore does not authorize
any model work.
