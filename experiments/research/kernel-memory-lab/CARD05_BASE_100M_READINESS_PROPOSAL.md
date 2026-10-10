# Retained-corpus admission and yield measurement — approval required

Assessment at merged main `1978ebeefed1a2ea7b9acf1c5050a0efddfb1016`, whose
PR #57 final reviewed head was `0de752b322e357d35bc64a6afdf03382528c58e8`.
This is retained-only inventory and a proposed admission/measurement milestone,
not corpus admission, a production ledger, a run configuration or runtime approval. The current 50M
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

**Single next requested allocation: retained-only admission and yield measurement.**
Use one new preparation identity, common-root baseline and persistent native
attempt ledger. Reuse and cold-verify all four snapshots; admit through native
inspection-backed item review, `normalizer-structure-v3` pre-freeze inventory,
family decisions, split freeze, build, leakage/protected-lineage audit and
explicit release acceptance. Then measure post-cleaning, post-admission unique
train positions with the unchanged C03-C1 tokenizer-origin binding. Stop with a
**cold-verified accepted release and authenticated per-stratum supply receipt**.
Report per-source and family yield from the accepted inventory alongside the
receipt; do not mistake those supplementary tallies for tokenizer-measured
positions. Keep protected held-out families out of training. **This allocation
excludes mixture declaration/materialization, prepared-run declaration and
prepared-bundle publication/verification regardless of measured supply.** Do
not fit a tokenizer or run a model.

The exact selected public-native plan is
[`card05-base-50m/current/measurement-only.json`](card05-base-50m/current/measurement-only.json),
SHA-256 `066d6e37c3703d120dbeabd89846dd9133483ad9ade2617b1832b862ddee952a`,
using the unchanged `sparselab-preparation-v1` interface. Its 26 phases are an
exact checked-in prefix of the current canonical plan through
`measure-accepted-supply`; the six downstream mixture and bundle phase labels
are absent and rejected by `attempt phase-command` / `attempt run-phase` when
this plan is selected. Its sole default difference is the
[`measurement-domains.yaml`](card05-base-50m/current/measurement-domains.yaml)
policy, SHA-256 `5954443818a492600fcb5d8f5f4b775b297f0989ab3983b7e06302409e0ea268`,
which selects the three domains with zero measurement thresholds.
This is **not** a zero-supply eligibility rule or a replacement for the 100M
floors. It keeps the native receipt COMPLETE and cold-verifiable even when a
domain falls short. The original 50M `token-floors.yaml` and full 32-phase
declaration remain unchanged for their prior scope. Pin the selected plan and
policy hashes in any later attempt contract/review record; derive commands
only with `attempt phase-command` and dispatch only the selected labels through
`attempt run-phase`. Do not treat the full 50M plan as authorized by this request.

Proposed hard limits for this *preparation-only* allocation: 14,400 seconds
whole attempt including review, tests and shutdown; 24 GiB process-tree RSS;
8 GiB added apparent bytes, 10,000 added entries and at least 2 GiB projected
free-space margin, all from one cumulative root baseline; at most 8 aggregate
agent-assisted reviewer-hours tracked separately. Zero network/metadata
requests, transferred bytes, acquisition retries, GPU, model forwards, updates,
generation, cloud use or spend; no attempt restart or resume. An unavailable or
invalid retained input, unresolved blocking review decision, lineage leak or
hard-cap failure stops with receipts. **A completed authenticated measurement
below one or more 100M comparison floors is a successful measurement with a
negative eligibility finding, not a failed preparation.** These limits are a
**request**, not authority.

**Provisional expansion plan, pending fresh measurement.** B7 already fell
short by 19.70M general and 0.16M incident unique positions, and the retained
extra Wikimedia bucket primarily addresses explanatory content. Its historical
~1.55M explanatory forecast could not close even that stratum's 5.84M B7 gap;
it adds no demonstrated general or incident supply. If fresh accepted v3
measurement confirms shortages, first pursue independent, rights-reviewed
general-language works or broad prose families; a distinct incident-response
documentation family; and additional independently sampled explanatory sources
only to the extent the measured explanatory gap remains. Prioritize useful new
language coverage over cosmetic stripping. Record expected filtering yield and
family independence before pinning sources; do not infer them from raw shard
bytes or count more pages from one family as independent supply. New-source
pins, rights/use terms, selection size, transport and storage budgets require
separate approval. Keep 81/18/1 and two exposures solely as the 100M comparison
baseline; any mix change needs explicit scientific review.

## Later 100M decision, outside this allocation

The 100M comparison still requires at least 40.5M general, 9M explanatory and
0.5M incident unique eligible content positions at 81/18/1 with no more than
two exposures. These are not the success criteria for admission or yield
measurement. Exact 100M supervised-target packing, tokenizer reuse, a fresh
scheduler/configuration, checkpoint selection, evaluation and runtime ceilings
must be declared **after** the new accepted release and supply receipt exist.
The earlier approximately 7.5-hour estimate remains a historical extrapolation
from B8, not a measured 100M runtime or authority to train. The original
C03-C1 tokenizer and fixed-evaluation origins, C05-N1 0/200 and Card 06 block
remain unchanged.

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
