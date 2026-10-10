# C05 50M execution binding — offline command audit, not a launch receipt

This is a command map for the unchanged [50M declaration](CARD05_BASE_50M_PROPOSAL.md).
It does not create an attempt, admit a corpus, or supersede any historical
artifact. The conditional execution remains **blocked before its ledger**:
the B8 fixed-validation selector cannot be invoked for a new run unchanged.
The new project, review, mixture, run, monitor and contract declarations must
also be frozen and checked against this map before any acquisition. No 50M
declaration or data artifact is claimed to exist yet.

The tested prepared-input CLI is `src/sparselab/cli/main.py` SHA-256
`d6299083bb887e17739a76c03e08dcbdba7945a3116296ded052660de91886a8`;
it delegates to the unchanged `src/sparselab/staging.py` SHA-256
`c625716a3f4327d3dbc14cf6daa08cc909a1e7ced6f1056262d4b00d7f2e43c7`.
The existing fixed-slice CLI and scorer are SHA-256
`fb2bc14cee4fca12aa7c88ee51d1500dd8e59a7ed18c64e164ae25ece58a2337`
and `75970adbb08b46a61f03896484bcaf5852111ba34dc938b74645b653c18a26b6`.
The v2 attempt ledger implementation is SHA-256
`9018246376cbcf47f43d9ced011ba80a3d8b9e0143aa983b57db6bfacccd1090`.
These are file identities at this offline review, not runtime qualifications.

The command shapes below use `S` for
`uv run --locked --no-sync sparselab --work-dir /srv/sparselab/state/experiments/kernel-memory-lab`,
`A` for a **new** attempt root, `P` for its reviewed prospective v3 project
declaration, `R` for its verified release, `F` for its finalized family inventory,
`M` for its 50M mixture declaration, `O` for the original C03-C1 release,
`T` for that release's unchanged tokenizer file, `TC` for its original tokenizer
fit configuration, `C` for the fresh seed-17 run config, `B` for the sealed
prepared bundle, `Q` for the frozen base profile, `H` for the original C03-C1
held-out release, and `N` for a new run ID. Every variable must be replaced by
an absolute path or exact identifier and pinned before the attempt. `Q` is
`CARD05_BASE_EVALUATION_PROFILE_V1.json` SHA-256
`bb4c1b719b29fa96c4429516022a5038f0e3595b9d8ef63394bf0a17a5c6851d`;
`H` and `O` are the original release ID
`e893cb2e3c65f7f6360f73e4daa159dcec5da1b4c2c6f911ff5d9d549b815818`,
not the new training release. `T` must hash to
`308b33a6edbed613f3caf5232b124a7c6105a7a3dc273c999725be51c3ffeaa9`.
The original frozen family inventory is
`/srv/sparselab/state/experiments/kernel-memory-lab/corpora/kernel-memory-lab-card03-scale-retry1/prep/family-inventory-v5.jsonl`
with SHA-256 `f3b506be3dcd9b99401c5dae72f6ef16b51441394e9a485913200e7eeaad1af5`.
`TC` is the original export's `lm/b57de1d5857b918120baa86d0caef8c7813cf50d422b621adee3d12a0561c944/tokenizer.yaml`
under release `O`, and `T` is that config's `output_dir/tokenizer.json`.

| Mandatory phase | Existing command or reviewed operation and binding | Effect / precondition |
| --- | --- | --- |
| Pre-ledger identity and limits | `git rev-parse HEAD`; `sha256sum CARD05_BASE_50M_PROPOSAL.md P Q T TC`; inspect `P`, source pins, `C`, monitor/contract policies and free-space baseline. | Read-only. Confirm one new identity/root, 36,000-second whole deadline and all unchanged phase/resource caps before writing. Do not treat the B8 scripts' old paths as live declarations. |
| Ledger and source transfer | `S attempt init --ledger A/attempt-ledger.sqlite --contract A/attempt-contract.json --contract-sha256 <sealed-hash>`; `S corpus budget-init P`; `S corpus acquire P`; `S corpus acquire P --offline`. | Initialize once under the common-root monitor. The native transport budget charges response bodies including the at-most-one source-transfer retry; offline acquire verifies retained snapshots. Never reset either ledger. The new project must declare the same pinned full Wikimedia shard with remainders `[0,1,2]`; other snapshots are retained inputs. |
| Source admission | `S corpus admission-draft P --template <new-application-template> --policy-document <accepted-policy-v1> --output A/prep/admission-draft.json`; review flagged rows, rights, source notices and spot sample; freeze a new reviewed admission declaration. | The retained B7 policy-binding and review are evidence, not automatic admission for new Wikimedia rows. Unknown coverage and material rights conflicts stop their affected records/source as declared. No training release exists before review. |
| Family and split | `S corpus split-inventory P --output A/prep/split-inventory.jsonl --json`; review exact-content components, prior family aliases, independent held-out assignments and clusters; `S corpus freeze-splits P --inventory A/prep/split-inventory.jsonl --clusters <reviewed-clusters.json> --output <reviewed-splits.yaml> --json`. | Native inventory/freeze execute the decision; cluster and split authoring are reviewed evidence. B7's `audit-and-declare-families.py` is hard-coded to its own release, so its code cannot be replayed with a new path without review. Incompatible accepted families stop. |
| v3 build, lineage and release | `S corpus build P --offline`; `S corpus freeze <verified-build-root>`; `S corpus audit R`; `S corpus near-duplicates R`; `S corpus finalize-family-inventory R --splits <reviewed-splits.yaml> --output F --json`. | `P` must declare `normalizer-structure-v3`. Review cleaning/source spans, rights, family and split lineage against accepted B7, plus exact/near cross-split leakage. B7's `audit-protected-lineage.py` and lexical-review scripts have fixed paths; a new reviewed audit record or adapted operation is required. Admission of `R` is an explicit research decision, not a CLI side effect. |
| Unique post-cleaning supply | `S corpus measure-tokens R --tokenizer T --tokenizer-origin-release O --family-inventory F --policy <50M-floor-policy.yaml> --output A/prep/unique-train.json --batch-source-bytes 8388608 --json`. | The tested cross-release path cold-authenticates both releases, tokenizer origin and eligible inventory. Stop unless measured distinct train positions are at least 20.25M general, 4.50M explanatory and 0.25M incident with protected held-out families intact. |
| Exact allocation | `S corpus materialize-mixture M --output A/prep/mixture --json`; `S corpus verify-mixture M --output A/prep/mixture --json`. | `M` must bind `R`, `F`, `TC`, original tokenizer origin, 81/18/1 quotas of 40.5M/9M/0.5M, seed 17 and at most two exposures. Check exactly 50M targets and the native receipt before preparing data. |
| Run declaration and prepared bundle | Review and save `C` with fresh seed-17 341,885,952-parameter configuration, 48,829-step scheduler and 50M target cap; `S corpus verify-export C`; `S data prepared-inputs publish C --output B --resource-envelope <prep-envelope.yaml> --tokenizer-batch-source-bytes 8388608`; `S data prepared-inputs verify C B`. | New CLI invokes native immutable publication and cold verifier; it reports bundle/data manifest identities and counts the authenticated loss mask. It rejects a count different from the verified mixture and run cap. Exclusive output; no model construction. |
| Runtime preflight and one train | `S inspect C --json`; `S workspace preflight C`; `S stage C --through validate --prepared-inputs B --cold-verify --output A/stage-validate --runtime rocm-7900xtx --resource-envelope <runtime-envelope.yaml>`; `S train C --stage-bundle A/stage-validate --run-id N --runs-dir A/runs --runtime rocm-7900xtx --resource-envelope <runtime-envelope.yaml>`. | Stage is zero-update validation. The sole fresh training invocation reserves at most 48,829 updates and exactly 50M targets in `attempt run`, with 48,828 full batches and one 128-target final batch; no smoke/warmup, retry or resume. Reuse the reviewed owned-process supervisor and monitor with new sealed path/identity declarations. |
| Validation scoring and selection | For each step `0,5000,10000,15000,20000,25000,30000,35000,40000,45000,48829`: `S evaluation fixed-slices score Q N --release H --tokenizer-config TC --family-inventory <original-frozen-family-inventory> --expected-profile-sha256 bb4c1b719b29fa96c4429516022a5038f0e3595b9d8ef63394bf0a17a5c6851d --expected-family-sha256 f3b506be3dcd9b99401c5dae72f6ef16b51441394e9a485913200e7eeaad1af5 --checkpoint <exact-generation> --runs-dir A/runs --backend rocm --runtime rocm-7900xtx --mode validation --max-forward-positions 3084 --json`. Then apply the missing 50M-bound selector before opening any test or prose result. | Score command charges the durable fixed-profile/aggregate forward ledger before forwards. Require all 11 verified native checkpoints, one-batch operational validation receipts, 12 exact fixed windows, finite token-weighted losses and the earliest-step minimum tie-break. **No reusable 50M selector command or reviewed operation exists yet.** |
| Held-out test, utility and continuations | After sealed selection only: the same `evaluation fixed-slices score` binding with `--checkpoint <selected-generation> --mode test --max-forward-positions 3084`, then `--mode utility --max-forward-positions 4608`; `S evaluation fixed-slices continuations Q N --release H --tokenizer-config TC --family-inventory <original-frozen-family-inventory> --expected-profile-sha256 <Q-hash> --expected-family-sha256 f3b506be3dcd9b99401c5dae72f6ef16b51441394e9a485913200e7eeaad1af5 --checkpoint <selected-generation> --runs-dir A/runs --backend rocm --runtime rocm-7900xtx --json`. Repeat `score` for the historical v2 run/checkpoint in validation, test and utility modes and `continuations` for its eight prompts with the same frozen binding; B8 remains read-only comparison evidence. The fixed-score sum is 11×3,084 + new test 3,084 + new utility 4,608 + v2 validation/test 2×3,084 + v2 utility 4,608 = 52,392 input positions. | Native scorer/generator retain fixed content, 64-token greedy prompts, per-call caps and aggregate ledger charges. Do not rerun the 200/400-item suites. |
| Final accounting and report | `S attempt status --ledger A/attempt-ledger.sqlite`; `S evidence N --runs-dir A/runs --json`; `S checkpoint verify A/runs/N/checkpoints/<selected-generation> --config C --json`; cold-verify `R`, `M` and `B` through the commands above; preserve monitor/owned completions and raw score/continuation records; write reviewed result and artifact index. | Reconcile native training receipt, final mask, all forward/generation charges, phase times, resource high waters and negative outcomes. Reporting does not promote model quality or unblock Card 06. |

The B8 selection file at
`/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-pretraining-v4/select-fixed-checkpoint.py`
has SHA-256 `1903a5c19c9d7ee85aeca774fc89069f7ef534311e0467ea3371f7769e164112`;
its historical output has SHA-256
`42f8bf15ecd5381708570e8fea2b807e7e3b8b735945896eb770e5ec636b7b10`.
Inspection confirms it chooses `min(STEPS, key=(fixed_loss, step))` only after
checking every score's profile, run/step/checkpoint, 12-window count, 3,072
targets, finite per-item/aggregate losses and successful native checkpoint
and monitor evidence. It writes an exclusive receipt before B8's test/utility/
continuation operations. It also hard-codes B8's 25M progress target, config
hash, eleven 2,500-step checkpoints, run paths and monitor labels. Editing or
copying it for 50M would be new, unreviewed Python. The historical 5M selector
and Campaign highest-step selection do not implement the declared rule.

No selector implementation is added here. Before a 50M ledger, a separate
reviewed identity-bound selector (or reviewed explicit retained operation)
must prove the exact eleven-step rule and before-test ordering with zero-model
negative tests for missing/duplicate/substituted/nonfinite scores and wrong
run/profile/checkpoint identities. The new admission/family/lineage and
monitor/contract declarations likewise require exact review; B7/B8's
hard-coded scripts cannot be run as-is. The CLI change is preparation-only;
it does not certify those future declarations, live ROCm or the 50M supply.

All numerical ceilings and stop conditions remain exactly those in the 50M
proposal: 36,000 seconds whole attempt; 14,400 preparation and 21,600 runtime
(18,000 stage/train/selection, 2,400 evaluation); transfer 7,200 seconds and
960,485,500 source plus 4,194,304 metadata response-body bytes with one
charged transfer retry; 20 GiB VRAM, 24 GiB process-tree RSS, 8 GiB
preparation/72 GiB total cumulative added apparent storage, 10,000 added
entries and 2 GiB free margin; 48,829 updates/50M targets; 11 operational
validation batches/11,264 positions, 55,000 fixed-profile and 170,000
aggregate nontraining forward positions, 16 generations/1,024 requested new
tokens and 64 MiB evaluation output; 8 preparation and 2 evaluation reviewer
hours; zero cloud use.
