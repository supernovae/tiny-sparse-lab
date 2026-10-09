# Prospective 50M-target base-language test — approval required

Planning base: merged `main` `58ac08ee1289c1446491c8082851f5be196dfbcd`
(PR #55). This declaration is **not** an acquisition or runtime approval. Keep
the accepted C05-B7 release, all earlier releases/attempts/checkpoints, frozen
Card 03 and base profile, tokenizer and C05-N1's 0/200 reader result unchanged.
C05-B8 completed 25M targets, but its 15/24 utility preferences and 0/8
sustained prose continuations did not meet the predeclared base-learning rule;
the incident test loss worsened. A 50M run tests whether more broad-language
exposure under the same architecture/mixture moves these fixed outcomes. It is
not a claim that duration alone solves repetition or instruction following.
Card 06 remains blocked.

## Supply and one smallest source addition

Use the same 341,885,952-parameter dense model, 32,768-token tokenizer file
SHA-256 `308b33a6edbed613f3caf5232b124a7c6105a7a3dc273c999725be51c3ffeaa9`
from the accepted C03-C1 origin release
`e893cb2e3c65f7f6360f73e4daa159dcec5da1b4c2c6f911ff5d9d549b815818`
(manifest SHA-256 `718a5f1df007e59a32717bc660708ab05e4c5c8b90da963e1e8a960208c4f3ca`),
context 1,024 and **81/18/1** target mixture. Bind that verified tokenizer
origin explicitly in the prospective mixture declaration; the new training
release is not the tokenizer's fitting release. For exactly **50,000,000**
supervised targets, the quotas are **40,500,000 general**, **9,000,000
explanatory** and **500,000 incident**. At most two exposures of any position
requires at least **20,250,000 / 4,500,000 / 250,000** distinct eligible train
positions before packing, plus independently assigned held-out families.

| Source stratum | Accepted B7 unique train content positions | Two-exposure capacity | Planned targets | Current margin or deficit |
| --- | ---: | ---: | ---: | ---: |
| General, Gutenberg | 20,802,818 | 41,605,636 | 40,500,000 | +1,105,636 |
| Explanatory, Wikimedia | 3,155,976 | 6,311,952 | 9,000,000 | **−2,688,048 targets; at least 1,344,024 more distinct positions** |
| Incident, PagerDuty/Scoutflo | 343,703 | 687,406 | 500,000 | +187,406 |

These are tokenizer positions in distinct retained training documents, not
vocabulary types, and exclude EOS; they conservatively expose the narrow
general margin. They describe the accepted B7 baseline, **not** the new
post-cleaning supply. The unadmitted C05-B9 structure candidate would give
about 20.800M / 3.156M / 0.339M, a similar constraint. Preserve B7 and B9
unchanged; neither the B9 candidate nor its measured positions are an
admission receipt for the prospective release.

The smallest currently supported addition is one more hash bucket from the
**same pinned Wikimedia complete shard**: change only its new declaration's
`hash_remainders` from `[0, 1]` to `[0, 1, 2]` at modulus 16. The preceding
bucket produced about **1.55M** unique eligible explanatory train positions
and 3,314 independent train page families after screening, so one comparable
bucket would bring supply to about **4.71M**, only about **0.21M** above the
mathematical minimum. This is a forecast, not an eligibility receipt. The old
two-bucket Wikimedia snapshot is 198,642,832 selected bytes; a third bucket
projects roughly 298 MB, so the prospective selection cap must be declared as
**384 MiB** rather than the prior 256 MiB. The complete compressed shard was
deleted after earlier selection; it must be fetched again in a separately
approved attempt. Its pinned revision is `0641bb84bd9b7162bcddf8be7836822161a9a342`,
path `wikimedia-0027.json.gz`, SHA-256
`0abe6c9be821ac100c604ac4b799faa00c90e39d139c7dccf9491e693b177ed0`,
and exact compressed size **480,242,750 bytes**. Reuse cold-verified unchanged
Gutenberg, PagerDuty and Scoutflo snapshots through native acquisition binding;
do not redownload them. No additional source or shard is proposed.

The prospective expanded training release **must** declare
`release.normalizer: normalizer-structure-v3`. The reviewed v3 implementation
and source-span review establish a candidate transformation, not corpus
admission. Apply the accepted local-research source policy, record-level
exceptions and spot audit to new rows, and review the expanded v3 output for
cleaning, rights and use before admitting this exact new release. Preserve
prior family IDs, splits and strata; freeze new independent families before
build, reject incompatible aliases, cross-split exact/near leakage and rights
conflicts. Preserve raw snapshots, quarantines and all B7/B9 artifacts.

Corpus admission is a mandatory pre-training gate: cold-verify the new v3
release and its lineage, complete rights/cleaning and held-out-family review,
and record a reviewed admission decision for its exact release identity. Do
not fit or change the tokenizer, alter the protected evaluation content, or
materialize a mixture before the new release passes its rights, family and
content checks. Measure **post-cleaning, post-admission, post-dedup** distinct
train content with the unchanged tokenizer; require **at least 20.25M general,
4.50M explanatory and 0.25M incident** unique positions, no loss of accepted
held-out families, and a native exact 50M-position mixture receipt showing at
most two exposures of each position. All of these are mandatory before staging
or model work. If one bucket falls short, stop and report the deficit; a second
bucket or changed mix needs a new plan.

For this gate, use the native `corpus measure-tokens` origin binding implemented
and tested at `18e90b82aa7b563183600d89156f0c10688b45f1` (measurement code
SHA-256 `d6fef1d9ee746096bb023a2bcecf1358a144cf9d311e72f2b54634e11f6c4bc5`).
After the v3 release and its finalized family inventory have been admitted,
the command shape is:

```text
uv run --locked --no-sync sparselab --work-dir /srv/sparselab/state/experiments/kernel-memory-lab corpus measure-tokens <new-v3-release> --tokenizer <unchanged-C03-C1-tokenizer.json> --tokenizer-origin-release <original-C03-C1-release> --family-inventory <new-finalized-family-inventory.jsonl> --policy <50M-token-floor-policy.yaml> --output <new-attempt>/prep/unique-train.json --batch-source-bytes 8388608 --json
```

Bind the resulting receipt to the new release manifest/documents, original
tokenizer fit release and export, tokenizer bytes and finalized inventory; cold
read it before using the measured floors. The 8 MiB tokenizer batch bound is an
operational per-document limit, not added training supply or a resource-cap
increase. If an eligible document exceeds it, stop rather than bypassing the
native measurement. This command is preparation work inside the existing
allocation; it does not create a new ledger or authorize an attempt restart.

## Fresh run and unchanged evaluation

Propose **fresh seed-17 initialization** with empty AdamW and scheduler state,
the unchanged dense architecture, effective batch one, ROCm/BF16, peak/floor
learning rates `3e-4`/`3e-5`, 500-step linear warmup and cosine decay through
update **48,829**. Do not resume C05-B8: its 25M scheduler ended at its floor,
and both the data release and horizon change. Its old 25M exposures remain
historical scientific cost, not charges to fresh weights; the new run's own
per-position exposure cap is two. Stage only through validation with the
authenticated prepared bundle and zero optimizer updates. Exactly 50M targets
need **48,828 full 1,024-target updates plus one 128-target update**, at most
**48,829 updates**, with **896** final padding targets masked. No smoke/warmup
optimizer work, training retry or resume is included.

Preserve full initial, every-5,000-update and terminal checkpoints (11 total),
with one held-out operational validation batch at each. Use the unchanged
frozen `CARD05_BASE_EVALUATION_PROFILE_V1.json` SHA-256
`bb4c1b719b29fa96c4429516022a5038f0e3595b9d8ef63394bf0a17a5c6851d`,
bound to the **original C03-C1 held-out release**
`e893cb2e3c65f7f6360f73e4daa159dcec5da1b4c2c6f911ff5d9d549b815818`
(manifest SHA-256 `718a5f1df007e59a32717bc660708ab05e4c5c8b90da963e1e8a960208c4f3ca`),
its frozen document identities and the same tokenizer. Verify this binding
against the original release, not the prospective v3 training release:
all 12 fixed validation windows at all 11 checkpoints, earliest verified
minimum finite token-weighted validation loss for checkpoint selection, then
the same 12 test windows and 24 true/decoy pairs. Repeat the profile's exact
historical C05-N1 v2 comparison and eight 64-token greedy prose prompts on
each selected checkpoint, with unchanged 64-token context, 64 requested new
tokens, cache/decoder recording, blind descriptive review and no instruction
framing. Retained C05-B8 profile outcomes provide a **read-only** second
comparison; they do not add evaluation calls or change selection. Preserve the
original prospective base-learning thresholds: ≥10% lower fixed test loss than
v2, ≥18/24 true-continuation preferences and ≥4 more than v2, and ≥6/8
coherent/relevant/nonrepetitive new prose continuations. A pass is reason to
consider instruction adaptation, not reader eligibility. Do not rerun the
frozen 200-item language suite or the 400 evidence items in this proposal.

## Proposed single conditional allocation, not yet approved

| Boundary | Proposed hard ceiling |
| --- | --- |
| Whole attempt | **36,000 elapsed seconds** from a new preparation ledger through final verification, with one fixed common-root storage baseline and no ledger reset, attempt restart, training retry or resume; only the charged source-transfer retry below is permitted |
| Preparation | **14,400 seconds**, 24 GiB process-tree RSS, 8 GiB added apparent bytes, 10,000 added entries, 8 aggregate agent-assisted reviewer-hours; no model work |
| One source transfer | **7,200 seconds**; at most **960,485,500 source response-body bytes** including **one bounded, charged retry of that transfer only**, **4,194,304 metadata/redirect/error-body bytes**, **964,679,804 total**; enforce before/during transfer under the same attempt ledger and deadline, without restart; ≤4 GiB expanded stream and ≤384 MiB selected Wikimedia output |
| Runtime | **21,600 seconds**, including ≤18,000 for stage/train/checkpoint selection and ≤2,400 for evaluation; one training invocation, ≤48,829 updates, exactly 50M targets for success |
| Memory/device | ≤20 GiB whole-device VRAM and ≤24 GiB process-tree RSS; fail closed on sensor, monitor, OOM or numerical failure |
| Common-root storage | ≤72 GiB new apparent bytes and ≤10,000 new entries across preparation, checkpoints, logs and evaluation; include retained artifacts made by this attempt; ≥2 GiB projected free-space margin; no deletion of old evidence |
| Nontraining work | ≤11 one-batch operational validations/11,264 inputs; ≤55,000 fixed-profile scoring inputs; ≤170,000 aggregate nontraining forward-input positions; ≤16 generation calls/1,024 requested new tokens; ≤64 MiB evaluation output; ≤2 aggregate agent-assisted evaluation reviewer-hours |
| Cloud | Zero use or spend |

The retained B8 attempt took about **7,960 seconds** end to end, of which the
training monitor took **6,348 seconds**, at 25M targets with the same number
of checkpoint events. Doubling that entire training-monitor duration and
carrying forward its other observed overhead gives a rough **14,308-second**
50M path estimate, before new-source preparation or changed long-run stalls.
It is an extrapolation, not a confidence bound; the 21,600-second runtime and
36,000-second aggregate are safety ceilings. B8's 11-checkpoint run added
46,863,235,311 apparent bytes and 1,879 entries after its common baseline;
same-count checkpoint retention plus a projected additional ~100 MB selected
Wikimedia bucket and 50M prepared mixture should be checked against the 72 GiB
ceiling and actual free space before any future launch. The previous
55,000/170,000 evaluation ceilings already include 52,392 fixed-profile
inputs, 11,264 operational validation inputs and conservative full-prefix
generation reservations, even if cache use reduces actual inputs.

**Decision requested later:** approve this single conditional preparation and
fresh 50M run only after a separate review of the declaration, native reuse
binding, measured headroom and exact identities. Admission of the exact
expanded v3 release and its measured post-cleaning unique-position floors
remain pre-training conditions, not assumptions. This document authorizes no
acquisition, mixture materialization, model forward, generation, GPU work,
optimizer update or cloud job. Stop on the first hard failure or token/family
shortfall; preserve all receipts and return for a new decision rather than
expanding a cap, data source or cleaning rule. Apart from the single charged
source-transfer retry, no attempt restart, training retry or resume is proposed.
