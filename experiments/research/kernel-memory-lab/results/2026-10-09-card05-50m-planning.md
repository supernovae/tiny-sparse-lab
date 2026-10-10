# C05-P5 — merged mainline checkpoint and prospective 50M plan

**State: planning only; no new data or runtime authority.** PR
[#55](https://github.com/supernovae/tiny-sparse-lab/pull/55) merged the reviewed
base-pretraining/shared-corpus branch at
`58ac08ee1289c1446491c8082851f5be196dfbcd`. The exact PR head was
`fdd4af9660c94bfebecf90393145355e09dd4f30`, its safe check passed,
GitHub reported clean mergeability and no required review rule, and the
manual model suites stayed skipped. The main-push safe check passed at the
merge SHA. [C05-B10](2026-10-09-card05-mainline-review.md) records the
whole-delta/source-span review and focused corrections. The C05-B9 cleaning
release remains NOT ADMITTED; C05-B8 model quality remains pending owner review;
C05-N1 remains 0/200 NOT ELIGIBLE and Card 06 is blocked.

[The fresh proposal](../CARD05_BASE_50M_PROPOSAL.md) retains the 341,885,952-
parameter architecture, tokenizer and frozen base evaluation while requiring
measured supply for a fresh 50M-target 81/18/1 run. At two exposures,
explanatory prose needs **4,500,000** distinct eligible training positions;
the accepted release has **3,155,976**, a deficit of **1,344,024**. One more
hash bucket from the same pinned Wikimedia shard is the smallest proposed
addition and is forecast to yield about 1.55M distinct positions, but the
small margin is a hard stop if screening yields less. The full shard is no
longer retained, so the proposal explicitly includes its one future accounted
download; no transfer or new mixture occurred in this task.

The prospective allocation is 36,000 seconds aggregate, with 14,400 seconds
preparation and 21,600 seconds runtime subcaps; source and metadata response
bodies ≤960,485,500 and ≤4,194,304 bytes; 8 GiB preparation and 72 GiB
whole-attempt added storage, 10,000 entries, 20 GiB VRAM, 24 GiB RSS, and
one fresh run capped at 48,829 updates for exactly 50M targets. Existing
fixed-profile and aggregate evaluation ceilings remain 55,000/170,000
forward-input positions and 16 calls/1,024 requested new tokens. These
numbers are a review request, **not authorization**. No staging, acquisition,
tokenizer fitting, model forward, generation, GPU use, optimizer update or
cloud spending occurred in this planning step.
