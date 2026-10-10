# DevMind v5: reproducible successor and dense MODEL-0

**Protocol ID:** `devmind-v5-reproducible-model0-2`.
**Authorization:** the reviewed request preserves v2/v3/v4 as nonreconstructable
historical releases and advances a separately identified v5 through recovery,
tokenizer selection, MODEL-0, ROCm training and evaluation. This is a new corpus
and protocol, not a repaired historical identity or a new historical replay.
No Developer SFT or model-weight publication is authorized here.
Revision 2 records the measured operational preparation bound in
`preparation-amendment.json` before tokenizer fitting or model outcomes; it
does not change source intent, scientific settings or acceptance criteria.

## Historical disposition and reviewed source intent

`historical-releases.json` records the unchanged historical build/release/producer
identities and available failure evidence. Nonreconstructable means the retained
recipes and available authenticated closure did not recover those historical
identities; it does not prove permanent impossibility if original evidence is
later recovered. Historical declarations, reports, expected digests, workspaces,
failed attempts, snapshots and receipts are read-only.

`corpora/devmind-v5/` copies all 123 reviewed final-v4 source declarations
byte-for-byte, including immutable revisions, bounded selection, rights,
obligations, exclusions, source IDs and whole-source train/validation/test
families. The splits and schema-3 metadata/reconstruction-only release policy
are also byte-identical. Only project ID `devmind-v5` and transform ID
`devmind_v5_lm` change. Source IDs retaining v2/v3/v4 prefixes identify reviewed
source families, not imported historical snapshots. Notes in these copied
sources describe their historical selection and do not assert that later v5
stages have already run. `source-intent.json` binds declaration-byte equality.
V4's source-supply, shape and lexical-screen observations remain historical;
v5 measures its own retained data rather than borrowing historical counts.

## Producer, corpus and independent recovery gates

Commit the declarations and tested necessary compatibility changes before
network acquisition. Record that full producer commit, tree, package/module
identities, uv lock/Python environment and effective declarations in external
operational evidence. Acquisition/build/freeze use that fixed source; later
reports do not become producer inputs. One producer acquires all 123 sources
from their declarations: no v2/v3/v4 snapshot imports, adapter re-stamping,
identity substitution, or hidden ancestry dependencies.

Primary mutable output is under
`/srv/sparselab/state/experiments/devmind-pretrain-v5/primary/`; independent
recovery uses a fresh `recovery/` sibling. Downloads, mutable indices, source
snapshots, exports, tokenizer bytes and checkpoints stay external. Preserve all
failed/interrupted attempts. Before each costly phase check bytes/inodes/RAM,
reserve projected additional use plus at least a 25% filesystem free margin,
and monitor the actual output filesystem during execution. A declaration cap
is not a bound on the remote Git cache. Never prune another task's data.

Independently authenticate primary acquisition, every snapshot, complete build
identity/artifact inventory, rights audit, LM lineage and frozen release. Archive
unaltered acquisition bytes and publish compact source-ID→snapshot-ID, producing
module provenance, complete build-identity metadata and release references. The
first valid v5 build/release become v5's immutable expectations, committed before
the recovery attempt; they are never called the old v4 identities.

Recover from exact Git producer/declarations and the locked environment into the
fresh independent corpus workspace. No primary snapshot/build/release copy and
no historical corpus reuse are allowed. Shared reconstructable download caches
are permitted only when upstream pinned bytes/checksums are independently
validated, and their use must be recorded. Compare all 123 snapshot IDs, project,
build and release IDs exactly; enforce expected build before freeze. Stop on a
mismatch or unverifiable closure, preserve it, and do not fit a tokenizer until
fresh recovery and current independent verification pass. Reproducibility is
bounded to the observed environment and available pinned upstream bytes, not an
unmeasured cross-host or indefinite-availability claim.

## Tokenizer selection gate

Before fitting, commit an actual-release-bound schema-2 bakeoff declaration:
three vocabulary candidates `[16384, 24576, 32768]`, `max_fit_bytes: 268435456`,
`eval_split: validation`, `eval_max_docs_per_group: 200`, `near_best_ratio: 0.98`.
Include every actual train/independent-validation pairing in fixed
`prose, python, go, rust, shell, yaml, json, toml, logs` order; record absent
pairings. Fit each candidate to the same hash-ordered train-only receipt and
score disjoint heldouts. Select the smallest vocabulary within 0.98 of the best
weighted bytes/token. No manual favorite, test-set selection or hidden refit.
Authenticate report, candidate bytes/manifests, fit receipt, heldout receipt,
release and chosen vocabulary; retain the authentic fit-sample revision when
using the selected tokenizer with the full-release LM export.

## MODEL-0 controls and measured budget

Carry forward the reviewed dense MODEL-0 controls: seed 42, tied dense decoder,
hidden dimension 640, 10 layers, 10 heads, FFN 1664, context 1024, no memory,
all-token next-token objective. Backend PyTorch/ROCm device 0, BF16, balanced
VRAM fraction 0.90, transformer-block activation checkpointing, no offload.
Microbatch 2 × accumulation 4 gives 8192 supervised targets/update. AdamW peak
`3e-4`, floor `3e-5`, warmup 2000 steps, weight decay 0.1, betas `[0.9, 0.95]`,
epsilon `1e-8`; decay horizon equals the measured whole-update budget.

Measure distinct normalized developer_systems train tokens by content SHA using
the selected tokenizer, excluding dropped documents and all heldouts. Let `D`
be that observed token count. Set
`max_steps = floor(floor(3*D/2)/8192)` and `max_tokens = 8192*max_steps`.
Stop if `D == 0` or `max_steps <= 2000`; no old token estimate or byte proxy is
an input. The release's requested mixture is descriptive, not an enacted sampler.
Train the entire verified LM export in its declared order, without inventing
oversampling or an implemented developer-share mixture.

The selected vocabulary is a preregistered variable, not an architecture tuning
choice. The 32768-vocabulary inventory reference is 69,317,760 parameters;
report the actual inventory at the deterministic winning vocabulary and verify
it against the fixed model settings. Unlike the unsuccessful v4 continuation's
reference-count stop, v5 does not reject its own preregistered winner solely for
changing tied-embedding parameters. Do not alter widths, layers, sequence,
optimizer, precision, data, seed or effective batch to chase runtime fit.

Validation/checkpoint cadence 2048 steps, validation maximum 64 batches,
periodic/best/latest/previous verified generations retained. Smoke 2 updates,
warmup 5, logging every 100. Derive full-release document/token ceilings from
actual rendered UTF-8 bytes/document counts as in Corpus Forge export; do not
prepare/train the unbound base config. Use the generated export dataset fields
unchanged and replace only the generated tokenizer path with the verified
bakeoff winner. Prepare sealed contiguous-eos-v6 train/validation arrays with
`all_tokens` loss and no redundant mask. Record counts and manifest digest.
Operational encoding batches contain at most 256 documents and 8 MiB of
decoded source text; the largest observed record is 4,943,884 bytes. No
truncation or segmentation is permitted. Use the version-1 resource envelope
with host-memory fraction 0.75, minimum available RAM 2 GiB, at most 4 workers,
queue depth 1 and disk spill; measure disk/inode thresholds before preparation.

## ROCm, campaign and evaluation gates

Use registered `rocm-7900xtx`, preserve vendor Torch/HIP packages, and run fresh
runtime doctor plus real full-shape inspect/validate/smoke/warmup before training.
A memory estimate or tiny doctor is not model-shape fit. Stop on a real fit or
integrity failure rather than CPU fallback or silently changed science. Record
warmed step distribution, supervised targets/sec, peak memory and headroom
separately from preparation, validation, checkpointing and end-to-end timings.

Commit reviewed scientific inputs before lock and long execution. Bind one
fresh `main:single` ExperimentPlan cell with verified corpus release/export,
selected tokenizer and prepared data; keep the machine runtime binding outside
the science SHA. A one-cell Campaign records artifact references, plan lock,
runtime acceptance, explicit approval, one experiment run, collection, ROCm
heldout evaluation and model readiness. Reconcile the same dispatched attempt;
no replacement training run to hide a failure. Require the exact declared
whole-update step/target totals and verified immutable terminal generation.

Preregister one checkpoint-bound heldout_lm validation evaluation. Require
finite loss and positive valid-target count, not an invented quality threshold.
Readiness requires a verified checkpoint, completed heldout evidence and at
least one evaluation; READY_FOR_NEXT_STAGE denotes evidence completeness only.
Run a descriptive fixed greedy generation panel on that exact checkpoint:
`def parse_config(path):`, `SELECT user_id FROM`, and `#!/bin/sh\nset -eu\n`;
temperature 0, top_k 0, max_new_tokens 64, seed 42. Preserve negative outputs.
No human quality verdict, promotion or publication license is inferred.

Bind final ModelFamily model-0 parent-null lineage to exact corpus, tokenizer,
scientific plan, architecture, measured budget, run, immutable checkpoint,
evaluation and readiness evidence. Record recoverable declarations separately
from externally required tokenizer/checkpoint bytes when deterministic
reconstruction is not demonstrated; publish and verify a thin archive without
pretending it contains source data or model weights.

## Outcome reporting

Report observed identities, recovery equality, source/rights/lineage audit,
actual tokenizer scores/winner, developer tokens/budget, prepared counts,
science/plan/runtime digests, hardware warmup, run/attempt IDs, actual training
coordinates/timings, terminal checkpoint, heldout loss/targets, raw generations,
readiness/family/recovery/archive verification and every unexercised gate.
An executed command is not useful model behavior. Fail closed at a scientific
identity/integrity gate, finish reachable evidence work, and name the exact
missing prerequisite; never relabel a partial run as MODEL-0 complete.

## Observed execution evidence

`primary-verification.json` records the independently authenticated first
primary freeze: all 123 v5-produced source snapshots, build
`7789f970526f3c07c005ce8e300e70fdc020e8bcbd088bd5bdf9f55706c1b75a`
and release
`72577dc6898c12caa3e17a731375573b5207d3f58a90963e4581531f3f1bf27b`.
The release envelope version is 1; its bound rights/publication policy is
schema 3. Typed corpus Artifact versions name the envelope, not that policy.
`build-identity.json` preserves the complete producing identity bytes.
`corpus-expectations.json` freezes the exact 123 snapshot/build/release gates
before fresh independent recovery. `corpus-recovery.yaml` pins the tested
producer and explicitly excludes later tokenizer/model closure from the
corpus-only deterministic recovery claim.
The two DNS-interrupted acquisition receipts remain evidence; the successful
primary reused only its own authenticated partial v5 snapshots under the same
producer with process-scoped resolver options. No historical snapshot import
occurred. Fresh recovery, tokenizer selection and MODEL-0 outcomes remain
unproven at this record boundary.

`tokenizer-group-supply.json` records 796,563 selected train documents and
16,411 independent validation documents, with zero cross-split normalized
content or family overlap. Actual paired groups are prose, Go, shell, YAML and
JSON; Python, Rust, TOML and logs lack independent pairings and are explicitly
unmeasured rather than supplied by new sources. `lm-view-measurement.json`
records full LM document/byte ceilings; these are not a token budget.
`typed-corpus-artifact-verification.json` records the exercised real large
release consumer gate: the typed Artifact verifier accepted the exact frozen
release and its retained snapshot closure with envelope version 1 and bound
schema-3 policy. This is not a tokenizer, training or quality result.

`recovery-verification.json` records completed fresh independent replay:
producer, project declaration digest, all 123 snapshot IDs, full build ID and
frozen release ID exactly match the committed primary expectations. Snapshot
imports were zero. The result is exact recovery on the observed host and locked
environment, not an unmeasured cross-host or indefinite-upstream claim.
The immutable first expectations are unchanged; the tokenizer-fitting gate
may now proceed.

`tokenizer-bakeoff-memory-stop.json` preserves the first actual bakeoff's
RAM-censored post-fit accounting attempt. All three candidate fits serialized,
but no complete report, winner or budget was accepted. The availability guard
stopped it below 2 GiB rather than allowing an OOM; its output remains intact.
The source fix streams unique token counts, document-domain flags and
unclassified supply counters without changing fitting, selection or source
scope. `streamed-accounting-verification.json` records 13 focused tests and
actual schema-3 synthetic CLI export/preparation/offline warmup/training plus a
verified 64-target full-state checkpoint. Those are compatibility proofs, not
DevMind scientific results; a fresh complete full-scale bakeoff is required.
The broader corpus/tokenizer/worker/preparation consumer suite passed 206 tests.
`local-readiness-post-streaming.json` binds formal dense CPU checkpoint/resume
smoke at the tested code revision. `runtime-post-streaming-prerequisite.json`
records actual BF16 forward/backward AdamW on the accepted RX 7900 XTX vendor
environment at the new package identity; neither substitutes for the later
selected-tokenizer full-shape MODEL-0 gates.

The fresh streamed full-scale bakeoff completed successfully with no RAM or
storage guard violation. `tokenizer-bakeoff-verification.json` authenticates the
release-bound report and all three candidate tokenizers. The preregistered
weighted bytes/token scores are 3.538705384367656 (16,384),
3.6951324084726047 (24,576), and 3.7870930862814522 (32,768).
The 0.98 near-best rule selected the authentic 32,768-token tokenizer
`ad186b251ca712e5deebf4cad2eda968a287a964a958604170b785cc380e2b56`.
Paired heldout samples contain prose 200, Go 200, shell 42, YAML 136 and JSON
one document. Python, Rust, TOML and logs remain explicitly unmeasured;
this is tokenizer efficiency evidence, not model quality.

`corpus-recovery-inspection.json` binds actual cold recovery CLI inspection:
the corpus is `PRESENT`, pinned producer implementation is `MATCH`, and the
recipe remains `BLOCKED` solely on its two explicit downstream external
boundaries. Independent fresh reacquisition/build/freeze already established
exact corpus reconstruction; the corpus recipe does not claim model recovery.
