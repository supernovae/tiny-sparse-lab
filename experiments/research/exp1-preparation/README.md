# Fresh EXP1 native preparation (unexecuted)

The [LAB86 package](lab86/README.md) now supplies pinned candidate shard/source
declarations, policies and job templates. It also records concrete native
compatibility boundaries and a separately sealable initial pilot session.

LAB89 software plan for LAB86. This plan has **not acquired or prepared a real
corpus**. It compiles native argv only. Missing bindings refuse; it contains no
owner approval, guessed shard hashes or historical corpus/tokenizer reuse.
Use `attempt phase-command --plan preparation.json --attempt-root ROOT --label
LABEL --bindings-json JSON` to inspect a phase, then the existing `attempt
run-phase` with the reviewed zero-model contract and resource policies. Inspect
public help for required ledger/baseline arguments. There is no run-all command.

## Bindings and sequence

1. The parent/operator qualifies exact source schemas, rights and full shard
   descriptors before initializing one acquisition project and ledger. Bind
   `ACQUISITION_PROJECT` to that immutable full-scope declaration. Pin
   `HuggingFaceTB/smollm-corpus` to
   `3ba9d605774198c5868892d7a8deda78031a781f`, and
   `codeparrot/github-code-clean` to
   `c48d40f9e70f0196f8236901ee35807f7d6c44c0`. Each admission source entry selects
   one exact shard; entries share reviewed strata and repository/seed families.
   Acquisition walks sorted declaration paths. Put the three reviewed pilot
   declarations first (web, synthetic, code), followed by the remaining scope;
   verify that the next reservation exceeds the remaining initial allowance.
   Without this ordering and size check, the gate selects an alphabetical
   prefix, not a three-stream pilot. Do not initialize until that check passes.
2. Initialize once, then acquire under the initial gate. Whole-file reservations
   crossing 4 GiB refuse before transfer; earlier completed snapshots and all
   failure evidence remain. Such a refused acquisition is **not success**. Read
   `pilot-budget-status`, preserve its ledger receipt and prepare explicit
   feasibility findings. Metadata and retries consume the same allowance.
   A refused full acquisition has no complete acquisition lock. Before
   inspection, bind a separate `PILOT_REUSE_PROJECT` containing only completed
   pilot snapshots with exact `reuse_only` source effects (origin project,
   declaration hash and snapshot hash). Retain the same source declarations;
   omit transport allocation from this local copy project. `pilot-reuse` uses
   native verified retained-copy acquisition without HTTP, then the pilot
   admission/inspection/inventory phases provide review evidence. Bind
   `PILOT_APPLICATION_POLICY` and `PILOT_INSPECTION_SELECTION` to this pilot view.
   Its separate zero-model attempt contract must bind that exact project; retain
   the whole-preparation disk baseline, wall deadline and resource envelope.
   Set `offline_retained_sources_only: true` in that project-bound contract.
   Initial local copy materialization uses `acquire` without `--offline` (that
   flag verifies an already published lock). Dispatch permits it only with a
   bound project, every source effect `reuse_only`, and no transport budget.
   Copies count against additional workspace bytes. This creates no second
   network allocation and does not change the full-scope transport ledger.
3. Stop for actual owner review of schema/rights, provenance, usable yields,
   family counts, qualified row/byte yields and measured resource use. Final
   tokenizer yield remains unknown before the full split and single fit; report
   conditional projections as estimates, never as measured unique tokens.
   No automatic gate release is included. A separately authorized
   `corpus budget-release-gate PROJECT --report REPORT --review REVIEW` records
   the report-bound decision; then `continue-acquire` uses the same ledger.
4. Bind reviewed `APPLICATION_POLICY`, `POLICY_DOCUMENT` and
   `INSPECTION_SELECTION`; draft and inspect admission. A human-reviewed native
   admission receipt is required before `PRE_FREEZE_PROJECT` is bound. The
   reviewed clusters set source strata and `split_weights` 98/1/1, merging
   duplicate/repository/seed families before fit. Bind `REVIEWED_CLUSTERS` and
   freeze; fewer than 100 eligible families per stratum refuse.
5. Bind `BUILD_PROJECT` to the reviewed admission and newly frozen split bundle,
   build offline, then bind actual `BUILD_RESULT` and freeze its release. Bind
   the returned `RELEASE`, finalize family inventory and audit. Required native
   project/release acceptance bindings remain mandatory; no acceptance is
   synthesized by this plan.
6. Export LM with reviewed `BASE_RUN_CONFIG` (sequence length 2048, native int32
   tokens, no model launch) and vocabulary 32768. Copy its generated tokenizer
   declaration to a new reviewed `TOKENIZER_CONFIG`, preserving dataset/export
   identity, adding 60/20/20 TRAIN byte sampling at most 512 MiB and a new output
   under the attempt root. Fit once; bind the actual `TOKENIZER`. Its manifest
   includes disjoint TRAIN audit and shortfall evidence. Never alter validation
   or test assignments to fill fit/audit quotas.
7. Bind `TOKEN_FLOORS`, measure available unique targets, and stop if conditional
   12B targets cannot be met without repetition. A new reviewed
   `MIXTURE_DECLARATION` binds the release, family inventory, tokenizer,
   60/20/20 target quotas and `max_exposures: 1`. Materialize/verify the unique
   export, then bind the corresponding native `RUN_CONFIG` and
   `RESOURCE_ENVELOPE`; publish and cold-verify packed inputs. This ends
   preparation. Training remains separately authorized.

The metadata-only candidate pilot totals 3,771,713,555 bytes (about 3.513 GiB):
FineWeb `train-00068-of-00234.parquet` (2,251,835,383), Cosmopedia
`train-00000-of-00104.parquet` (1,174,424,341), and CodeParrot
`data/train-00570-of-00880.parquet` (345,453,831). These basenames and counts are
planning evidence, **not complete source declarations**. Full paths, verified
hashes, schemas and qualification decisions must be supplied before execution.
Minimum-sized shards do not prove representativeness. Full selection cannot
silently reuse the smallest-shard pilot as a representative corpus.

## Envelope and remaining operational gates

Use one immutable transport budget with `initial_body_bytes: 4294967296` and
`max_total_body_bytes: 107374182400`. These are response-body charges, not TLS/wire overhead. Source/metadata individual allowances also
remain enforced. Declare the full source scope before initialization: changing
sources, pins or budgets changes ledger identity and cannot reset spent bytes.
The wall ceiling is 172800 seconds; the whole preparation monitor binds at most
322122547200 **additional** workspace bytes, at most 8 CPU threads, and a 16 GiB
RSS observation/stop target. Sampling and monitoring are not hard memory caps.
Native memory-heavy preparation paths still require pilot observation; this
software does not establish that a 12B corpus fits the target. Runtime/container
hard enforcement must be qualified separately by the assigned CPU runner.

Existing LAB39/LAB70 holds remain. Actual CPU preparation belongs to
`lab-gpu-runner` only after reviewed source/software/runtime pins and explicit
job approval. No download, tokenizer fit on real data, GPU job or model training
is authorized by this file. Full qualified acquisition/application declarations,
owner review and runtime feasibility evidence are still required bindings.
