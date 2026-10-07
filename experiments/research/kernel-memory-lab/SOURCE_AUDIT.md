# Current main source audit

Audit date: 7 October 2026
Pinned main: `06efc4db82ecf3da97b50cff518cba605ad27b33`
Previous document baseline: `c54b17a59cc59a81e51949ff699ab859e707fc28`
Current main includes PR49, merged 7 October at 14:14:58 UTC. The [exact comparison](https://github.com/supernovae/tiny-sparse-lab/compare/c54b17a59cc59a81e51949ff699ab859e707fc28...06efc4db82ecf3da97b50cff518cba605ad27b33) contains 18 commits and merge PRs 46, 47, 48 and 49. PRs 44 and 45 are not new in this delta; the refreshed documents now account for newly shipped native SDPA in PR49.

This pass inspected repository source and documentation through the read-only connector. No repository commands, tests, GPU probes, training or hosted acceptance checks were executed. Existing tests and contracts in source are not new passing test receipts. The actual WSL checkout and hardware remain unverified.

## Place the new experiment correctly

[Root AGENTS.md](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/AGENTS.md) requires real campaigns under `experiments/research/<campaign>/` and teaching material under `experiments/samples/`. Keep mutable or large runtime outputs under the external persistent root's `experiments/<campaign>/`. Small protocols, configs, reviewed summaries and evidence references can be checked in.

The default persistent root resolves from absolute `XDG_DATA_HOME` to `$XDG_DATA_HOME/sparselab`, otherwise `~/.local/share/sparselab`. `SPARSELAB_WORK_DIR=/data/sparselab` is a documented suggestion for substantial campaigns; global `--work-dir` takes precedence. Do not set or change it automatically. Resolve and record the actual root before a run. This pack proposes `experiments/research/kernel-memory-lab/`, not an already created directory. There is no verified requirement for STATE.md, HYPOTHESES.md or CANDIDATES.md; STATUS.md is a human pointer, not a new native campaign authority.

## Reuse shipped work

### Typed authoring from PR46

`sparselab config derive SOURCE --set FIELD=JSON --output NEW.yaml --json` and `sparselab experiment derive SOURCE --set FIELD=JSON --output NEW.yaml --json` author fresh validated declarations with `.derivation.json` receipts. Experiment derive also supports `--prepared RECEIPT` and `--max-runs 1000`. These are writes. Receipts declare `executable: false` and `runtime_verified: false`. They do not derive checkpoint tensors, train a model or prove arbitrary dense-to-reader transfer. Use source schemas and help before authoring.

Sources: [main CLI](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/cli/main.py), [experiment CLI](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/experiments/cli.py), [iteration contracts](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/iteration.md).

### Sealed staging and exact state comparison

`stage CONFIG --prepared-inputs PRIOR_STAGE/prepared --through warmup --output NEW_STAGE` authenticates a sealed prior staging bundle and requires a fresh destination. It is not an arbitrary data-prepare array cache. Missing/mismatched inputs fail without downloading or repacking; allow_runtime_drift is rejected in this mode. Stage can perform runtime work and has no `--json` flag.

`canonical_training_state(snapshot, observation=observed_update)` and `compare_training_states(left, right)` are Python APIs, not a checkpoint-digest CLI. They compare supported exact training-state bytes without numeric tolerance, excluding run/file/source identities and operational paths/timing. Enabled time-based cadence or unknown fields fail closed. The digest does not authenticate a checkpoint, establish artifact lineage or authorize resume. Keep hash/authentication and tolerance-based output-equivalence checks separate.

Sources: [dataset contracts](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/datasets.md), [checkpoint contracts](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/checkpointing.md), [state-digest API](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/training/state_digest.py).

### Semantic controls and trainer support

`semantic probe DECLARATION.yaml --json` is a read-only CPU inference probe using supplied vectors. It does not train, create a run or encode text. Reuse it for bounded vector controls, not as evidence of natural-language reading.

The current trainer **does** build `SemanticQueryBatch` from verified FP32 `[B,T,K]` query and boolean `[B,T]` mask sidecars, with ownership checks in allocation mode; evaluation has an analogous path. Some semantic-memory prose is stale. Use the actual trainer code rather than treating batch construction as missing. The retriever supports top-k 1–16, while the adapter specifically uses top-1. Queries detach to CPU float64, and keys are FP64 on CPU. Default bounds are 65,536 entries, 256 MiB tensor assets, 4,194,304 comparisons and 64 MiB results. This lookup does not train a content router or implement raw-text encoding/SSD paging.

Sources: [semantic probe contract](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/research/semantic-memory.md), [training sidecar construction](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/engines/pytorch.py#L1186-L1227), [semantic retrieval](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/engram/semantic.py).

### Source overlap and preparation benchmarking

`memorization analyze INPUT.yaml --json` is read-only descriptive supplied-continuation/source overlap. Inputs bind file SHA or an authenticated retained generation-panel row. N-gram size is 1–10 (default 3); edit comparison cap is 1–2,048 (default 512). It is explicitly not a quality gate or complete contamination proof. Keep the proposed train/eval split, answer-exclusion and exact/near-duplicate safeguards.

`data benchmark-preparation DECLARATION.yaml --json` mutates a fresh workspace/report and runs offline generated-corpus/tokenizer preparation in isolated subprocesses. Declared limits include ≤32 MiB per size, ≤64 MiB sum, ≤32 cases, document batches 1–256, source batches ≤8 MiB, threads no more than logical CPUs, and worker timeout ≤3,600 seconds (default 120). These are implementation bounds, not this project's approved runtime. Results concern generated-input throughput/parity, not real-corpus preparation or GPU/model speed.

Sources: [overlap CLI](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/corpus/memorization_cli.py), [prep benchmark](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/data/preparation_benchmark.py).

### Attention and hosted recovery contracts

Native `attention.implementation: sdpa` is opt-in. Reference remains the default. Configuration requires PyTorch, dense attention and equal query/KV head counts. `runtime attention CONFIG --worker WORKER --store RUNS --json` executes disposable forward/backward and forced backend candidates. It is active runtime work. Report the actually selected backend separately from supported MATH/EFFICIENT/FLASH/CUDNN candidates; an SDPA call does not guarantee a fused implementation. T4 source gates require observed SM75, FP16 GradScaler and efficient forward/backward. No FlashAttention package integration or TPU execution is established here.

PR49 supplies hosted transport/relay contracts, including notebook-cell authoring, session/SSH inspection, setup, registration, relay checks, runs and collection. Setup/registration/relay/runtime/run/collect are active operations with separate authority and resource requirements. Provider allocation is not established. Relays commit synchronously at finalized checkpoint boundaries and preserve the last verified committed checkpoint, not in-flight work. UNKNOWN/disconnection is not proof the worker is dead or permission to retry. Recovery collection with explicit confirmed worker loss preserves the original UNKNOWN state. The pinned backlog still requires live S3/T4/Drive loss-recovery acceptance; local round-trip evidence cannot close it.

Sources: [SDPA configuration](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/config/models.py#L501-L510), [dense attention](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/model/attention/dense.py), [hosted environments](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/hosted-environments.md), [project TODO](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/TODO.md).

## Safe observation versus active work

Verified observational examples include `inspect CONFIG --json`, `checkpoint verify GENERATION --json`, `evidence RUN --runs-dir ROOT --json`, `triage RUN --runs-dir ROOT --json`, and `iteration check DECLARATION_OR_LOCK --parent GENERATION --json` with optional `--cold-verify`. Iteration check does not prepare/download/train or write proof receipts. Workspace preflight has no `--json` flag, but it is not classified as read-only by the main CLI: it can initialize the persistent work root and scratch directories. Treat it as capacity inspection with setup side effects, outside the strict first-session read-only workflow. Campaign validate/status/next/explain accept CAMPAIGN and --json; plan is also read-only. These commands observe state; their proposed next actions are not approved operations.

Campaign apply/resume can mutate and pilot/evaluate/generate even without `--execute-runs`. Config/experiment derive, stage, prep benchmarks and readiness smoke are active. `readiness smoke --family dense --output FRESH` has no `--json` and performs tiny train/resume. Do not hide these under a read-only audit. The workbook's commands remain templates for actual existing validated inputs in an already available locked environment.

## Mixture accounting remains a project responsibility

The corpus pipeline’s requested_mixture is reporting metadata, with actual counts computed from retained records. Corpus pipeline/export is unchanged across this delta; an enacted sampler has not been added. Declare realized ordering and target-token accounting rather than treating the requested mixture as sampled evidence. Source: [corpus pipeline](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/corpus/pipeline.py#L3069-L3110).

## What the audit does not establish

The fresh main configuration, approved corpora/tokenizer, comprehension-capable checkpoint, natural-language chunk reader, lexical evidence baseline, dense-to-new-reader transfer, learned router, frozen-weight update test and cold-NVMe frontier remain project work. All thirteen cards are NOT STARTED. Fresh initialization, language nuance/evidence use, matched controls, actual receipts and bounded approval remain the governing design.
