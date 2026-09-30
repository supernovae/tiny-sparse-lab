# Campaign v1: declared orchestration

**The Campaign engine executes declared science. It does not design science.**

`campaign_version: 1` is a narrow orchestration language above Corpus Forge and
[`ExperimentPlan(plan_version: 1)`](experiments.md#authored-experiment-plans-and-resolved-locks).
Corpus Forge owns source acquisition, normalized documents, lineage and frozen
release semantics. ExperimentPlan owns concrete scientific model configuration,
artifact identities, comparisons and resolved locks. Campaign adds dependencies,
readiness policies, recoverable execution and explicit authorization; it never
chooses architecture, data mixture, tokenizer, training budget or quality threshold.
Existing corpus, tokenizer, experiment, controller and training CLIs remain usable.

The [Campaign JSON Schema](../schemas/campaign-plan-v1.schema.json) describes the
strict authoring format. YAML/JSON is data only: duplicate keys, executable tags,
nonfinite numbers, unknown fields, duplicate stage/dependency IDs, unknown or
self dependencies, cycles and incompatible producer kinds/scopes are rejected.
Campaign file references must be declaration-relative, without absolute paths,
parent traversal or symlinks. A dependency in `requires` is not an artifact binding:
each typed input explicitly names its compatible producer in `requires`.
Approval bindings may name transitive ancestors. Topological execution preserves
authored order among available stages.

## A complete small corpus workflow

[`examples/tiny-campaign.yaml`](../examples/tiny-campaign.yaml) is a valid partial
campaign and a complete, runnable corpus/readiness/approval workflow. Its recipe
contains twelve short local text files: ten train documents in two source families
and two test documents in one heldout family, across `technical_docs` and
`developer` domains. The selected LM training view uses the `raw_document` shape.
It contains no external tokenizer/model artifact placeholders and starts no training.

```yaml
campaign_version: 1
id: tiny-campaign
stages:
  - id: corpus
    kind: corpus_release
    scope: corpus
    project: tiny-campaign/corpus.yaml
  - id: readiness
    kind: corpus_readiness
    scope: corpus
    requires: [corpus]
    corpus: corpus
    policy:
      min_unique_train_bytes_by_domain:
        technical_docs: 100
        developer: 100
      required_nonzero_languages: [en]
      required_nonzero_shapes: [raw_document]
      min_heldout_families: 1
  - id: corpus-approval
    kind: approval
    scope: release
    requires: [readiness]
    bind: [corpus, readiness]
```

Use a new named workspace; the global `--work-dir` precedes `campaign`:

```sh
uv run --locked sparselab --work-dir sparselab-work/campaign-example campaign validate examples/tiny-campaign.yaml --json
uv run --locked sparselab --work-dir sparselab-work/campaign-example campaign plan examples/tiny-campaign.yaml --json
uv run --locked sparselab --work-dir sparselab-work/campaign-example campaign status examples/tiny-campaign.yaml --json
uv run --locked sparselab --work-dir sparselab-work/campaign-example campaign next examples/tiny-campaign.yaml --json
uv run --locked sparselab --work-dir sparselab-work/campaign-example campaign apply examples/tiny-campaign.yaml --json
uv run --locked sparselab --work-dir sparselab-work/campaign-example campaign explain examples/tiny-campaign.yaml --json
uv run --locked sparselab --work-dir sparselab-work/campaign-example campaign approve examples/tiny-campaign.yaml corpus-approval --note 'Reviewed these bound inputs' --json
uv run --locked sparselab --work-dir sparselab-work/campaign-example campaign resume examples/tiny-campaign.yaml --json
```

`validate` performs static checks only. `plan`, `next` and `explain` verify requisite
existing inputs and committed receipts without acquiring/building/training or
creating a workspace. Once initialized, `status` reads the historical durable
index, not artifacts: `verified_at` and `verification: last_committed` identify
the last committed verification, not a fresh rehash. Before initialization it
checks declared static inputs and reports an unexecuted projection, blocking
missing inputs without creating a workspace. The campaign CLI's own `PATH` is
normalized; authored artifact references still cannot contain parent traversal.
`explain` includes authored declarations, policy, dependencies and available gate
bindings. All eight verbs accept `--json`; stdout is one sorted
`sparselab-campaign-command-v1` object. Progress goes to stderr. Text mode lists
stage ID/kind/scope/state/outcome, reasons, deficits, identities and next action.

For example, after corpus readiness the next action is structurally:

```json
{"action":"approve","stage":"corpus-approval","reason":"approval required","identities":{"outputs":[]}}
```

A gate's stage row separately includes its binding SHA and bound identities.
Scope labels are domain ownership, not measurements of success.

## Stage forms and binding

Every stage has `id`, `kind`, `scope` and optional `requires` (default `[]`).

| Kind | Scope | Explicit inputs/settings |
|---|---|---|
| `artifact_reference` | Derived from artifact kind | External ExperimentPlan `artifact`, including full SHA, identifier, version, producer and relative path; `from_phase: null` |
| `corpus_release` | `corpus` | `project`; only local/deterministic-generator sources accepted, never network acquisition |
| `corpus_readiness` | `corpus` | `corpus`, `policy`, optional `tokenizer`; token policies require a tokenizer dependency |
| `tokenizer_reference` | `tokenizer` | External `artifact` of kind `tokenizer`; provenance manifest required |
| `token_measurement` | `tokenizer` | Verified `corpus` release and `tokenizer`; actual selected-view/split counts |
| `experiment_plan` | `model` | `source`, `mode: lock|reference`, `tokenizer`, `prepared`, optional `corpus`; reference mode requires `lock` |
| `runtime_acceptance` | `runtime` | `plan`; CPU/PyTorch runtime and actual storage-headroom checks |
| `experiment_run` | `model` | `plan`, matching `runtime`, exact `cell` |
| `experiment_collect` | `evaluation` | `plan`, matching `run`; selected cell must have complete ingested evidence |
| `evaluation` | `evaluation` | `collect`; existing checkpoint-bound heldout evidence only |
| `model_readiness` | `model` | `evaluation`, optional finite nonnegative `max_heldout_loss` |
| `approval` | `release` or `model` | Nonempty `bind` of declared ancestors |

External artifact scopes: source snapshot/build/release/export → corpus;
tokenizer → tokenizer; prepared data/stage bundle/checkpoint → model;
capability card/prompt set → evaluation. Artifacts lacking a domain verifier,
notably `outcome`, are not accepted. Directory presence never proves identity.
Missing local inputs block with their path; existing corrupt/mismatched inputs fail.

Lock mode reuses existing preparation, resolution and immutable lock publication
APIs. Reference mode reopens the supplied lock. Selected cells must bind exactly
the upstream tokenizer and prepared-data artifact digests. Non-synthetic cells
also require an upstream corpus release whose identity matches the lock's selected
variant; Campaign never silently substitutes a variant. Synthetic cells must omit
`corpus`: a separate corpus/readiness stage can be a research-workflow prerequisite,
but is not thereby the model's training data. Every plan consumer reopens the lock.
An explicit run cell selects exactly one locked cell; there is no automatic
multi-cell dispatch. Local workers use the existing lock-bound request,
controller receipts, checkpoint verification and ingestion machinery.

`tests/test_campaign.py` generates its own full fixture under `tmp_path`, adding
real synthetic tokenizer training (seed 7, vocabulary 260), prepared data, an
ExperimentPlan, CPU worker execution (two steps / 32 supervised targets), evidence
collection and model readiness. These synthetic model inputs are **independent of
its Corpus Forge release**. `max_heldout_loss: 100.0` is fixture-only wiring
acceptance, not a generally useful model-quality threshold. No mock controller
completion or checked-in tokenizer/cache/run artifact is used.

## Corpus policy facts and projected passes

The policy can declare `min_unique_train_bytes_by_domain`,
`min_unique_train_tokens_by_domain`, `required_nonzero_languages`,
`required_nonzero_shapes`, `min_heldout_families`, and/or `passes`. At least one
requirement is necessary; minima are nonnegative integers.

Measurements first verify the frozen release. For each domain, the denominator
is normalized source-document UTF-8 text from `documents.jsonl`, restricted to
`split: train` and `drop_reason: null`, deduplicated by `content_sha256`
independently within that domain. It is **not** raw snapshot bytes or the build
report's overlapping/repeated mixture byte total. Token minima encode those exact
same kept unique documents with the pinned tokenizer and count `.encode(text).ids`;
tokens are never inferred from bytes. Languages use these documents; shapes use
selected views that explicitly include `train` in `training_splits`; heldout
coverage counts distinct nonempty source-family IDs in kept test lineage.
An absent measurement produces a typed deficit, not a pass.

```yaml
passes:
  basis: bytes  # or tokens, requiring the pinned tokenizer
  requested_total: 1000000
  mixture: {technical_docs: 0.5, developer: 0.5}
  max_required: 4
```

For each requested mixture domain `d`:

```text
required_passes[d] = ceil(requested_total * mixture[d] / unique_available[d])
```

Authored numeric weights retain exact decimal precision; numeric strings are
rejected. Integer rational ceilings avoid rounding across exact boundaries.
Weights must be positive, finite and sum to one; total and maximum are positive
integers. Zero or unavailable denominator blocks. The maximum per-domain estimate
must not exceed `max_required`. This mixture is policy input, never a rewrite of
Corpus Forge's release mixture. Overlap across domains means these are per-domain
upper-bound reuse estimates, **not train-order or actual ingestion passes**.

Satisfied policies return `COMPLETE / READY_FOR_TOKENIZER` with no deficits.
Unsatisfied policies return `BLOCKED / EXPAND_MORE`, sorted deficits with
`dimension`, `observed`, `required` and optional `domain`. Descendants remain
blocked. The engine never adjusts sources, mixture or thresholds automatically.

## Durable state, authorization and replay

State vocabulary is exactly `NOT_STARTED`, `READY`, `RUNNING`, `COMPLETE`,
`BLOCKED`, `AWAITING_APPROVAL`, `INTERRUPTED`, `FAILED`, `INCONCLUSIVE`.
Result outcomes are `EXPAND_MORE`, `READY_FOR_TOKENIZER`, `READY_FOR_NEXT_STAGE`,
`DO_NOT_ADVANCE`, `INCONCLUSIVE`. Next actions are typed objects with reasons and
identities. A prerequisite that cannot advance blocks descendants.

A domain-separated canonical declaration SHA creates its own namespace:

```text
<work-dir>/campaigns/<id>/<declaration-sha>/state.json
```

Stage-input hashes bind the declaration, stage and sorted immutable dependency
results, including transitive input hashes. Machine-local availability is separate
from scientific identities; absolute paths do not enter Campaign declaration,
stage-input or approval-binding digests. A new declaration with changed policy or
artifact SHA gets a new state/approval namespace, even with the same campaign ID.
Old approvals never authorize it.

`RUNNING` is durable before side effects. Immutable canonical stage receipts at
`receipts/<stage>/<input-sha>.json` publish before the atomic fsynced state index.
Resume reconciles receipts after a crash between receipt and index publication;
pre-receipt content-addressed lower-level outputs can be verified and reused.
Interrupted attempts remain visible. A per-campaign file lock serializes mutation.
Noncanonical/tampered records, conflicting output for one input SHA, missing or
changed completed artifacts fail closed: the engine never silently reruns them.
Repeated completed apply verifies without rebuilding, dispatching, training or
rewriting state/receipts.

`apply` initializes or reuses state; `resume` requires existing state. Both accept
`--max-wait-seconds` (default 120). Timeout leaves `RUNNING` with a `resume` next
action. The foreground budget bounds sequential protocol waits, not local hashing
or verification work. Runtime headroom is checked on the campaign's actual
execution workspace. Before reuse and after completion, the controller attempt
must match the full lock-bound config, phase, coordinate, science digest and
continuation binding, not just lock/cell labels. Multiple/conflicting attempts fail.
There is no automatic optimizer retry after a terminal worker interruption:
the stage remains `INTERRUPTED` with a `wait` action.

Approval waits for complete bound ancestors and binds their immutable outputs,
transitive stage inputs and the gate's own declaration/policy identity. `approve`
defaults to `--decision approve`; `--decision reject` blocks with `DO_NOT_ADVANCE`.
Receipts at `approvals/<gate>/<binding-sha>.json` are immutable. Identical decision
replay retains the original UTC timestamp and note; a conflicting decision needs
a new declaration, not an overwrite. Approval authorizes **only those inputs**;
it is not a quality finding or model promotion.
Resume also reconciles an approval receipt published before its gate-stage receipt,
without changing its original decision, timestamp or note.

Model readiness accepts only `checkpointed_held_out` evidence from the existing
verified evaluation API. It selects the greatest step, rejects conflicting latest
observations, requires a finite loss, and compares `loss <= max_heldout_loss`.
An exceeded threshold blocks. No threshold, `partial_held_out` or `artifact_only`
evidence is `INCONCLUSIVE`, never approval. Deterministic software acceptance
and declared-policy outcomes do not replace human research judgment.

## Explicit v1 limits

No tokenizer bakeoff, architecture selection, auto-generated evaluation protocol,
accelerator execution or worker-farm scheduling stage is provided. V1 uses pinned
tokenizer references and verified local CPU/PyTorch plan/worker adapters. Capacity
checks are local execution readiness, not evidence of fit on other hardware,
throughput superiority, useful model behavior, causality or portability.
