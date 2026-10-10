# Lab mode: from a YAML delta to a scored comparison

Lab mode is the default path for normal, authorized local experiments. One
command trains a baseline and a candidate, scores both on the same held-out
split and writes one compact record:

```sh
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
cat > wider-ffn.yaml <<'YAML'
question: Does a wider FFN lower held-out loss at a fixed token budget?
set:
  model.ffn_dim: 128
YAML
uv run --locked --extra cpu sparselab try wider-ffn.yaml --vs BASELINE.yaml
uv run --locked --extra cpu sparselab report TRY_ID      # reread later, no compute
```

`BASELINE.yaml` is any RunConfig whose tokenizer already exists. The candidate is
either a **lab delta** (`set:` with dotted fields, optional `question:`) or a full
RunConfig. Output looks like:

```text
LAB TRY try-20261010T052417Z-b9361b7d  status: completed  (lab mode)
  question: Does a wider FFN lower held-out loss at a fixed token budget?
  delta model.ffn_dim: 96 -> 128
  baseline  lab-base-759596ad32b5bf50  held-out loss 3.8764  ppl 48.25  targets 512
  candidate lab-try-20261010T052417Z-b9361b7d  held-out loss 3.8451  ppl 46.76  targets 512
  verdict: CANDIDATE_LOWER_LOSS  Δ held-out loss -0.0313
```

Use `--json` for the full record. A baseline is reused only when an earlier try
*completed* with the same reuse key: effective config, code (source identity)
and the tokenizer, dataset input files and external memory packages
(`model.memory_package_path`, `training.portability_manifest_path`) hashed
**now**, plus a matching prepared tokenizer digest. Editing a training file,
validation file, tokenizer or memory package therefore retrains the baseline;
interrupted or failed tries are never reused. A delta that points at a different
memory package is an explicit changed variable (`memory_packages` in the
record); otherwise both arms must have trained with identical package contents.
Pass `--fresh-baseline` to retrain anyway.

`--seed N` is authoritative: both arms use seed N, overriding any seed in the
baseline or the delta. Without it, a delta that sets `seed` is an explicit
changed variable. The record keeps `seed` (`override`, `baseline`,
`candidate`, `changed_by_delta`) and `changed_variables`.

## What lab mode skips

ExperimentPlan locks, Campaign approvals and reconciliation, corpus admission
reviews, attempt ledgers and proposal/binding/stop documents. Those remain the
release path ([training programs](experiment-programs.md),
[campaigns](campaigns.md)) for results that will be promoted, published or used
as release parents, and for paid compute.

## What lab mode keeps

| Rail | How |
|---|---|
| Resource limits | Native storage preflight before training; optional `--resource-envelope` (checked before any work); optional `--max-wall-seconds` per arm. There is no implicit short timeout. |
| Data identity | Each arm records tokenizer, train and validation digests and the prepared-data manifest digest; the record also carries the source identity, Git commit/dirty state, seed and both configs (the candidate is derived with a `config derive` receipt). |
| Safe cancellation | Covers the whole try: setup and input hashing, training, scoring and the record write. Before any parsing or hashing, `try` installs its SIGINT/SIGTERM/CANCEL handling and writes a sealed `started` record; a stop during setup finalizes it as `interrupted` at `phase: setup`. Ctrl-C, SIGTERM or `touch <try>/CANCEL` stops training at a checkpointed step boundary; outside training (loading, scoring) `try` handles SIGINT/SIGTERM itself and checks `CANCEL` between phases. The record is published atomically (temp file, fsync, exclusive link) as `interrupted` with `arm`, `phase` (`setup`, `training` or `scoring`) and `reason`; interrupted tries are kept and never reused. |
| Held-out checks | Both arms are scored by native checkpoint-bound evaluation under **one shared evaluation protocol**: the baseline's eval `seq_len` (context windows), batch size and batch count over the same tokenized validation bytes. Each arm records `eval_protocol` (split, validation and tokenizer digests, packing version, objective mode and validation loss-mask digest, windowing, batching, scored blocks, evaluator) and its sha256, so assistant-only and whole-transcript scoring on identical tokens are never compared. The comparison requires the same protocol, validation bytes, tokenizer, scored targets and training data (unless the delta changes `dataset.*`/`tokenizer.*`), and a validation split distinct from train; otherwise the verdict is `NOT_COMPARABLE` (exit 3) and no delta is reported. A candidate whose model cannot take the baseline's eval window (e.g. smaller `model.max_seq_len`) is `NOT_COMPARABLE`. |

Records live in `WORK_DIR/lab/tries/<try_id>/try.json` and runs in
`WORK_DIR/lab/runs` (override with `--lab-dir`). A record carries its own
`record_sha256`; `report` rejects edited records. Exit status is 0 for a
completed comparison, 3 for `NOT_COMPARABLE`, 130 for an interrupted try and
1 for a failure (the record keeps the error).

## Limits

One seed per arm and one held-out loss: a lab verdict is a direction to follow
up, not a significance test or a promotion. Training runs in the `try` process
on one device; use a runtime profile (`--runtime-profile`/`--runtime`) for
accelerators exactly as with `train`. Remote and multi-seed sweeps still go
through workers or Campaigns. The `lab-loop` CI job runs
`tests/test_lab_mode.py::test_cpu_smoke_loop_yaml_to_report_within_budget`, which
times this path on CPU (tokenizer → try → report) against a generous budget
(`SPARSELAB_LAB_LOOP_BUDGET_SECONDS`, 900 s).
