# Finding a lab result

Keep the record ID printed by `try` or `probe`. It works across the everyday
result commands:

```sh
sparselab report TRY_ID
sparselab compare TRY_ID --references
sparselab probe TRY_ID --vs BASELINE_RUN --backend cpu --tier standard
sparselab explore TRY_ID
```

`report` reads the whole sealed record. `compare` reads its candidate's
measurements, preserving comparison groups and the explicit `--final` boundary.
Neither command loads a model. `probe` performs new scoring; `explore` performs
descriptive inspection. Both select the candidate checkpoint recorded by a try,
or the target checkpoint recorded by a probe. Supplying a record does not
implicitly select its baseline, tier, final split, backend or resource envelope.
`--vs OTHER_RECORD` selects the **other record's candidate**.

The headline probe example pins only the candidate. `--vs BASELINE_RUN` loads
the baseline run's current `latest`, which may have advanced since the try.
To use the same baseline the try scored, copy its `arms.baseline.run_dir` and
`arms.baseline.checkpoint` from `report TRY_ID --json` and combine them:

```sh
sparselab probe TRY_ID --vs /path/to/BASELINE_RUN/checkpoints/GENERATION --backend cpu --tier standard
```

This pins both sides; there is no record selector for a try's baseline arm.

## IDs, paths and missing evidence

All four commands use the same lookup:

- A record ID is looked up under `LAB/tries` and `LAB/probes`, where `LAB`
  defaults to `WORK_DIR/lab`. `--lab-dir` selects another lab for `try`, `probe`, `compare` and `report`;
  `explore` uses the `lab_dir` field of its [declaration](explore-config.md).
- Bare IDs also match existing entries in the current working directory,
  preserving the previous local-path lookup. If that is the only match, it is
  used; if a lab location also matches, lookup reports `ambiguous_result`.
  This includes unrelated same-named local directories. Use an absolute path
  to choose a location unambiguously; do not assume a bare ID is lab-only.
- A record file or directory can be passed explicitly, including an exported
  sealed JSON file with a different filename. Its digest is verified.
- `probe` and `explore` also accept run IDs under `LAB/runs` or an extra runs root (`probe --runs-dir`;
  `explore` declaration `runs_dir`),
  run directories and checkpoint paths. A direct run still uses the native
  checkpoint loader's default (`latest`); an explicit checkpoint uses native
  checkpoint verification.
- A record selects its **exact recorded generation and digest**, even if the
  run has advanced. The native loader verifies it before model loading. If
  its saved location is gone, restore the run there or explicitly supply the
  relocated checkpoint path. No fallback to another run or `latest` occurs.
- If an ID matches multiple locations, pass an explicit path. A directory
  containing both record filenames requires an explicit file.

A run or checkpoint does not name a unique scientific record: it can have
several tiers, protocols and final verdicts. `report` and `compare` therefore
require a record ID/path, with an actionable error for a run/checkpoint input.
They do not guess the newest record or start scoring. Records remain readable
without the original checkpoint, including interrupted/resource-stopped records.
A record without a scored candidate cannot be used to load a checkpoint.

Lookup failures exit with status 1. With `--json`, the same text diagnostic is
returned as `status`, `result`, `reason` and `hint`. Status is `missing_evidence`,
`ambiguous_result` or `invalid_evidence`; successful record JSON and stored hashes
are unchanged. Probe cancellation, resource and OOM stops retain their existing
partial records and incomplete verdicts.

## Explicit devices

Standalone `probe` inherits each run's **saved backend** when `--backend` is
omitted. Installing `--extra cpu` only chooses dependencies; it cannot convert
a saved CUDA/ROCm run into CPU inference. A saved accelerator without runtime
authorization fails before loading with `runtime_required` and instructions
to select a compatible `--runtime`/`--runtime-profile` or explicitly choose
`--backend cpu` for a compatible PyTorch run. No unavailable device is silently
replaced. `explore` retains its documented CPU default; select an existing runtime
profile with `--runtime` or `--runtime-profile` to choose another backend. Backend overrides do not
rewrite saved configs or checkpoint identities. MLX cannot be converted into
PyTorch inference by a backend override.

## Scope of the everyday-command audit

This change consolidates lookup and diagnostics. It does **not** complete the
option-reduction work: `try` and `probe` still exceed five relevant
options in beginner help. Follow-up work must move scientific controls into
validated declarations and runtime controls into the existing profiles/envelopes,
with caller and documentation migration, rather than hiding executable flags.
`explore` now has five options and a validated descriptive declaration; see
[explorer declarations](explore-config.md).

| Command | Distinct semantics to retain in that follow-up |
| --- | --- |
| `try` | Derive a candidate, paired seed override, baseline reuse, held-out scoring, optional probe tier, checkpointed wall/resource stops. |
| `probe` | Explicit baseline, tier and final split; early-stop policy; benchmark tasks/limit change benchmark identity. |
| `explore` | Descriptive text diagnostics, refreshable cache, CPU default, resource stops. |
| `compare` | Read-only recorded metrics, comparison groups, references and explicit final-only reports. |
| `report` | Whole sealed record, including failed/interrupted/missing observations. |
