# Probe declarations

The common journeys need no custom file:

```sh
sparselab probe RESULT                         # fast checks, saved device
sparselab probe RESULT --vs BASE --backend cpu --config standard
sparselab probe RESULT --config full            # optional lm-eval extra
sparselab probe RESULT --vs BASE --backend cpu --config final
```

`--config` selects a validated named declaration (`fast`, `standard`, `full`,
`final`, `final-full`) or a YAML file. Default is `fast`. `final` selects the
standard tier on the untouched final split; `final-full` selects full final
scoring. Choose the candidate first: never use final declarations to select or
rank ideas, and never change the idea before repeating a missing-evidence final
request. `compare --final` remains the explicit read-only final report.

For reusable custom policy, save `probe.yaml`:

```yaml
probe_config_version: 1
tier: standard
fast_fail: false
# Optional lookup roots and native operational declarations:
# runs_dir: ../runs
# lab_dir: ../lab
# runtime_profile: ../profiles/my-runtime.yaml
# resource_envelope: ../bounds.yaml
```

Then run `sparselab probe RESULT --vs BASE --config probe.yaml --json`.
`--vs` always names the baseline explicitly; a record selects its candidate,
never its baseline arm. Use an explicit checkpoint path to pin a baseline.
Paths in a declaration expand `~` and anchor to that file. Bare preset names
are reserved; use `./full` to load a file actually named `full`.

All fields are optional. Empty/comment-only files, YAML `null` and `{}` use the
same defaults as `fast`. The version must be integer 1, booleans must be actual
booleans, and unknown fields and nonmapping documents fail before runtime
selection, model loading or output creation. `final: true` requires standard or
full. `fast_fail: false` keeps running after a hard failure without converting
failures or missing evidence into passes.

Full-tier benchmark overrides are `lm_eval_tasks: [piqa, winogrande]` and
`lm_eval_limit: 200`. Tasks must be distinct nonempty names and the limit a
positive integer. These fields require `tier: full`; they change the existing
benchmark group, never the suite digest. Only matching groups are comparable.

Five executable options remain: `--vs`, `--config`, `--backend`, `--runtime`
and `--json`. `--backend cpu` explicitly overrides a saved PyTorch accelerator
backend; installing CPU dependencies alone never does. `--runtime ID` uses a
registered native runtime. Direct profile files use declaration
`runtime_profile` and are mutually exclusive with `--runtime`. Their interpreter,
engine, backend and device-index authorization remains native. A declaration
references the existing ResourceEnvelope; it defines no new resource schema.
No unavailable device is silently substituted, and MLX cannot become PyTorch
through an override.

Migration: `--tier`, `--final`, `--no-fast-fail`, `--lm-eval-tasks` and
`--lm-eval-limit` become `tier`, `final`, `fast_fail: false`, `lm_eval_tasks`
and `lm_eval_limit`; comma-separated tasks become a YAML list. `--lab-dir`,
`--runs-dir`, `--runtime-profile` and `--resource-envelope` become file fields.
These executable flags are removed. Named declarations replace the common
`--tier standard|full` and `--final` journeys without an extra setup step.

The sealed CLI record includes the validated effective `request`, baseline,
explicit backend selection, loaded native profile and envelope values. Native
checkpoint/digest binding, scoring, suite identity, resource checks, partial
records and safe cancellation are preserved. Historical records remain readable
without adding or rewriting fields. `report` returns the whole sealed record.

This is a bounded continuation of option reduction. Counts excluding help are
try 13, probe 5, explore 5, compare 5, report 2. Try's paired-seed and probe policy,
per-arm wall stops and preparation batching remain the next migration scope.
