# Explorer declarations

`explore` performs descriptive inspection, with five options: `--config`,
`--runtime`, `--runtime-profile`, `--resource-envelope` and `--json`.
The runtime selectors are mutually exclusive. Without either selector it uses
CPU, including for compatible saved PyTorch accelerator runs. An explicit
profile selects its backend through native runtime authorization; its engine
and device index must match the saved run. There is no device fallback.

```sh
sparselab explore TRY_ID
sparselab explore TRY_ID --config explore.yaml --json
sparselab explore TRY_ID --runtime MY_RUNTIME --resource-envelope bounds.yaml
```

Save a reusable descriptive request as `explore.yaml`:

```yaml
explore_config_version: 1
text: Once upon a time, a little dog found a ball.
refresh: true
# Optional lookup roots, relative to this file:
# lab_dir: ../lab
# runs_dir: ../runs
```

The declaration accepts only these fields. Version defaults to 1, text defaults
to the existing short story, refresh defaults to false, and roots default to
native work-directory lookup. Unknown fields, unsupported versions, non-string
or empty text and non-boolean refresh values fail before runtime preparation or
model loading. Text remains subject to the explorer's existing 128-token limit;
checkpoints over 60 million parameters are refused before weight loading.

Migration: `--text`, `--refresh`, `--lab-dir` and `--runs-dir` become `text`,
`refresh`, `lab_dir` and `runs_dir` fields. These flags are removed, including
from executable parsing. `--backend` is removed: CPU needs no option and other
devices use an existing [runtime profile](runtime.md) or registered runtime.
Resource bounds remain in the existing ResourceEnvelope, not in the descriptive
request. No new runtime or resource schema is introduced.

Record inputs still pin the recorded checkpoint generation and digest. Refresh
recomputes diagnostics over that same checkpoint and declared text. The sealed
cache remains keyed by checkpoint digest and text; configuration version and
lookup locations do not change scientific identities. The native loader,
cancellation sentinel, per-stage resource checks and OOM cleanup are unchanged.

This is a scoped part of option reduction. `try` (13 options) and `probe`
(13 options) still need validated declaration migrations; `compare` (5) and
`report` (2) already meet the beginner-option bound. Full sealed reporting and
read-only comparison boundaries remain unchanged.
