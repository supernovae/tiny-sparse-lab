# Validate an iteration with the lab

Use SparseLab's declared inputs, native verifiers and retained evidence between
runs. A passing check establishes its stated scope; it does not approve model
quality or promote a child. Keep the baseline and parent generation immutable.
Record the intended scientific delta before compute, then compare the actual
locked delta rather than reconstructing it in a Python notebook or agent script.

## Start a runner session

Use one external persistent work root and a fixed, tested source revision.
Examples below use shell variables `WORK`, `CONFIG`, `CAMPAIGN`, `GENERATION`
and `RUN_ID` for your actual paths and identifiers; they are not literal inputs.
Set `SPARSELAB_WORK_DIR` to `WORK`, independently of checkout and branch.

Bash setup:

```sh
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
WORK="$SPARSELAB_WORK_DIR"
```

PowerShell setup:

```powershell
$env:SPARSELAB_WORK_DIR = Join-Path $env:USERPROFILE '.local/share/sparselab'
$WORK = $env:SPARSELAB_WORK_DIR
uv run --locked --extra cpu sparselab readiness smoke --family dense --output "$WORK/experiments/session-check-001"
```

In PowerShell use `$env:SPARSELAB_WORK_DIR` for the environment variable and
`$CONFIG`, `$CAMPAIGN`, `$GENERATION`, `$RUN_ID` for shell variables. Write
continued Bash commands on one line instead of using `\`. For the TinyStories
copy step, set `$WORK = Join-Path $env:SPARSELAB_WORK_DIR 'experiments/tinystories-microlab'`,
create a fresh `$WORK/inputs` directory with `New-Item -ItemType Directory`,
and copy the four sample YAML files with `Copy-Item` into it. Use forward slashes
or quoted absolute paths in YAML. Do not edit the checked-in templates.

For a dense-model session, run the bounded CPU wiring check once into a new,
task-owned output directory:

```sh
uv run --locked --extra cpu sparselab readiness smoke --family dense \
  --output "$WORK/experiments/session-check-001"
```

This check includes tiny training and child resume; it creates disposable runs
and evidence. Its `readiness.json` records commands, outcomes and identities.
It has no `--json` flag. Select other affected families when appropriate; CPU
success does not establish accelerator acceptance. See [lab readiness](lab-readiness.md).
Do not repeat the entire test suite for each budget or config iteration on the
same tested revision. Code changes need nearest behavior tests and the relevant
broader gate; PRs still require the full CPU suite in [test-speed.md](test-speed.md).

## Check the actual proposed configuration

```sh
uv run --locked --extra cpu sparselab inspect "$CONFIG"
uv run --locked --extra cpu sparselab workspace preflight "$CONFIG"
uv run --locked --extra cpu sparselab stage "$CONFIG" --through smoke \
  --output "$WORK/experiments/candidate-001-stage"
```

For agents, `inspect "$CONFIG" --json` gives machine-readable output.
`workspace preflight` and `stage` have no `--json` flag. Inspect estimates shape
and memory; preflight checks storage assumptions; staging verifies inputs and
executes a disposable pilot without advancing the eventual training state.
Use `--through warmup` for a measured full-shape runtime gate when required.
Provide the declared runtime profile using `--runtime-profile PROFILE` where
appropriate. Use a new stage output for changed intent.

On a provisioned vendor environment, use the documented
`uv run --locked --no-sync sparselab ...` commands instead of synchronizing the
CPU extra into that environment. Follow [runtime acceptance](runtime.md).
Neither a memory estimate nor the session smoke proves that the actual model
fits or that its selected backend is stable.

## Let Campaign explain the next action

```sh
uv run --locked --extra cpu sparselab campaign status "$CAMPAIGN"
uv run --locked --extra cpu sparselab campaign next "$CAMPAIGN"
uv run --locked --extra cpu sparselab campaign explain "$CAMPAIGN"
uv run --locked --extra cpu sparselab campaign apply "$CAMPAIGN"
# Once state exists, reconcile or continue it:
uv run --locked --extra cpu sparselab campaign resume "$CAMPAIGN"
```

Every Campaign verb above accepts `--json`: agents should consume the native
states, reasons, deficits, identities and typed next actions instead of scraping
logs or recomputing them. Text mode exposes the same decision context to humans.
`status` reports last-committed outcomes plus fresh recoverability; it is not a
fresh rehash of every historical artifact. `next` and `explain` verify requisite
existing inputs without building or training.

`apply` and `resume` mutate orchestration state and can perform declared
deterministic stages, pilots, evaluation and generation. Without `--execute-runs`
they do not submit a new optimizer run; existing submissions can be reconciled.
Explicit submission still requires committed declarations, bound approval,
verified inputs and accepted runtime. Submission is an authorization boundary,
not proof that remote compute has completed. Read [Campaign](campaigns.md) for
worker/runtime binding and approval commands. Never infer permission to retry an
interrupted optimizer attempt from a missing output.

## Verify the parent and read outcomes

Use the finalized generation directory, not a moving `latest.json` or `best.json`
pointer, as the parent/evaluation binding:

```sh
uv run --locked --extra cpu sparselab checkpoint verify "$GENERATION" --json
uv run --locked --extra cpu sparselab evidence "$RUN_ID" --runs-dir "$WORK/runs" --json
uv run --locked --extra cpu sparselab triage "$RUN_ID" --runs-dir "$WORK/runs" --json
```

These commands also have human-readable text mode: omit `--json`.
Full checkpoint verification checks supported native continuation state; do not
use `--weights-only` to claim optimizer-resume readiness. Evidence authenticates
retained checkpoint-bound heldout observations, with its own weights-only scope.
Triage reads the immutable advisory report without rerunning inference; missing
triage is `UNKNOWN`. Keep negative and interrupted results. Campaign collection,
evaluation, generation panels and model readiness provide declared result closure;
none automatically promotes a model. See [checkpointing](checkpointing.md),
[evidence](evidence.md) and [triage](research/post-train-triage.md).

## Repeat checks according to what changed

| Change | Repeat before the next run |
|---|---|
| Only inspecting completed results | Native status/evidence/triage reads; do not train again to reconstruct evidence. |
| Training budget or other scientific config | Inspect the effective config and declared comparison; resolve a new immutable declaration/lock; check parent compatibility, storage, and candidate stage/runtime gates. Preserve unchanged identities. |
| Corpus, tokenizer or prepared bytes | Authenticate the changed artifact and affected dependency closure; prepare/resolve affected outputs and rerun config/runtime gates. Never relabel changed bytes as the baseline. |
| Lab implementation | Nearest contract tests, relevant broader checks and affected readiness families; refresh source-bound locks/compatibility and runtime acceptance as required. |
| Backend, environment, device or worker | Fresh acceptance on the actual runtime, full-shape pilot and storage check; CPU smoke cannot stand in for it. |
| New host/store, transfer, relocation or final archive | Cold authentication at the trust boundary; then only supported authenticated reuse. Check availability separately from historical identity. |

Verified reuse is already supported at selected native boundaries, not as a
universal skip-validation switch. Eligible unchanged artifacts in an owner-only
registered persistent store use signed host-local proofs; changed fingerprints,
closure/source identity, unsafe paths or missing/foreign proofs fall back to cold
verification. `stage --cold-verify` explicitly disables proof reuse; supported
ExperimentPlan commands also expose `--cold-verify`. Independent archive,
recovery and Family verification remain cold. There is no Campaign-wide
`--cold-verify` option. See [capacity-aware execution](capacity-aware-execution.md)
for exact support and trust limitations.

A consolidated typed iteration-check command is **not shipped yet**. The native
commands above are the current path; proposed consolidation and a complete
declarative TinyStories acceptance demo are tracked under
[rapid iteration](../TODO.md#rapid-iteration). The [direct TinyStories walkthrough](tinystories-microlab.md)
already teaches baseline, exposure child and optional fresh width comparison
without Python scripting.
