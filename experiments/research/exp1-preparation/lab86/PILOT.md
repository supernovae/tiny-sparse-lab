# Executable-through-pilot boundary

Use `pilot-session.template.json`, `pilot-contract.template.json`, `pilot-content.json`
and the four `pilot-jobs/` templates. All scientific/source/byte-limit decisions are
concrete. `HOST_*` and `PACKAGE_ROOT` substitutions are pre-launch host facts, not
future dataset receipts. The chief owns binding and launch; none occurred here.

1. Use the qualified Linux amd64 Docker path and existing lease/hold namespace.
   Bind the reviewed Lab commit containing timestamp serialization, a smoke-tested
   immutable image containing that commit, and a reviewed Harness pin allowing
   that digest. The previous cached image cannot execute the corrected path.
   Do not work around the image allowlist, overlay installed code, or install packages.
2. Bind actual runtime/version/executable SHA (and optional venv object consistently
   in session and jobs), boot ID, owner approval reference, source roots and fresh
   work/receipt roots. Measure actual capacity under the existing profile: 24 GiB
   effective available RAM, 310 GiB disk and 100k inodes. These are preflight facts.
3. Capture the native workspace baseline before creating attempt receipts:
   exact argv `python -m sparselab monitor-baseline WORK --output WORK/baseline.json --json`.
   Render the pilot contract with that baseline SHA. Its content identity is already
   the SHA256 of `pilot-content.json`; never replace it with an arbitrary digest.
   Hash the rendered contract into the init argv. `HOST_REQUIRED_FILES_OBJECT` is
   the absolute-path/SHA256 map of actual inputs: package-sha256.json entries,
   `../preparation.json`, rendered contract, baseline, and reviewed runtime bindings.
   Keep caches, temp files and native outputs in the declared work root.
4. Render session and four job templates as data, commit them in the work-root Git
   repository, and verify no `${...}` remains. No shell expansion is used. All argv
   are fixed now; no model config, release ID, tokenizer or dataset receipt is needed.
   Preserve the single outer 48-hour origin, whole workspace baseline, 16 GiB RSS
   observation and 300 GiB retained ceiling. Acquire has a four-hour allowance;
   this is a stricter phase timeout, not a promise that the source fits or finishes.
5. Launch the reviewed docker_session.py once and submit slots in this order:
   `attempt-init`, `transport-init`, `pilot-acquire`, `pilot-budget-status`.
   Submit each `{"slot":"LABEL","job":"ABSOLUTE_COMMITTED_JOB_PATH"}` only after
   inspecting the preceding receipt. The acquisition uses the full immutable project
   and the original cumulative ledger, with the three pilot declarations sorted first.
6. Expect the fourth whole-shard reservation to fail at the 4 GiB gate. Preserve its
   real nonzero exit and prior snapshots. Only after complete evidence and verified
   cleanup, acknowledge that exact failed job using the Harness protocol and retain
   the native gate receipt as evidence. Then run budget status. A different failure,
   missing pilot snapshot, timeout, interruption or resource stop is not pilot success.
   End the initial session with `{"finish":true}`; do not send a continuation job or
   release the gate. This pilot session deliberately contains no continuation slot.

Initial blockers are the reviewed software/image/Harness release and measured host
bindings/qualification. No downstream model config or yet-unproduced receipt blocks
these four slots. Initial rights remain quarantine-only. Missing CodeParrot language,
Cosmopedia family granularity, admission decisions, projected yield and disk feasibility
are findings for the pilot review. No license acceptance or language/family assumption
is silently supplied. Continuation requires its reviewed boundary and the same native
transport ledger; it must not reset the original 48-hour or disk accounting. If ending
this initial session prevents a supported continuation within those constraints,
report that remaining orchestration gap before continuing, not before acquiring pilot
evidence. The pilot is useful even if subsequent continuation is blocked.

## Materialize host object bindings as JSON data

`HOST_RUNTIME_OBJECT` and `HOST_REQUIRED_FILES_OBJECT` bind to absolute paths of
pre-launch JSON files containing the measured runtime object and exact required-file
mapping. The later skeleton uses `GENERATED_REQUIRED_FILES_OBJECT` for the same
operation after its inputs exist. These are not native `_JSON` array fragments.
The native renderer fills string slots only; do not submit that intermediate file
to Harness. Materialize the two object fields with Python's existing JSON library,
then commit the final job. For example, after native template rendering:

```python
import json
from pathlib import Path

job = json.loads(rendered_job.read_text())
for field in ("runtime", "required_files"):
    value = json.loads(Path(job[field]).read_text())
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} requires a nonempty JSON object")
    job[field] = value
with final_job.open("x") as stream:
    json.dump(job, stream, indent=2)
    stream.write("\n")
```

`rendered_job` and `final_job` are distinct absolute Paths inside the work root.
Populate the runtime object's native venv block when the actual interpreter uses
one, and copy that same block to `session.runtime_venv` before final serialization.
The measured runtime JSON and input map are host facts, not future dataset evidence.
No subprocess, shell, evaluation, automatic approval or workload launch is involved.
