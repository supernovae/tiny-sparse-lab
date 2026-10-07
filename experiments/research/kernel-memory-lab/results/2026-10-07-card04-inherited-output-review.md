# KML-20261007-C04-S4 — bound inherited output and orphaned sampler children

- 2026-10-07; **READY FOR REVIEW for code only**. Base is rolling commit
  `705a2d8323f1fcbf7cb1af34c8ced1ce1fd8fcb1`, which includes PRs #51 and #52.
  Work is isolated on `codex/kernel-memory-lab-bounded-output`; no rolling-branch
  writes, merges, GPU work, acquisition, training or allocation are authorized.
- Trigger: the owner reported local failures in the `preflight_hang` and
  `midrun_hang` cases despite earlier cloud success. Those observations remain
  failures; cloud tests cannot overrule them. Their exact WSL process trace has
  not been independently inspected here.
- Reproduced mechanism on the exact base: a measurement leader exits while a
  detached, TERM-ignoring child retains captured stdout. GNU timeout no longer
  supervises that exited leader, while Bash command substitution still waits
  for pipe EOF. The mock exceeded the 12-second harness deadline; the fixture's
  cleanup killed its group and separately attributed detached child. This is a
  reproduced failure mode, not a claim to have inspected the live AMD library.
- Fix: a dedicated Linux child subreaper invokes each measurement with regular
  output files and null stdin. It reuses native process-identity discovery and
  signalling, adopts orphaned descendants, requires leader and children to finish,
  and never calls `communicate` or drains a subprocess pipe. It polls leader exit
  before the ownership snapshot to avoid missing newly adopted children. Failure
  kills attributed children and uses only nonblocking reaping, with bounded cleanup.
  The wrapper parses bounded completed files and exports canonical paths to the
  helper. Resource caps, stage/train reservations and scientific inputs are unchanged.
- Bounds: nominal five-second measurement deadline, 64 KiB output ceiling,
  half-second cleanup, and an outer six-second supervisor/startup guard with
  TERM-to-KILL escalation after one further second. The outer attempt budget
  still enforces its original shared deadline. Scheduling delays, uninterruptible
  kernel operations, and SIGKILL of the whole supervisor are not hard-real-time
  guarantees; this is not a security boundary against hostile children.
- Corrected launcher SHA-256:
  `38dcadc4212d32cccd85911035bce673523905dba706183a9e4c7eb2b8a13ce5`.
  Measurement helper SHA-256:
  `9a3265c1bfafe8a710c4f634c677db89da39b6f05466216d868322fc511ae81b`.
  Reader and launch-input validator are unchanged. Retry2 binds these tracked
  versions; S2/S3 hashes and historical experiment records remain unchanged.
- Regression scope: retained `preflight_hang`/`midrun_hang` cases, detached output
  holders, early leader exit, TERM-ignoring children, explicit supervisor TERM,
  missing executable, oversized output, nonzero exit and successful measurement.
  Tests check that fixture descendants are gone or dead before test cleanup.
  Failed-base and corrected runs use CPU mocks only; no AMD SMI call is made.
- Validation: 131 offline tests passed across tool, budget, native-monitor and
  corpus suites on the corrected code. Ruff check/format, shell syntax and
  `git diff --check` passed. CI and local WSL status must be reported separately;
  passing cloud tests are not a replacement for the owner's failed local gates.
- Next gate: review this completed correction, then rerun the exact focused suite
  on WSL and perform separately scoped idle/no-training checks using the fixed
  tracked tools. Stop on any local failure and preserve full traces. Request
  fresh Retry2 runtime approval only after these gates; no synthetic fit,
  throughput, numerical result or GPU memory peak has been established here.
