# KML-20261007-C04-S2 — tracked monitor supervision review

- Date/status: 2026-10-07; **READY FOR REVIEW for code**, not GPU execution.
  Reviewed base `75f9a8d7667fd046cb909de88c184a9b24ce6209` and incorporated it
  into the existing PR #51 branch by a normal merge commit, preserving both
  histories. No remote rolling-branch update or PR merge was performed.
- Corrections to S1's review availability: P5 now supplies tracked scripts.
  The missing-source blocker is resolved. External originals, P3/P4 evidence,
  and both historical seven-update reservations remain unchanged.
- Material findings in P5: unbounded API calls could stall supervision; waiting
  solely for the native monitor ignored watchdog death; TERM-only shutdown
  could leave a resistant child alive. Disk/inode limits were not checked before
  launch or at completion. File existence did not establish a valid reserved
  budget, adequate native RSS/headroom policy, or monitored output locations.
- Correction: five-second GNU timeout process groups bound API/storage calls,
  including a reader child of uv. The supervisor observes both children and
  rejects watchdog failure even when native work finishes first. Failure sends
  TERM then KILL after one second to its native-work process group; cap evidence
  is retained and unsuccessful shutdown is recorded as shell status 137.
  Sensor groups have independent five-second deadlines; they are not training
  groups. A successful phase joins its watchdog and takes a final resource sample.
  Phase claims prohibit replay. NUL-based inode counting handles newline names.
- The new small validator uses native AttemptBudget and monitor/envelope readers.
  It requires 120 total updates, no more than 1,800 seconds, stage reservation 7
  and then train reservation 113, with no extra reservations. It requires the
  24 GiB RSS ceiling and 2 GiB/1,000-inode projected free margins, checks output
  path containment, and gates train on successful stage supervision. Native
  train remains responsible for authenticating its stage bundle. The validator
  is not a runtime authorization mechanism; explicit owner approval is required.
- Reviewed tool SHA-256 values: reader
  `1ff1e180cccf01e8c84d8a3a2af5fbf21c6d843a487522c268fada19a938948a`;
  corrected launcher
  `848eff0ad1fd968f1a4993bdab9c9524dbb944607a1117fb228d31e1c1c001fa`;
  input validator
  `ae4d9f06c62f3222dd44bb316cec6baf3550a1e872631d685f892b6ba8e07502`.
  Retry2 now binds these tracked files instead of the historical external copies.
- Validation: 122 offline tests passed across tracked tools, attempt budget,
  native monitor, acquisition, corpus identity and pipeline. New cases include
  hung sensor subprocesses, watchdog death, a TERM-resistant native child,
  bool/nonfinite/invalid metrics, missing/incorrect reservations, corrupt ledger,
  weak RSS/storage policy, output escape and absent successful stage evidence.
  Ruff check/format and shell syntax passed. The Linux fast CI job now includes
  the tools, native monitor and budget regression suites. Shell tests are scoped
  to Linux/WSL GNU tools; reader/validator unit tests are portable.
- Limitations: no live WSL/ROCm/GPU, acquisition or training occurred. SIGKILL of
  the supervisor is handled only by the outer native budget runner's whole-group
  deadline; deliberately detached training descendants are unsupported. Sampled
  readings cannot prove instantaneous peak bounds. Task-root baselines must be
  taken once and preserved; no sampler can recover historical high-water usage
  from a stale or reset baseline. Full fit, numerical and throughput receipts
  remain absent. No native campaign/run or GPU resource measurement was created.
- Recommendation: **NO-GO on the original P5 launcher**. Corrected code is ready
  for review, followed by local idle/no-training supervision verification on the
  fixed revision and a separate decision on the unchanged Retry2 allocation.
  No retry, fit promotion or Card 05 authority is granted by this review.
