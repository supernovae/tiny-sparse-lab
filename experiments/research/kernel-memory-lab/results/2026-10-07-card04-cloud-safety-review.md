# KML-20261007-C04-S1 — cloud code review, no runtime allocation

- Status: **READY FOR REVIEW** for the focused budget-runner correction;
  Retry2 and Card 04 measured fit remain **BLOCKED**. This review grants no
  acquisition, training, GPU, retry, cloud-use or spending authority.
- Reviewed rolling-branch head: `683c225e56c92c3bb3a29b723c0b8d3383b01c52`,
  compared with main `cf29aff79ec7259f7ec93988cd6a87065f4350e5`. Work used a
  separate normal branch, `codex/kernel-memory-lab-cloud-review`, in the saved
  cloud environment. No user-machine files or historical attempt ledgers changed.
- Diagnosis from the retained P3/P4 records: the AMD SMI CLI raised
  `AttributeError: 'AMDSMICommands' object has no attribute 'list_devices'`.
  P3's external watchdog exited without stopping staging; actual updates remain
  UNKNOWN. P4 stopped before staging with zero observed updates. Each keeps its
  seven-update reservation. Neither provides an OOM or model-fit conclusion.
- Review blocker: refreshed remote tree contains neither `read-vram-bytes.py`
  nor `run-profile-phase.sh`. The reader and wrapper referenced in Retry2 live
  under the owner's external `/srv` workspace; recorded hashes and test summaries
  do not expose their implementation. Review sanitized, versioned copies before
  another approval. In particular, verify bounded sensor-call timeouts, sensor
  failure and watchdog death, owned descendants, unavailable disk/inode readings,
  cap crossings, and termination before a successful completion can be reported.
- Independently reproduced defect: `AttemptBudget.run` launches a separate
  session but previously cleaned it up only on timeout/budget exceptions.
  `KeyboardInterrupt`, `SystemExit`, or an unexpected wait error could orphan
  work without its supervisor. The focused correction kills the owned process
  group and reaps the child before re-raising; timeout behavior and conservative
  reservations remain unchanged. Three mocked regressions failed before the fix.
  This is not a solution for uncatchable supervisor death, default SIGTERM, or
  descendants that deliberately leave the owned process group.
- Validation: 85 offline tests passed across attempt budget, operational monitor,
  corpus acquisition, corpus identity and corpus pipeline. Ruff check and format
  checks passed; research lint validated 228 tracked declarations/references.
  Commands used the existing locked environment with `--no-sync`; no dependency
  replacement, real source acquisition, GPU use or training was performed.
  Readiness smoke and model fixtures were deliberately excluded because they
  train, outside this review's scope. External runtime evidence was not available
  for independent hash verification in the cloud.
- Remaining limits: the native monitor still has no device-memory sampler or
  added-workspace cap implementation; those are existing TODO items. Cloud/mock
  results cannot certify AMD SMI 1.2.2 on WSL2/ROCm or establish training peaks.
  Retry2's proposed 120 updates, 1,800 seconds, 20 GiB VRAM, 24 GiB RSS,
  20 GiB additional disk, 1,000 inodes and zero spend are unchanged and unapproved.
