# KML-20261007-C04-S3 — stop when sample evidence cannot be written

- 2026-10-07; **READY FOR REVIEW for code only**. PR #51 was merged externally
  at `df2f0a29c885fc01d87c37d5828abafe70d6d679` while final review continued.
  This minimal follow-up is based on that exact rolling head, on the fresh
  branch `codex/kernel-memory-lab-log-failure`. It is not part of PR #51.
- Finding: Bash disables implicit `errexit` inside a function called in an
  `||` conditional. The watchdog's `check_resources || return 1` could therefore
  ignore a failed sample-CSV write and continue native work without recording
  resource evidence. The merged-base mock exceeded its 12-second test deadline;
  the test harness terminated only its owned process group.
- Fix: explicitly return failure when writing the sample record fails. Existing
  supervisor failure handling then terminates the native-work group. Also ensure
  that a failed cleanup grace sleep cannot skip the final group KILL. No phase,
  budget, scientific configuration, source data or approval boundary changed.
- Corrected launcher SHA-256:
  `3064f3e406b82ef4945e6c7e1cc6650048dafd5ee94059416b9396b238ec8e8e`.
  The unchanged reader and validator retain S2's hashes. Retry2 references this
  new launcher hash; S2's historical record remains unchanged.
- Validation: 123 offline tests passed across tracked tools, attempt budget,
  native monitor and corpus suites on the corrected code. The new regression
  makes the sample path unwritable after native launch and requires failure
  rather than success. Ruff check/format and shell syntax passed. No real GPU,
  training, acquisition, runtime receipt or resource allocation was performed.
- Next gate: review and integrate this correction, then verify the exact fixed
  revision locally using idle telemetry and CPU/no-training supervision checks.
  Only after those pass should an operator request separate fresh Retry2
  approval. Fit remains unverified; neither this fix nor passing mock tests grants
  GPU execution authority or proves an instantaneous peak bound.
