# C05-B12 — fresh 50M attempt stopped at retained-source reuse

- **Gate:** FAILED during live acquisition, before Wikimedia transfer, corpus
  admission, preparation, model initialization, training or evaluation. The
  exact approved checkout was clean
  `cbd69d087885c020fbc20c849a9d29604581b462` on
  `codex/kernel-base-50m-proposal`. The owner approved one fresh conditional
  attempt under the revised launch packet and unchanged proposal, with no
  restart, training retry or code change during execution. This is that one
  attempt; it is stopped and cannot resume under that authority.
- **Identity and preflight:** Proposal SHA-256
  `de968f0dc80d1cd3f640b6193c90aaa23281fae068e2ce92fc7a02495b8607eb`,
  packet SHA-256
  `46ecfc2451f73dfa204c575fc60c6d3e231abd8eccc54c3dd99f98cef5244ad8`,
  acquisition-project byte SHA-256
  `6326d66589fa787fd3e85d834417d968b21eb6f808f1b6c85635ac255714aa51`.
  The retained PagerDuty, Gutenberg and Scoutflo snapshots cold-verified at
  their pinned identities; the original held-out release cold-verified, and
  its manifest, tokenizer, family inventory, fixed profile and frozen suite
  hashes matched the packet. The registered ROCm environment was PyTorch
  `2.13.0+rocm10.0.0`/HIP `7.15.26333`; idle UUID-matched VRAM was
  3,433,029,632 bytes. This was a device preflight, not a model fit.
- **Attempt:** New root
  `/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-50m-attempt2-cbd69d0/`.
  One pre-write common-root baseline measured 126,417,739,451 apparent bytes,
  42,828 entries, 898,411,675,648 free bytes and 66,995,218 free inodes.
  Baseline identity `0df76dc0b29807def3eed94d5ee17e84048aad86e3e45e773a996342ea92c558`;
  static content identity `1c48c347aacc1c8218c33627bf1032fcce49366708b3601cc8807bae81e6551e`;
  contract byte SHA-256
  `02313f9e0e5929dbd941ce7f60dfb050046b6c8206b6370f0ddad3027e87f1de`.
  The one persistent 36,000-second ledger began
  `2026-10-09T18:20:31.004628Z`. C05-B11's distinct stopped ledger retained
  SHA-256 `09a5b2a4348417bb2b90d2d56603e5a02208c07c7ea46863795c0f0ae0ea3d2d`.
- **Operations:** Three `corpus alias-snapshot` calls and `corpus budget-init`
  completed through the actual native CLI → attempt ledger → owned supervisor
  → whole monitor → preparation monitor → native operation path, each with
  zero-descendant owned receipts. The first `corpus acquire` then ran through
  that same path. Its persistent transport budget began with zero charges.
  Before reaching Wikimedia, its ledger showed retained-source transfers:
  PagerDuty 38 transfers/317,526 actual bytes, Gutenberg one
  transfer/362,929,669 bytes, and Scoutflo 284 transfers/1,361,984 actual
  bytes, including one started but unfinished transfer. This violates the
  approved instruction to reuse those snapshots and acquire only the
  additional Wikimedia selection. The owner stop file was set immediately;
  no retry, resume or further preparation command ran.
- **Cause established offline, without a code change:** The new native alias
  creates a symlink and cold-verifies it. Live `corpus acquire` uses the
  path-bound `verified_reuse` proof path; `verify_snapshot` rejects that
  symlinked artifact path. The acquisition loop catches that verification
  error and falls through to reacquisition. A read-only reproduction on the
  pinned PagerDuty alias returned `ValueError: invalid source_snapshot artifact
  ... symlinked artifact path`. The alias fixture had exercised cold reuse,
  so it did not qualify this live proof mode.
- **Actual charges:** The transport ledger retained 364,609,179 actual and
  **364,614,231 charged** source-body bytes; 227,403 actual and **399,555
  charged** metadata-body bytes. The attempt ledger has five zero-model phase
  reservations; the interrupted acquisition reservation has null actuals.
  Charged optimizer updates, supervised targets, nontraining forwards,
  operational validation batches, generation calls and requested tokens are
  all **zero**. No transfer to Wikimedia, acquisition lock, v3 release,
  mixture, prepared bundle, model run, learning curve or fixed-profile score
  exists for this attempt.
- **Resources and shutdown:** The last independent common-root observation
  found 132,074,142 added apparent bytes and 482 added entries against the
  original baseline. The whole monitor sampled through 86.84 seconds with
  observed peaks of 2,321,211,392 B process-tree RSS, 3,493,838,848 B
  whole-device VRAM, 484,284,967 added apparent bytes and 474 entries;
  sampled violations were zero. The preparation monitor sampled through
  83.36 seconds with observed peaks of 1,413,681,152 B RSS and 493,359,863
  added bytes. These are **sampled peaks, not terminal monitor receipts**:
  the intentional supervisor stop left both monitor completion files absent.
  The outer owned receipt says `launcher requested stop`, return code 1,
  zero living descendants; the inner owned receipt says signal 15, return
  code 1, zero living descendants. A process-list check found no continuing
  acquisition or model worker.
- **Retained evidence:** The attempt root contains immutable preflight and
  contract files, the fixed baseline, one ledger, three alias and budget-init
  receipts, partial acquisition traces, stop file, `final-attempt-status.json`,
  `final-transport-status.json`, `final-storage-observation.json`,
  `stop-record.json` (SHA-256
  `963129447f10153be07e24233e09a0ff07f85d81e3473f4cdf05ddee1cfb6c63`)
  and a 425-file `evidence-sha256.txt` inventory (SHA-256
  `5e17409c40e5c59ad7028221b2e1a1b6004cb0065c0b075d879707543238cd90`).
  Partial source files and the transport ledger under the new project's
  corpus root are preserved. Final attempt-ledger SHA-256 is
  `19a88098595d69613e50bb707ac39c211d88a831125832995825ac915c3f7358`;
  transport-ledger SHA-256 is
  `4515d902886b140eef4cc450c5a7d8c53153b03d78f44a3e342d2896bb2f465a`.

**Next decision:** Repair and test the existing native retained-snapshot
reuse path so the normal path-bound proof mode accepts a verified immutable
reuse binding without weakening live verification or reacquiring retained
bytes. That is separate code work. A future 50M attempt would need a new
reviewed allocation; this attempt's charges and evidence remain intact.
The B7/B9/B8 artifacts, original fixed evaluation, C05-N1 0/200 reader
result and Card 06 block remain unchanged. Model-quality eligibility is
unestablished because no model work occurred.
