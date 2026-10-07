# Card 03 scaled preparation retry 1 — approval request

Status: **PROPOSED, NOT APPROVED**. [C03-S2](results/2026-10-07-card03-scale-metadata-stop.md) records the one KML-D15 attempt stopped in its Git metadata phase. Its SQLite ledger and monitor traces remain immutable in the external work root. The stopped attempt consumed 66,430 metadata response-body bytes, reserved 133,427 metadata bytes, and transferred zero source-body bytes. No content snapshot, tokenizer, split, mixture, evaluation set, GPU work or optimizer update resulted.

The sole native code change since the fixed `5c3bdb0` checkout is `d996633`: a per-read transport-deadline fallback for live urllib responses whose socket is unavailable through the existing private traversal. It retains the strict body-byte accounting, fails closed when it cannot enforce a deadline, and has an offline hidden-socket response test. Focused acquisition/budget/Scoutflo tests pass (44); the prior integrated corpus suite passed (371). No network validation of the fix has been attempted.

Approve **one new preparation attempt** using [the retry-1 declaration](corpus-scale/corpus-acquire-retry1.yaml), with a new project ID and durable ledger. Use exactly the same four immutable sources, files, revision pins and expected digests as [the original proposal](CARD03_SCALE_PREPARATION_PROPOSAL.md). Do not resume, reset or delete the failed ledger. There are no additional source candidates, revised model settings, or training authorization. The retry's resource ceilings are **independent new caps**, not remaining balances from the failed attempt:

| Limit | Retry-1 ceiling |
| --- | ---: |
| Source response bodies, one interrupted retry per file | 1,691,154,896 bytes |
| Shared metadata, redirect and error response bodies | 4,194,304 bytes |
| Combined response bodies | 1,695,349,200 bytes |
| Transfers / per-file interrupted retries | 946 / 1 |
| Transport wall time from new ledger initialization | 7,200 seconds |
| Decompressed HF bytes | 4 GiB each shard, 8 GiB aggregate |
| Retained PG / Wikimedia / pinned Git content | 256 MiB / 256 MiB / 4 MiB; 516 MiB combined |
| Added physical disk / inodes for transport and preparation | 8 GiB / 100,000 |
| Post-transport preparation time / peak RSS / tokenizer input | 28,800 CPU wall seconds / 24 GiB / 516 MiB |
| Agent-assisted reviewer time | 20 aggregate hours; no hired help or owner time commitment |
| GPU / optimizer updates / cloud spend | zero |

Apply the owner's KML-D15 conditions unchanged: quarantine individual exceptions; stop an affected source on a source-wide rights conflict; preserve verified reusable artifacts; release 5,000,000 positions only after measuring at least 1,625,000 general, 625,000 explanatory and 250,000 incident **unique train** positions and at least 100/1,000/75 eligible independent train families, with nonempty validation and test families, 80/10/10 family splits and at most two exposures. Do not draw from held-out data or expand a cap to satisfy a floor. Freeze the train-only 32,768-entry tokenizer, realized 65/25/10 mix and reviewed 200/400-item evaluation sets only if their actual prerequisites pass. Stop at the first checksum, metadata, rights, monitoring, resource, supply or numerical failure; retain traces and request a new decision. A completed preparation still stops before Card 05 runtime approval.
