# Durable training reporting connection reuse

The trainer previously opened and closed an SQLite connection for every store
operation, including every optimizer update's metric set. Training now owns a
thread-local store session through its existing `ExitStack`. Connections and
SQLite's statement cache survive between operations. Every operation still
commits its projection and outbox atomically before returning. There is no
in-memory write queue, singleton, new thread/process, schema change, or change to
SQLite durability settings. The existing heartbeat thread uses its own connection.
The session closes after progress reporter cleanup, including exception exits.
SQLite's existing automatic WAL checkpoint policy remains enabled; the session
avoids closing the last connection after every metric write.

## Bounded offline evidence

Measured in the isolated Linux cloud checkout on 2026-10-31, Python 3.14.7,
starting from main `4e00a706f6baa5a279d02bf8e5968585363d8d44`.
No accelerator, model, dataset, network, or dependency installation was used.

Reproduce with the provisioned environment:

```sh
uv run --locked --no-sync python benchmarks/training_reporting.py
```

Five alternating trials per case, each with a fresh disposable store and 1,000
updates of three scalar metrics (all 1,001 outbox records checked after timing).
The session timing includes opening and closing the session.

| Mode | Seconds, five trials | Median seconds |
| --- | --- | --- |
| Per-call connections | 0.3913, 0.4032, 0.3809, 0.4001, 0.4121 | 0.4001 |
| Training session | 0.0454, 0.0526, 0.0469, 0.0491, 0.0563 | 0.0491 |

A separate cProfile sample of the original 1,000-write path took 0.568s:
`sqlite3.connect` accumulated 0.072s, connection `close` 0.072s, and
`executemany` 0.222s. Keeping a separate reader connection open, while retaining
per-call writer connections, reduced that profiled sample to 0.422s. With the
session, a profiled sample took 0.080s (`executemany` 0.009s, commits 0.013s).
This supports amortizing connection lifecycle and associated SQLite work;
it does not separately quantify statement preparation, WAL cleanup, or storage
latency. Profiled and unprofiled timings are separate measurements.

The trainer's reporting category also includes progress JSON publication,
validation reports, checkpoint metadata/events, and final audits/receipts.
These measurements isolate metric persistence only. They neither establish the
root cause of a host reporting total nor predict end-to-end training speedup.
Host filesystem, model, heartbeat contention, and accelerator effects remain
unmeasured; coordinated host validation is still needed.

## Verification

- 31 focused store/throughput tests passed, including immediate independent-reader
  visibility, nested sessions, rollback after outbox rejection, heartbeat thread
  writes, and recovery after a process exits without closing its session.
- Repository Ruff check, format check, and `git diff --check` passed.
- The attempted training suite could not collect because the supplied environment
  lacks `torch`; two existing manifest transport tests failed for that same
  missing import. The passing focused rerun excluded those two cases. No
  dependencies were installed. Training/resume and broader tests are left to CI.
