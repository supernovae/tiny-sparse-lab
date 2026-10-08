# One fresh hosted owned-root qualification request

**Status: approval request only; do not dispatch from this document.** Pin `serving_only=true` to the final clean, pushed implementation SHA. Keep PR #54 draft. This is a new allocation; C05-I6 and C05-I8 are spent and cannot be retried or resumed.

Dispatch exactly once using `gh workflow run ci.yml --ref codex/kernel-memory-lab -f serving_only=true -f qualification_sha=<final-clean-implementation-SHA>`. Before model work, require the Actions workflow API head and checked-out commit to match that SHA, and retain the actual workflow run ID/head. Stop all five jobs at the first declared failure; no rerun or resume. The unchanged frozen test selections and parameter guards must pass.

| Job | Max updates | Actual targets | Generation calls | Requested generated-token allowance | Job timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linux serving | 20 | 640 | 113 | 355 | 10 min |
| macOS serving | 20 | 640 | 113 | 355 | 10 min |
| macOS archive | 2 | 32 | 0 | 0 | 10 min |
| Linux fast | 0 | 0 | 0 | 0 | 10 min |
| Linux lint | 0 | 0 | 0 | 0 | 5 min |
| **Aggregate maximum** | **42** | **1,312** | **226** | **710** | **45 runner-min** |

One 1,800-second deadline starts at workflow creation and includes queueing, provider setup, guarded work and shutdown; each job also has its listed Actions timeout. During lab execution, each job enforces 8 GiB Runner.Worker descendant RSS and 4 GiB live bytes/50,000 live inodes across the entire checkout and its distinct `$RUNNER_TEMP/kml-ci-<job>` root, including dependency environment, caches, temp files, logs and SQLite/WAL/SHM. A complete sweep has 15 seconds; 15 seconds remain reserved for shutdown. Before lab entry, fail if observed free capacity is below 8 GiB or 100,000 inodes on checkout or runner-temp filesystems. Provider bootstrap has exact SHA/deadline checks and a time-only watcher; toolcache/home/runner-temp setup writes are observed through path/version and free-capacity receipts, **not** exact added-storage accounting. No recursive toolcache scan.

The guard must precharge requested model work into the persistent job ledger before execution, including Python CLI children, retain charges on failure, and report actual counters separately. Preserve startup, provider observation, watcher, storage, ledger, model-test and cleanup receipts. A watcher loss, changed selection, exhausted sampling/deadline, cap breach, escaped required lab output or unverified descendant cleanup is a failure. No full CPU suite, GPU work, research experiment, source acquisition, paid runner upgrade, PR-ready transition or merge is included. Hosted Linux/macOS path confinement, provider setup footprint, watcher latency and model-test behavior remain the uncertainties this one dispatch would measure.
