# Card 03 scaled source-policy audit protocol v1

Status: preregistered before the KML-D15 content attempt. This applies the
accepted [source-admission policy v1](CARD03_SOURCE_ADMISSION_POLICY_V1.md)
to the exact selection in [the scale proposal](CARD03_SCALE_PREPARATION_PROPOSAL.md).
An automatic pass is a candidate decision, not source-text publication or
model-quality evidence. Preserve the original pilot audit separately.

Review the pinned component terms and the acquired PagerDuty/Scoutflo LICENSE
files before making any source eligible. A conflict in source-wide terms stops
the affected source. For covered sources, the native admission draft screens
every selected row/file; a material row/file conflict or unknown coverage is
quarantined individually. Optional descriptive gaps are logged as issues.

Freeze the audit population and candidate decisions from verified snapshot
digests before selecting samples. Use the seed string
`kml-card03-scale-spot-audit-v1` and rank candidates within each stratum by
SHA-256 of UTF-8 `seed | source_id | stratum | immutable row-or-file digest`,
ascending, breaking ties by stable row index/path. Review these clear-screen
samples, or all available if the stratum is smaller:

| Source | Preregistered clear-screen sample |
| --- | ---: |
| Gutenberg | 12 works, stratified 6 above and 6 below the source median retained-text byte length |
| Wikimedia | 24 pages: 6 each with/without optional attribution-detail gaps and above/below median retained-text byte length; redistribute an empty cell deterministically across available cells |
| PagerDuty | 8 files across `docs/before`, `docs/during`, `docs/after` and remaining docs, at least one from each nonempty group |
| Scoutflo | 24 files: 8 each from AWS, Kubernetes and Sentry playbook directories, including proactive and direct-response topics where available |

Also manually review **every** automatically flagged exception proposed for
qualification. Keep unresolved flags quarantined; no quota may force their
release. Verify source/license scope, locator and attribution/notice route,
content identity, third-party inserts, personal or secret data, and document
family against the acquired bytes and upstream source. Record reviewer, date,
evidence URL, finding and any issue for each audited candidate. If a material
error is found, quarantine that record or family, expand the affected stratum
sample to locate the boundary, and stop the source only if policy coverage
itself is unreliable. Do not replace negative samples.

The aggregate agent-assisted review limit is **20 hours** for source-policy
review, flags, spot audits and evaluation-item review together. Track elapsed
review time; stop and report unfinished work if the ceiling is reached.
