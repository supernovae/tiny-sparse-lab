# Result template

Copy into a new file under `results/` only within an authorized task. Placeholder fields are not receipts. Preserve every failure; correct earlier results with new linked entries.

- Record ID and UTC timestamp: [real values]
- Card and bounded action: [one action]
- Operator and reviewer: [actual values or review pending]
- Starting status and decision reference: [values]
- Repository revision, branch and diff: [verified values]
- Native campaign ID/path and run/receipt IDs: [verified values or NOT APPLICABLE]
- Input artifacts, roles, byte sizes, hash algorithm and hashes: [real values or UNVERIFIED]
- Exact approval reference and limits: [actual authorization or NOT APPLICABLE]
- Device, environment/framework versions and precision: [observed values]
- Command or action actually executed: [exact value]
- Steps, target tokens, wall time, VRAM/RSS, disk and spend/credits: [measured values or NOT MEASURED]
- Output artifacts and hashes, including failed logs/checkpoints: [actual values]
- Tests, denominators, uncertainty and metrics: [exact receipts]
- Gate assessment: [READY FOR REVIEW / EVIDENCE VERIFIED / FAILED / BLOCKED]
- Reviewer and evidence checked: [actual review or pending]
- Known limits or unexplained differences: [values]
- One next decision: [bounded question]
- Correction to an earlier record: [ID and reason, if any]

Report the result before updating state. `STATUS.md` points to this entry; it does not replace machine-readable native receipts. Secrets, private credentials and signed URLs do not belong in records.
