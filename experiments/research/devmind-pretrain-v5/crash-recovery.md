# DevMind v5 WSL restart recovery checkpoint

Recorded 2026-10-02 (CDT, UTC-05:00). Evidence-only recovery; no scientific
continuation or measurement implementation change. Status:
`V5_RECOVERY_STATE_UNCERTAIN`. Incident classification: `UNKNOWN_AFTER_RESTART`.

## Git protection

- Checkout: `/home/byron/src/tiny-sparse-lab`.
- Branch: `research/devmind-v5-model0`; pre-checkpoint HEAD:
  `6a04ee9a8408d36dd2bbbf6951564ba46004c522`.
- Local `origin/main` and GitHub `main`:
  `365e2d804bcb69c69de27aa78a2651a817a5eb0a`; local branch was 14 commits ahead,
  zero behind. Initial staged, unstaged and untracked inventories were empty;
  no incomplete new declarations/evidence/TODO files required exclusion.
- GitHub branch initially returned HTTP 404. Immediately protected the existing
  tip with the following command; GitHub branch SHA then equaled local HEAD:

  ```sh
  git -c credential.helper='!gh auth git-credential' push https://github.com/supernovae/tiny-sparse-lab.git HEAD:refs/heads/research/devmind-v5-model0
  gh api repos/supernovae/tiny-sparse-lab/branches/research/devmind-v5-model0 --jq .commit.sha
  ```

- Configured SSH remote and persistent credential configuration were unchanged.
  This note is the only checkpoint input to be committed and pushed on the same
  branch. No merge, rebase, branch switch or force push.
- Read-only inventory executed before changes: `git status --short --branch`,
  `git branch --show-current`, `git rev-parse HEAD`, `git remote -v`,
  `git log -14 --oneline`, `git rev-parse origin/main`,
  `git rev-list --left-right --count origin/main...HEAD`, `git branch -vv`, and
  `git status --porcelain=v1 --untracked-files=all`. The original user's eight
  command spellings were not supplied in the recovery plan; this records the
  actual inventory, not a claim of verbatim reproduction of unavailable text.

## Persistent state inaccessible

Intended artifact root: `/srv/sparselab/state/experiments/devmind-pretrain-v5/`.
`stat -c '%d:%i %n' / /srv/sparselab` returned `2112:2` for both paths.
`findmnt -R /srv/sparselab` showed only `/dev/sde` ext4 mounted there, with no
child mount; `findmnt -T /` showed the same source. Process inventory showed no
running v5 job. Unprivileged `fuser -vm` reported the root mount and user
processes; privileged process visibility required authentication.

The single `sudo -n umount /srv/sparselab` attempt, from the checkout, failed
with `sudo: interactive authentication is required`. No retry, force/lazy
unmount, remount, device change, replacement workspace or `state` creation.
Reading `/srv/sparselab/state/` returned path not found. This is an operational
visibility failure, **not evidence that the underlying artifacts were deleted**.

Consequently `primary/`, `recovery/`, `tokenizer-bakeoff-streamed/`,
`logs/monitored/`, measurement outputs, acquisition/build/freeze/failure receipts,
fit/heldout receipts and source provenance cannot be freshly inspected or hashed.
MODEL-0 final declarations, export, prepared data, Campaign state, token-budget
result and stage receipts: external presence and completion **unverified**.
The tracked v5 directory has protocol/policies/evidence, but no final MODEL-0
run/plan/Campaign declaration. No external artifact was modified.

## Prior durable evidence and bounded current checks

These are committed prior SparseLab verification records, not new verification
of inaccessible corpus/tokenizer bytes:

- `primary-verification.json`: `VERIFIED_PRIMARY`.
- `recovery-verification.json`: `EXACT_RECOVERY_MATCH`, zero snapshot imports.
- Producer: `51be77ba7fb186b3a4af47e9bfe487c7afa84273`.
- Project digest: `ba0ebec178b9451054b7f6a2d538b36354c2494d8bd7879ef4a8e16a2e1e6573`.
- Build: `7789f970526f3c07c005ce8e300e70fdc020e8bcbd088bd5bdf9f55706c1b75a`.
- Release: `72577dc6898c12caa3e17a731375573b5207d3f58a90963e4581531f3f1bf27b`.
- Fresh comparison of committed expectations/primary/recovery JSON confirmed
  identical 123-entry source snapshot maps, producer/project/build/release IDs.
- Fresh SHA-256 of tracked `build-identity.json`:
  `cd803725672e8f9c9572d6d194861213bb03dd4c77d33b59982c6cf83c408864`, matching
  the recorded primary/recovery build-identity binding.
- `corpus-recovery-inspection.json` records cold corpus `PRESENT`, implementation
  `MATCH`, recipe closure `BLOCKED` on explicit downstream tokenizer/model
  external boundaries. Its external report digest is
  `94b6d228d04e450b047224e020b65e57d889d8cdb6a0d03f9c8f7cf86128aaba` (unverified
  external bytes now). Prior DNS failures and RAM-censored bakeoff remain intact.

`tokenizer-bakeoff-verification.json` records
`VERIFIED_V5_BAKEOFF_AND_SELECTED_TOKENIZER`, winner `32768`:

| Bound input/artifact | Recorded SHA-256 | Current check |
| --- | --- | --- |
| Tracked `tokenizer-bakeoff.yaml` | `5538d3e6cc2c3efb0fc5f96ca671d98c23fccd522bfa36ed0ef3f1139f31470b` | Fresh bytes match |
| Tracked `corpus-expectations.json` | `c67cd5c5082cd9948e5673b795b959a92b47b2bb6483a70acda52842a428e743` | Fresh bytes match |
| External bakeoff `report.json` | `6927c81a734a266ada296ea7fdd9e138f1887678482ef73815068f6ab8bcb092` | Inaccessible; no fresh hash |
| External winner `tokenizer.json` | `ad186b251ca712e5deebf4cad2eda968a287a964a958604170b785cc380e2b56` | Inaccessible; no fresh hash |
| External winner manifest | `da5b295c42133ef2c31114a4fd81602e144b0113d2ad81cbddd2c0c722665aa5` | Prior candidate evidence; no fresh hash |

Winner path:
`/srv/sparselab/state/experiments/devmind-pretrain-v5/tokenizer-bakeoff-streamed/candidates/32768/tokenizer.json`.
Report path: `tokenizer-bakeoff-streamed/report.json` under the artifact root.
The 16,384 and 24,576 candidates and their manifests are also inaccessible.
The committed bakeoff count `983520212` is explicitly **not** a MODEL-0 budget.

Bounded runtime check passed from the repository root:

```sh
uv run --locked python -c 'from pathlib import Path; from sparselab.corpus.tokenizer_bakeoff import load_declaration; print(load_declaration(Path("experiments/research/devmind-pretrain-v5/tokenizer-bakeoff.yaml")).model_dump(mode="json"))'
```

Parsed schema 2, exact pinned primary release path, vocabularies
`[16384, 24576, 32768]`, fit cap `268435456`, validation cap 200/group, groups
`prose, go, shell, yaml, json`, near-best ratio `0.98`. No `verify_release`,
`describe`, `verify_tokenizer_artifact`, `bakeoff` or corpus reconstruction ran.

## Stage boundary and interrupted command

Last completed stage established by committed records: successful streamed
bakeoff and authenticated selected tokenizer, plus cold corpus inspection.
Present survival of those external bytes is unverified. The recovery plan's
last UI observation said budget measurement was waiting; that is not a durable
completion receipt. No surviving accessible launch/completion JSON establishes
the interrupted measurement's exact command, start time, PID, return code,
progress, stdout/stderr, result identity or successful finish. All are **unknown**;
no measurement result is accepted. Primary and independent recovery completion
references in their committed records point into inaccessible `logs/monitored/`.
The earlier guarded bakeoff's return code 143 is a separate recorded attempt,
not this restart's measurement return code.

Bounded current-source audit (not proof of the interrupted process's route):

- `campaign/engine.py:1362` routes `token_measurement` to `release.describe`.
  `release.py:1411` calls full `verify_release`, then `measure_views`, then
  `measure_source_rights` for schema-2/3 reports with a tokenizer.
- All four `_streaming_v3` conditions hold for the tracked bound build identity:
  schema 3, chat unselected, only `lm_text` transforms, no fraction. Thus this
  identity selects schema-3 streaming stage checks and `_validate_rows_v3`, not
  legacy `_validate_rows`, if used by the interrupted process. External manifest
  and actual launch remain inaccessible, so actual dispatch is unverified.
- Schema-3 verification still hashes/scans release files and snapshots; its
  evidence SQLite database is at the release parent, `temp_store=FILE`, cache
  size 8192 KiB. It stores document text/data on disk, not a whole Python text map.
- Legacy `_rows` uses `read_text().splitlines()` and legacy validation keeps whole
  document/lineage collections. Do not attribute that legacy peak to v5 without
  command/schema evidence.
- `measurement._lineage_index` streams lineage into a disk-backed SQLite file at
  the release parent. It does not explicitly set SQLite `temp_store`.
  `_measure_views` retains generation/scenario maps, family/dimension counters,
  and per-view/split concentration digest/prefix sets; parent counts are SQLite
  keyed, with at most 100,000 entries materialized in the report.
- `_Concentration` retains exact/normalized digest and 32-character prefix sets,
  not complete rendered records. Its n-gram census clears after 20 million
  characters. `measure_source_rights` retains one distinct-content SHA set.
  Tokenizer encoding runs over rendered views and separately over eligible
  distinct training source documents; `len(model.encode(text).ids)` does not
  retain encoded token lists between rows. No subprocess/executor/fanout calls
  were found in release/measurement modules; inaccessible launch scripts may
  differ. These are corpus-size-dependent allocations, not evidence of a leak.

## Journal evidence and classification

Likely crash boot: `96ebea96e91e43dfa9c35956a8b9bb0a` (`-1`),
2026-10-02 10:43:30–10:54:21 CDT. Bounded kernel journal, shutdown journal and
OOM/kill/shutdown search show no crash-window OOM kill; systemd reached poweroff
at 10:54:21. Earlier lines include WSL operation cancellation at 10:54:18 and
`/dev/sde: Can't lookup blockdev` at 10:53:50. None ties an accessible measurement
PID/command to the shutdown cause. No crash-window monitor samples are accessible.

Separate earlier boot: `7da50c42c7644c8287c2fb1ae90a4738` (`-2`),
2026-10-01 11:35:46–2026-10-02 06:44:53 CDT. At 06:43:13 on October 2, kernel
logged `Out of memory: Killed process 12660 (python)` with total VM 61742552 KiB
and anonymous RSS 31393076 KiB; reaped at 06:43:14. This confirms an OOM **in that
other boot**, not the likely 10:43–10:54 incident or a particular v5 command.

Current boot `2f4a45c355f545c7ac320a58eb8563c3`: fresh `free -h` showed 31 GiB
RAM, 29 GiB available, 8 GiB swap and zero swap used; `swapon --show` identified
`/dev/sdc`. Current warning/error `dmesg -T` contained startup/WSL/DXG/journal
warnings but no OOM kill. Current resources do not reconstruct prior pressure.
Evidence commands: `journalctl --list-boots`, `journalctl -k -b -1 -n 100`,
`journalctl -b -1 -n 100`, filtered `journalctl -b -1` and `journalctl -k -b -2`,
all `--no-pager`; `free -h`, `swapon --show`, `dmesg -T --level=err,warn`.

Classification remains `UNKNOWN_AFTER_RESTART`: no command-to-crash correlation,
no accessible pressure samples, and no corroborated measurement failure cause.
An orderly poweroff alone is not sufficient to label a non-memory failure.

## Safe handoff

Only bounded design investigation is authorized next; **no measurement retry**.
Restoring access to the original state requires an operator with sudo
authentication; do not create a substitute or infer deletion. Once original
visibility is restored, inspect launch/completion/monitor receipts and small
artifact hashes before accepting a stage boundary. No full corpus verifier,
exporter, tokenizer refit, data preparation, MODEL-0 plan, training or code fix
was run during this checkpoint. MODEL-0 and scientific inputs remain unchanged.

## Post-mount recovery — 2026-10-02

This later observation preserves the preceding inaccessible-state record.
External-state result: `V5_EXTERNAL_STATE_VERIFIED`; measurement:
`PARTIAL_INTERRUPTED`; command memory behavior: `UNKNOWN`.

### Protected source and restored volume

- Initial branch `research/devmind-v5-model0`, clean working tree, local HEAD
  and freshly queried GitHub branch both
  `e199ee8e4b68085a971d9416ec5d941fdb18e123`. This evidence-only commit descends
  from that protected tip; no branch switch, merge or mount mutation.
- `findmnt /srv/sparselab`, `df -hT`, `df -B1`, `df -i`, and
  `lsblk -o NAME,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS` identify `/dev/sdd`, ext4,
  label `sparselab`, UUID `ecf1fe73-0a20-4ac6-87e3-0fef2ab51148`.
  The block device is 1 TiB; filesystem capacity is 1,081,101,176,832 bytes,
  used 178,507,436,032, available 847,601,385,472 (human display 1007G/167G/790G).
  Free inodes: 64,986,509 of 67,108,864.
  WSL root is separately `/dev/sde`, ext4,
  UUID `5e06339b-ab11-486d-a9eb-694f762fa21a`.

### Survival and cold authentication

Task root remains `/srv/sparselab/state/experiments/devmind-pretrain-v5`.
Both `primary/` and `recovery/`, their exact recorded release directories,
metadata and replay receipts exist. Before any cold scan, 22 bounded hash
comparisons passed against committed primary/recovery/tokenizer/inspection
evidence (including repeated references). In particular:

| Artifact | Fresh matching SHA-256 |
| --- | --- |
| Both release `manifest.json` files and metadata copies | `003da8f335899d68391497847aa32f03043110acf68b04e2a55e132e5843ffcb` |
| `tokenizer-bakeoff-streamed/report.json` | `6927c81a734a266ada296ea7fdd9e138f1887678482ef73815068f6ab8bcb092` |
| `tokenizer-bakeoff-streamed/candidates/32768/tokenizer.json` | `ad186b251ca712e5deebf4cad2eda968a287a964a958604170b785cc380e2b56` |
| Selected `tokenizer_manifest.json` | `da5b295c42133ef2c31114a4fd81602e144b0113d2ad81cbddd2c0c722665aa5` |
| `cold-corpus-recovery-inspection.json` | `94b6d228d04e450b047224e020b65e57d889d8cdb6a0d03f9c8f7cf86128aaba` |

The 141,116,322/141,116,445-byte acquisition metadata copies were deliberately
deferred by the initial 16 MiB bound, then stream-hashed after that gate; both
match their committed SHA values. Restored primary/recovery results agree with
committed producer, project, build, release and all 123 snapshot identities.

One locked-environment Python process called `verify_release(primary_release,
expected_id=release_id)` exactly once. Complete release/stage/snapshot/row
authentication passed in 493.71 seconds for release
`72577dc6898c12caa3e17a731375573b5207d3f58a90963e4581531f3f1bf27b`
and 123 snapshots. It exercised the schema-3 disk-backed verifier, including
temporary SQLite evidence under the release parent; no durable payload changed.
The recovery copy's manifests/metadata/receipts were authenticated, but a second
full scan of its payload was intentionally not performed.

In the same process, `_verify_tokenizer_manifest` and `load_tokenizer` accepted
the selected tokenizer and actual vocabulary 32,768. The unchanged committed
report hash authenticates the prior accepted bakeoff evidence; `choose_candidate`
reapplied the existing near-best rule without encoding any corpus text.
Candidate manifest, report release binding, primary release path/ID/manifest
hash, selected path and tokenizer SHA agree. Streamed sample hashes also match:
fit `a9678289862316f93846b3ddace4dce9e4f7e3b621eb13f21ce730db63fc1006`,
heldout `375d745529ad5a26c26fdfe32be0bb26a1a2a521b4560643ce532f4f0275e50d`.
No fitting, heldout remeasurement or full token measurement ran. Public tokenizer
verification would repeat the release scan and sample/score reconstruction via
`recovery.evidence._selection`; exact committed evidence reuse avoided that work.

### Interrupted measurement identity and result boundary

Known log prefix, relative to the task root:
`logs/monitored/v5-measured-model0-budget-1790939923303015023`.
Surviving launch SHA:
`227bfc28e6fe4c904ffbc037734061483d8b25316e8bb5e9cff2557d2f7ed3b3`.
Surviving samples SHA:
`9e03b66287dacf51ce44eb18a177b4d8860a35bae84d87c4200ed3b4e8ca62c1`.
`.stdout` and `.stderr` are each zero bytes; their SHA is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
`.completion.json` and task-root `measured-model0-budget.json` do not exist.
Bounded task-root result names, known monitored logs and both replay receipt
directories revealed no alternative budget completion/result. No orphan accepted.

Launch time: `2026-10-02T11:18:43.303042+00:00` (06:18:43 CDT);
source commit `90a2e7a39e59113f3d83abaead5e6eaa340c6ace`, cwd the checkout.
Exact recorded argv, rendered as a shell command:

```sh
uv run --locked --extra cpu python \
  /srv/sparselab/state/scratch/devmind-v5/tmp/measure_tokens_v5.py \
  --primary-result /srv/sparselab/state/experiments/devmind-pretrain-v5/primary-result.json \
  --recovery-result /srv/sparselab/state/experiments/devmind-pretrain-v5/recovery-result.json \
  --selection-report /srv/sparselab/state/experiments/devmind-pretrain-v5/tokenizer-bakeoff-streamed/report.json \
  --measurement-policy /home/byron/src/tiny-sparse-lab/experiments/research/devmind-pretrain-v5/budget-measurement-policy.yaml \
  --output /srv/sparselab/state/experiments/devmind-pretrain-v5/measured-model0-budget.json
```

The surviving script SHA is
`1c6f2e4b3468c8c310b012f45c6975c16d2068a8adbcabbfbf109dea1bcb7240`;
mtime is 2026-10-01 19:01:45 CDT, before launch. It was not hash-bound in the
launch receipt, so its historical byte identity is not independently established.
Its route is direct `verify_release` → tokenizer `verify_artifact` →
`campaign.policy.measure_readiness`, **not** Campaign `token_measurement` →
`release.describe`. Relevant policy/release/tokenizer/selection source files are
unchanged between the recorded launch commit and protected HEAD. This route can
verify the release three times: explicit script gate, tokenizer selection gate,
and readiness gate. No durable evidence identifies the last reached phase.
Launch/samples record no PID, process start identity or return code; all remain
unknown. Current process inventory contains no surviving measurement command.
Status is `PARTIAL_INTERRUPTED`, not complete or not-started. Canonical `D`,
`max_steps` and `max_tokens` remain unavailable; `983520212` is still not `D`.

### Memory evidence and safe next step

49 samples at approximately 30-second intervals survive, spanning
11:18:43.310545–11:42:43.328173 UTC. Available system RAM starts at
32,083,738,624 bytes, reaches its recorded minimum 23,551,811,584 at
11:34:13.321125, and ends at 26,208,464,896. The samples contain no
process-tree RSS, PID, swap, token progress or phase counters; system-available
memory cannot be converted into this process's RSS.

Newly available launch timestamps place this attempt in boot
`7da50c42c7644c8287c2fb1ae90a4738`, not the later 10:43–10:54 CDT restart window.
Targeted kernel journal inspection shows the previously recorded OOM at
06:43:13 CDT, 30 seconds after the last sample: Python PID 12660, anonymous RSS
31,393,076 KiB, file RSS 916 KiB, total VM 61,742,552 KiB; host swap was entirely
used (0 KiB free of 8,388,608 KiB). No launch PID binding or journal entries for
`_PID=12660` establish that it was this script. This is temporal correlation,
not a proven command-specific OOM cause or leak. Command classification:
`UNKNOWN`. Current `free -b`/`swapon --show --bytes` observed 33,610,706,944-byte
RAM, 32,333,824,000 available, 8,589,934,592-byte swap, zero used; these current
values do not reconstruct historical process pressure.

The next unfinished MODEL-0 gate is its developer-token denominator.
Generic `describe()` is unnecessarily broad for it, but was not this launch's
recorded route. The actual readiness route is also unnecessarily broad:
`policy.py:130` calls `_rows` on the 6,123,770,983-byte `documents.jsonl`;
`release.py:27-32` uses `read_text().splitlines()` and materializes every parsed
row. `policy.py:154-172` additionally materializes the 1,079,569,336-byte lineage
ledger, selected train IDs, shapes and heldout families. It tokenizes all domains,
although this policy only requires `developer_systems`. Those corpus-sized
allocations establish an unbounded-memory design risk, not a demonstrated leak.

Safe next task: implement and verify a bounded denominator-only measurement
before authorizing another full measurement. Preserve normalized, non-dropped,
train-only `developer_systems` documents, content-SHA deduplication and raw
selected-tokenizer encoding (no packing EOS/padding); stream rows and use
disk-backed deduplication rather than a whole-corpus text/lineage map. Bind any
future result to the authenticated release/tokenizer and preserve the existing
8192-target update and budget formula. This recovery did not implement or rerun
that measurement. No acquisition/build/freeze, refit, export, preparation, new
MODEL-0 plan, ROCm or training was performed. Only this compact source evidence
is authorized for commit/push; main remains unmerged.
