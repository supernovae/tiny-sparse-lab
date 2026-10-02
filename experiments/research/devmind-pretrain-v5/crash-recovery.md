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
