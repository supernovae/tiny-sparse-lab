#!/usr/bin/env bash
# One zero-update diagnostic; the immutable ledger's deadline starts before this launcher.
set -euo pipefail
umask 077
root=${KML_DIAG_ROOT:?}
task_root=${KML_TASK_ROOT:?}
checkout=${KML_CHECKOUT:?}
uuid=${KML_EXPECTED_GPU_UUID:?}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:?}
[[ -d "$root" && -d "$task_root" && -d "$UV_PROJECT_ENVIRONMENT" ]] || exit 2
root=$(realpath -e -- "$root")
task_root=$(realpath -e -- "$task_root")
checkout=$(realpath -e -- "$checkout")
case "$root/" in "$task_root/"*) ;; *) exit 2;; esac
cd "$checkout"
export SPARSELAB_WORK_DIR=$task_root
ledger=$root/attempt.json
baseline=$root/baseline.txt
declaration=$root/declaration.json
[[ -f "$ledger" && -f "$baseline" && -f "$declaration" ]] || exit 2
read -r baseline_bytes baseline_inodes < "$baseline"
read -r deadline_ns declaration_sha < <(uv run --locked --no-sync python - "$ledger" "$declaration" <<'PY'
import json, sys, time
from pathlib import Path
from sparselab.training.manifest import sha256_file
ledger=json.loads(Path(sys.argv[1]).read_text())
assert ledger['format']=='kml-card05-matched-attempt-v1'
assert ledger['max_optimizer_updates']==0 and ledger['max_generations']==18
assert ledger['max_generated_tokens']==1152 and ledger['max_wall_seconds']==600
assert ledger['max_vram_bytes']==21474836480 and ledger['max_rss_bytes']==25769803776
assert ledger['max_added_bytes']==1073741824 and ledger['max_added_inodes']==100
assert ledger['declaration_sha256']==sha256_file(Path(sys.argv[2]))
assert time.time_ns()<ledger['deadline_ns']
print(ledger['deadline_ns'], ledger['declaration_sha256'])
PY
)
[[ "$baseline_bytes" =~ ^[0-9]+$ && "$baseline_inodes" =~ ^[0-9]+$ && "$deadline_ns" =~ ^[0-9]+$ ]] || exit 2
mkdir "$root/launch-claim"
stop_file=$root/launch-claim/stop
abort_file=$root/launch-claim/abort
monitor_pid=
watchdog_pid=
cleanup() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then
    trap '' TERM INT HUP
    touch "$abort_file" || true
    printf '%s\n' "$status" > "$root/failed-exit-code.txt" || true
  fi
  touch "$stop_file" || true
  if [[ -n "$monitor_pid" ]]; then wait "$monitor_pid" || true; fi
  if [[ -n "$watchdog_pid" ]]; then wait "$watchdog_pid" || true; fi
  if [[ -n "$monitor_pid" && ! -s "$root/launch-claim/owned-completion.json" ]]; then
    printf '%s\n' 'owned shutdown unverified' > "$root/shutdown-unverified.txt"
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
measure() {
  local vram counts bytes inodes now_ns
  now_ns=$(date +%s%N)
  (( now_ns < deadline_ns )) || { printf '%s\n' 'deadline' > "$root/cap-event.txt"; return 1; }
  vram=$(timeout --signal=TERM --kill-after=1s 6s uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/bounded-measurement.py" vram --phase evaluate --output "$root/launch-claim/vram.txt" >/dev/null 2>>"$root/sensor-errors.log" && cat "$root/launch-claim/vram.txt") || return 1
  counts=$(timeout --signal=TERM --kill-after=1s 6s uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/sample-task-root.py" --root "$task_root" --kind both 2>>"$root/sensor-errors.log") || return 1
  read -r bytes inodes <<< "$counts"
  [[ "$vram" =~ ^[0-9]+$ && "$bytes" =~ ^[0-9]+$ && "$inodes" =~ ^[0-9]+$ ]] || return 1
  printf '%s,%s,%s,%s\n' "$(date -u +%FT%TZ)" "$vram" "$((bytes-baseline_bytes))" "$((inodes-baseline_inodes))" >> "$root/resource-samples.csv"
  (( vram <= 21474836480 && bytes-baseline_bytes <= 1073741824 && inodes-baseline_inodes <= 100 )) || { printf '%s\n' 'VRAM, byte or inode cap exceeded' > "$root/cap-event.txt"; return 1; }
}
watchdog() {
  while [[ ! -e "$stop_file" ]]; do
    measure || { printf '%s\n' 'resource sample unavailable or cap exceeded' > "$root/watchdog-failure.txt"; return 1; }
    sleep 1
  done
}
# The first measurement is pre-inference. The native monitor independently caps RSS.
measure
remaining=$(( (deadline_ns-$(date +%s%N))/1000000000 ))
(( remaining > 1 )) || exit 1
setsid --wait uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/run-owned-phase-command.py" \
  --deadline-ns "$deadline_ns" --stop-file "$abort_file" --owner-pid "$$" \
  --completion "$root/launch-claim/owned-completion.json" -- \
  timeout --signal=TERM --kill-after=2s "${remaining}s" \
  uv run --locked --no-sync sparselab monitor \
  --policy "$root/monitor-policy.yaml" --log-dir "$root/monitor" \
  --workspace "$task_root" --reserve-bytes 1073741824 --reserve-inodes 100 --json -- \
  uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/run-card05-matched-diagnostic.py" \
  --declaration "$declaration" --declaration-sha256 "$declaration_sha" \
  --selected "$task_root/card05-full-tranche-v2/selected-checkpoint.json" \
  --runs "$task_root/card04-synthetic/runs" --output "$root/generations.jsonl" \
  > "$root/monitor-result.json" 2> "$root/monitor-error.log" &
monitor_pid=$!
watchdog &
watchdog_pid=$!
set +e
finished=
wait -n -p finished "$monitor_pid" "$watchdog_pid"
status=$?
set -e
if [[ "${finished:-}" != "$monitor_pid" ]] || (( status != 0 )); then exit 1; fi
touch "$stop_file"
wait "$watchdog_pid" || exit 1
measure
[[ ! -e "$root/cap-event.txt" && ! -e "$root/watchdog-failure.txt" ]] || exit 1
uv run --locked --no-sync python - "$root" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
completion=json.loads((root/'monitor/completion.json').read_text())
owned=json.loads((root/'launch-claim/owned-completion.json').read_text())
rows=[json.loads(x) for x in (root/'generations.jsonl').read_text().splitlines()]
assert completion['status']=='COMPLETE' and completion['returncode']==0 and not completion['violations']
assert owned['returncode']==0 and owned['living_descendants']==0
assert len(rows)==18 and [row['ordinal'] for row in rows]==list(range(18))
assert sum(len(row['token_ids']) for row in rows)<=1152
assert all(len(row['token_ids'])<=64 for row in rows)
PY
printf '%s\n' 0 > "$root/exit-code.txt"
