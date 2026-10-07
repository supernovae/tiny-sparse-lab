#!/usr/bin/env bash
# Card 04 operational wrapper; invoke only through the native attempt budget.
set -euo pipefail
umask 077

root=${KML_PROFILE_ROOT:?set KML_PROFILE_ROOT to a fresh external attempt root}
task_root=${KML_TASK_ROOT:?set KML_TASK_ROOT to the external Kernel Memory Lab root}
uuid=${KML_EXPECTED_GPU_UUID:?set KML_EXPECTED_GPU_UUID from device inventory}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT to the registered ROCm environment}
ledger=${SPARSELAB_ATTEMPT_BUDGET_LEDGER:?invoke through the native attempt budget}
checkout=${KML_CHECKOUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}
export KML_CHECKOUT=$checkout
config=$checkout/experiments/research/kernel-memory-lab/card04-synthetic-profile.yaml
reader=$checkout/tools/kernel-memory-lab/read-vram-bytes.py
phase=${1:?phase required}
case "$phase" in stage|train) ;; *) exit 2 ;; esac

[[ -d "$root" && -d "$task_root" && -f "$ledger" && -f "$config" && -f "$reader" ]] || {
  printf '%s\n' 'Missing attempt root, task root, ledger, config, or reader' >&2
  exit 2
}
[[ -d "$UV_PROJECT_ENVIRONMENT" ]] || {
  printf '%s\n' 'Registered ROCm environment is missing' >&2
  exit 2
}
root=$(realpath -e -- "$root")
task_root=$(realpath -e -- "$task_root")
ledger=$(realpath -e -- "$ledger")
case "$root/" in "$task_root/"*) ;; *)
  printf '%s\n' 'Attempt root must be inside the monitored task root' >&2
  exit 2
  ;;
esac
case "$ledger" in "$root/"*) ;; *)
  printf '%s\n' 'Attempt ledger must be inside the attempt root' >&2
  exit 2
  ;;
esac
cd "$checkout"
export SPARSELAB_WORK_DIR=$task_root
baseline_bytes=$(cat "$root/profile-baseline-bytes.txt")
baseline_inodes=$(cat "$root/profile-baseline-inodes.txt")
[[ "$baseline_bytes" =~ ^(0|[1-9][0-9]*)$ && "$baseline_inodes" =~ ^(0|[1-9][0-9]*)$ && ${#baseline_bytes} -le 18 && ${#baseline_inodes} -le 18 ]] || {
  printf '%s\n' 'Invalid task-root baseline' >&2
  exit 2
}
pgid=$(ps -o pgid= -p $$ | tr -d ' ')
[[ "$pgid" == "$$" ]] || {
  printf '%s\n' 'Launcher must own its process group' >&2
  exit 2
}

vram_cap_bytes=21474836480
added_bytes_cap=21474836480
added_inodes_cap=1000
monitor_pid=
# Claim before installing cleanup so a rejected replay cannot overwrite evidence.
mkdir "$root/$phase-launch-claim" || exit 2
# Native work stays in the attempt runner's group. Each bounded measurement
# uses timeout's own group so killing uv also kills a stuck Python reader.
# The leader survives TERM long enough to escalate, even for a TERM-ignoring child.
cleanup() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then
    trap '' TERM INT HUP
    if [[ ! -e "$root/$phase-cap-event.txt" ]]; then
      printf '%s\n' 'Supervisor, watchdog, or native command failed' > "$root/$phase-cap-event.txt" || true
    fi
    printf '%s\n' 137 > "$root/$phase-exit-code.txt" || true
    kill -TERM -- "-$pgid" 2>/dev/null || true
    sleep 1 || true
    kill -KILL -- "-$pgid"
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
stop_file=$root/$phase-launch-claim/stop
stop_for_cap() {
  printf '%s\n' "$1" > "$root/$phase-cap-event.txt"
  return 1
}
read_vram() {
  timeout --signal=KILL 5s uv run --locked --no-sync python "$reader" --expected-uuid "$uuid"
}
check_resources() {
  local used_bytes current_bytes current_inodes added_bytes added_inodes
  if ! used_bytes=$(read_vram 2>> "$root/$phase-device-errors.log"); then
    stop_for_cap 'Device-memory measurement unavailable'
    return 1
  fi
  if [[ ! "$used_bytes" =~ ^(0|[1-9][0-9]*)$ || ${#used_bytes} -gt 11 ]] || (( used_bytes > vram_cap_bytes )); then
    stop_for_cap "VRAM unavailable or above 20 GiB limit: $used_bytes bytes"
    return 1
  fi
  if ! current_bytes=$(timeout --signal=KILL 5s du -sbx "$task_root" | awk '{print $1}'); then
    stop_for_cap 'Disk measurement unavailable'
    return 1
  fi
  # Count NUL-delimited entries, so newlines in names cannot change accounting.
  if ! current_inodes=$(timeout --signal=KILL 5s find "$task_root" -xdev -printf '\0' | wc -c); then
    stop_for_cap 'Inode measurement unavailable'
    return 1
  fi
  if [[ ! "$current_bytes" =~ ^(0|[1-9][0-9]*)$ || ! "$current_inodes" =~ ^(0|[1-9][0-9]*)$ || ${#current_bytes} -gt 18 || ${#current_inodes} -gt 18 ]]; then
    stop_for_cap 'Disk or inode measurement invalid'
    return 1
  fi
  added_bytes=$((current_bytes-baseline_bytes))
  added_inodes=$((current_inodes-baseline_inodes))
  printf '%s,%s,%s,%s\n' "$(date -u +%FT%TZ)" "$used_bytes" "$added_bytes" "$added_inodes" >> "$root/$phase-resource-samples.csv" || return 1
  if (( added_bytes > added_bytes_cap || added_inodes > added_inodes_cap )); then
    stop_for_cap "Added disk/inodes cap: $added_bytes bytes, $added_inodes inodes"
    return 1
  fi
}
watchdog() {
  while [[ ! -e "$stop_file" ]]; do
    check_resources || return 1
    sleep 1
  done
}
# Validate native accounting and operational policy before any GPU-facing read.
timeout --signal=KILL 5s uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/validate-profile-phase.py" "$phase"
check_resources

if [[ "$phase" == stage ]]; then
  uv run --locked --no-sync sparselab monitor --policy "$root/profile-monitor-policy.yaml" --log-dir "$root/monitor-stage" --workspace "$task_root" --reserve-bytes 21474836480 --reserve-inodes 1000 --json -- uv run --locked --no-sync sparselab stage "$config" --through warmup --output "$root/stage-v3" --runtime rocm-7900xtx --resource-envelope "$root/profile-resource-envelope.yaml" > "$root/stage-monitor-result.json" 2> "$root/stage-monitor-error.log" &
else
  uv run --locked --no-sync sparselab monitor --policy "$root/profile-monitor-policy.yaml" --log-dir "$root/monitor-train" --workspace "$task_root" --reserve-bytes 21474836480 --reserve-inodes 1000 --json -- uv run --locked --no-sync sparselab train "$config" --stage-bundle "$root/stage-v3" --run-id kml-card04-synthetic-profile-v3 --runtime rocm-7900xtx --resource-envelope "$root/profile-resource-envelope.yaml" > "$root/train-monitor-result.json" 2> "$root/train-monitor-error.log" &
fi
monitor_pid=$!
watchdog &
watchdog_pid=$!
# Observe either child, rather than ignoring a dead watchdog until training ends.
set +e
finished=
wait -n -p finished "$monitor_pid" "$watchdog_pid"
status=$?
set -e
if [[ "${finished:-}" != "$monitor_pid" ]] || (( status != 0 )); then
  exit 1
fi
# Cooperative shutdown lets an in-flight sensor finish under its own timeout.
# Its status must be checked even when the native command finishes first.
touch "$stop_file"
wait "$watchdog_pid" || exit 1
check_resources
printf '%s\n' 0 > "$root/$phase-exit-code.txt"
