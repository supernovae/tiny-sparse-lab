#!/usr/bin/env bash
# Card 05 full-tranche wrapper; invoke only through the native attempt budget.
set -euo pipefail
umask 077

root=${KML_PROFILE_ROOT:?set KML_PROFILE_ROOT to a fresh external attempt root}
task_root=${KML_TASK_ROOT:?set KML_TASK_ROOT to the external Kernel Memory Lab root}
uuid=${KML_EXPECTED_GPU_UUID:?set KML_EXPECTED_GPU_UUID from device inventory}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT to the registered ROCm environment}
ledger=${SPARSELAB_ATTEMPT_BUDGET_LEDGER:?invoke through the native attempt budget}
checkout=${KML_CHECKOUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}
checkout=$(realpath -e -- "$checkout")
export KML_CHECKOUT=$checkout
config=$checkout/experiments/research/kernel-memory-lab/card05-full-tranche-v1.yaml
reader=$checkout/tools/kernel-memory-lab/read-vram-bytes.py
export KML_MEASUREMENT_KIND=card05_full
phase=${1:?phase required}
case "$phase" in stage|train|evaluate) ;; *) exit 2 ;; esac

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
export KML_PROFILE_ROOT=$root KML_TASK_ROOT=$task_root SPARSELAB_ATTEMPT_BUDGET_LEDGER=$ledger
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
baseline_digest=$(printf '%s\n%s\n' "$baseline_bytes" "$baseline_inodes" | sha256sum | awk '{print $1}')
[[ "$baseline_digest" == "$(cat "$root/profile-baseline-sha256.txt")" ]] || {
  printf '%s\n' 'Task-root storage baseline changed' >&2
  exit 2
}
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
added_bytes_cap=68719476736
added_inodes_cap=2000
monitor_pid=
watchdog_pid=
# Claim before installing cleanup so a rejected replay cannot overwrite evidence.
mkdir "$root/$phase-launch-claim" || exit 2
stop_file=$root/$phase-launch-claim/stop
abort_file=$root/$phase-launch-claim/abort
# A future baseline must come from the same live-tree sampler used below.
[[ "$(cat "$root/profile-baseline-sampler.txt")" == sample-task-root-v1 ]] || exit 2
cleanup() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then
    trap '' TERM INT HUP
    (set -C; printf '%s\n' 'Supervisor, watchdog, or native command failed' > "$root/$phase-cap-event.txt") 2>/dev/null || true
    printf '%s\n' 137 > "$root/$phase-exit-code.txt" || true
    touch "$abort_file" || true
  fi
  touch "$stop_file" || true
  # The owner supervisor adopts monitor/worker orphans across process groups.
  # Wait for its zero-survivor receipt before this launcher can return.
  if [[ -n "$monitor_pid" ]]; then
    wait "$monitor_pid" || true
    if [[ ! -s "$root/$phase-launch-claim/owned-completion.json" ]]; then
      printf '%s\n' 'Owned-process exit verification is missing' > "$root/$phase-shutdown-unverified.txt" || true
    fi
  fi
  if [[ -n "$watchdog_pid" ]]; then wait "$watchdog_pid" || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
phase_deadline_ns=$(uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/full-tranche-phase-deadline.py" "$phase")
[[ "$phase_deadline_ns" =~ ^[0-9]{18,19}$ ]] || exit 2
printf '%s\n' "$phase_deadline_ns" > "$root/$phase-launch-claim/deadline-ns.txt"
if [[ "$phase" == evaluate ]]; then
  eval_dir=$task_root/card04-synthetic/runs/kml-card05-full-tranche-v2/evaluations
  [[ -d "$eval_dir" ]] || exit 2
  eval_start_bytes=$(timeout --signal=TERM --kill-after=1s 6s uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/sample-task-root.py" --root "$eval_dir" --kind bytes)
  [[ "$eval_start_bytes" =~ ^[0-9]+$ ]] || exit 2
fi
stop_for_cap() {
  (set -C; printf '%s\n' "$1" > "$root/$phase-cap-event.txt") 2>/dev/null || true
  return 1
}
measure() {
  measurement_file=$root/$phase-launch-claim/measurement-$1.txt
  timeout --signal=TERM --kill-after=1s 6s uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/bounded-measurement.py" "$1" --phase "$phase" --output "$measurement_file" </dev/null > /dev/null 2>> "$root/$phase-measurement-errors.log"
}

check_resources() {
  local used_bytes current_bytes current_inodes added_bytes added_inodes now_ns eval_bytes
  now_ns=$(date +%s%N) || return 1
  if (( now_ns >= phase_deadline_ns )); then
    stop_for_cap 'Full-tranche phase deadline reached'
    return 1
  fi
  if ! measure vram; then
    stop_for_cap 'Device-memory measurement unavailable'
    return 1
  fi
  used_bytes=$(cat "$measurement_file") || return 1
  if [[ ! "$used_bytes" =~ ^(0|[1-9][0-9]*)$ || ${#used_bytes} -gt 11 ]] || (( used_bytes > vram_cap_bytes )); then
    stop_for_cap "VRAM unavailable or above 20 GiB limit: $used_bytes bytes"
    return 1
  fi
  if ! measure bytes; then
    stop_for_cap 'Disk measurement unavailable'
    return 1
  fi
  current_bytes=$(awk '{print $1}' "$measurement_file") || return 1
  # Count NUL-delimited entries, so newlines in names cannot change accounting.
  if ! measure inodes; then
    stop_for_cap 'Inode measurement unavailable'
    return 1
  fi
  current_inodes=$(cat "$measurement_file") || return 1
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
  if [[ "$phase" == evaluate ]]; then
    if ! eval_bytes=$(timeout --signal=TERM --kill-after=1s 6s uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/sample-task-root.py" --root "$eval_dir" --kind bytes); then
      stop_for_cap 'Evaluation output measurement unavailable'
      return 1
    fi
    if [[ ! "$eval_bytes" =~ ^[0-9]+$ ]] || (( eval_bytes - eval_start_bytes > 67108864 )); then
      stop_for_cap "Evaluation output exceeds 64 MiB: $eval_bytes bytes"
      return 1
    fi
  fi
}
watchdog() {
  while [[ ! -e "$stop_file" ]]; do
    check_resources || return 1
    sleep 1
  done
}
# Validate native accounting and operational policy before any GPU-facing read.
measure validate
check_resources
remaining_ns=$((phase_deadline_ns-$(date +%s%N)))
(( remaining_ns > 1000000000 )) || exit 1
phase_seconds=$((remaining_ns/1000000000))
supervisor=(uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/run-owned-phase-command.py" --deadline-ns "$phase_deadline_ns" --stop-file "$abort_file" --owner-pid "$$" --completion "$root/$phase-launch-claim/owned-completion.json" --)

if [[ "$phase" == stage ]]; then
  setsid --wait "${supervisor[@]}" timeout --signal=TERM --kill-after=2s "${phase_seconds}s" uv run --locked --no-sync sparselab monitor --policy "$root/profile-monitor-policy.yaml" --log-dir "$root/monitor-stage" --workspace "$task_root" --reserve-bytes 68719476736 --reserve-inodes 2000 --json -- uv run --locked --no-sync sparselab stage "$config" --through validate --output "$root/stage-validate" --runtime rocm-7900xtx --resource-envelope "$root/profile-resource-envelope.yaml" > "$root/stage-monitor-result.json" 2> "$root/stage-monitor-error.log" &
elif [[ "$phase" == train ]]; then
  setsid --wait "${supervisor[@]}" timeout --signal=TERM --kill-after=2s "${phase_seconds}s" uv run --locked --no-sync sparselab monitor --policy "$root/profile-monitor-policy.yaml" --log-dir "$root/monitor-train" --workspace "$task_root" --reserve-bytes 68719476736 --reserve-inodes 2000 --json -- uv run --locked --no-sync sparselab train "$config" --stage-bundle "$root/stage-validate" --run-id kml-card05-full-tranche-v2 --runtime rocm-7900xtx --resource-envelope "$root/profile-resource-envelope.yaml" > "$root/train-monitor-result.json" 2> "$root/train-monitor-error.log" &
else
  setsid --wait "${supervisor[@]}" timeout --signal=TERM --kill-after=2s "${phase_seconds}s" uv run --locked --no-sync sparselab monitor --policy "$root/profile-monitor-policy.yaml" --log-dir "$root/monitor-evaluate" --workspace "$task_root" --reserve-bytes 68719476736 --reserve-inodes 2000 --json -- uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/run-full-tranche-evaluation.py" > "$root/evaluate-monitor-result.json" 2> "$root/evaluate-monitor-error.log" &
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
if [[ "$phase" == train ]]; then
  remaining_ns=$((phase_deadline_ns-$(date +%s%N)))
  (( remaining_ns > 1000000000 )) || exit 1
  phase_seconds=$((remaining_ns/1000000000))
  setsid --wait uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/run-owned-phase-command.py" --deadline-ns "$phase_deadline_ns" --stop-file "$abort_file" --owner-pid "$$" --completion "$root/$phase-launch-claim/select-owned-completion.json" -- timeout --signal=TERM --kill-after=2s "${phase_seconds}s" uv run --locked --no-sync sparselab monitor --policy "$root/profile-monitor-policy.yaml" --log-dir "$root/monitor-select" --workspace "$task_root" --reserve-bytes 68719476736 --reserve-inodes 2000 --json -- uv run --locked --no-sync python "$checkout/tools/kernel-memory-lab/select-full-tranche-checkpoint.py" > "$root/select-monitor-result.json" 2> "$root/select-monitor-error.log" || exit 1
  check_resources
fi
printf '%s\n' 0 > "$root/$phase-exit-code.txt"
