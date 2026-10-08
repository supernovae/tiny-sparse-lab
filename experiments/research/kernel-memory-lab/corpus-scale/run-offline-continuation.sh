#!/usr/bin/env bash
# One-command wrapper for the approved C03 offline preparation continuation.
set -euo pipefail

record_dir=/srv/sparselab/state/experiments/kernel-memory-lab/card03-scale-operations/continuation1
workspace=/srv/sparselab/state
policy=/srv/sparselab/state/experiments/kernel-memory-lab/card03-scale-operations/retry1-monitor-policy.yaml
record="$record_dir/attempt.json"
[[ -f "$record" && -f "$policy" && $# -gt 0 ]] || exit 64

started=$(sed -n 's/.*"started_at_epoch":\([0-9]*\).*/\1/p' "$record")
[[ "$started" =~ ^[0-9]+$ ]] || exit 65
deadline=$((started + 28800))
remaining=$((deadline - $(date +%s)))
(( remaining > 0 )) || { echo 'Card 03 continuation deadline exhausted' >&2; exit 124; }

read -r free_blocks free_inodes block_size < <(stat -f -c '%a %d %S' "$workspace")
(( free_blocks * block_size >= 994399727616 && free_inodes >= 66936207 )) || {
  echo 'Card 03 combined disk or inode ceiling exhausted' >&2
  exit 125
}

export SPARSELAB_PREPARATION_ONLY=1
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SPARSELAB_WORK_DIR="$workspace/experiments/kernel-memory-lab"

run_name=${CONTINUATION_RUN_NAME:?Set a unique CONTINUATION_RUN_NAME}
[[ "$run_name" =~ ^[a-z0-9][a-z0-9-]*$ ]] || exit 65
log_dir="$record_dir/operations/$run_name"
[[ ! -e "$log_dir" ]] || { echo "Operation already exists: $run_name" >&2; exit 65; }

uv run --locked --no-sync sparselab monitor \
  --policy "$policy" --log-dir "$log_dir" --workspace "$workspace" \
  --cwd "$PWD" -- \
  timeout --signal=TERM --kill-after=10 "$remaining" "$@"
