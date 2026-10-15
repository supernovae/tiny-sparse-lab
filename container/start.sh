#!/bin/sh
set -eu
mkdir -p "$HOME" "$TMPDIR" "$SPARSELAB_WORK_DIR" "$SPARSELAB_WORKER_LEASE_DIR" \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" /state/evidence
exec "$@"
