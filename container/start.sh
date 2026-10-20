#!/bin/sh
set -eu
mkdir -p "$HOME" "$TMPDIR" "$SPARSELAB_WORK_DIR" "$SPARSELAB_WORKER_LEASE_DIR" \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" /state/evidence
# Empty invocation prints help. Installed executables remain usable verbatim;
# otherwise treat argv as SparseLab verbs/options without shell interpolation.
if [ "$#" -eq 0 ]; then
    set -- sparselab --help
elif ! command -v "$1" >/dev/null 2>&1; then
    set -- sparselab "$@"
fi
exec "$@"
