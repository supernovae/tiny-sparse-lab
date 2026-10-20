#!/bin/sh
set -eu
mkdir -p "$HOME" "$TMPDIR" "$SPARSELAB_WORK_DIR" "$SPARSELAB_WORKER_LEASE_DIR" \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" /state/evidence
# Bare words are Lab verbs, regardless of PATH executables or shell builtins.
# Explicit interpreter/shell commands and executable paths retain their argv.
if [ "$#" -eq 0 ]; then
    set -- sparselab --help
else
    case "$1" in
        sparselab|python|python3|sh|bash|streamlit|*/*) ;;
        *) set -- sparselab "$@" ;;
    esac
fi
exec "$@"
