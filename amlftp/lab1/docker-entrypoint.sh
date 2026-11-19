#!/bin/sh
set -eu

mode="${1:-gui}"

case "$mode" in
  gui)
    exec uv run gui.py --weights /weights --host 0.0.0.0 --port 7860
    ;;
  train)
    shift
    exec uv run train.py --weights-dir /weights --reports-dir /reports "$@"
    ;;
  *)
    echo "Ожидался режим gui или train, получено: $mode" >&2
    exit 1
    ;;
esac
