#!/bin/sh
set -eu
ROOT=$(dirname "$0")/..
if [ -x "$ROOT/.venv/bin/python3" ]; then
  exec "$ROOT/.venv/bin/python3" "$ROOT/code/test_step6.py" "$@"
fi
exec python3 "$ROOT/code/test_step6.py" "$@"
