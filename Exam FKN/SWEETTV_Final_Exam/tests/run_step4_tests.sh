#!/bin/sh
set -eu
exec python3 "$(dirname "$0")/../code/test_step4.py" "$@"
