#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    printf '%s\n' 'Usage: ./manage.sh run <script.py> [arguments]' \
        'Example: ./manage.sh run scripts/train.py --task wheelleg_flat --headless' \
        'Documentation: docs/LEGGED_GYM.md and docs/USAGE.md'
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
RUNTIME_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
export PATH="$RUNTIME_PREFIX/bin:$PATH"
# Isaac Gym's Python 3.8 extension needs libpython from the active environment.
export LD_LIBRARY_PATH="$RUNTIME_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" "$@"
