#!/usr/bin/env bash
# 统一站立/驾驶/跳跃：空格单跳，X停止，方向键驾驶，Q/E调腿长。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" legged_gym/scripts/stand.py "$@"
