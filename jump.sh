#!/usr/bin/env bash
# 独立跳跃演示：空格单跳，X停止；既有站立/抬轮入口保持原语义。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" legged_gym/scripts/jump.py "$@"
