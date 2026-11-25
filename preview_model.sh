#!/usr/bin/env bash
# 新 CAD 模型检查：固定机身观察机构，不加载旧策略。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" legged_gym/scripts/preview_model.py "$@"
