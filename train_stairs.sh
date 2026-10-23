#!/usr/bin/env bash
# 用保存的阶梯配置继续训练，始终写入新实验目录。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="$SCRIPT_DIR/logs/stair_training_20261022_103417/input/stage4_start.pt"
LOG_DIR="$SCRIPT_DIR/logs/stairs/$(date +%Y%m%d_%H%M%S)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" train.py --allow-approximate-loop \
  --checkpoint "$CHECKPOINT" --sim-device cuda:0 --num-envs 512 --headless \
  --max-iterations 400 --save-interval 10 --log-dir "$LOG_DIR" "$@"
