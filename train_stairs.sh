#!/usr/bin/env bash
# 新 CAD 阶梯任务从头训练；续训需显式提供匹配模型的 --resume --checkpoint。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" legged_gym/scripts/train.py --task wheelleg_stairs \
  --sim_device cuda:0 --num_envs 128 --headless "$@"
