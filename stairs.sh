#!/usr/bin/env bash
# 查看已通过初级阶梯复测的模型。可用 --checkpoint 指定其他权重。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR/logs/stair_training_20261022_103417"
CHECKPOINT="$EXPERIMENT_DIR/evaluation/verified_level0.pt"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" play.py --allow-approximate-loop \
  --checkpoint "$CHECKPOINT" --sim-device cpu --physics-threads 0 \
  --viewer-width 1280 --viewer-height 720 --command 0.25 0 0 0.14 \
  --stair-min-level 0 --stair-level 0 --no-stair-curriculum --num-steps 100000 "$@"
