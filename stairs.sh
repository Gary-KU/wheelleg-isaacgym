#!/usr/bin/env bash
# 新 CAD 阶梯播放：必须通过 --checkpoint 指定匹配新模型的权重。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" play.py --allow-approximate-loop \
  --sim-device cpu --physics-threads 0 \
  --viewer-width 1280 --viewer-height 720 --command 0.25 0 0 0.10 \
  --stair-min-level 0 --stair-level 0 --no-stair-curriculum --num-steps 100000 "$@"
