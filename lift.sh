#!/usr/bin/env bash
# 先站稳，再原地交替抬轮；现有RL仅负责轮子平衡。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" lift.py --allow-approximate-loop \
  --checkpoint "$SCRIPT_DIR/logs/standing_support_20261022_093823/input/standing_hold.pt" \
  --wheel-collision sphere_tread --wheel-observations sagittal "$@"
