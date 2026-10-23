#!/usr/bin/env bash
# 低蹲起立/键盘演示：RL平衡差模 + 显式航向控制，Q/E调有效腿长。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$SCRIPT_DIR/run.sh" play.py --allow-approximate-loop \
  --checkpoint "$SCRIPT_DIR/logs/standing_support_20261022_093823/input/standing_hold.pt" \
  --sim-device cpu --num-envs 1 --physics-threads 0 --initial-leg-length 0.05 \
  --viewer-width 1280 --viewer-height 720 \
  --wheel-control heading --heading-kp 6 --heading-kd 0.6 \
  --command 0 0 0 0.08 --keyboard --keyboard-speed 0.6 \
  --keyboard-turn-speed 2.0 --yaw-acceleration 4.0 --keyboard-leg-range 0.06 0.20 \
  --keyboard-leg-speed 0.02 --num-steps 100000 "$@"
