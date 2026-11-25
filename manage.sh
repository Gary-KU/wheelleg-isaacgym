#!/usr/bin/env bash
# 工程唯一启动入口：新增日常命令时在下面增加一条分支，实现放 scripts/。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

usage() {
  cat <<'HELP'
用法：./manage.sh <命令> [参数]

日常使用：
  preview        查看 CAD 模型（默认固定机身，约 10 秒退出）
  check          检查 URDF、网格、惯量和闭环初始姿态
  tasks          列出可用训练任务
  train          注册任务训练
  play           注册任务播放，需指定匹配新模型的权重
  stand          站立/驾驶/跳跃演示，需指定 198 维站立权重
  train-stairs   从头训练阶梯任务
  stairs         阶梯演示，需指定匹配权重
  lift           交替抬轮实验，需指定匹配权重

开发工具：
  smoke          有限步数的物理集成检查
  run            执行指定 Python 脚本或 -m 模块
  legacy-train   旧参数接口训练（供兼容工具使用）
  legacy-play    旧参数接口播放
  evaluate       旧参数接口评估

示例：
  ./manage.sh preview --length 0.10 --steps 15000
  ./manage.sh train --task wheelleg_sensor_tracking --num_envs 128 --headless
  ./manage.sh play --task wheelleg_sensor_tracking --checkpoint <新权重路径>
  ./manage.sh run tools/check_model.py --report validation/model_check.json

命令后加 --help 查看参数。先激活 isaacgym 环境，或设置 PYTHON_BIN。
HELP
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
runner="$ROOT/scripts/run.sh"
case "$command" in
  help|-h|--help) usage ;;
  preview) exec "$runner" scripts/preview_model.py "$@" ;;
  check) exec "$runner" tools/check_model.py "$@" ;;
  tasks) exec "$runner" scripts/list_tasks.py "$@" ;;
  train|play|stand) exec "$runner" "scripts/$command.py" "$@" ;;
  train-stairs) exec "$runner" scripts/train.py --task wheelleg_stairs --sim_device cuda:0 --num_envs 128 --headless "$@" ;;
  stairs) exec "$runner" scripts/legacy/play.py --allow-approximate-loop --sim-device cpu --physics-threads 0 --viewer-width 1280 --viewer-height 720 --command 0.25 0 0 0.10 --stair-min-level 0 --stair-level 0 --no-stair-curriculum --num-steps 100000 "$@" ;;
  lift) exec "$runner" scripts/legacy/lift.py --allow-approximate-loop --wheel-collision mesh --wheel-observations sagittal "$@" ;;
  smoke) exec "$runner" scripts/legacy/smoke_test.py "$@" ;;
  legacy-train) exec "$runner" scripts/legacy/train.py "$@" ;;
  legacy-play) exec "$runner" scripts/legacy/play.py "$@" ;;
  evaluate) exec "$runner" scripts/legacy/evaluate.py "$@" ;;
  run) exec "$runner" "$@" ;;
  *) printf '未知命令：%s\n执行 ./manage.sh --help 查看可用命令。\n' "$command" >&2; exit 2 ;;
esac
