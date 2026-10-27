# 工具索引

日常训练与播放使用 [README 中的入口](../README.md#先运行)。本目录用于资产处理、检查点转换、专用评估和有限实验；命令从 `IsaacGym/` 执行。每个脚本可用 `./run.sh tools/脚本.py --help` 查看参数，USD 导出需独立 `pxr` 环境。

机器人计算和检查点转换实现现位于 `legged_gym/envs/wheelleg/`，工具脚本负责参数、运行与报告。训练循环位于 `legged_gym/learning/`，根目录命令仍可使用。模块位置见[项目结构](../docs/PROJECT_LAYOUT.md)。

## 当前诊断、转换与评估

| 脚本 | 用途 | 使用边界 |
| --- | --- | --- |
| [benchmark_closure.py](benchmark_closure.py) | 固定机身、零重力下对比闭合点载荷与步长 | 机构诊断；写报告，不改生产默认值 |
| [upgrade_observations.py](upgrade_observations.py) | 186→198 维权重与 Adam 动量补零 | 保留初始预测，生成新文件；新增输入仍需学习 |
| [prepare_support_checkpoint.py](prepare_support_checkpoint.py) | 保留轮子策略，清零髋输出，重置价值网络/优化器 | 创建 198 维支撑学习起点，拒绝覆盖源文件 |
| [prepare_stair_checkpoint.py](prepare_stair_checkpoint.py) | 198→252 维地形观测初始化，髋动作转为目标角修正 | 创建新任务起点，重置价值网络/优化器，拒绝覆盖 |
| [evaluate_support.py](evaluate_support.py) | 统计实际腿长、左右差异、被动限位、力矩、姿态与接地 | 默认 CPU、20 秒零速；区分启动与持续阶段，完整计数失败 |
| [evaluate_drive_stability.py](evaluate_drive_stability.py) | 固定静止、前后、启停与转向序列，统计航向和姿态 | 默认 CPU 单环境、28 秒；独立积分转向要求，保留终止前诊断 |
| [evaluate_stairs.py](evaluate_stairs.py) | 核对完整阶梯通过、轮子接地、机身支撑、倒地与超时 | 默认 CPU、4 环境各 26 秒；每个环境真实通过且无失败/超时才通过筛查 |
| [export_usd_asset.py](export_usd_asset.py) | 源 USD→URDF、网格、闭合点 JSON 与几何报告 | 离线资产工具；需 `pxr` 和源 USD，日常运行使用已导出资产 |

程序成功退出和控制质量分别判断；评估需查看失败、超时、接触、实际运动与完整协议结果。具体工况见[运行与驾驶复测](../docs/USAGE.md)、[阶梯评估](../docs/STAIRS.md#用完整课程评估)。

## 兼容实验流程

以下脚本仍围绕根目录 `train.py`、`evaluate.py` 的参数和实验文件格式组织流程。

| 脚本 | 用途 | 使用边界 |
| --- | --- | --- |
| [optimize_training.py](optimize_training.py) | 基线评估→可选观测升级→有限稳定性训练→同条件复测 | 仅支持 186/198 维前馈检查点；支持 `--dry-run`，实际运行会启动子训练并写新实验目录，见[训练优化](../docs/TRAINING_OPTIMIZATION.md) |
| [benchmark_parallelism.py](benchmark_parallelism.py) | 比较不同并行环境数的短 PPO 吞吐 | 读取共用引擎平铺的 `config.json` 与 `training_summary.json`，仍通过旧入口启动短训练；必须指定已有训练 PID，会暂挂它并在 `finally` 恢复 |

`benchmark_parallelism.py` 的环境数量、少量更新与计时只衡量该配置的采样吞吐，不能推断达到相同控制质量所需时间。运行前确认 PID、检查点与配置来自同一次实验；脚本读取 `config.json`，不读取嵌套的 `task_config.json`，也不比较收敛质量。

## 历史脚本

[archive/train_and_evaluate_186d.py](archive/train_and_evaluate_186d.py) 是历史 794 轮、186 维实验流程，原名 `train_and_evaluate.py`，已归档。它只透传部分旧环境参数，不完整恢复新增观测、稳定奖励、指令调度与评估物理参数，还写入固定的历史状态/报告路径。保留它用于追溯实验；新增工作使用明确的训练与评估命令。

## 常用检查

闭环载荷诊断：

```bash
./run.sh tools/benchmark_closure.py --static-duration 0.5 --driven-duration 1 \
  --report validation/closure_check_new.json
```

已包含的精选阶梯权重复测：

```bash
./run.sh tools/evaluate_stairs.py --allow-approximate-loop \
  --checkpoint logs/stair_training_20261022_103417/evaluation/verified_level0.pt \
  --stair-level 0 --seconds 26 --num-envs 4 --physics-threads 0 \
  --report validation/my_stair_level0.json
```

观测转换时明确源、目标文件，目标应为新路径：

```bash
./run.sh tools/upgrade_observations.py --source /实际路径/model_N.pt \
  --target /实际路径/model_N_with_leg_inputs.pt
```

重新导出资产时，在装有 `pxr` 的 Python 环境中显式传入源 USD 和输出目录：

```bash
python tools/export_usd_asset.py --source /实际路径/Robot_Model.usd \
  --output /新的输出目录/wheelleg
```

默认源路径面向相邻 Isaac Lab 工程；独立克隆不包含源 USD。先校验新导出的几何和闭合点，再决定是否替换运行资产。

`logs/` 与 `validation/` 的旧 PID 和状态 JSON 是历史记录。Git 保留精选权重和摘要，其余本机产物被忽略；文件边界见[项目结构](../docs/PROJECT_LAYOUT.md)。
