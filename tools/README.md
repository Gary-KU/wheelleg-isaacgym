# 辅助工具索引

日常从上级目录的 `train.py`、`play.py`、`evaluate.py`、`smoke_test.py` 进入。
本目录是资产处理、诊断和实验辅助脚本，不应把工具执行成功当作机器人控制稳定。
以下命令均从 `IsaacGym/` 目录执行。

| 脚本 | 用途 | 依赖 / 状态 |
| --- | --- | --- |
| [export_usd_asset.py](export_usd_asset.py) | 原 USD → URDF、网格、闭合点 JSON、几何报告 | 离线资产工具；需要独立 `pxr` 环境，日常运行不需要 |
| [benchmark_closure.py](benchmark_closure.py) | 固定机身、零重力下的闭合点载荷与步长对比 | CPU 诊断；会生成报告，不改生产默认值 |
| [benchmark_parallelism.py](benchmark_parallelism.py) | 对比不同并行环境数的 PPO 采样吞吐 | **会运行短训练**；指定现有训练 PID 时会暂挂并在 finally 恢复，阅读代码后再执行 |
| [upgrade_observations.py](upgrade_observations.py) | 186 → 198 维观测权重转换 | CPU 文件转换；生成新文件，不覆盖源权重；转换不等于重新训练 |
| [optimize_training.py](optimize_training.py) | 基线评估 → 观测升级 → 有限稳定性训练 → 同条件复测 | 新流程；会实际启动训练，状态和对比报告写入新实验目录，见 [训练优化](../docs/TRAINING_OPTIMIZATION.md) |
| [prepare_support_checkpoint.py](prepare_support_checkpoint.py) | 保留轮子初始策略、清零旧折叠髋输出，重置价值网络/优化器，建立独立支撑学习起点 | 198 维 CPU 文件转换；新目标初始化，不是等价权重转换；拒绝覆盖 |
| [evaluate_support.py](evaluate_support.py) | 检查实际腿长、左右差异、被动关节止挡、髋力矩、姿态及接地 | 默认 CPU 20 秒零速；完整计数物理失败，区分启动和后续，程序退出成功不代表初筛通过 |
| [evaluate_drive_stability.py](evaluate_drive_stability.py) | 连续静止、前后、启停与左右转向，统计偏航误差、角速度、俯仰/横滚、速度误差及实际距离 | 固定CPU单环境、线程0、种子42、28秒；独立积分转向目标，终止前诊断与失败完整保留 |
| [prepare_stair_checkpoint.py](prepare_stair_checkpoint.py) | 198 → 252 维地形观测初始化，髋动作改为目标角修正，重置价值网络和优化器 | 创建独立新任务起点，不覆盖源文件；初始化不等于已经学会阶梯 |
| [evaluate_stairs.py](evaluate_stairs.py) | 固定阶梯级别，核对整段通关、轮子接地、机身支撑、倒地和超时 | 默认 CPU 4 环境、26 秒；每个环境真实通关且无失败/超时才通过有限筛查，见 [阶梯训练](../docs/STAIRS.md) |
| [train_and_evaluate.py](train_and_evaluate.py) | 历史 794 轮实验的有限训练/评估流程 | **旧配置实验工具**；只传递部分环境参数，尚未支持新增 198 维/稳定性配置完整续接 |

## 资产重新导出

一般不需要重新导出。原 USD 改动后，在包含 `pxr` 的独立 Python 环境中运行：

```bash
python IsaacGym/tools/export_usd_asset.py
```

该命令从仓库根目录执行，路径由脚本定位。会写出导出资产与校验报告，不改源 USD。
具体依赖与坐标转换见 [资产说明](../assets/wheelleg/README.md)。

## 闭环标定

```bash
./run.sh tools/benchmark_closure.py --static-duration 0.5 --driven-duration 1 --report validation/closure_check_new.json
```

比较相同载荷下的锚点位移、振荡、未截断恢复力和限力比例。它检查的是柔性闭环模型，
不能把结果当作刚性关节验证或策略成绩。

## 观测升级

```bash
./run.sh tools/upgrade_observations.py --source logs/forward_backward_1024_recovered_20261021/model_794.pt --target logs/model_794_with_leg_inputs.pt
```

转换器给每帧添加两列零权重，同时扩展 Adam 动量，使新增输入启用前的预测保持一致。
新权重尚未学会使用腿长；正式续训仍须传入对应环境参数，见 [运行说明](../docs/USAGE.md#续训与观测兼容)。

## 历史自动流程的边界

`train_and_evaluate.py` 会实际启动子训练并写入固定报告路径。目前不传递
`observe_leg_lengths`、`stability_rewards`、回合内重采样与限加速度等新增字段；
它的评估也不完整重建所有物理参数。当前仅保留用于理解历史实验，新增实验优先使用明确的训练/评估命令。

`validation/` 下旧 PID 和状态 JSON 是历史运行产物，不是后台任务管理服务。
