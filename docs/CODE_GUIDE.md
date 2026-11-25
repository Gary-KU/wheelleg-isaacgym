# 按运行顺序读代码

先看这一条主线：**配置 → 模型 → 物理 → 控制 → 观测 → 奖励与重置 → PPO**。当前统一使用 2026-11-25 导入的 CAD URDF，腿长范围 8–13 cm，轮半径 32.5 mm；旧模型的训练结果不能直接使用。

## 先读哪些文件

| 顺序 | 文件 | 解决什么问题 |
| --- | --- | --- |
| 1 | [config/base.py](../legged_gym/envs/wheelleg/config/base.py) | 环境数量、物理步长、控制方式、指令范围、奖励和 PPO 参数 |
| 2 | [model/geometry.py](../legged_gym/envs/wheelleg/model/geometry.py) | 从资产元数据读取尺寸，给全部计算提供共同来源 |
| 3 | [model/kinematics.py](../legged_gym/envs/wheelleg/model/kinematics.py)、[initial_pose.py](../legged_gym/envs/wheelleg/model/initial_pose.py) | 髋角与腿长怎样转换，初始姿态怎样闭合并落在地面上 |
| 4 | [robot.py](../legged_gym/envs/wheelleg/robot.py) | 创建仿真、读取状态、执行 `step()`、终止和重置 |
| 5 | [control/leg_control.py](../legged_gym/envs/wheelleg/control/leg_control.py)、[wheel_control.py](../legged_gym/envs/wheelleg/control/wheel_control.py) | 髋 PD、径向支撑前馈、轮子力矩和航向控制 |
| 6 | [observations/sensors.py](../legged_gym/envs/wheelleg/observations/sensors.py)、[tasks/sensors.py](../legged_gym/envs/wheelleg/tasks/sensors.py) | 编码器/IMU 状态、噪声与延迟、132 维策略输入 |
| 7 | [rewards/flat.py](../legged_gym/envs/wheelleg/rewards/flat.py)、[tasks/tracking.py](../legged_gym/envs/wheelleg/tasks/tracking.py) | 奖励分项、停车/直行目标和速度课程 |
| 8 | [learning/ppo_trainer.py](../legged_gym/learning/ppo_trainer.py) | 采样、计算回报、PPO 更新、日志和保存 |

不必先读 `tools/`、`diagnostics/` 或 `compat/`。它们分别负责检查工具、统计报告和专项工具参数。

## 一次训练怎样启动

```text
scripts/train.py                 解析参数
  → utils/task_registry.py       选择任务，复制配置，校验检查点
  → tasks/base.py                  将配置接到物理环境和奖励接口
  → WheelLegRobot                创建 PhysX 场景和机器人
  → learning/task_runner.py      适配任务接口
  → learning/ppo_trainer.py      采样并训练
```

[register.py](../legged_gym/envs/wheelleg/register.py) 列出任务对应的环境类和配置类。传感器任务用 `config/sensors.py`，跳跃控制参数用 `config/jump.py`；子任务只覆盖自己的差异。

## 一个控制周期怎样执行

从 `WheelLegRobot.step()` 开始，按四段读：

1. `_prepare_control()`：六维动作转换为四髋目标和两轮力矩。
2. `_simulate_control()`：执行 40 个物理子步，每步更新髋 PD、闭环力并推进 PhysX。
3. 刷新状态和观测，计算奖励、失败、超时与阶梯成功；先保存终止前信息，再重置结束的环境。
4. 准备下一拍指令，有窗口时绘制画面。

默认物理步长为 `0.0005 s`（2 kHz），控制周期为 `0.02 s`（50 Hz）。RL 动作在整个控制周期保持；PD 和闭环反馈使用子步内最新状态。支撑前馈每个控制周期计算一次。

`WheelLegRobot.step()` 返回五项，任务 `WheelLegged.step()` 再提供第六项历史输入：

```python
obs, privileged_obs, rewards, dones, infos, obs_history
```

观测已经堆叠六帧，训练器不能再次堆叠。评估结束姿态要读取 `infos["pre_reset_..."]`，此时普通状态张量可能已被自动重置。

## 数据和单位

- 坐标：X 前、Y 左、Z 上；Gym 四元数为 **XYZW**。
- 指令：`[vx, vy, yaw_rate, leg_length]`，单位依次为 `m/s、m/s、rad/s、m`。
- 动作顺序：左前髋、左后髋、右前髋、右后髋、左轮、右轮，见 `constants.py::ACTION_JOINTS`。
- `hip_pd`：RL 提供额外力矩；`hip_target`：前四维提供髋角增量。最终力矩按 `loop_constraints.json::actuator_limits` 分别限幅；髋关节峰值 0.68 N·m、连续 0.32 N·m，累计超连续值 2 秒后本回合内降额。每个轮子限幅 ±0.1 N·m。
- 支撑前馈：径向力通过雅可比变成髋力矩，不是给机身施加向上的外力，也不是完整的分布质量重力补偿。
- 闭环：`model/constraints.py` 从两侧锚点计算成对弹簧阻尼力，仍存在有限闭合误差。

普通站立观测为 198 维，阶梯为 252 维；面向实机的传感器任务为 132 维，其字段和缩放只在 `observations/sensors.py` 定义。传感器任务的策略不读取绝对位置或绝对 yaw；奖励和验收指标可以使用仿真真值。

## 想改功能时去哪里

| 修改目标 | 入口 |
| --- | --- |
| 模型尺寸、碰撞、惯量 | `assets/wheelleg/`；同步检查 URDF、网格和闭环元数据 |
| 训练参数、动作模式、奖励权重 | `config/base.py` / `config/sensors.py` |
| 新增奖励计算 | 任务 `_reward_<名称>()`，公共计算放 `rewards/flat.py` |
| 地形生成与高度查询 | `terrain.py`；运行配置在 `cfg.terrain` |
| 方向键和腿长按键 | `control/keyboard.py`；跳跃/停车逻辑在 `control/jump.py` |
| 新增同一机器人的任务 | 配置继承 + 必要的环境覆盖 + `register.py` 注册 |
| 新增其他机器人 | 新建与 `wheelleg/` 同级的目录，提供自己的配置和模型校验 hook |

日常先运行 `./manage.sh preview --length 0.10` 检查模型，再决定是否训练。预览固定机身，默认约 10 秒退出；`--steps 15000` 约 5 分钟。训练和播放命令见[README](../README.md)，更详细的扩展规则见[EXTENDING.md](EXTENDING.md)。
