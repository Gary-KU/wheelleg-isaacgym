# 代码导读

日常从注册任务的训练、播放入口开始。机器人默认值在 [wheelleg_config.py](../legged_gym/envs/wheelleg/wheelleg_config.py)，共用单位和旧入口默认值在 [constants.py](../legged_gym/envs/wheelleg/constants.py)。加载权重时还会恢复它保存的配置，因此复现运行需要同时核对任务、检查点和显式参数。

日常手动演示统一运行 `./stand.sh`：入口为 [scripts/stand.py](../legged_gym/scripts/stand.py)，参数为 [jump_config.py](../legged_gym/envs/wheelleg/jump_config.py)，站立/停车/跳跃状态在 [control/jump.py](../legged_gym/envs/wheelleg/control/jump.py)。空格跳跃，X 停止；详细操作见 [站立与跳跃](JUMP.md)。

## 阅读路线

| 顺序 | 文件 | 关注什么 |
| --- | --- | --- |
| 1 | [wheelleg_config.py](../legged_gym/envs/wheelleg/wheelleg_config.py) | 平地、站立、阶梯任务的控制、观测、奖励和 PPO 参数 |
| 2 | [register.py](../legged_gym/envs/wheelleg/register.py)、[task_registry.py](../legged_gym/utils/task_registry.py) | 任务注册、独立配置副本、检查点恢复和命令行覆盖 |
| 3 | [wheelleg.py](../legged_gym/envs/wheelleg/wheelleg.py) | 嵌套配置到环境参数的映射、奖励注册、任务接口 |
| 4 | [wheelleg_robot.py](../legged_gym/envs/wheelleg/wheelleg_robot.py) | `__init__()`、`step()`、`_observations()`、重置与终止 |
| 5 | [kinematics.py](../legged_gym/envs/wheelleg/model/kinematics.py)、[constraints.py](../legged_gym/envs/wheelleg/model/constraints.py) | 五连杆几何、对称逆解、闭合点外力 |
| 6 | [leg_control.py](../legged_gym/envs/wheelleg/control/leg_control.py)、[wheel_control.py](../legged_gym/envs/wheelleg/control/wheel_control.py) | 髋 PD、支撑前馈、轮子平衡与航向控制 |
| 7 | [ppo_runner.py](../legged_gym/learning/ppo_runner.py)、[on_policy_runner.py](../legged_gym/learning/on_policy_runner.py) | 共用 PPO 循环与任务接口适配 |

只修改奖励时，先看任务的 `rewards.scales`、`WheelLegged._reward_<名称>()` 与 [rewards.py](../legged_gym/envs/wheelleg/rewards.py)；阶梯项的计算见 [stair_rewards.py](../legged_gym/envs/wheelleg/stair_rewards.py)。各目录职责见[项目结构](PROJECT_LAYOUT.md)，添加任务和机器人见[扩展指南](EXTENDING.md)。

## 从入口到更新

```text
legged_gym/scripts/train.py
  → register_tasks()：显式加载机器人任务注册表
  → task_registry：任务配置、检查点、设备
  → WheelLegged(cfg)：配置适配、奖励选择
  → WheelLegRobot：真实物理与状态
  → OnPolicyRunner：任务 API 适配
  → PPORunner：采样、PPO 更新、日志、保存
```

[learning/policy.py](../legged_gym/learning/policy.py) 按调用方传入的网络配置创建策略，提供共用推理辅助函数。网络、PPO 和 rollout 存储位于 `third_party/rsl_rl/`。注册任务与旧入口共用训练实现，训练从策略分布采样，播放与评估使用确定性输出。

`WheelLegged.step()` 返回六项：

```python
obs, privileged_obs, rewards, dones, infos, obs_history
```

`obs` 已包含当前帧和五帧历史；当前前馈策略的 `obs_history_length=1`，最后一项指向同一份已堆叠输入。根目录命令使用的 `WheelLegRobot.step()` 返回前五项，`privileged_obs` 均为 `None`。终止机器人会自动重置；评估使用 `infos` 的终止前状态、`terminal_observation` 与失败原因，避免把重置后的直立姿态计作终止姿态。

## 一个控制步

按以下顺序阅读 `WheelLegRobot.step()`：

1. 检查 `[num_envs, 6]` 动作，按腿轮控制模式计算目标和力矩。
2. 每个物理子步重新计算闭环外力、施加力矩、推进 PhysX、刷新刚体状态。
3. 更新腿长、姿态、机体速度、连续航向和堆叠观测。
4. 按本次动作使用的指令计算奖励、物理终止、超时与阶梯成功。
5. 保存终止前诊断、上一动作和力矩，只重置已结束的机器人。
6. 重采样并平滑下一步指令，更新最新观测字段；有窗口时绘制画面。

`WheelLegRobot` 的 `create_training_metrics()` 创建 [diagnostics/training_metrics.py](../legged_gym/envs/wheelleg/diagnostics/training_metrics.py) 中的 collector，向训练引擎注入姿态、支撑、闭环与课程统计。`reset()`、`update()`、`compute()` 负责每轮指标；可选 checkpoint hooks 保存和恢复课程状态。通用训练引擎不导入机器人常量或诊断代码。

默认物理步长 `0.0005 s`，控制周期 `0.02 s`，每次动作推进 40 个物理子步。指令加速度和航向积分使用仿真时间。

## 模型与闭合点

[assets/wheelleg/](../assets/wheelleg/) 保存 11 个刚体、10 个树关节的 URDF、网格、闭合点 JSON 和导出校验。主动关节通过力矩驱动，被动关节不继承位置 PD。

`ClosureForces` 从刚体状态求闭合点位置、点速度与铰链轴误差，施加成对力及绕质心的力矩。弹簧阻尼连接具有有限刚度，恢复力默认限于 `500 N`；轴对齐力矩保留绕共同轴的转动自由度。闭环误差须单独检查。

坐标为 X 前、Y 左、Z 上，Gym 四元数顺序为 **XYZW**。刚体位置参考连杆原点，线速度参考质心，外力力臂从质心量到锚点。详见[资产说明](../assets/wheelleg/README.md)与[闭链核对](SCUT_CLOSED_CHAIN.md)。

## 动作与控制

六维动作顺序由 `constants.py::ACTION_JOINTS` 固定，旧权重依赖该顺序：

| 下标 | 关节 | 旧力矩模式缩放 |
| --- | --- | --- |
| 0、1 | 左前髋 `L_F_Joint1`、左后髋 `L_H_Joint1` | `0.5 Nm / 动作单位` |
| 2、3 | 右前髋 `R_F_Joint1`、右后髋 `R_H_Joint1` | `0.5 Nm / 动作单位` |
| 4、5 | 左轮 `L_lun_Joint`、右轮 `R_lun_Joint` | `10 Nm / 动作单位` |

最终主动关节力矩裁剪到 `±9 Nm`。`leg_control=hip_pd` 使用腿长逆解目标、髋 PD 与可选支撑前馈；`hip_target` 将前四维改为 `clamp(action, -1, 1) × hip_target_scale` 的髋角增量，叠加在逆解目标上并裁剪到真实限位。阶梯默认角度系数 `0.2 rad`、髋 PD `80/4`、每腿近似支撑前馈 `22.4649 N`，后两维仍控制轮子。相同输出维度下，控制模式决定动作含义。

`wheel_control=heading` 用显式航向反馈处理转向：镜像轮轴下，策略差模 `drive=(右−左)/2` 提供前后平衡，共模提供转向。转向力矩限于 `±1 Nm` 且受 `9−abs(drive)` 的余量限制；内部航向参考相对实际航向限于 `±0.35 rad`。`stand.sh` 使用航向增益 `6/0.6`。`wheel_control=policy` 保留策略轮子输出，并可叠加配置的航向偏置；具体设置跟随任务和检查点。

## 指令与观测

指令为 `[vx, vy, yaw_rate, leg_length]`，单位为 `m/s`、`m/s`、`rad/s`、`m`。软件腿长目标范围为 `0.06–0.20 m`；注册平地任务默认随机范围、速度与重采样周期以任务配置为准。`commands.py` 处理采样与加速度限制，`set_commands()` 更新指令和最新帧，不推进物理或历史。

每帧布局如下，区间采用 Python 左闭右开写法：

| 区间 | 内容 | 缩放 |
| --- | --- | --- |
| `0:3` | 机体线速度 | × 1.5 |
| `3:6` | 机体角速度 | × 0.5 |
| `6:9` | 机体单位重力方向 | 无 |
| `9:13` | 四维指令 | × `[1.5, 0.5, 1, 5]` |
| `13:17` | 主动髋角相对腿长对称逆解的偏差 | 无 |
| `17:23` | 六个主动关节速度 | × 0.05 |
| `23:29` | 刚执行的六维动作 | 无 |
| `29` | 实际前后速度减目标速度 | × 1.5 |
| `30` | 连续航向减累计目标航向 | 无 |
| `31:33`，可选 | 左右实际腿长 | × 5 |
| `33:42`，阶梯且开启腿长输入 | 前方 9 点高度减轮下地面高度均值 | 裁剪到 `±0.5 m` 后 × 5 |

旧布局 `31×6=186` 维，开启实际腿长后 `33×6=198` 维，阶梯再加入 9 点地形成为 `42×6=252` 维。注册平地/站立任务默认 198 维，阶梯默认 252 维；旧入口保留 186 维默认。改变布局需要转换或重新训练权重，加载时会检查输入尺寸。[observation_upgrade.py](../legged_gym/envs/wheelleg/observation_upgrade.py) 负责 186→198，阶梯初始化见[准备工具](../tools/prepare_stair_checkpoint.py)。

## 奖励、终止与课程

任务通过非零 `rewards.scales` 注册显式 `_reward_<名称>()`，零值移除该项，未知名称提前报错。平地奖励计算集中在 `rewards.py`，供任务和 `WheelLegRobot` 共用；阶梯命名项在 `stair_rewards.py`。奖励按每个控制步累计，配置系数不再额外乘 `dt`。阶梯帮助函数中的项已包含原系数，任务的 `1.0` 是倍率。

平地物理终止检查俯仰/横滚 `0.35 rad`、闭合点误差 `0.03 m`、无效几何和非有限数；默认 16 秒回合。`time_outs` 只标记正常超时，PPO 按终止前观测进行超时估值。

阶梯使用双轮中心位移与速度衡量进度，以轮下地面均值作为扫描参考。俯仰阈值为 `0.65 rad`、横滚仍为 `0.35 rad`，另检查通道、停滞和机身碰撞，默认 24 秒回合。真正完成课程才升档，早期物理失败降档且受最低档限制。成功、超时、物理失败分别记录，评估方法见[阶梯说明](STAIRS.md#用完整课程评估)。

## 按需继续读

| 修改目标 | 文件 |
| --- | --- |
| 任务、网络与 PPO 默认值 | `legged_gym/envs/wheelleg/wheelleg_config.py` |
| 关节顺序、物理常量、观测尺寸常量 | `legged_gym/envs/wheelleg/constants.py` |
| 根目录命令的参数恢复与 `--profile` | `legged_gym/compat/runtime_config.py`、`cli.py` |
| 地形网格、扫描、课程判据 | `legged_gym/envs/wheelleg/terrain.py`、`wheelleg_robot.py::_stair_diagnostics()` |
| 键盘输入、播放日志 | `envs/wheelleg/keyboard.py`、`diagnostics/playback_log.py` |
| 原地抬轮 | `lift.py`、`envs/wheelleg/control/wheel_lift.py`，见[抬轮说明](WHEEL_LIFT.md) |
| 速度/姿态/接触评估 | `evaluate.py`、`envs/wheelleg/diagnostics/evaluation_metrics.py`、[专用工具](../tools/README.md) |

注册任务的 `task_config.json` 保存嵌套配置；共用引擎每次训练的 `config.json`、`progress.json`、`training_summary.json` 和 `model_N.pt` 分别保存解析配置、进度、统计与检查点。`--max-iterations` 表示新增更新次数；环境数改变会改变每轮样本量。保存轮数和测试通过只覆盖对应范围，控制质量还需匹配权重、物理配置和工况复测，见[运行说明](USAGE.md)与[训练优化](TRAINING_OPTIMIZATION.md)。
