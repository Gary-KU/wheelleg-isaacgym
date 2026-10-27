# 阶梯训练与复测

已加入五档阶梯地形、地形观测和 PPO 训练入口。**目前只验证了最简单的 1 cm 单阶课程；复杂阶梯尚未学会。**

修正为双轮进度与轮下地面参考后，默认第 80 轮策略配合航向反馈 `6 / 0.3`，在 4 个环境各运行 26 仿真秒，均完成一次四级上阶、平台、四级下阶，全程无物理失败或超时。此前同一策略在约 2 cm 单阶上没有完成课程，共发生 22 次俯仰失败；该 2 cm 数字属于修正前的历史复测。这里的改善包含航向控制器参数调整，不能归因于 RL 单独学会了爬楼梯。

| 固定工况 | 完整通过 | 物理失败 | 超时 | 原始报告 |
| --- | --- | --- | --- | --- |
| 当前代码：1 cm，航向反馈 `6 / 0.3` | 4 / 4 个环境 | 0 | 0 | [新复测](../logs/stair_training_20261022_103417/evaluation/verified_level0_wheel_reference.json) |
| 历史：1 cm，航向反馈 `2 / 0.1` | 3 / 4 个环境 | 1 次偏离通道 | 0 | [报告](../logs/stair_training_20261022_103417/stage2_model80_level0.json) |
| 历史：1 cm，航向反馈 `6 / 0.3` | 4 / 4 个环境 | 0 | 0 | [报告](../logs/stair_training_20261022_103417/stage2_model80_heading6_level0.json) |
| 历史：约 2 cm，航向反馈 `6 / 0.3` | 0 / 4 个环境 | 22 次俯仰失败 | 0 | [报告](../logs/stair_training_20261022_103417/stage2_model80_heading6_level1.json) |

以上是有限、无初态扰动的确定性复测，不代表任意速度、楼梯或扰动下均能通过。五连杆仍采用弹簧阻尼外力近似闭合，存在有限刚度和闭合误差。

## 查看效果

全部命令从本机 `IsaacGym/` 执行：

```bash
conda activate isaacgym
cd /home/gary/wheelleg_work/WheelLeg_RL_IsaacLab-main/IsaacGym
./stairs.sh
```

默认加载 `logs/stair_training_20261022_103417/evaluation/verified_level0.pt`，自动以目标 `0.25 m/s` 前进，目标有效腿长 `14 cm`，固定在第 0 档。窗口采用单机器人 CPU 物理、1280×720 画面。此文件保存已复测的第 80 轮策略和航向反馈 `6 / 0.3`，不会自动挑选最新训练权重。

想看其他权重，在末尾追加 `--checkpoint`；例如复现尚未通过的 2 cm 工况：

```bash
./stairs.sh \
  --checkpoint logs/stair_training_20261022_103417/stage2/model_80.pt \
  --heading-kp 6 --heading-kd 0.3 --stair-level 1
```

查看自己训练的结果时，把路径替换成该次实验实际存在的 `model_N.pt`。平地键盘与变腿长演示仍使用 `./stand.sh`，详见 [运行说明](USAGE.md#先看当前起立与站立效果)。

## 继续训练

```bash
./train_stairs.sh
```

脚本从 `logs/stair_training_20261022_103417/input/stage4_start.pt` 开始，使用本机 RTX 4060 的 `cuda:0`、512 个并行环境、无窗口训练，新增 400 次 PPO 更新，每 10 次保存。新结果写入 `logs/stairs/日期_时间/`。保存文件编号表示累计更新数，不能当作通过阶梯的证明。

本次起点把初始和最低难度设为第 1 档、最高为第 4 档，避免失败后一直退回已通过的 1 cm 课程。真实通过后升档，早期物理失败后降档，降档不会低于第 1 档。地形在创建时固定，只在回合重置后换到相应通道。

`stage4_start.pt` 是修正奖励后的训练起点；髋/轮探索标准差分别设为 `0.25 / 0.02`，PPO 每次更新使用 2 个 epoch、4 个 minibatch，航向反馈 `6 / 0.3`。它并未通过 2 cm 阶梯，来源与参数见[准备记录](../logs/stair_training_20261022_103417/input/stage4_preparation.json)。本轮实验进度写入 `logs/stair_training_20261022_103417/stage4/progress.json`，脚本另行启动的训练则查看各自目录的 `progress.json`。

第 3 阶段因进度度量问题提前停止，共完成 39 次更新、累计到第 119 轮，最后完整保存的检查点为第 110 轮；相关文件保留为历史。机身前倾而双轮卡在台阶前时，旧版仍可能按机身前移给进度奖励；机身投影跨过台阶也会提前抬高扫描参考。第 4 阶段改用双轮中心的世界 X 位移与世界 XY 速度计算阶梯进度/跟踪奖励，并以双轮下地面高度的均值作为扫描参考，防止把前倾当作爬上台阶。

继续自己的检查点，例如：

```bash
./train_stairs.sh --checkpoint logs/stairs/你的实验目录/model_N.pt --max-iterations 100
```

将示例路径换成实际文件。`--max-iterations 100` 表示再更新 100 次。续训会恢复优化器、探索噪声、训练参数和课程状态；不要在续训时使用 `--init-noise-std` 或 `--learning-rate` 重设初值。可以通过 `--log-dir` 指定新的输出目录。训练速度会受到同时运行的仿真与 GPU 程序影响。

## 地形与控制含义

| `--stair-level` | 名义单阶高度 | 课程 |
| --- | --- | --- |
| 0 | 1 cm | 等高、规则踏面，四上四下，最高平台 4 cm |
| 1 | 2 cm | 踏面宽度与单阶高度有固定变化 |
| 2 | 3.5 cm | 同上，整体增高 |
| 3 | 5 cm | 增加左右高度差 |
| 4 | 7 cm | 增加左右高度差，整体更高 |

第 1～4 档的单阶并非严格等高。地形种子固定，各环境生成不同的可复现变体；第 3、4 档左右两侧的台阶高度也有差别。每条课程四级上阶、顶部平台、四级下阶，终点相对起点为 `x=3.8 m`。高度不是整段楼梯的总高度。

阶梯策略使用 `42 × 6 = 252` 维观测：每帧原有 31 维，加左右实际腿长 2 维，再加前方 9 个地形高度点。采样位置是机体前方 `0.10 / 0.25 / 0.45 m` 与横向 `-0.12 / 0 / 0.12 m` 的组合，随航向旋转；高度减去双轮各自下方地面高度的均值。此处直接查询仿真地形，并非真实相机或雷达输入。地形参考修改不改变252维输入尺寸，仍会改变旧模型收到的数值，应重新评估。

`leg_control=hip_target` 下，前四维动作表示髋目标角增量：`clamp(action, -1, 1) × 0.2 rad`，叠加在目标腿长的对称逆解上，再裁剪到真实关节限位。髋 PD 为 `80 / 4`，另有每腿 `22.4649 N` 的近似支撑前馈；后两维仍由策略控制轮子力矩。最终主动关节力矩均限于 `±9 Nm`。因此这里是基础控制加 RL，髋动作语义与旧的纯力矩策略不同。

旧站立权重先经过 [准备工具](../tools/prepare_stair_checkpoint.py)，从 198 维扩展到 252 维。新增地形输入列置零以保留原有预测，髋输出头置零以启用新的角度动作语义，价值网络及优化器重新开始。转换本身不是训练。

## 用完整课程评估

```bash
./run.sh tools/evaluate_stairs.py --allow-approximate-loop \
  --checkpoint logs/stair_training_20261022_103417/evaluation/verified_level0.pt \
  --stair-level 0 --seconds 26 --num-envs 4 --physics-threads 0 \
  --report validation/my_stair_level0.json
```

更换检查点和 `--stair-level`，逐档复测；末尾添加 `--record-trace` 可记录每个控制步，报告会明显增大。评估默认 CPU、固定前进 `0.25 m/s`、腿长 `0.14 m`，禁用课程升降，保留 24 秒回合时间限制。`--seconds` 为每个环境的仿真时长，须是 `0.02 s` 的整数倍。

报告应同时查看：

- `verified_course_crossings`：经过独立检查的完整通过次数。
- `physical_failure_resets`、`termination_reason_counts`：物理失败次数及俯仰、横滚、出界、闭合误差等原因。
- `time_limit_resets`：未完成课程而正常超时的次数。
- `summary`：每个环境的最远进度、最高到达地面、倾角、腿长、力矩与闭合误差。
- `screening.stair_climbing_screen_passed`：本次有限工况是否所有环境均真正通过且没有物理失败或超时。

完整通过要求双轮到达终点、保持在通道内、轮子接触地面且高度合理、没有机身或连杆撑地，并在同一回合走过高处。只存活、只到第一阶、训练奖励变高，均不算学会爬阶梯。评估命令正常结束也不代表质量通过。

报告中的`forward_velocity_body_*`仍诊断机体坐标系速度；阶梯奖励的速度使用双轮中心的世界XY差分，两者含义不同。

历史 2 cm 工况主要在第一阶受阻后前倾，最终触发俯仰失败；这与已经完整爬过后偶发摔倒不同，后续应以完整通过率检验训练是否解决了问题。

## 按代码学习

| 想理解什么 | 文件与入口 |
| --- | --- |
| 日常播放、训练默认参数 | [stairs.sh](../stairs.sh)、[train_stairs.sh](../train_stairs.sh) |
| 五档阶梯网格、采样点、精确高度查询 | [terrain.py](../legged_gym/envs/wheelleg/terrain.py) 的 `StairTerrain` |
| 地形创建、42 维单帧观测、髋角动作与重置 | [env.py](../legged_gym/envs/wheelleg/wheelleg_robot.py) 的 `__init__()`、`_observations()`、`step()` |
| 支撑 PD 与前馈换算 | [leg_control.py](../legged_gym/envs/wheelleg/control/leg_control.py)、`env.py::step()` |
| 双轮中心进度/速度、轮下地面扫描参考 | `env.py::_kinematics()`、`_stair_diagnostics()`、`_observations()` |
| 进度、姿态、力矩、失败与通过奖励 | [stair_rewards.py](../legged_gym/envs/wheelleg/stair_rewards.py)、`env.py::_rewards()` |
| 真实通过、失败判据、课程升降 | `env.py::_stair_diagnostics()` 与 `step()` 的重置部分 |
| 预设与检查点恢复优先级 | [runtime_config.py](../legged_gym/compat/runtime_config.py) 的 `STAIRS_ENVIRONMENT`、[cli.py](../legged_gym/compat/cli.py) |
| PPO、课程状态保存、训练指标 | [training.py](../legged_gym/learning/ppo_runner.py) 的 `PPORunner` |
| 198→252 维起点转换 | [prepare_stair_checkpoint.py](../tools/prepare_stair_checkpoint.py) |
| 终止前诊断、独立通过检查与报告 | [evaluate_stairs.py](../tools/evaluate_stairs.py) |

先沿 `stairs.sh → play.py → runtime_config.py → env.py::step()` 看一次推理，再看 `terrain.py` 与 `_observations()`，最后阅读奖励和 PPO。闭环原理仍见 [代码导读](CODE_GUIDE.md#3-机器人与闭合连接)。
