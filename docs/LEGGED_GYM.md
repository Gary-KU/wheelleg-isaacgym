# legged_gym 任务入口

本项目借用 [Wheel-Legged-Gym 固定版本](https://github.com/clearlab-sustech/Wheel-Legged-Gym/tree/c354431e5633eb98d9663fa3fbe444201d2de383) 的任务注册、嵌套配置、奖励方法和 runner 接口惯例，沿用 `legged_gym` 包名，按当前轮腿任务实现所需部分。机器人使用本项目的五连杆模型、髋 PD、近似支撑前馈和弹簧阻尼闭环；新旧入口共用同一套物理、奖励计算和 PPO 训练循环。

## 先运行，再读三个文件

```bash
conda activate isaacgym
# 本机路径；公司电脑进入克隆的 wheelleg-isaacgym 即可
cd /home/gary/wheelleg_work/WheelLeg_RL_IsaacLab-main/IsaacGym
./run.sh legged_gym/scripts/list_tasks.py
./run.sh legged_gym/scripts/play.py --task wheelleg_stand
```

播放默认加载仓库精选站立权重，使用 CPU、单机器人，点击仿真窗口后用方向键前后/转向、Q/E 调腿长、空格停下、R 重置。这是通用播放入口；统一的安静站立、驾驶与空格跳跃使用 `./stand.sh`（X 停止），详见 [站立与跳跃](JUMP.md)。原地抬轮仍用独立的 `./lift.sh`（N/空格），两套按键不同。

| 阅读顺序 | 文件 | 修改什么 |
| --- | --- | --- |
| 1 | [wheelleg_config.py](../legged_gym/envs/wheelleg/wheelleg_config.py) | 机器人环境与 PPO 默认值，以及站立、阶梯的继承覆盖 |
| 2 | [wheelleg.py](../legged_gym/envs/wheelleg/wheelleg.py) | 任务、`_reward_...` 奖励、六返回值接口、旧检查点字段映射 |
| 3 | [wheelleg_robot.py](../legged_gym/envs/wheelleg/wheelleg_robot.py) | 真实物理步进、URDF、闭环力、状态/观测、终止与自动重置 |

日常默认值集中在 [wheelleg_config.py](../legged_gym/envs/wheelleg/wheelleg_config.py)。[register.py](../legged_gym/envs/wheelleg/register.py) 是当前三个任务的唯一注册清单；入口显式调用 [register_tasks()](../legged_gym/envs/__init__.py)，导入常量或工具不会自动加载机器人和 Gym。

## 目录和调用关系

```text
legged_gym/
  envs/base/base_config.py   嵌套配置实例化
  envs/wheelleg/            任务、配置、WheelLegRobot、常量、地形
    model/                 几何、闭环与初始姿态
    control/               腿轮控制、抬轮状态机
    diagnostics/           训练统计、评估、播放记录
  learning/                共用 PPO 循环、OnPolicyRunner 适配与策略创建
  utils/                   配置/权重工具与通用任务注册器
  compat/                  根目录命令的参数集与历史 profile
  scripts/                 注册任务 train.py / play.py / list_tasks.py
third_party/rsl_rl/        ActorCritic / PPO / rollout 存储与来源许可
assets/wheelleg/           同一份机器人资产
```

调用路线为 `register_tasks() → task_registry → WheelLegged(cfg) → OnPolicyRunner → PPORunner → PPO`。[WheelLegged](../legged_gym/envs/wheelleg/wheelleg.py) 继承 [WheelLegRobot](../legged_gym/envs/wheelleg/wheelleg_robot.py)，负责配置映射、奖励选择和任务接口；机器人实现负责真实物理与状态。[OnPolicyRunner](../legged_gym/learning/on_policy_runner.py) 适配六项任务返回接口，调用 [learning/ppo_runner.py](../legged_gym/learning/ppo_runner.py) 的共用循环。

轮腿专用的 `configure_execution()` 与 `restore_legacy_environment()` hook 在任务类中处理播放/命令行规则及旧检查点配置；注册器不再直接导入轮腿实现。机器人训练指标由 `create_training_metrics()` 提供 collector，姿态、支撑、闭环和课程状态留在 `diagnostics/`。添加任务、奖励和新机器人见[扩展指南](EXTENDING.md)，完整目录见[项目结构](PROJECT_LAYOUT.md)。

## 训练与继续训练

4060 上的起步命令（从头训练，持续运行到配置的迭代上限）：

```bash
./run.sh legged_gym/scripts/train.py --task wheelleg_flat --num_envs 128 --headless
```

短检查可以加 `--max_iterations 3`。GPU 默认 `cuda:0`；CPU 检查加 `--sim_device cpu --rl_device cpu`。仿真与策略必须使用同一设备，当前明确拒绝 `--sim_device` 与 `--rl_device` 分处不同设备的配置。

```bash
# 同一实验最近保存的 run 和数值最大的 model_N.pt；后续新增 100 次更新
./run.sh legged_gym/scripts/train.py --task wheelleg_flat --num_envs 128 --headless \
  --resume --max_iterations 100

# 明确指定检查点继续训练
./run.sh legged_gym/scripts/train.py --task wheelleg_flat --num_envs 128 --headless \
  --checkpoint /绝对路径/model_100.pt --max_iterations 100
```

权重默认写入 `logs/<experiment_name>/<日期时间>_<run_name>/model_<已完成更新次数>.pt`。同目录的 `task_config.json` 保存注册任务的嵌套环境/训练配置；共用引擎生成的 `config.json` 保存兼容旧入口的平铺环境与训练配置。`progress.json` 是最近一轮的完整指标，包含 `Motion/`、`Support/`、`Reward/`、失败和闭环误差；同一引擎还写入 `training_summary.json` 与 TensorBoard 事件。继续训练读取优化器、更新计数和课程状态；`--max_iterations` 表示这次新增次数。`--load_run` 选具体运行目录，`--checkpoint 100` 选对应数字编号；也可直接给检查点路径。

```bash
./run.sh legged_gym/scripts/play.py --task wheelleg_flat \
  --checkpoint /绝对路径/model_100.pt
./run.sh legged_gym/scripts/play.py --task wheelleg_stand --headless \
  --num_steps 3000 --report validation/my_stand.json
```

`--sim_device`、`--rl_device`、`--num_envs`、`--max_iterations` 等同时接受连字符拼法。完整选项用 `./run.sh legged_gym/scripts/train.py --help` 和播放入口的 `--help` 查询。

## 三个任务

| 名称 | 观测 / 动作 | 任务与默认权重 |
| --- | --- | --- |
| `wheelleg_flat` | 198 / 6 | 平地变速/转向/腿长指令训练；直接播放时默认用精选站立权重作检查 |
| `wheelleg_stand` | 198 / 6 | 零速站立；默认播放 `standing_hold.pt` |
| `wheelleg_stairs` | 252 / 6 | 五档阶梯课程；默认播放已验证的 1 cm 权重 |

```bash
./run.sh legged_gym/scripts/play.py --task wheelleg_stairs --no_keyboard
./run.sh legged_gym/scripts/train.py --task wheelleg_stairs --num_envs 128 --headless
```

从头训练的阶梯任务默认开启课程；加载历史检查点会恢复它保存的课程配置。`--stair_level` 选择初始级别；播放关闭课程升级。任务并不会因为换框架就自动掌握 2 cm 或更高阶梯。

## 奖励和观测怎么改

在 `WheelLeggedCfg.rewards.scales` 中设置某项系数；非零项才注册对应的 `_reward_<名称>()`，设为零会移除该项，未知名称会直接报错。这些方法显式定义在任务类中，平地公式集中在 [rewards.py](../legged_gym/envs/wheelleg/rewards.py)，供任务和 `WheelLegRobot` 共用；阶梯项集中在 [stair_rewards.py](../legged_gym/envs/wheelleg/stair_rewards.py)，由任务配置决定启用和倍率。

奖励仍按每个 20 ms 控制步累计，**没有额外乘 dt**，以保持旧奖励量级。阶梯的 `progress` 等帮助函数内部已包含各自原有系数，配置中的 1.0 是其倍率。`only_positive_rewards` 默认为关闭。

198 维已经是 33 × 6 帧，252 维是 42 × 6 帧；当前前馈网络的 `obs_history_length=1`，不会再堆叠一次。任务的 `step()` 返回 `obs, privileged_obs, rewards, dones, infos, obs_history` 六项，适配后送入共用循环；`WheelLegRobot` 保留五项接口。当前只支持前馈 `ActorCritic/PPO`，没有接入估计器或 RNN。

## 权重和兼容范围

三个精选权重保留原网络大小、动作顺序、观测尺度和力矩含义。共用引擎保存旧 `environment_config` / `training_config`，注册任务适配层另附 `legged_gym_env_config` / `legged_gym_train_config`；新旧入口都可读取相应尺寸的前馈权重。加载时检查观测尺寸，198/252 不能混用。

根目录 `train.py`、`play.py`、`stand.sh`、`stairs.sh`、`lift.sh` 等启动命令继续保留，内部导入统一使用 `legged_gym`。原 `wheelleg_gym` Python 包和假通用物理基类已移除；自定义 Python 脚本需按[目录表](PROJECT_LAYOUT.md)更新导入。原项目已有的横滑、俯仰波动、近似闭环误差和步态限制仍需单独优化。SCUT 原生闭链方案的核对见 [闭链参考说明](SCUT_CLOSED_CHAIN.md)。

已有的 [框架迁移验收](../validation/legged_framework_work.json) 记录了当时的测试、CPU/GPU 短训练续训及精选权重复测。这些是对应配置下的有限检查；本次共用循环与目录整理继续沿用已有权重。控制能力与实际限制见[运行说明](USAGE.md)、[阶梯说明](STAIRS.md)和[抬轮说明](WHEEL_LIFT.md)。
