# legged_gym 框架入口

当前工程提供 `legged_gym` 的任务、嵌套配置、任务注册和 `rsl_rl.OnPolicyRunner` 训练入口。参考的是 [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym/tree/c354431e5633eb98d9663fa3fbe444201d2de383)；这里沿用常见的 `legged_gym` 包名。机器人仍使用本项目的五连杆模型、髋 PD、近似支撑前馈和弹簧阻尼闭环。

## 先运行，再读三个文件

```bash
conda activate isaacgym
# 本机路径；公司电脑进入克隆的 wheelleg-isaacgym 即可
cd /home/gary/wheelleg_work/WheelLeg_RL_IsaacLab-main/IsaacGym
./run.sh legged_gym/scripts/list_tasks.py
./run.sh legged_gym/scripts/play.py --task wheelleg_stand
```

播放默认加载仓库精选站立权重，使用 CPU、单机器人，点击仿真窗口后用方向键前后/转向、Q/E 调腿长、空格停下、R 重置。这是普通站立和行驶播放；原地抬轮仍用独立的 `./lift.sh`（N/空格），两套按键不同。

| 阅读顺序 | 文件 | 修改什么 |
| --- | --- | --- |
| 1 | [wheelleg_config.py](../legged_gym/envs/wheelleg/wheelleg_config.py) | 机器人任务的平地、站立、阶梯配置，使用继承覆盖基础默认值 |
| 2 | [wheelleg.py](../legged_gym/envs/wheelleg/wheelleg.py) | 任务、`_reward_...` 奖励、六返回值接口、旧检查点字段映射 |
| 3 | [legged_robot.py](../legged_gym/envs/base/legged_robot.py) | 真实物理步进、URDF、闭环力、状态/观测、终止与自动重置 |

基础默认值在 [legged_robot_config.py](../legged_gym/envs/base/legged_robot_config.py)。任务在 [envs/__init__.py](../legged_gym/envs/__init__.py) 中通过 `task_registry.register(...)` 注册；注册器位于 [task_registry.py](../legged_gym/utils/task_registry.py)。算法循环在 [on_policy_runner.py](../third_party/rsl_rl/runners/on_policy_runner.py)，PPO/网络/存储仍在同一份 `third_party/rsl_rl/` 中。

## 目录和调用关系

```text
legged_gym/
├── envs/
│   ├── base/
│   │   ├── base_config.py
│   │   ├── legged_robot.py          # 已迁入的完整物理环境
│   │   └── legged_robot_config.py   # 嵌套环境配置与 PPO 配置
│   ├── wheelleg/
│   │   ├── wheelleg.py             # 任务接口、奖励函数
│   │   └── wheelleg_config.py      # 平地 / 站立 / 阶梯配置继承
│   └── __init__.py                 # 注册任务
├── utils/                         # 配置、权重路径、随机种子、任务注册器
└── scripts/                       # train.py / play.py / list_tasks.py
third_party/rsl_rl/                 # ActorCritic / PPO / OnPolicyRunner
wheelleg_gym/                       # 几何、闭环、腿部控制等共用模块和旧入口兼容
assets/wheelleg/                    # 同一份模型，无重复复制
```

`train.py → task_registry → WheelLegged(cfg) → OnPolicyRunner.learn() → PPO`。`WheelLegged` 继承基础 `LeggedRobot`；任务负责配置和奖励，基础环境负责实际物理过程。旧的 `wheelleg_gym/env.py` 现在是兼容导入，物理主体在上面的新位置。

## 训练与继续训练

4060 上的起步命令（从头训练，持续运行到配置的迭代上限）：

```bash
./run.sh legged_gym/scripts/train.py --task wheelleg_flat --num_envs 128 --headless
```

短检查可以加 `--max_iterations 3`。GPU 默认 `cuda:0`；CPU 检查加 `--sim_device cpu --rl_device cpu`。目前验证了同设备 CPU 与 GPU；未验证物理和策略分处不同设备的训练。

```bash
# 同一实验最近保存的 run 和数值最大的 model_N.pt；后续新增 100 次更新
./run.sh legged_gym/scripts/train.py --task wheelleg_flat --num_envs 128 --headless \
  --resume --max_iterations 100

# 明确指定检查点继续训练
./run.sh legged_gym/scripts/train.py --task wheelleg_flat --num_envs 128 --headless \
  --checkpoint /绝对路径/model_100.pt --max_iterations 100
```

权重默认写入 `logs/<experiment_name>/<日期时间>_<run_name>/model_<已完成更新次数>.pt`，同目录有 `config.json`、`progress.json` 和 TensorBoard 事件。继续训练读取优化器、更新计数和课程状态；`--max_iterations` 表示这次新增次数。`--load_run` 选具体运行目录，`--checkpoint 100` 选对应数字编号；也可直接给检查点路径。

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

在 `WheelLeggedCfg.rewards.scales` 中设置某项系数；非零项才注册对应的 `_reward_<名称>()`，设为零会移除该项，未知名称会直接报错。平地奖励已从旧环境拆成这些函数，默认计算与旧奖励做了数值对照。阶梯沿用已验证的独立奖励项，再由任务配置决定启用和倍率。

奖励仍按每个 20 ms 控制步累计，**没有额外乘 dt**，以保持旧奖励量级。阶梯的 `progress` 等帮助函数内部已包含各自原有系数，配置中的 1.0 是其倍率。`only_positive_rewards` 默认为关闭。

198 维已经是 33 × 6 帧，252 维是 42 × 6 帧；当前前馈网络的 `obs_history_length=1`，不会再堆叠一次。任务的 `step()` 返回 `obs, privileged_obs, rewards, dones, infos, obs_history` 六项，与参考 Wheel-Legged-Gym runner 接口对应。旧入口保留五项接口。没有增加估计器、RNN 或其他未验证算法。

## 权重和兼容范围

三个精选权重保留原网络大小、动作顺序、观测尺度和力矩含义。新 runner 同时保存旧 `environment_config` / `training_config` 与新嵌套配置；新旧入口都可读取相应尺寸的前馈权重。加载时检查观测尺寸，198/252 不能混用。

旧入口 `train.py`、`play.py`、`stand.sh`、`stairs.sh`、`lift.sh` 继续保留。原项目已有的横滑、俯仰波动、近似闭环误差和步态限制仍需单独优化。SCUT 原生闭链方案的核对见 [闭链参考说明](SCUT_CLOSED_CHAIN.md)。

2026-10-27 验证：238 项测试，237 通过、1 跳过；CPU 8 环境训练 2 次后再续训 1 次，计数为 3；4060 GPU 128 环境训练 3 次后再续训 1 次，计数为 4。以上是框架短检查，新策略尚未收敛。旧站立权重连续 60 秒无物理失败、无超时重置；旧 1 cm 阶梯权重在 4 环境各 26 秒测试中全部通过一次。记录见 [框架迁移验收](../validation/legged_framework_work.json)。
