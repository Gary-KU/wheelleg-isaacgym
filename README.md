# WheelLeg · Isaac Gym Preview 4

五连杆轮腿机器人的 Isaac Gym 项目，包含独立运行所需的 URDF、网格、闭环近似、PPO 和精选权重。运行需要 NVIDIA Isaac Gym Preview 4 与 Python 3.8；安装和换电脑使用见[环境指南](docs/COMPANY_SETUP.md)。

## 先运行

在本项目 `IsaacGym/` 目录、已安装 Isaac Gym 的环境中执行：

```bash
conda activate isaacgym
./run.sh legged_gym/scripts/list_tasks.py
./run.sh legged_gym/scripts/play.py --task wheelleg_stand
```

默认加载仓库精选站立权重，使用 CPU、单机器人。点击仿真窗口后，方向键前后/转向，Q/E 调腿长，空格停止，R 重置。

| 入口 | 用途 |
| --- | --- |
| `./stand.sh` | 从 5 cm 低蹲撑到 8 cm，再用键盘驾驶与调腿长 |
| `./jump.sh` | 安静站立与小跳；空格单跳，X 停止，方向键驾驶，见[跳跃说明](docs/JUMP.md) |
| `./stairs.sh` | 已复测的 1 cm 四上四下阶梯演示 |
| `./lift.sh` | 原地交替抬轮；N 请求下一侧，空格切换连续动作；`--auto` 自动演示 |
| `./train_stairs.sh` | 从精选的 2 cm 课程起点继续 GPU 训练，写入新实验目录 |

新训练使用注册任务入口，例如从头训练平地任务：

```bash
./run.sh legged_gym/scripts/train.py --task wheelleg_flat --num_envs 128 --headless
```

`wheelleg_flat`、`wheelleg_stand`、`wheelleg_stairs` 分别对应平地指令、零速站立、阶梯课程。训练默认 `cuda:0`；加 `--max_iterations 3` 可做有限接口检查。续训、指定权重和参数说明见[任务入口](docs/LEGGED_GYM.md)。根目录 `train.py`、`play.py`、`evaluate.py` 和现有 shell 脚本继续可用。

## 代码从哪里读

Python 源码统一放在 `legged_gym/`：`envs/wheelleg/` 负责当前机器人，`learning/` 负责共用训练，`utils/` 负责注册与配置。建议按以下顺序阅读：

1. [wheelleg_config.py](legged_gym/envs/wheelleg/wheelleg_config.py)：机器人任务默认值和 PPO 配置。
2. [wheelleg.py](legged_gym/envs/wheelleg/wheelleg.py)：任务配置适配、显式奖励方法与六项返回接口。
3. [wheelleg_robot.py](legged_gym/envs/wheelleg/wheelleg_robot.py)：`WheelLegRobot` 的物理、动作、观测、终止和重置。
4. [model/](legged_gym/envs/wheelleg/model/)、[control/](legged_gym/envs/wheelleg/control/)、[learning/](legged_gym/learning/)：几何闭环、腿轮控制和共用 PPO 循环。

完整阅读路线与动作/观测单位见[代码导读](docs/CODE_GUIDE.md)，目录职责见[项目结构](docs/PROJECT_LAYOUT.md)，添加任务、奖励或机器人见[扩展指南](docs/EXTENDING.md)。现有启动命令保留；Python 导入改用统一的 `legged_gym` 路径。

## 已验证范围

- 平地航向反馈已改善固定驾驶序列中的偏航；静止俯仰晃动和长时间慢漂仍待改善，见[运行与复测](docs/USAGE.md)。
- 精选阶梯策略通过了 1 cm 课程的有限复测；2 cm 及更高阶梯尚未通过验收，见[阶梯训练与报告](docs/STAIRS.md)。
- 抬轮采用现有轮子策略与显式腿部控制，依赖专用胎面代理和输入适配，尚未向前迈步，见[抬轮说明](docs/WHEEL_LIFT.md)。
- 五连杆采用弹簧阻尼外力近似闭合，具有有限刚度与闭合误差，见[资产说明](assets/wheelleg/README.md)和[SCUT 闭链核对](docs/SCUT_CLOSED_CHAIN.md)。

这次结构整理沿用已有权重，测试与复测结果见[整理验收记录](validation/readability_cleanup_work.json)；具体实验参数和测量留在[训练优化](docs/TRAINING_OPTIMIZATION.md)、[实验记录](docs/EXPERIMENTS.md)与[整理前 README](docs/archive/README_20261027_before_readability_cleanup.md)。

## 检查与资料

```bash
# 固定机身、零动作检查机构
./run.sh play.py --allow-approximate-loop --zero-actions --fix-base --num-steps 500
# 回归测试，不启动策略训练
./run.sh -m unittest discover -s tests -q
```

工具及其适用范围见[工具索引](tools/README.md)。Git 保留三份精选权重和部分评估摘要；其他本机 `logs/`、`validation/` 产物被忽略，历史文档中的部分路径可能不随克隆提供。历史状态文件中的 PID、`completed` 和训练轮数都应结合对应权重与评估报告理解。
