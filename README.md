# WheelLeg · Isaac Gym Preview 4

五连杆轮腿机器人的旧版 Isaac Gym 项目，可独立于 Isaac Sim / Isaac Lab 运行。
这里包含导出的 URDF、闭环近似、环境、PPO 和播放入口，克隆后不需要上级 Isaac Lab 工程。

**换电脑使用：先看[公司电脑安装与学习指南](docs/COMPANY_SETUP.md)。** 仓库包含三份精选权重；NVIDIA Isaac Gym SDK 需要单独安装。

**当前状态：平地前后行驶偏航已修正，静止俯仰晃动和长时间慢漂仍待改善；已加入五档阶梯训练，复杂阶梯仍待学习。**
运行 `./stairs.sh` 查看已复测的 1 cm 单阶、四上四下演示；`./train_stairs.sh` 从 2 cm 档继续 GPU 训练。
当前阶梯代码采用双轮中心进度与轮下地面扫描参考；第80轮策略配合航向反馈6/0.3，经新复测在4个环境各26仿真秒内均完整通过1 cm课程，无物理失败或超时。
修正前的2 cm历史复测为0次通过、22次俯仰失败；后续 stage4 的400次新增更新已结束，最终策略在4环境、各26秒的2 cm复测中仍为0次通过、24次俯仰失败。当前没有训练运行，更高阶梯仍未通过验收。
这不代表已掌握复杂楼梯；默认演示固定使用已复测权重。命令、结果与代码入口见[阶梯训练](docs/STAIRS.md)。

在已激活的`isaacgym`环境中运行`./stand.sh`，可查看从5 cm低蹲撑到8 cm的效果。
方向键前后目标±0.6 m/s、自旋目标±2.0 rad/s；Q/E在6～20 cm之间缩腿/伸腿，松开保持目标。
单机器人窗口默认1280×720、CPU物理线程0；闭环计算精简后，本机热身后实测约53～57 FPS，原设置约34 FPS。
`stand.sh`现在显式使用`--wheel-control heading --heading-kp 6 --heading-kd 0.6`：保留RL轮子差模平衡，由显式航向反馈取代RL共模转向；髋PD和近似支撑前馈保持原配置，权重没有重训。
固定28秒驾驶序列中，前进/后退偏航误差RMS从10.78°/4.25°降至0.149°/0.087°，无物理失败，转向目标仍为2 rad/s；静止俯仰尚无改善。
复测命令见[运行说明](docs/USAGE.md#平地驾驶复测)，完整数据见[驾驶对比](logs/drive_stability_20261022/heading_gain_comparison_seq28.json)。
此前混合站立版本通过60秒零速检查：腿长约8.01～8.03 cm，俯仰RMS约0.67°、横滚RMS约0.09°，零速最大位置漂移约1.6 cm；
前向0.15 m/s速度扰动后保持站姿但位移约7 cm，未通过5 cm原位保持判据。
配置、报告与限制见[站立控制说明](docs/TRAINING_OPTIMIZATION.md)和[验收索引](validation/standing_work.json)。

此前的纯RL策略使用198维观测，从794轮续训至994轮，
前后俯仰与一般指令跟踪改善，但横滚增大、快速换向和低蹲仍有问题。
腿长观测、稳定奖励和回合内指令切换已接入 `--profile stability`，
使用方式见 [训练优化](docs/TRAINING_OPTIMIZATION.md)，实测见 [前后对比报告](logs/stability_optimization_20261021_215150/reviewed_comparison.md)。
闭合铰链使用弹簧阻尼近似，具有有限刚度和闭合误差。

原地慢漂已做 5 分钟连续检查，仍未根治；可选接触参数、实时位移显示和限制见[慢漂排查](docs/USAGE.md#原地慢漂排查尚未根治)。

## 先看什么

| 文档 | 内容 |
| --- | --- |
| [公司电脑安装与学习](docs/COMPANY_SETUP.md) | 克隆、环境安装、精选权重与日常同步 |
| [代码导读](docs/CODE_GUIDE.md) | 阅读顺序、控制流程、动作/观测布局、各模块职责 |
| [运行与参数](docs/USAGE.md) | 环境、模型检查、训练、键盘、评估、测试和兼容事项 |
| [阶梯训练与复测](docs/STAIRS.md) | 阶梯演示、GPU续训、五档地形、252维观测、真实通过判据与限制 |
| [实验与权重状态](docs/EXPERIMENTS.md) | 已有权重、测量结果、已知问题与未完成工作 |
| [训练优化与复测](docs/TRAINING_OPTIMIZATION.md) | 稳定性预设、配置恢复、超时估值修复和有限对比实验 |
| [工具索引](tools/README.md) | 资产导出、诊断、观测升级及历史自动流程 |
| [机器人资产说明](assets/wheelleg/README.md) | 连杆、坐标系、关节与闭合点数据 |

## 目录

```text
IsaacGym/
├── train.py                  # 训练入口
├── play.py                   # 模型检查 / 权重播放 / 键盘控制
├── evaluate.py               # 无窗口策略评估
├── smoke_test.py             # 有限动作下的机构数值检查
├── run.sh                    # 使用当前 Python 环境启动上述入口
├── stand.sh                  # 当前混合控制的低蹲起立与零速站立演示
├── stairs.sh                 # 已复测的1 cm阶梯演示，可指定其他权重
├── train_stairs.sh           # 从2 cm档继续GPU课程训练
├── wheelleg_gym/             # 自己实现的环境和训练适配
│   ├── config.py             # 默认参数、动作顺序、PPO 配置
│   ├── runtime_config.py     # 检查点恢复、稳定性预设与命令行覆盖
│   ├── env.py                # 仿真、观测、奖励、终止与重置
│   ├── terrain.py            # 五档四上四下阶梯与地形高度查询
│   ├── stair_rewards.py      # 阶梯进度、稳定性、失败与通过奖励
│   ├── leg_control.py        # 髋目标角、支撑PD与前馈
│   ├── wheel_control.py      # RL差模平衡、显式航向与力矩余量
│   ├── kinematics.py         # 五连杆几何计算
│   ├── constraints.py        # 弹簧阻尼闭环外力
│   ├── commands.py           # 指令采样、限加速度
│   ├── rewards.py            # 可选跟踪/稳定奖励
│   ├── training.py           # PPO 收集、更新、日志、权重
│   ├── keyboard.py           # 按键状态 → 目标指令
│   ├── evaluation_metrics.py # 评估调度与统计
│   ├── playback_log.py       # 按键与失稳事件记录
│   └── observation_upgrade.py # 186 → 198 维权重转换
├── assets/wheelleg/          # URDF、网格、闭合点参数
├── third_party/rsl_rl/       # 保留许可的 PPO / ActorCritic / rollout storage
├── tests/                   # CPU 数值、接口和 PhysX 回归测试
├── tools/                   # 辅助工具，非日常入口
├── docs/                    # 中文说明；archive/ 保存整理前 README
├── logs/                    # Git仅保留精选权重和报告；其他本机实验被忽略
└── validation/              # 精选评估摘要与历史状态索引
```

建议阅读顺序：`config.py → train.py → env.py → constraints.py → training.py`。
只想改奖励就先看 `env.py::_rewards()` 与 `rewards.py`；只想改键盘就看 `keyboard.py → play.py`。

## 常用入口

从本仓库根目录进入，使用已经安装 Isaac Gym 的 Python 3.8 环境：

```bash
conda activate isaacgym
cd wheelleg-isaacgym
./run.sh train.py --help
./run.sh play.py --help
./run.sh evaluate.py --help
```

固定机身查看机构，不加载策略：

```bash
./run.sh play.py --allow-approximate-loop --zero-actions --fix-base --num-steps 500
```

运行 CPU 回归测试，不进行策略训练：

```bash
./run.sh -m unittest discover -s tests -q
```

训练和评估的完整示例在 [运行说明](docs/USAGE.md)。播放默认使用 CPU 物理以减少单机器人开销，
训练默认使用 `cuda:0`；可通过 `--sim-device` 明确选择。
运行时先导入 `isaacgym`，再导入 `torch`，入口已处理这一要求。

## 修改前先确认

- 单帧观测默认31维，连同五帧历史共186维；开启腿长输入后为198维，阶梯权重再加9个地形点后为42×6=252维。
- 旧力矩模式按腿/轮系数转换动作；阶梯`hip_target`模式前四维改为髋目标角增量，后两维仍控制轮子。最终主动关节力矩裁剪到±9 Nm。
- 默认物理步长 0.0005 秒、控制周期 0.02 秒，每个控制步重新计算 40 次闭环外力。
- 续训/推理会恢复相关检查点环境参数；`--profile stability` 和显式命令行可覆盖，观测尺寸不符会提前报错，参见 [训练优化](docs/TRAINING_OPTIMIZATION.md)。
- `logs/` 和 `validation/` 是实验产物；历史 PID 或 `completed` 字段不代表当前仍在运行，也不代表策略稳定。

完整历史日志保留在原电脑；Git 只包含精选权重和报告。历史文档中的其他实验路径可能不随仓库提供。后续优化请使用独立实验目录，记录实际参数和前后复测结果。
