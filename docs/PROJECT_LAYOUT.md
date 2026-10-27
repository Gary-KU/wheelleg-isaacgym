# 项目结构与扩展边界

Python 源码统一位于 `legged_gym/`。机器人代码按任务集中，共用训练、注册工具和历史参数入口单独放置。先从[README](../README.md)运行，再按[代码导读](CODE_GUIDE.md)读实现；添加任务或机器人见[扩展指南](EXTENDING.md)。

## 当前目录

```text
legged_gym/
  scripts/                       注册任务 train.py / play.py / list_tasks.py；统一演示 stand.py
  envs/
    __init__.py                  显式 register_tasks()，调用各机器人 register()
    base/base_config.py          嵌套配置实例化
    wheelleg/
      register.py                flat / stand / stairs 注册清单
      wheelleg_config.py         机器人环境与 PPO 默认值、任务覆盖
      wheelleg.py                任务接口、配置 hooks、显式奖励方法
      wheelleg_robot.py          WheelLegRobot：物理、观测、终止、重置
      constants.py               关节顺序、单位常量、旧入口默认值
      commands.py / keyboard.py  指令采样、平滑与键盘状态
      terrain.py                 阶梯生成与高度查询
      rewards.py                 平地奖励计算
      stair_rewards.py           阶梯命名奖励项
      observation_upgrade.py     186→198 维权重转换
      model/                     kinematics.py / constraints.py / initial_pose.py
      control/                   leg_control.py / wheel_control.py / wheel_lift.py
      diagnostics/               training_metrics.py / support_metrics.py
                                 evaluation_metrics.py / playback_log.py
                                 optimization_plan.py
  learning/
    ppo_runner.py                共用 PPO 采样、更新、日志、检查点
    on_policy_runner.py          注册任务配置与六项返回接口的薄适配
    policy.py                    按调用方配置创建前馈策略
  utils/                         通用参数、配置序列化、权重选择、任务注册器
  compat/                        根目录命令的 cli.py / runtime_config.py
third_party/rsl_rl/               ActorCritic / PPO / rollout 存储与来源许可
assets/wheelleg/                  同一份 URDF、网格、闭合点与导出校验
```

`envs/base/` 只保留 `BaseConfig`，物理类明确命名为 `WheelLegRobot`。几何、闭环、腿轮控制和诊断都在轮腿目录内，其他机器人可以建立自己的同级目录。

日常配置从 `wheelleg_config.py` 开始；`constants.py::TRAIN_CONFIG` 与 `compat/` 的 `--profile` 保留历史命令语义，可能与注册任务默认值不同。导入常量不会注册任务或导入 Gym；训练/播放入口先导入 `isaacgym`，再显式调用 `register_tasks()`。

## 训练与机器人如何连接

`learning/ppo_runner.py` 接收环境接口、策略/PPO 配置和可选 hooks，不导入轮腿几何或常量。`WheelLegRobot.create_training_metrics()` 返回 `diagnostics/training_metrics.py` 中的 collector，其 `reset()`、`update()`、`compute()` 注入姿态、支撑、闭环、奖励和课程统计；可选 checkpoint hooks 保存和恢复机器人状态。

注册器负责配置副本、检查点选择和执行参数；任务类的 `restore_legacy_environment()` 与 `configure_execution()` 负责机器人相关的旧配置导入、命令和地形覆盖。添加同一机器人的任务通常只需配置继承、任务继承与注册，不需要改训练循环。

## 继续使用的命令

| 入口 | 职责 |
| --- | --- |
| 根目录 `train.py` | 原参数集和 `--profile` 训练，调用共用 `PPORunner` |
| 根目录 `play.py`、`evaluate.py` | 现有播放与评估，供演示和工具调用 |
| `stand.sh`、`stairs.sh`、`train_stairs.sh` | 固定权重/参数的演示与续训命令 |
| `lift.py`、`lift.sh` | 原地抬轮专用入口 |
| `smoke_test.py` | 有限动作下的机构数值检查 |

这些命令已改用 `legged_gym` 内的新模块。旧 `wheelleg_gym` Python 导入路径和 `LeggedRobot` 配置别名已移除；自定义脚本应更新为上表目录中的实际模块。`WheelLegRobot.step()` 返回五项，`WheelLegged.step()` 加上已堆叠观测成为六项，由 `OnPolicyRunner` 适配到共用循环。

## 工具、测试与证据

[tools/](../tools/README.md) 区分当前诊断/转换/评估、兼容实验流程和归档脚本。[tests/](../tests/) 覆盖几何、控制、奖励、配置恢复、超时估值、任务接口和物理回归。新任务与机器人还需针对自身工况验证。

Git 中保留的三份检查点有明确用途：

| 文件 | 用途 |
| --- | --- |
| `logs/standing_support_20261022_093823/input/standing_hold.pt` | 平地站立/驾驶与抬轮演示的轮子策略 |
| `logs/stair_training_20261022_103417/evaluation/verified_level0.pt` | 已复测 1 cm 阶梯演示 |
| `logs/stair_training_20261022_103417/input/stage4_start.pt` | 2 cm 及以上课程的独立续训起点，尚未通过相应验收 |

`logs/` 另保留驾驶对比、稳定性比较和精选准备/评估报告；`validation/` 保留精选验收摘要。其余本机实验产物由 `.gitignore` 忽略。结构调整继续使用这些资产、权重和证据；历史文档中的部分本机路径可能不随克隆提供。

[整理前 README](archive/README_20261027_before_readability_cleanup.md) 保存当时完整的状态叙述和旧目录名称，历史正文保持原样。阅读历史数字与路径时以记录时间和对应检查点为准。
