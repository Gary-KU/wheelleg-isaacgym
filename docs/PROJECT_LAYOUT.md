# 项目结构与扩展边界

Python 源码统一位于 `legged_gym/`。机器人代码按任务集中，共用训练、注册工具和历史参数入口单独放置。先从[README](../README.md)运行，再按[代码导读](CODE_GUIDE.md)读实现；添加任务或机器人见[扩展指南](EXTENDING.md)。

## 当前目录

```text
manage.sh                        唯一运行入口
scripts/                         train / play / stand / preview_model / list_tasks
  legacy/                        旧参数接口实现
  run.sh                         内部 Python 启动器
legged_gym/
  envs/
    __init__.py                  显式 register_tasks()，调用各机器人 register()
    base/base_config.py          嵌套配置实例化
    wheelleg/
      register.py                flat / stand / stairs / sensor / sensor_tracking 注册清单
      wheelleg_config.py         机器人环境与 PPO 默认值、任务覆盖
      wheelleg.py                任务接口、配置 hooks、显式奖励方法
      wheelleg_robot.py          WheelLegRobot：物理、观测、终止、重置
      constants.py               关节顺序、单位常量、旧入口默认值
      commands.py / keyboard.py  指令采样、平滑与键盘状态
      terrain.py                 阶梯生成与高度查询
      rewards.py                 平地奖励计算
      stair_rewards.py           阶梯命名奖励项
      observation_upgrade.py     186→198 维权重转换
      model/                     geometry.py / kinematics.py / constraints.py / initial_pose.py
      sensor_config.py           传感器站立与速度课程配置
      sensor_task.py             132 维可实机获取状态的任务
      tracking_task.py           停车/直行目标及速度课程
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
assets/wheelleg/                  同一份 URDF、网格、闭合点元数据与模型指纹
```

`envs/base/` 只保留 `BaseConfig`，物理类明确命名为 `WheelLegRobot`。几何、闭环、腿轮控制和诊断都在轮腿目录内，其他机器人可以建立自己的同级目录。

日常配置从 `wheelleg_config.py` 开始；`constants.py::TRAIN_CONFIG` 与 `compat/` 的 `--profile` 保留历史命令语义，可能与注册任务默认值不同。导入常量不会注册任务或导入 Gym；训练/播放入口先导入 `isaacgym`，再显式调用 `register_tasks()`。

## 训练与机器人如何连接

`learning/ppo_runner.py` 接收环境接口、策略/PPO 配置和可选 hooks，不导入轮腿几何或常量。`WheelLegRobot.create_training_metrics()` 返回 `diagnostics/training_metrics.py` 中的 collector，其 `reset()`、`update()`、`compute()` 注入姿态、支撑、闭环、奖励和课程统计；可选 checkpoint hooks 保存和恢复机器人状态。

注册器负责配置副本、检查点选择和执行参数；任务类的 `restore_legacy_environment()` 与 `configure_execution()` 负责机器人相关的旧配置导入、命令和地形覆盖。添加同一机器人的任务通常只需配置继承、任务继承与注册，不需要改训练循环。

## 继续使用的命令

| 入口 | 职责 |
| --- | --- |
| `scripts/legacy/train.py` | 原参数集和 `--profile` 训练，调用共用 `PPORunner` |
| `scripts/legacy/play.py`、`scripts/legacy/evaluate.py` | 现有播放与评估，供演示和工具调用 |
| `manage.sh preview` | 无策略的 CAD 模型检查 |
| `manage.sh stand`、`manage.sh stairs`、`manage.sh train-stairs` | 显式指定新权重播放，或从头训练 |
| `lift.py`、`manage.sh lift` | 原地抬轮专用入口 |
| `smoke_test.py` | 有限动作下的机构数值检查 |

这些命令已改用 `legged_gym` 内的新模块。旧 `wheelleg_gym` Python 导入路径和 `LeggedRobot` 配置别名已移除；自定义脚本应更新为上表目录中的实际模块。`WheelLegRobot.step()` 返回五项，`WheelLegged.step()` 加上已堆叠观测成为六项，由 `OnPolicyRunner` 适配到共用循环。

## 工具与历史记录

[tools/](../tools/README.md) 保存资产检查、转换和评估工具，运行统一经过 `manage.sh`。按用户清理要求，`tests/` 与 Git 跟踪的旧精选权重已移除；历史内容可从 Git 历史追溯。当前 CAD 模型需要重新训练，新权重必须通过模型指纹校验。
