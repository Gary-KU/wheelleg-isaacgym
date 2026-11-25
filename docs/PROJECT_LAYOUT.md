# 项目结构

```text
manage.sh                    唯一运行入口
scripts/                     训练、播放、预览、站立、抬轮和物理检查
legged_gym/
  envs/wheelleg/
    wheelleg_config.py       基础任务与 PPO 参数
    sensor_config.py         传感器站立和速度课程参数
    wheelleg.py              注册任务接口与奖励选择
    wheelleg_robot.py        物理、状态、控制、终止与重置
    sensor_task.py           IMU/编码器观测任务
    tracking_task.py         停车、直行和速度课程
    model/                   CAD 尺寸、运动学、初始姿态、闭环外力
    control/                 腿部、轮子、跳跃与抬轮控制
    diagnostics/             训练、支撑与运动指标
    terrain.py               阶梯生成和高度查询
    rewards.py               平地奖励
    stair_rewards.py         阶梯奖励
    commands.py / keyboard.py 指令平滑与键盘输入
  learning/
    task_runner.py           任务配置、环境接口和检查点元数据适配
    ppo_trainer.py           采样、回报计算、PPO 更新、日志与保存
    policy.py                创建和初始化策略网络
  utils/                     注册器、配置、检查点校验
  compat/                    抬轮和专项评估仍在使用的参数解析/恢复
assets/wheelleg/              当前 CAD URDF、网格、闭环和电机参数
tools/                       模型检查、闭环诊断、专项评估
third_party/rsl_rl/           策略网络、PPO 和轨迹存储
```

训练只通过 `scripts/train.py`，播放和普通评估只通过 `scripts/play.py`。`manage.sh stairs`、`train-stairs` 只是同一入口的参数预设。

任务配置统一在 `wheelleg_config.py` 和 `sensor_config.py`。`constants.py` 保存单位、关节顺序和共享常量，不再保存重复的 PPO 训练配置。模型参数的唯一来源是 `assets/wheelleg/loop_constraints.json`。

`compat/` 仍被抬轮、物理检查和专项评估调用，所以保留；它已不包含旧训练预设。旧训练/播放链路、权重转换、旧 USD 导出和历史一键实验已删除，历史内容从 Git 追溯。

阅读顺序见[代码导读](CODE_GUIDE.md)，运行命令见[USAGE.md](USAGE.md)，扩展方法见[EXTENDING.md](EXTENDING.md)。
