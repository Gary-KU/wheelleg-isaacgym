# 项目结构

```text
manage.sh                    唯一运行入口
scripts/                     训练、播放、预览、站立、抬轮和物理检查
legged_gym/
  envs/wheelleg/
    register.py              唯一任务注册清单
    robot.py                 创建仿真、物理步进、状态与重置
    constants.py             单位、动作顺序和共享常量
    terrain.py               地形生成与高度查询
    config/                  改参数
      base.py                基础、站立、阶梯与 PPO 参数
      sensors.py             传感器任务与速度课程参数
      jump.py                跳跃控制参数
    tasks/                   定义任务行为
      base.py                注册任务接口、配置映射与奖励选择
      sensors.py             生成传感器策略观测
      tracking.py            停稳、直行与速度课程
    observations/sensors.py  IMU/编码器字段、归一化、噪声与延迟
    rewards/                 奖励计算
      flat.py                平地奖励
      stairs.py              阶梯奖励
    control/                 动作与输入控制
      commands.py            指令采样和平滑
      keyboard.py            键盘驾驶输入
      leg_control.py         腿部 PD 和支撑前馈
      wheel_control.py       轮子力矩与航向控制
      jump.py / wheel_lift.py 跳跃与交替抬轮状态机
    model/                   CAD 尺寸、运动学、初始姿态与闭环外力
    diagnostics/             训练、支撑与运动指标
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

任务配置统一在 `config/base.py` 和 `config/sensors.py`。`constants.py` 保存单位、关节顺序和共享常量，不再保存重复的 PPO 训练配置。模型参数的唯一来源是 `assets/wheelleg/loop_constraints.json`。

`compat/` 仍被抬轮、物理检查和专项评估调用，所以保留；它已不包含旧训练预设。旧训练/播放链路、权重转换、旧 USD 导出和历史一键实验已删除，历史内容从 Git 追溯。

阅读顺序见[代码导读](CODE_GUIDE.md)，运行命令见[USAGE.md](USAGE.md)，扩展方法见[EXTENDING.md](EXTENDING.md)。
