# 按 STM32 可获取状态训练

> 2026-11-25：活动资产已替换为新 CAD URDF。本文中旧权重路径、旧尺寸和性能记录属于替换前模型；请先阅读[当前模型与运行入口](../README.md)。新模型需重新训练。

`wheelleg_sensor` 是独立的平地训练任务。它在仿真中生成 IMU/编码器能够提供的观测，供以后部署使用；本次不读取实时串口、不向电机发送指令，也不修改或烧录 STM32 工程。

```bash
conda activate isaacgym
# 从头训练；4060 可先用64个并行环境
./manage.sh train --task wheelleg_sensor --num_envs 64 --headless

# 只做两次更新的接口检查，不代表已经学会站立
./manage.sh train --task wheelleg_sensor --num_envs 16 \
  --max_iterations 2 --headless

# 本任务自己的权重保存到 logs/wheelleg_sensor_balance/，续训同一任务
./manage.sh train --task wheelleg_sensor --num_envs 64 \
  --resume --max_iterations 100 --headless
```

训练后查看本任务的权重（将路径替换为自己的 `model_N.pt`）：

```bash
./manage.sh play --task wheelleg_sensor \
  --checkpoint logs/wheelleg_sensor_balance/你的运行目录/model_N.pt --keyboard
```

## 先站稳的训练配置

当前 `wheelleg_sensor` 默认固定8 cm腿长，前进和转向指令都为零，先学静止平衡；不会自动进入运动课程。站立达到要求后，再单独扩展速度和腿长范围。原来的随机运动实验仍在 `logs/wheelleg_sensor/`，新的平衡实验保存在 `logs/wheelleg_sensor_balance/`。

这次调整同时包含奖励与优化器配置，不能将新旧 reward 数值直接比较。姿态奖励使用 `exp(-(pitch²+roll²)/0.1²)`，15°附近不会再接近满分；速度/角速度跟踪采用较宽的早期训练尺度。直立、腿长等常量正奖励降低，倒地扣分由100改为10，减小早期critic回报量级；去掉对不可观测绝对航向的奖励约束。

初始actor输出层缩小到原来的0.01倍且偏置清零，髋部/轮子探索标准差分别为0.08/0.20；学习率从1e-4开始并限制上限为1e-4，每次更新3个epoch、4个minibatch。这是待持续评估的平衡配置，不是已经收敛的策略。

**从头启动本配置时不要加 `--resume` 或指定旧权重**，否则保存的旧指令、奖励与PPO配置会覆盖新默认值。新目录训练出的权重才使用上面的续训命令。已经运行中的Python进程不会自动采用文件修改，需要在原训练终端按Ctrl+C后重新启动。

新任务每帧22维，使用5帧历史加当前帧，共132维。字段顺序、缩放、历史帧数和传感器扰动配置一起保存在权重的 `legged_gym_env_config.sensors` 中。旧的198/252维权重不能直接加载；入口会拒绝观测尺寸不匹配的权重。原有 `manage.sh stand`、精选权重和其他任务保留原来的观测与控制方式。

## 已核对的固件来源

核对的是 [Gary-KU/mylegwheel](https://github.com/Gary-KU/mylegwheel/tree/2d395d6e69eb8c2a7b81ceace42066630b736aa0) 的固定提交 `2d395d6e69eb8c2a7b81ceace42066630b736aa0`，不是根据变量名猜测传感器。

- [imu_task.h](https://github.com/Gary-KU/mylegwheel/blob/2d395d6e69eb8c2a7b81ceace42066630b736aa0/App/Tasks/Inc/imu_task.h)：1 kHz IMU采样，姿态为度、原始传感器轴角速度为rad/s；一致性快照带时间戳和状态。
- [wheel_leg_observer.c](https://github.com/Gary-KU/mylegwheel/blob/2d395d6e69eb8c2a7b81ceace42066630b736aa0/App/Tasks/Src/wheel_leg_observer.c)：从已标定的四个关节反馈计算左右腿长、腿角以及雅可比速度，保留左右轮轴角速度和有效标记。
- [wheel_leg_config.h](https://github.com/Gary-KU/mylegwheel/blob/2d395d6e69eb8c2a7b81ceace42066630b736aa0/App/Tasks/Inc/wheel_leg_config.h)：俯仰来自传感器X轴的负方向，轮速方向为左负右正，转向反馈使用右转为正。

## 一帧输入及顺序

策略采用统一的物理方向：机头向下俯仰为正、向左转为正、两条腿向机身前方偏离下垂竖直为正。所有量先变为SI单位，再按下表缩放。左右顺序始终为左、右。

| 帧内索引 | 策略输入 | 对应固件状态 | 缩放 |
| --- | --- | --- | --- |
| 0 | 机身俯仰 rad | `body_pitch_rad` | 1 |
| 1 | 俯仰角速度 rad/s | `body_pitch_rate_rad_s` | 0.5 |
| 2 | 左转为正的偏航角速度 rad/s | `-yaw_feedback_direction * gyro_rad_s[2]`；核对版本方向为-1 | 0.5 |
| 3–4 | 左右腿相对下垂竖直的前偏角 rad | `pi/2 - leg[side].angle` | 1 |
| 5–6 | 左右腿长 m | `leg[side].length` | 5 |
| 7–8 | 左右腿前偏角速度 rad/s | `-angle_speed_rad_s[side]` | 0.05 |
| 9–10 | 左右伸腿速度 m/s | `length_speed_m_s[side]` | 5 |
| 11–12 | 左右前进方向轮轴角速度 rad/s | `wheel_speed_rad_s * [-1,+1]` | 0.05 |
| 13–15 | 目标前进速度、左转角速度、腿长 | 上层指令，不是测量值 | 1.5、1、5 |
| 16–21 | 上一拍六维策略动作 | 部署端缓存上一拍限幅后的网络动作，不是实测力矩 | 1 |

这是一份平面状态配置，没有假设已经完成IMU三维安装标定。原始传感器X/Y/Z不能直接当作机体X/Y/Z；如以后加入横滚或完整重力向量，应先完成安装坐标系标定。

腿角与腿长来自主动关节编码器正解，速度来自同一几何的解析雅可比，而不是从物理世界中的轮子位置或机体速度倒推。左右URDF的轴向不同，映射已用真实轮子运动方向检查；共享状态转换也有已知值、单位及符号测试。

## 不向策略提供的仿真真值

机体三维线速度、全局位置、绝对航向、地形高度扫描、接触力和被动关节真值不进入132维输入。原来依赖绝对航向误差的外部轮子反馈也在这个任务中关闭；轮子平衡和转向由策略决定，髋关节PD/支撑仍使用可由编码器获得的状态。

奖励、终止条件和训练评价仍可使用仿真真值，例如评价是否跟随实际前进速度。奖励中原本依赖旧观测列号的四项已改为读取明确的物理状态，速度评价也读取重置前的机体速度，避免新布局读错列。当前actor和critic使用同一132维输入；没有给critic额外添加特权观测。

`config/sensors.py` 提供逐传感器白噪声、每回合固定偏置和固定延迟。默认延迟1个控制周期，即20 ms。噪声数值是训练用示例，并非测量到的BMI088噪声或真实通信延迟，后续应根据实机日志标定。局部重置只重置该机器人的传感器队列；修改指令不重复采样传感器或推进历史帧。

## 当前还不能直接部署的部分

1. 模型不同：固件[几何](https://github.com/Gary-KU/mylegwheel/blob/2d395d6e69eb8c2a7b81ceace42066630b736aa0/Control/WheelLeg/Src/leg_pos.c)为短杆50 mm、长杆105 mm、髋距60 mm，轮半径26 mm；当前仿真为105/160/88 mm，胎面半径约62.5 mm。本次只对齐可获取状态，没有改URDF、质量惯量或力矩上限。观测可部署不代表当前仿真模型已经匹配实机。
2. 六维动作还需与真实电机顺序、零点、安装方向、力矩/电流后端、限幅和控制周期一致。当前仿真的9 Nm上限不能直接当作小型实机的安全限幅。
3. 现有[robot24遥测](https://github.com/Gary-KU/mylegwheel/blob/2d395d6e69eb8c2a7b81ceace42066630b736aa0/App/Services/Src/robot_telemetry_service.c)并不包含全部上述状态，默认ANO输出更只选择少数显示通道。支撑力、LQR力矩及电压字段是控制指令，不能当作测得的载荷/力矩。MCU内部已具备本任务所需状态；以后部署可直接在MCU中组帧，若由电脑推理则需扩展通信快照。
4. `from_mylegwheel()` 只转换完整、已验证的状态快照，检查有效标记和非有限值；它不是robot24或UART接收器。时间新鲜度、版本及掉线处理须由真正的接收/控制层实现。

## 代码位置

| 文件 | 职责 |
| --- | --- |
| [observations/sensors.py](../legged_gym/envs/wheelleg/observations/sensors.py) | 明确字段顺序、单位、归一化、固件状态转换、编码器正解/雅可比、延迟与噪声 |
| [config/sensors.py](../legged_gym/envs/wheelleg/config/sensors.py) | 新任务、传感器扰动和独立实验目录 |
| [tasks/sensors.py](../legged_gym/envs/wheelleg/tasks/sensors.py) | 在原物理后端上生成新观测，处理重置与指令更新 |

回归和短训练记录见[传感器任务验收](../validation/stm32_sensor_work.json)。这里只验证观测接口、CPU/GPU更新、保存/恢复及有限播放，没有声称已经训练出可站稳的新策略。

平衡配置的短训练对比见[平衡训练调整记录](../validation/sensor_balance_work.json)。

## 减少漂移并训练直行到1.0 m/s

旧第600轮权重虽然不倒地，零速度复测仍有漂移：2个环境、30秒，前向速度均值约0.058 m/s，速度误差RMS约0.079 m/s；相对初始直线最大横向偏移2.34 m、最大航向变化2.47 rad。没有倒地不等于原地停稳，横向偏移还可能由缓慢转向累计而来。

新增 `wheelleg_sensor_tracking`，继续使用相同132维传感器输入，不给策略增加全局位置或仿真线速度。奖励增加前向速度误差、横向速度、偏航角速度以及停车时轮速的软惩罚。轮子仍可以转动来维持平衡，没有固定机体或强行锁轮。

课程为 `0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0 m/s`，每个环境独立晋级。晋级考核使用单独的无动作探索回合，需要跑满16秒、至少300个稳定目标采样，停车级速度误差RMS<0.02 m/s、移动级<0.08 m/s，横向速度RMS<0.02 m/s、偏航角速度误差RMS<0.06 rad/s、结束时航向偏离<0.10 rad，且无物理失败、结束姿态合格。未达标不升速；物理失败降一级。移动阶段25%的回合仍练习停车，停车回合不能代替移动考核晋级。加速度限制0.4 m/s²。

从旧第600轮策略开始新课程，用 `--warm-start`，它只载入actor和探索标准差，保留新任务配置，并重新初始化critic、优化器和训练计数；普通 `--resume` 会恢复旧配置，不适合跨课程初始化。

```bash
./manage.sh train --task wheelleg_sensor_tracking \
  --warm-start logs/wheelleg_sensor_balance/2026-11-23_15-45-20_291561_/model_600.pt \
  --num_envs 64 --headless
```

当前第二版记录在 `logs/wheelleg_sensor_tracking_v2/`，第一版保留在 `logs/wheelleg_sensor_tracking/`。下次继续**同一课程**时才用：

```bash
./manage.sh train --task wheelleg_sensor_tracking \
  --resume --num_envs 64 --headless
```

检查 `Curriculum/max_speed_m_s`、`Curriculum/mean_speed_level`、实际 `Motion/mean_vx_m_s`、速度跟踪误差和横向速度。课程上限为1.0 m/s，不代表权重已经达到这个速度。

训练到相应级别后再用实际测量验收，示例（替换为本课程权重）：

```bash
# 零指令：检查停稳与漂移
./manage.sh play --task wheelleg_sensor_tracking \
  --checkpoint logs/wheelleg_sensor_tracking_v2/你的运行目录/model_N.pt \
  --command 0 0 0 0.08 --headless --num_steps 3000 --report /tmp/standing_motion.json

# 1.0 m/s直行：检查实际速度和偏航/横移，不仅看给定值
./manage.sh play --task wheelleg_sensor_tracking \
  --checkpoint logs/wheelleg_sensor_tracking_v2/你的运行目录/model_N.pt \
  --command 1.0 0 0 0.08 --headless --num_steps 3000 --report /tmp/forward_motion.json
```

播放报告增加 `motion`：前向速度均值、速度误差RMS、横向速度RMS、停车速度RMS、各连续回合相对起点的位移和航向变化；排除重置瞬移，速度统计跳过重置后的前2秒。跨多回合的最大位移不能证明连续站立，应同时看失败次数。课程检查点保存各环境级别，播放时冻结晋级，固定前进指令也从零按训练时的加速度斜坡进入。

代码：`tasks/tracking.py` 管理奖励与课程，`config/sensors.py` 管理阈值和速度级别，`diagnostics/motion_evaluation.py` 只做真值评价。没有修改URDF，也没有实机部署。

## 第二版：把晋级考核从探索训练中分离

第一版训练到580轮仍在停车级；独立2环境、30秒无探索复测，速度RMS约0.054 m/s、前后最大位移1.83 m、横向轨迹偏移0.57 m，没有倒地。它确实没有达到停车标准，不只是考核方式的问题。同时，训练中的探索轮子标准差仍约0.09，随机力矩会干扰严格停车考核，因此不能用带动作探索的训练回合直接判断最终表现。

现在普通探索回合只用于PPO学习，不再直接升降级。在第1次更新和每100次更新后，使用当前策略均值动作做最多16秒的单独考核。传感器噪声、偏置、延迟仍保留；只关闭策略动作探索。考核开始与结束都重置环境，结束后PPO重新读取重置观测；考核的状态、动作与回报不写入PPO storage。每台机器人只按考核中的第一次回合结束评分，考核移动级别时不会抽到停车任务。

门槛没有放宽，仍需满足停车/移动速度、横向速度、偏航、结束航向及姿态要求。升降级和保存的课程等级只由这些考核回合更新。

曾测试加大速度误差惩罚（归一化尺度0.20降到0.05）、降低value_loss_coef到0.1并取消熵奖励的组合。16环境100次更新后，完整无探索考核的平均前向速度误差RMS由约0.045升到0.072 m/s，0台晋级，候选未通过，因此这组奖励/PPO参数已经撤回。当前默认仍用原来的0.20尺度、value_loss_coef=1和entropy_coef=0.001，只保留独立考核修复。没有把失败试验权重作为默认模型。

第二版必须通过 `--warm-start` 开始新实验，不能用 `--resume` 加载第一版配置，否则旧奖励与优化器配置会覆盖新默认值。可选择自己的同布局权重；本机保留的80轮微调权重例子：

```bash
./manage.sh train --task wheelleg_sensor_tracking \
  --warm-start logs/sensor_tracking_initialization_20261123/tracking_refine_80.pt \
  --num_envs 64 --headless
```

`Starting curriculum assessment ...` 表示正在考核，期间不会输出普通iteration行。结果写到控制台、TensorBoard、progress.json和 `curriculum_evaluations.jsonl`，包括 `Assessment/promotions`、速度误差、物理失败次数和考核耗时。训练总步数只统计PPO数据，考核步数单独记录。普通探索回合的RMS可以高于考核RMS，两者不应混为一谈。

第二版验收记录见[独立课程考核修复](../validation/sensor_tracking_v2_work.json)。考核流程修复不等于停车控制已经解决，也不意味着已达到1.0 m/s。
