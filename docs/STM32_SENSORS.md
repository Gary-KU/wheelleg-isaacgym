# 按 STM32 可获取状态训练

`wheelleg_sensor` 是独立的平地训练任务。它在仿真中生成 IMU/编码器能够提供的观测，供以后部署使用；本次不读取实时串口、不向电机发送指令，也不修改或烧录 STM32 工程。

```bash
conda activate isaacgym
# 从头训练；4060 可先用64个并行环境
./run.sh legged_gym/scripts/train.py --task wheelleg_sensor --num_envs 64 --headless

# 只做两次更新的接口检查，不代表已经学会站立
./run.sh legged_gym/scripts/train.py --task wheelleg_sensor --num_envs 16 \
  --max_iterations 2 --headless

# 本任务自己的权重保存到 logs/wheelleg_sensor_balance/，续训同一任务
./run.sh legged_gym/scripts/train.py --task wheelleg_sensor --num_envs 64 \
  --resume --max_iterations 100 --headless
```

训练后查看本任务的权重（将路径替换为自己的 `model_N.pt`）：

```bash
./run.sh legged_gym/scripts/play.py --task wheelleg_sensor \
  --checkpoint logs/wheelleg_sensor_balance/你的运行目录/model_N.pt --keyboard
```

## 先站稳的训练配置

当前 `wheelleg_sensor` 默认固定8 cm腿长，前进和转向指令都为零，先学静止平衡；不会自动进入运动课程。站立达到要求后，再单独扩展速度和腿长范围。原来的随机运动实验仍在 `logs/wheelleg_sensor/`，新的平衡实验保存在 `logs/wheelleg_sensor_balance/`。

这次调整同时包含奖励与优化器配置，不能将新旧 reward 数值直接比较。姿态奖励使用 `exp(-(pitch²+roll²)/0.1²)`，15°附近不会再接近满分；速度/角速度跟踪采用较宽的早期训练尺度。直立、腿长等常量正奖励降低，倒地扣分由100改为10，减小早期critic回报量级；去掉对不可观测绝对航向的奖励约束。

初始actor输出层缩小到原来的0.01倍且偏置清零，髋部/轮子探索标准差分别为0.08/0.20；学习率从1e-4开始并限制上限为1e-4，每次更新3个epoch、4个minibatch。这是待持续评估的平衡配置，不是已经收敛的策略。

**从头启动本配置时不要加 `--resume` 或指定旧权重**，否则保存的旧指令、奖励与PPO配置会覆盖新默认值。新目录训练出的权重才使用上面的续训命令。已经运行中的Python进程不会自动采用文件修改，需要在原训练终端按Ctrl+C后重新启动。

新任务每帧22维，使用5帧历史加当前帧，共132维。字段顺序、缩放、历史帧数和传感器扰动配置一起保存在权重的 `legged_gym_env_config.sensors` 中。旧的198/252维权重不能直接加载；入口会拒绝观测尺寸不匹配的权重。原有 `stand.sh`、精选权重和其他任务保留原来的观测与控制方式。

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

`sensor_config.py` 提供逐传感器白噪声、每回合固定偏置和固定延迟。默认延迟1个控制周期，即20 ms。噪声数值是训练用示例，并非测量到的BMI088噪声或真实通信延迟，后续应根据实机日志标定。局部重置只重置该机器人的传感器队列；修改指令不重复采样传感器或推进历史帧。

## 当前还不能直接部署的部分

1. 模型不同：固件[几何](https://github.com/Gary-KU/mylegwheel/blob/2d395d6e69eb8c2a7b81ceace42066630b736aa0/Control/WheelLeg/Src/leg_pos.c)为短杆50 mm、长杆105 mm、髋距60 mm，轮半径26 mm；当前仿真为105/160/88 mm，胎面半径约62.5 mm。本次只对齐可获取状态，没有改URDF、质量惯量或力矩上限。观测可部署不代表当前仿真模型已经匹配实机。
2. 六维动作还需与真实电机顺序、零点、安装方向、力矩/电流后端、限幅和控制周期一致。当前仿真的9 Nm上限不能直接当作小型实机的安全限幅。
3. 现有[robot24遥测](https://github.com/Gary-KU/mylegwheel/blob/2d395d6e69eb8c2a7b81ceace42066630b736aa0/App/Services/Src/robot_telemetry_service.c)并不包含全部上述状态，默认ANO输出更只选择少数显示通道。支撑力、LQR力矩及电压字段是控制指令，不能当作测得的载荷/力矩。MCU内部已具备本任务所需状态；以后部署可直接在MCU中组帧，若由电脑推理则需扩展通信快照。
4. `from_mylegwheel()` 只转换完整、已验证的状态快照，检查有效标记和非有限值；它不是robot24或UART接收器。时间新鲜度、版本及掉线处理须由真正的接收/控制层实现。

## 代码位置

| 文件 | 职责 |
| --- | --- |
| [sensor_observations.py](../legged_gym/envs/wheelleg/sensor_observations.py) | 明确字段顺序、单位、归一化、固件状态转换、编码器正解/雅可比、延迟与噪声 |
| [sensor_config.py](../legged_gym/envs/wheelleg/sensor_config.py) | 新任务、传感器扰动和独立实验目录 |
| [sensor_task.py](../legged_gym/envs/wheelleg/sensor_task.py) | 在原物理后端上生成新观测，处理重置与指令更新 |

回归和短训练记录见[传感器任务验收](../validation/stm32_sensor_work.json)。这里只验证观测接口、CPU/GPU更新、保存/恢复及有限播放，没有声称已经训练出可站稳的新策略。

平衡配置的短训练对比见[平衡训练调整记录](../validation/sensor_balance_work.json)。
