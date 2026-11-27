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


## 后置电池模型的安静站立修正

2026-11-25：第 1300 轮策略在确定性 60 秒检查中没有倒地，但零速时前后速度 RMS 约 0.183 m/s、偏航角速度 RMS 约 0.678 rad/s，不能算安静站稳。该结论来自单机器人、seed=42 的有限检查；训练 reward 不能代替这项检查。

候选任务 `wheelleg_sensor_quiet` 的调整在 `config/sensors.py`，原 `wheelleg_sensor` 默认参数保留：缩小速度和偏航奖励容忍范围、增加零速跟踪和腿长支撑权重、增加零指令下的轮速软惩罚，并降低熵系数。轮速惩罚不会锁轮，运动指令下不启用。没有增加绝对位置/航向策略输入，也没有增加电机力矩上限。

这一早期候选曾同时调整髋阻尼，旧闭环参数下髋阻尼 0.04 在轻连杆上频繁限幅；同一冻结策略的短对照中，降低到 0.005 改善了腿长支撑和限幅占比，但不能直接解决漂移，必须重新微调并验证。该低髋阻尼调整现已撤回，见下方 2026-11-26 数值诊断。

**修改奖励和控制参数后使用 `--task wheelleg_sensor_quiet --warm-start` 开新实验**，保留 Actor 初始化并重新训练 Critic/优化器；`--checkpoint`/`--resume` 会恢复旧配置。原始权重和诊断报告保存在对应实验的 `diagnosis/` 下。新结果需要重新做连续站立、漂移、腿长、轮速和逐关节限幅检查：

```bash
./manage.sh run tools/evaluation/evaluate_sensor_balance.py --task wheelleg_sensor \
  --checkpoint <新权重> --headless --sim_device cpu --num_steps 3000 \
  --report logs/quiet_stand_result.json
```

候选任务目前仅完成有限微调和对照，不等于已经修复漂移；未通过连续站立门槛前，不恢复新的长训练或替换默认策略。

本次 60 轮候选微调的 60 秒确定性复测无倒地，腿长平均约 7.6 cm、偏航角速度 RMS 约 0.305 rad/s，但前后速度 RMS 约 0.711 m/s，比原策略更差，因此未采用为默认配置。候选第 50、60 轮均未通过安静站立门槛，不能把奖励数值或局部指标改善当作整体修复。


## 2026-11-26：先修正闭环数值模型，再训练

进一步固定机身、关闭策略和支撑前馈后发现：闭环阻尼 25 N·s/m 在新轻量 CAD、0.5 ms 外部步长下仍产生约 22 rad/s 的髋速度，原先降低髋阻尼只是缓解了电机限幅，未消除该振荡。

保持刚度 50000 N/m，把闭环阻尼改为 2 N·s/m 后，同一测试的髋速度降至约 0.006 rad/s，闭环误差由约 1.7 mm 降至 0.02 mm。独立无重力 20 N 拉开负载检查的静态间隙约 0.400 mm，符合 F/K；0.5 ms 与 0.25 ms 结果一致。原髋阻尼 0.04 在修正闭环后也能稳定运行，CPU/GPU 带前馈的固定机身检查均无重置且力矩限幅保持不变。

默认闭环参数改为 K=50000、D=2，配置统一引用 `constants.py`，候选站立任务恢复继承髋阻尼 0.04，并降低从头训练时的探索噪声。**旧检查点记录了 D=25，续训/播放会恢复旧参数；本次从头训练，不把旧策略当作修正后模型的结果。** 这些是近似闭环的数值参数，不是要写入 MCU 的电机参数，也没有改成原生闭链。

可复现的受载对照：

```bash
./manage.sh run tools/diagnostics/benchmark_closure.py \
  --case 50000,25,0.0005 --case 50000,2,0.0005 --case 50000,2,0.00025 \
  --report logs/closure_damping_comparison.json
```

静态或固定机身检查只验证物理模型，不能证明自由机身已经学会站立。新策略仍须通过前述连续站立、漂移、腿长和偏航检查。


站立候选增加了按 URDF 全机质量计算的初始俯仰（当前 8 cm 腿长约 2.2°），让重心投影靠近轮轴；只改变重置姿态，不固定机身或施加额外力。基础任务和旧检查点保留原重置方式。仅靠初始姿态仍不能平衡倒立摆。

`bootstrap_standing.py` 可把只读取 IMU/轮编码器的简单反馈写为现有 ELU 网络的初值：髋动作零，由原 PD/质量前馈支撑；轮子使用俯仰、俯仰角速度和偏航角速度反馈。网络推理没有额外教师控制器，也不读取位置或仿真线速度。**初始化文件不是 RL 训练成果**，元数据记录 `rl_updates_at_initialization=0`；随后才能用统一 PPO 入口微调。

```bash
./manage.sh run tools/policy/bootstrap_standing.py --output logs/my_standing/init.pt
# 此处恢复的是刚生成的新模型配置，不是旧 D=25 的检查点。
./manage.sh train --task wheelleg_sensor_quiet \
  --checkpoint logs/my_standing/init.pt --num_envs 32 --headless \
  --max_iterations 100 --log-root logs/my_standing/finetune
```

初始化的单环境、60 秒确定性检查（保留配置的传感器扰动）零倒地/零超时：前后速度 RMS 0.00178 m/s，横向 0.000103 m/s，偏航角速度 0.000456 rad/s，前后/横向位移峰值 5.7/1.4 mm，闭环误差峰值 0.168 mm。它只证明该初态的一次检查通过，不代表已经通过多初态、运动、跳跃或实机测试；PPO 微调结果必须重新评估，不能用训练轮数替代验收。记录位于 `logs/closure_fix_20261126_105745/`。


后续多环境反例：同一初始网络在 seed=17、4 个环境、60 秒中虽无倒地，却出现约 0.738 m/s 的前后速度 RMS；其 100 轮 PPO 微调也仍约 0.703 m/s，未采用。只加较弱轮速反馈（0.03）降至约 0.101 m/s，但仍不合格。该结果说明单环境通过不能当作鲁棒性证明。

进一步候选在网络初值中加入较强的轮编码器速度反馈、编码器左右差速反馈，并测试 5 ms 更新周期。差速转向估计基于平地近似滚动，打滑时不保证准确；这些仍只是网络初始化，没有在运行时额外叠加控制器。新周期的传感器单步延迟变为 5 ms，历史窗也变短；它不是保持 20 ms 延迟不变的单变量对照，今后部署应按真实采样/推理延迟重新配置和验收。暂未替代其他任务的默认周期。


同原点对照进一步发现：原 1.5 m 间距网格中的 4 环境、60 秒最差横向位移约 21.4 cm；只把 `env.spacing` 改为 0，让碰撞组隔离的机器人共用同一原点，同一个初始网络的 4 环境、60 秒检查全部通过，前后/横向位移峰值约 8.4/2.2 mm。该现象表明数值结果依赖环境坐标；固定机身诊断并不能覆盖这项接触与坐标精度问题。

当前候选默认 200 Hz、初始重心平衡、同原点，基础任务和旧检查点仍保留原布局。**同原点 GPU 模式目前只验证至 32 个并行环境**；128 个环境触发 PhysX `foundLostAggregatePairsCapacity` 警告，增大公开缓冲倍率未消除，相关实验不作为有效训练结果。构造器拒绝该模式超过 32 个 GPU 环境，防止继续使用带警告的数据。CPU 多环境也按碰撞组隔离；可视化请使用单环境，多个同原点机器人会在画面中重叠。不同任务的地形布置不受此项候选设置影响。


最终 32 环境 GPU 短训练无容量警告，完成 25 轮 PPO 微调。使用最终检查点在 CPU 确定性推理（保留传感器扰动），seed=17/42、各 4 个机器人分别连续 60 秒，8 个机器人均零倒地、零超时并通过全部安静站立门槛：最差前后位移峰值 8.3 mm，横向 2.3 mm，前后速度 RMS 不超过 0.00039 m/s，横向不超过 0.00022 m/s。这里是反馈初始化后的有限 RL 微调结果，不是从随机策略自主学会所有动作。

已验收本机权重：`logs/closure_fix_20261126_105745/standing_verified.pt`；完整结果为同目录 `standing_verified.json`。检查覆盖 8 cm 原地站立，未验证其他腿长、行走、跳跃或实机，也没有替换原 `stand` 的旧精选权重。

```bash
./manage.sh play --task wheelleg_sensor_quiet \
  --checkpoint logs/closure_fix_20261126_105745/standing_verified.pt --sim_device cpu
# 当前 5 ms 周期，12000 步是每个机器人 60 秒。
./manage.sh run tools/evaluation/evaluate_sensor_balance.py --task wheelleg_sensor_quiet \
  --checkpoint logs/closure_fix_20261126_105745/standing_verified.pt \
  --sim_device cpu --headless --seed 17 --num_envs 4 --num_steps 12000 \
  --report logs/my_standing/verified_60s.json
```

权重和实验日志保留在本机，GitHub 只同步源码和使用说明；在其他电脑可按前述初始化和训练命令重建，仍须在对应环境重新验收。

### 2026-11-26：转弯腿部阻尼修正

转弯对照发现左右腿长反相起伏（相关系数约 -0.93），伴随机身横滚。原地转向腿长标准差约0.28–0.32 mm，主要振荡约8 Hz，未触发电机限幅。单独将髋刚度2加倍到4改善很小；保持刚度2，把髋PD阻尼从0.04提高到0.08 N·m·s/rad，原地转向腿长波动下降约30–40%，0.6 m/s边走边转下降约15–25%。对照不含额外RL更新；传感器噪声、支撑前馈、闭环参数和力矩上限保持不变。

此处0.08是**髋关节PD阻尼**，闭环约束阻尼仍为2 N·s/m，两者单位和作用不同。Actor张量没有改变，修改的是检查点中保存的外部PD参数。部署此配置时MCU也要复现相同的PD、前馈及限幅，单独导出网络不会包含这次阻尼调整。原检查点仍会恢复自己的0.04，不会因修改任务默认配置自动更新。

最终验收和播放权重见 `docs/KEYBOARD_DRIVE.md` 的转弯阻尼修正节。除了正常峰值控制的运动检查，还测试了将2秒峰值额度设为耗尽、只使用0.32 N·m连续髋力矩的组合运动；通过仿真检查并不代表完成实机热验证。


### 1.0 rad/s转向版的部署差异

`logs/turn_refine_20261126/keyboard_play.pt` 中的yaw误差反馈校准已写入Actor第一层，MCU仍输入原132维传感器历史，不要再对yaw输入乘2。该版没有额外RL训练，转向参考上限1.0 rad/s，前后0.65 m/s，原指令斜坡不变。

该版另含网络外的左右腿差动阻尼。MCU需以仿真相同的关节坐标/速度符号和LF,LH,RF,RH顺序，在原髋PD（Kp=2、Kd=0.08）与支撑前馈上叠加：

```text
v_diff = 0.5 * (hip_velocity - reverse(hip_velocity))
tau_diff = -0.08 * v_diff
```

这里镜像对应LF↔RH、LH↔RF；`reverse`反转四个分量。共同前后运动的附加阻尼为0，左右反向运动耗散能量。每个物理/低层控制步以最新关节速度计算，叠加后统一执行原0.68 N·m峰值/2秒与0.32 N·m连续限幅。仿真闭环约束阻尼仍为2 N·s/m，URDF物理属性没有调整。只导出网络不包含该差动阻尼；原版缺少 `leg_differential_kd` 时取0，保持历史行为。

伸缩腿播放档另可启用 `control/standing_trim.py` 的慢速修正，详细顺序、门控和限幅见[键盘说明](KEYBOARD_DRIVE.md#伸缩腿与提速试用档)。MCU 需保存每台机器人的积分系数、低通轮速、腿长稳定计时和上一腿长；R/安全状态重置时一起清零。低通只供修正器使用，不替换网络原传感器历史。网络输入中的目标前进速度为用户请求经斜坡后加该修正，训练/评价中的目标仍保留原请求。网络结构与权重未因该播放配置改变；只移植网络文件无法复现整套混合控制。

另一个可选候选是轮角里程外环（`control.encoder_hold=True`）：Actor仍接收原132维历史，但MCU还需同步提供左右轮轴的原始编码器角度rad。CAN反馈已有电机角度，当前13维 `from_mylegwheel()` 适配器只输出轮速，部署该候选时须在生产者快照中另外暴露两个轮角及有效/时间戳信息，不可把缺失角度填0。补偿安装连杆旋转及伸腿平移后积分相对路程/航向；轮角跨±pi先计算最短角差。保存上一轮角、上一腿部投影、相对里程/航向、停车/直行参考及有效状态，失能/重置时清零并重新捕获，不跨越丢帧复用旧参考。原轮速积分修正与该模式互斥。它依赖平地滚动假设，未完成实机验证，见[轮角外环说明](KEYBOARD_DRIVE.md#可选轮角里程外环)。

该模式的 `encoder_hold_turn_reference_scale` 默认为1；若候选单独标定此参数，MCU需在原指令斜坡后计算 `(scale-1)*yaw_command`，与航向修正相加并统一限幅±0.2rad/s，然后叠加到策略转向参考。它不修改遥控器原目标，也不放宽电机输出；验收仍比较原目标。

可选 `encoder_hold_gyro_heading` 还需移植停车检测、至少1秒的陀螺积分与轮角航向差标定、零偏及航向状态。未标定时用轮角增量，运动时冻结零偏学习并积分去偏陀螺；它只改变外环状态估计，不要在MCU重复修改网络原IMU输入。`sensor_pitch_rate_kd` 则需在原策略轮力矩上叠加 `[-kd*pitch_rate,+kd*pitch_rate]`，之后统一执行原总力矩限制。上述选项默认关闭，需按候选配置移植，尚不代表实机验收通过。

若候选还启用 `sensor_pitch_guard_kp`，同样把 `kp * sign(pitch) * max(abs(pitch)-sensor_pitch_guard_angle,0)` 加到上述drive反馈，再映射到左右轮并统一限幅。它是轮力矩反馈，不是MCU或仿真机体的姿态钳制；默认kp为0，原检查点不启用。
