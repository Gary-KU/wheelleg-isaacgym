# 传感器策略的键盘运动训练

`wheelleg_sensor_drive` 基于后置电池模型已经验收的站立配置，固定 8 cm 腿长，训练前进、后退、左右转向以及松键停车。左右是转向，不是横向平移。当前新任务默认速度为 ±0.3 m/s、转向为 ±0.8 rad/s，暂不代表达到 1.0 m/s 或学会跳跃。

键盘请求先经过速度斜坡，再进入网络的目标速度输入；网络输出六维动作，由原髋部 PD/质量前馈与电机力矩限幅执行。没有键盘直接覆盖轮子力矩，也没有向策略增加仿真位置、线速度或接触力。

## 训练与检查

```bash
conda activate isaacgym
# 用已验收的站立网络初始化新任务：调整转向阻尼与速度指令参考。
./manage.sh run tools/bootstrap_motion.py \
  --standing-checkpoint logs/closure_fix_20261126_105745/standing_verified.pt \
  --output logs/my_drive/motion_init.pt
./manage.sh train --task wheelleg_sensor_drive \
  --checkpoint logs/my_drive/motion_init.pt --num_envs 32 --headless \
  --max_iterations 50 --log-root logs/my_drive/train
```

初始化只改新网络副本的权重，不覆盖站立文件；它不是已经训练完成的运动策略。由于转向阻尼也调整过，零速稳定性必须重新验收。Actor 结构仍是原来的 132 输入、6 输出 ELU 网络，部署时不需要额外的教师控制器。每帧目标速度缩放沿用同一个 `COMMAND_SCALES`。

训练采样方向键单按、组合按以及 30% 的停车目标，每 3 秒重新采样；当前新任务前后加速度限制 0.4 m/s²，转向角加速度限制 1.8 rad/s²。训练和键盘播放使用同一斜坡。固定腿长训练期间，Q/E 不改变腿长。

```bash
# 四环境分别执行前进、后退、左转、右转，再松键、反向、松键。
./manage.sh run tools/evaluate_sensor_drive.py \
  --checkpoint <本任务权重> --report logs/my_drive/motion.json
# 加 --combined 可检查前进/后退与转向同时按下的四种组合。
# 另做连续60秒零速检查，不能把运动序列中的停车片段当作完整站立验收。
./manage.sh run tools/evaluate_sensor_balance.py --task wheelleg_sensor_drive \
  --checkpoint <本任务权重> --headless --sim_device cpu \
  --num_envs 4 --num_steps 30000 --report logs/my_drive/standing_60s.json
```

站立步数要按权重保存的控制频率换算：200 Hz 的 60 秒为 12000 步，500 Hz 为 30000 步。旧权重仍恢复自己的控制频率和速度设置，不会因修改新任务默认值而自动提速。

运动检查默认每台连续 27 秒，包含两次 6 秒运动和三次 5 秒停车。速度与角速度 RMS 在每段前 2 秒后统计；停车漂移也从减速后的第 2 秒起计量，完整减速过程仍保留在报告中。物理失败和超时逐一计数，任何倒地都不能用重置后的时长抵消。前后速度误差门槛为运动 0.05 m/s、停车 0.02 m/s；偏航误差门槛分别为 0.12/0.06 rad/s，并检查横移、直行航向、停车漂移和闭环误差。通过有限检查不等于覆盖所有地形或实机。

## 播放与修改位置

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint <已验收的运动权重> --sim_device cpu --keyboard --num_steps 120000
```

上述 120000 步在 200 Hz 下为 10 分钟、500 Hz 下为 4 分钟仿真时间，也可以随时关闭窗口。点击仿真窗口后，↑/↓ 前进后退，←/→ 左右转向，松开停止该方向；空格停车，R 重置。本任务没有启用跳跃。

| 文件 | 修改内容 |
| --- | --- |
| `legged_gym/envs/wheelleg/config/sensors.py` | `WheelLeggedDriveCfg` 的训练速度、加速度、键盘速度与腿长设置 |
| `legged_gym/envs/wheelleg/tasks/tracking.py` | `DrivingWheelLegged` 的方向组合采样与播放斜坡 |
| `scripts/play.py` | 读取任务键盘配置，连接窗口事件与目标速度 |
| `legged_gym/envs/wheelleg/control/keyboard.py` | 按下、松开、停车和重置事件 |
| `tools/bootstrap_motion.py` | 新网络的指令参考与转向阻尼初始化 |

当前仍使用近似闭环与同原点环境；同原点 GPU 训练最多使用已验证的 32 个并行环境。实验和权重留在本机，源码与说明同步 GitHub。


## 本机当前结果

实验 `logs/keyboard_drive_20261126_120433/` 的第三版完成 32 环境、50 轮运动微调，训练物理失败为零、无 GPU 容量警告。最终权重在 seed=17/42、各 4 个机器人连续 60 秒静止检查中全部通过，前后/横向位移峰值约 7.5/2.1 mm。运动序列中四方向实际响应、反向和松键停车均无倒地。

完整运动精度门槛尚未全部通过：一项反向转向的角速度误差 RMS 为 0.1245 rad/s，略高于 0.12 rad/s 门槛；组合按键也有约 0.122 rad/s 的超限项。因此本机保存为 `keyboard_play.pt`，定位为低速可操作候选，没有标记成全面验收权重。完整报告为同目录 `keyboard_play.json`，原站立验收权重保持不变。

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/keyboard_drive_20261126_120433/keyboard_play.pt \
  --sim_device cpu --keyboard --num_steps 120000
```

## 传感器噪声与高频抖动

`tools/filter_sensor_policy.py` 可把最新传感器帧的短窗口均值滤波折叠进 Actor 第一层（默认保留原转向反馈延迟），利用已有历史帧减小噪声驱动的力矩跳变。它生成新权重，不改电机限幅、噪声配置、指令输入或网络尺寸；原网络已有的其他历史通道保持不变。偏航陀螺仪和两轮差速反馈直接透传，只平滑两轮共同速度分量，避免滤波延迟削弱转向阻尼。

```bash
./manage.sh run tools/filter_sensor_policy.py \
  --checkpoint <原权重> --output logs/my_drive/filtered.pt --frames 3
```

滤波窗口为 N 帧时，额外平均延迟为 `(N-1) × 控制周期 / 2`；200 Hz 下三帧为 5 ms，六帧为 12.5 ms。窗口越大，响应越可能受影响，因此新文件必须重新做连续站立、四方向及组合运动检查，不能仅凭输出平滑就验收。

滤波已包含在新网络权重里，部署仍输入原始的六帧传感器历史，MCU 不要重复加这道滤波。输出文件保留来源 SHA256 与变换等价性检查结果；后续 PPO 使用 `--warm-start` 初始化 Actor，不能沿用变换前的优化器状态直接续训。该工具拒绝对自己已处理过的权重重复滤波。

当前策略不建议启用实验选项 `--filter-yaw-feedback`：完整三帧滤波会增大转向波动，六帧窗口也会损害停车恢复。可选 `--yaw-command-scale` 仅校准转向指令参考（默认 1），不改变零指令站立；调整后必须重新检查运动。

### 2026-11-26 高频抖动修复版

本机 `logs/vibration_20261126/keyboard_play.pt` 使用三帧选择性滤波，转向参考乘 0.98；新增平均延迟 5 ms 仅作用于被平滑的观测，转向阻尼不增加延迟。没有额外 RL 训练，没有关闭噪声或提高力矩。旧权重保留。

对照静止时序，轮力矩逐步变化 RMS 降低约 30–31%，横滚角标准差降低约 62%；仍有残余高频运动。seed17/42 各四台连续 60 秒站立全部通过，无倒地/超时，前后/横向位移峰值约 7.71/1.97 mm。两组四方向及一组组合按键检查全部通过。仅验证固定 8 cm、平地、前后 0.2 m/s、转向 0.6 rad/s，不能外推到实机或跳跃。结果及模型等价性说明见该目录 `fix_summary.json`。

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/vibration_20261126/keyboard_play.pt \
  --sim_device cpu --keyboard --num_steps 120000
```

重建此修复需从未滤波的旧权重导出：

```bash
./manage.sh run tools/filter_sensor_policy.py \
  --checkpoint logs/keyboard_drive_20261126_120433/keyboard_play.pt \
  --output logs/my_drive/smooth.pt --frames 3 --yaw-command-scale 0.98
```

滤波权重导出时清空旧 Adam 动量；后续微调仍建议用 `--warm-start` 开始新实验。上述实验权重和报告留在本机，源码与文档可通过 GitHub 获取。

### 2026-11-26 提速与转向校准版（当前推荐）

本机权重 `logs/drive_speed_20261126/keyboard_play.pt`：前后 ±0.3 m/s，左右转向 ±0.8 rad/s；加速度限制为 0.4 m/s²、1.8 rad/s²。策略从 200 Hz 提高到 500 Hz，物理保持 2 kHz。原三帧选择性滤波和传感器噪声保留；单帧传感器延迟、滤波平均延迟各从 5 ms 缩短到 2 ms。六帧观测相邻间隔 2 ms，最旧与最新相差 10 ms。这是仿真时序调整，实机延迟仍需测量对齐。

转向参考在上一版基础上乘 0.93，校正实际转速偏高。此版没有额外 RL 训练：已有网络做参考校准与时序适配，并完成确定性检查；不能称为重新训练完成的高速策略。旧权重保留自己的配置，不会自动提速。

验收：seed17/42 的四方向运动及 seed42 的组合运动全部通过，包含反向和松键停车；转向误差 RMS 为约 0.041–0.089 rad/s。四台机器人各连续站立 60 秒，无倒地、无超时，前后/横向漂移峰值约 7.18/1.90 mm。完整报告为该目录 `summary.json`。站立检查使用转向参考校准前的同配置网络，零偏航指令下逐层输出严格等价；此等价性另以 2048 组观测验证。

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/drive_speed_20261126/keyboard_play.pt \
  --sim_device cpu --keyboard --num_steps 300000
```

500 Hz 下 300000 步对应 10 分钟仿真时间。↑/↓ 前后，←/→ 左右转向，松键减速，空格停车，R 重置；腿长仍固定 8 cm，不含跳跃。当前验证范围为 CPU 平地仿真，转向手感仍需使用者主观确认。

绘图与控制解耦，`config/base.py` 的 `viewer.viewer_fps` 默认每仿真秒 60 帧；实际墙钟帧率取决于机器负载，不保证实时。物理积分和策略推理都不会因为跳过绘图而跳步，键盘仍按每个控制步处理。窗口创建和有限步播放已检查。

以后新建训练使用当前 `WheelLeggedDriveCfg` 默认速度和 500 Hz；每轮250步仍采样0.5秒，gamma/GAE衰减按物理时间换算。当前播放权重保存了对应配置；进一步训练建议 `--warm-start` 开新实验。权重导出和验收记录保存在实验目录，未覆盖原文件。

### 1 m/s 探索结果（未通过稳定性验收）

`logs/speed1_20261126/` 保存逐档提速检查。本次没有额外训练，也没有修改电机限制。0.6 m/s 的四方向、反向和停车检查通过，保存为 `keyboard_06.pt`；没有据此宣称0.6的长距离或组合转向均已验收。

1.0 m/s 指令下，原网络首段平均实际速度约 +0.964/-0.964 m/s，没有倒地，但直行横偏约64 mm，超过50 mm门槛，不能标记为稳定1 m/s。单独的 `trial_1ms.pt` 为试用文件，原0.3 m/s推荐权重未覆盖。尝试融合轮差速与陀螺仪反馈的另一个候选，在10秒运动段出现更大偏航，已弃用，没有作为试用权重。

```bash
# 通过四方向检查的0.6 m/s版本
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/speed1_20261126/keyboard_06.pt \
  --sim_device cpu --keyboard --num_steps 300000
# 仅试用1.0 m/s：把上述权重改为 logs/speed1_20261126/trial_1ms.pt
```

高速评估还修正了固定2秒等待的不足：1 m/s以0.4 m/s²减速，仅指令斜坡就需要2.5秒。新评估按本阶段命令差和限加速度计算斜坡预算，稳态等待取 `max(2秒, 斜坡预算+1秒)`，并要求之后至少1秒测量。全程轨迹仍记录，倒地/超时仍全程计数，速度、横移、航向及漂移门槛未放宽。原先1 m/s报告的停车误差包含部分正常制动，不能把这些条目全部归因于策略不稳定。

0.3 m/s等原低速检查的等待仍为2秒；更高速度的稳态统计窗口可能改变，比较报告时应核对 `settle_seconds`。完整结果为该目录 `summary.json`；目前尚未解决1 m/s的直行保持和停车恢复精度。

### 转弯时两腿反相振荡：阻尼修正

原地转向的对照时序中，两腿长度相关系数约 -0.93，伴随约8 Hz横滚模态；腿长标准差约0.28–0.32 mm。把髋刚度2增至4效果有限，提高了振荡频率。保持刚度2，将髋PD阻尼从0.04增至0.08 N·m·s/rad，原地转向腿长标准差下降约30–40%，0.6 m/s边走边转下降约15–25%；仍有残余振荡。比较窗口为四台各18秒中的第6–13秒，传感器噪声、支撑前馈、闭环参数和力矩上限未改变，没有额外RL训练。

本机实验目录 `logs/turn_legs_20261126/` 保留基线、刚度与两档阻尼对照的时序和 `comparison.json`。新文件只修改检查点的外部髋PD阻尼，Actor参数逐项保持原值；旧权重不会自动更新。源配置位于 `config/sensors.py` 的 `WheelLeggedDriveCfg.control.leg_kd`，控制计算在 `control/leg_control.py`。

长期操作还应检查峰值额度耗尽的情况：

```bash
./manage.sh run tools/evaluate_sensor_drive.py \
  --checkpoint <权重> --speed 0.6 --turn-speed 0.8 --combined \
  --continuous-hip-limit --report logs/my_drive/continuous_limit.json
```

该标志仅用于评估，令每一步开始时所有髋关节的2秒峰值额度均已用尽，因此自动重置也不会重新获得峰值力矩。实际输出须遵守0.32 N·m连续上限，原播放逻辑与电机峰值上限不受该标志修改。不能把这个对照当成正常峰值控制或实机热测试。

本次修正已通过：seed42的0.6 m/s四方向和组合运动、seed17的0.3 m/s四方向，均包含反向与停车；四台各连续站立60秒，倒地及超时为零。另通过0.6 m/s组合运动的连续力矩对照（所有髋关节输出不超过0.32 N·m）。这不等于验证实机温升或任意长时间运行。

新版0.6 m/s播放：

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/turn_legs_20261126/keyboard_play.pt \
  --sim_device cpu --keyboard --num_steps 300000
```

慢速版将权重改为同目录 `keyboard_03.pt`（0.3 m/s），两版左右转向均为0.8 rad/s，固定8 cm腿长。验收汇总为 `summary.json`。本次验证CPU平地仿真，旧0.3/0.6文件和1 m/s试用文件均未覆盖；1 m/s的整体稳定性仍未通过，不能因为阻尼修正就宣称已解决高速偏移。
