# 传感器策略的键盘运动训练

`wheelleg_sensor_drive` 基于后置电池模型已经验收的站立配置，默认训练固定 8 cm，训练前进、后退、左右转向以及松键停车。左右是转向，不是横向平移。新任务默认速度为 ±0.3 m/s、转向为 ±0.8 rad/s；播放检查点会恢复自己的设置。最新伸缩腿与速度档见文末[当前配置](#伸缩腿与提速试用档)。本任务未启用跳跃。

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

### 按实际时间刷新画面与 FPS 参数

当前播放默认按墙钟时间安排绘图，`viewer.viewer_wall_clock=True`，目标 `viewer.viewer_fps=60`。旧版按仿真时间每秒画60帧，单机器人仿真慢于实时的时候，实际绘图帧率也跟着降低。现在控制和物理仍逐步计算，绘图按实际时间选取最新状态；不跳物理步、不改变策略频率、传感器时间、PD增益或电机限制。

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/turn_legs_20261126/keyboard_play.pt \
  --sim_device cpu --keyboard --viewer-fps 60 --num_steps 300000
# 可将 --viewer-fps 60 改为 --viewer-fps 120，画面更顺但仿真推进可能更慢。
```

本机单机器人、CPU、2000步（4秒仿真）顺序短测，启动/载入时间不计入：

| 绘图方式 | 实测绘图调用/墙钟秒 | 仿真秒/墙钟秒 |
|---|---:|---:|
| 旧版：按仿真时间60帧 | 25.1 | 0.417 |
| 新版：实际时间目标60帧 | 50.5 | 0.379 |
| 新版：实际时间目标120帧 | 85.0 | 0.330 |

因此默认建议60档。提高绘图帧率不代表物理仿真变快或已达到实时；实际显示还受显示器刷新率和机器负载影响。原物理2kHz、策略500Hz仍指仿真时间内的频率。本次另试单CPU线程，收益不明显，未修改原4线程配置。

退出播放时的JSON报告新增 `rendered_frames`、`measured_viewer_fps`、`simulation_speed_ratio` 和 `viewer_wall_clock`。无窗口时绘图FPS为null。旧绘图方式可通过配置 `viewer_wall_clock=False` 保留。`--viewer-fps` 覆盖检查点的目标绘图频率，不需要修改或重新训练网络。

基准保存在 `logs/fps_20261126/`；相同4秒运行的运动统计、姿态峰值、闭环误差及失败次数在两种绘图方式和120档间逐项完全一致。本次没有重新进行站立训练或改写权重。

### 键盘响应慢：速度斜坡与仿真慢放

画面FPS与仿真推进速度是两个指标。原0.6 m/s权重的线加速度限制为0.4 m/s²，指令从零升到0.6需要1.5秒**仿真时间**；仿真慢于实时后，实际等待更长。播放可显式调整斜坡，不需要改网络或复制权重：

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/turn_legs_20261126/keyboard_play.pt \
  --sim_device cpu --keyboard --viewer-fps 60 \
  --linear-acceleration 0.8 --yaw-acceleration 2.5 --num_steps 300000
```

这两项分别为m/s²和rad/s²，加速、减速共用；0.6 m/s指令斜坡缩短到0.75秒仿真时间，0.8 rad/s转向斜坡为0.32秒。它们表示**指令**变化速度，实际车速仍受策略跟踪、惯性和电机力矩限制，不等于按键到达目标车速的实测延迟。参数不指定时保留权重原值，设0表示取消该项斜坡，不建议直接取消。训练使用同一参数入口，部署MCU时也应保持一致。

闭环几何及力/力矩计算改为TorchScript内核，减少高频Python调度；仍逐物理步计算，维持2kHz物理、500Hz控制，以及原刚度、阻尼和力矩限制。没有额外RL更新。由于编译计算存在float32舍入差异，测试核对数值容差与运动表现，不宣称长轨迹逐位一致。响应实验及数值对照保存在 `logs/response_20261126/`。

本次0.6 m/s、0.8 rad/s四方向及组合运动各四台、各27秒检查通过，倒地/超时均0；组合运动额外限制髋连续输出0.32 N·m，轮子仍限制0.1 N·m。俯仰峰值约0.20 rad，闭环误差峰值约0.19 mm；加速更快会增加起停俯仰，不应理解为任意加速度均稳定。

单机器人有窗口顺序短测（各6秒仿真、目标60FPS）：优化前后仿真时间比约0.383→0.399，绘图约50.2→51.2FPS。整机提升较小，仍明显慢于实时；不能把闭环内核微基准的加速倍数当成整机收益。这里主要的操作响应改善来自缩短指令斜坡。完整结果见实验目录 `summary.json`。

### 小幅提速到0.65 m/s

本机 `logs/speed065_20261126/keyboard_play.pt` 将前后速度从0.6提高到0.65 m/s（约8%），转向仍0.8 rad/s，保存线加速度0.8 m/s²、转向加速度2.5 rad/s²。网络参数逐项未变，没有额外训练；旧版文件保留。

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/speed065_20261126/keyboard_play.pt \
  --sim_device cpu --keyboard --viewer-fps 60 --num_steps 300000
```

seed42四方向、组合运动各四台各27秒检查通过，倒地/超时均0；组合测试全程限制髋连续力矩0.32 N·m。四方向直行最大横向偏移约4.16 cm、俯仰峰值约0.207 rad。0.7 m/s候选虽未倒地，但反向后的直行横向偏移约6.22 cm，超过原5 cm门槛，因此未作为推荐版本。完整报告在该目录 `summary.json`，这只是上述短程仿真范围的验证。

### 转向1.0 rad/s与左右腿差动阻尼

推荐本机 `logs/turn_refine_20261126/keyboard_play.pt`：前后仍0.65 m/s，转向从0.8提高到1.0 rad/s（约25%）。线加速度0.8 m/s²、转向加速度2.5 rad/s²保持原值。原0.65版本保留。

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/turn_refine_20261126/keyboard_play.pt \
  --sim_device cpu --keyboard --viewer-fps 60 --num_steps 300000
```

统一提高四髋阻尼虽然压低腿长波动，却加重边走边转时的yaw波动，因此没有采用。新版保持公共髋Kp=2、Kd=0.08，仅增加 `control.leg_differential_kd=0.08`，抑制左右腿反向运动。实现见 `control/leg_control.py`；每个物理步使用关节速度计算，再与RL、原PD和支撑前馈一起经过原电机限幅。该项默认0，已有检查点不会自动启用。

网络第一层的yaw误差反馈乘2，并同时补偿同帧转向参考列；由 `tools/tune_turning.py` 生成，维持132维原始历史输入、6维输出和原网络结构，没有额外滤波或RL更新。这是反馈校准，不是重新训练；重新学习应使用Actor warm-start。可复现候选（输出后仍需评估）：

```bash
./manage.sh run tools/tune_turning.py \
  --checkpoint logs/speed065_20261126/keyboard_play.pt \
  --output logs/my_turn/candidate.pt \
  --feedback-gain 2 --turn-speed 1 --differential-kd 0.08
```

seed42、四台机器人14秒时序对照，第5–11秒的原地转向腿长标准差下降约44–50%，横滚标准差下降约一半，转向角速度标准差下降约23%。边走边转腿长波动下降约9–21%，角速度标准差在提高25%转速后仍比原版高约3–6%，不能称为完全消除振荡。原始时序、各候选及 `selected_comparison.json` 保存在实验目录。

四方向及组合运动各四台各27秒验收通过（含反向/松键停车），组合运动强制全程0.32 N·m髋连续限幅；轮子仍0.1 N·m、髋峰值仍0.68 N·m/2秒。四台各30000控制步，即连续60秒站立，倒地/超时均0；包含启动过程的最大前后位移约7.3 mm、横向约1.8 mm，静止轮速RMS约0.051–0.052 rad/s。float32回合年龄累加显示59.99秒是舍入误差；该次零重置，连续时长按步数乘dt计算。汇总见 `summary.json`。只验证上述CPU仿真条件，不代表实机或任意工况稳定。

### 伸缩腿与提速试用档

本机 `logs/telescopic_20261126/keyboard_play.pt` 为保守速度的伸缩腿试用档：保留前后 0.65 m/s、转向 1.0 rad/s，开放 Q/E 调整 8–10 cm 腿长；`keyboard_fast.pt` 为前后 0.7 m/s 试用档。两档线加速度仍为 0.8 m/s²，转向加速度 2.5 rad/s²。**网络参数没有变化，也没有额外 RL 训练**；原来满意的固定 8 cm 版本仍在 `logs/turn_refine_20261126/keyboard_play.pt`。

```bash
./manage.sh play --task wheelleg_sensor_drive \
  --checkpoint logs/telescopic_20261126/keyboard_play.pt \
  --sim_device cpu --keyboard --viewer-fps 60 --num_steps 300000
# 提速试用：把上面的文件名改成 keyboard_fast.pt。
```

点击仿真窗口后用方向键驾驶，按住 E 伸腿、Q 缩腿，松开保持高度，每仿真秒变化 1 cm，终端每变化 2 mm 显示目标与实际左右腿长。空格停车并停止继续调整高度；R 重置到 8 cm。与旧 `stand` 演示不同，本传感器任务的空格仍是停车，没有跳跃。

伸腿后的静止修正在 `control/standing_trim.py`，由 `DrivingWheelLegged` 自动调用。仅用 IMU 俯仰角速度和两轮编码器估计低速运动；用 0.1 秒低通抑制瞬时轮速振动，再在零速、腿长不变至少 0.5 秒、估计速度小于 0.04 m/s 时积分。它修正的是网络速度参考，增益 0.4、上限 ±0.08 m/s；原 8 cm 时修正严格为零。行驶和伸缩时冻结学习系数，已学补偿随伸腿量变化；重置只清除对应机器人的状态。没有使用仿真全局位置/线速度，也不锁轮。

这是自动混合控制。用户只提供驾驶和腿长指令，不需要手动调力矩；MCU 部署需一起移植该修正器、指令斜坡、髋 PD/支撑前馈、差动阻尼和原电机限幅。网络仍接收 132 维原布局，只是每帧 `command_vx` 现在为斜坡后的请求加修正；缩放仍为 1.5。评价仍对比原请求，不能把修正后的参考当作用户目标来美化误差。关闭修正使用 `control.standing_height_trim_ki=0`，旧检查点默认关闭。

0.7 m/s 仍是试用档：先前四方向检查出现约 5.6–5.8 cm 直行横向偏移，超过原 5 cm 门槛，因此不标记为全面稳定版。12 cm 长时间停车仍有振荡，当前没有开放为默认范围。模型、近似闭环和轮子/髋部力矩上限均未放宽。

最终 CPU、seed42 复测：0.65 m/s、8 cm 的四方向27秒检查通过；0.7 m/s、10 cm 的四种边走边转组合36秒检查通过（强制髋连续限幅0.32 N·m），均零倒地/超时。8/9/10/10 cm 四台机器人先伸腿，再各连续停车60秒，最大前后漂移约3.57 cm、横向3.70 mm，但最大航向变化0.189 rad，超过0.1 rad标准。0.65 m/s、10 cm 四方向检查的前进后停车有一组继续调整8.2 cm，超过5 cm标准。**伸腿后仍有漂移/偏航，不能把零倒地或部分序列通过称为全面站稳。** 12台扩展检查中10 cm停车漂移不超过约5 cm，12 cm有一台达10.5 cm，故限在10 cm。原始报告与最终汇总见本机实验目录 `summary.json`。

自己生成配置副本时，不需要改网络或散落的启动脚本：

```bash
./manage.sh run tools/configure_driving.py \
  --checkpoint logs/turn_refine_20261126/keyboard_play.pt \
  --output logs/my_height/candidate.pt \
  --speed 0.65 --leg-range 0.08 0.10 --leg-speed 0.01 --standing-trim-ki 0.4
./manage.sh run tools/evaluate_sensor_drive.py \
  --checkpoint logs/my_height/candidate.pt --report logs/my_height/drive_height10.json \
  --speed 0.65 --turn-speed 1 --leg-length 0.10 --stop-seconds 8 --combined --continuous-hip-limit
```

生成器只复制配置并记录来源，不能代替训练或验收。只修改播放档位时使用该工具；开始新训练应使用合适的任务配置和 Actor warm-start。

### 变腿长训练候选

`wheelleg_sensor_telescopic` 保留前后 ±0.7 m/s、转向 ±1.0 rad/s，训练时随机请求 8–10 cm 腿长，包含两个端点，每 4 秒重新采样；50% 请求为停车。腿长从实际重置姿态的 8 cm 开始，以 1 cm/s 变化，训练和播放使用同一斜坡。原 `wheelleg_sensor_drive` 的固定高度采样以前没有使用 `leg_length_range`；现在变高度范围会参与采样，固定范围保持原行为。

配置在 `config/sensors.py` 的 `WheelLeggedTelescopicCfg`，仍使用相同 IMU/编码器 132 维输入、髋 PD/支撑前馈及实际电机限幅。该候选关闭旧的高度积分修正，由网络学习不同高度的平衡；奖励额外惩罚实际侧向速度和速度/转速跟踪误差，降低停车锁轮惩罚，允许必要的轮子平衡修正。仿真真值只用于奖励与验收。

```bash
./manage.sh train --task wheelleg_sensor_telescopic \
  --warm-start logs/turn_refine_20261126/keyboard_play.pt \
  --num-envs 32 --max-iterations 100 --headless --sim-device cuda:0
./manage.sh run tools/evaluate_sensor_drive.py \
  --task wheelleg_sensor_telescopic --checkpoint <候选权重> \
  --report logs/my_height/drive.json --speed 0.7 --turn-speed 1 \
  --leg-length 0.10 --stop-seconds 8
```

当前实验位于本机 `logs/telescopic_training_20261126/`。训练完成不等于验收通过；须检查多种随机种子的连续站立，以及 8/9/10 cm 下的四方向、组合转向、反向、松键停车。运动报告同时保存各电机限幅占比、髋关节超过连续力矩的累计时长及峰值额度余量，用于区分策略问题和执行器饱和。

### 变高度恢复实验

上一轮100次微调完成后，10cm仍有持续漂移，未被接受为稳定版本。恢复实验保存在本机 `logs/height_rescue_20261126/`，仍保留0.7m/s、8–10cm和原电机限幅。

`tools/normalize_height_policy.py` 可从固定高度策略生成待验收的初始化：把腿长观测和目标转换为相对8cm的变化量，并补偿CAD模型随高度变化的名义重心平衡俯仰。参照取最新有效目标，因此8cm启动时的空历史也保留原语义。变换折叠进网络第一层权重和偏置，输入仍是原132维传感器历史，MCU不要重复做同样处理。这是输入校准，不能称为新增RL训练；径向支撑前馈仍是原近似，未被替换成完整重力补偿。

```bash
./manage.sh run tools/normalize_height_policy.py \
  --checkpoint logs/turn_refine_20261126/keyboard_play.pt \
  --output logs/my_height/normalized.pt --speed 0.7 --leg-range 0.08 0.10
./manage.sh train --task wheelleg_sensor_telescopic \
  --warm-start logs/my_height/normalized.pt --num-envs 32 \
  --max-iterations 50 --headless --sim-device cuda:0
```

新的 `WheelLeggedTelescopicCfgPPO` 为新Critic设置独立学习率0.001，Actor仍为0.00001，并关闭价值损失裁剪，避免新Critic的值变化被限制在旧预测附近。`algorithm.critic_learning_rate=None` 保留共用学习率的行为；旧检查点缺少该字段时恢复为None，因此播放和旧实验续训不会自动启用新设置。新的Actor warm-start会采用恢复配置。独立Critic步长在Actor的自适应学习率变化和优化器恢复后保持不变。

价值网络在固定目标上的拟合通过，仅证明优化器能学习，不能证明策略站稳；候选仍须通过长时间静止、变高度、驾驶和停车验收。速度误差反馈并非越大越好：本次提高该反馈的诊断出现倒地，未采用。

后置电池还使“机体俯仰为零”与“整机重心在轮轴上方”不完全相同。新的变高度任务设置 `rewards.upright_about_com=True`，使用CAD质量和仿真刚体状态计算整机重心相对轮轴的前后倾角，作为upright奖励；横滚项保持原定义。该真值仅供奖励使用，不进入132维Actor输入，也不修改物理模型或施加额外支撑。旧检查点缺少该字段时保持原奖励。

`wheelleg_sensor_height_stand` 是站立课程：一回合保持一个8–10cm目标高度，伸腿后约有14秒静止数据，并收紧实际速度误差的奖励尺度。先用该任务学习停车，通过多种子长时间检查后，再用Actor warm-start进入 `wheelleg_sensor_telescopic` 的0.7m/s完整运动课程；站立中间权重不能直接称为驾驶版。恢复流程和验收参数保存在实验目录 `pipeline_plan.json`。

Actor的输入校准信息保存在检查点 `actor_metadata`，warm-start时跟随Actor写入新检查点，避免把高度校准误加两遍。该元数据不参与推理。
