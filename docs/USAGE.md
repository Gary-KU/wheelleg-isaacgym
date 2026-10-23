# 运行与参数

本页收录日常运行命令。稳定性优化和已完成实验另见 [训练优化与复测](TRAINING_OPTIMIZATION.md)。
全部命令默认从 `IsaacGym/` 目录执行。

## 先看当前起立与站立效果

```bash
conda activate isaacgym
cd /home/gary/wheelleg_work/WheelLeg_RL_IsaacLab-main/IsaacGym
./stand.sh
```

从5 cm低蹲起步，初始零速、目标腿长8 cm。默认开启键盘，松开方向键回到零速度，空格停止，R复位。
`stand.sh`按住↑/↓的目标为±0.6 m/s，←/→为±2.0 rad/s（约115°/s），偏航加速度4 rad/s²，约0.5仿真秒升至目标。
这是策略的目标值，实际速度取决于跟踪效果。
按住Q缩腿、E伸腿，以2 cm/仿真秒调整五连杆屈伸后的有效腿长，范围6～20 cm；松开Q/E保持当前腿长目标。
空格同时停止行驶和腿长调整；R恢复默认8 cm目标，并复位机器人；按住的Q/E需松开后重按。
程序使用单机器人CPU物理计算来减少开销，窗口仍由显卡渲染。
默认`--physics-threads 0 --viewer-width 1280 --viewer-height 720`。优化闭环几何计算后，本机相同1600×900窗口由约34提高至53 FPS；1280×720热身后约57 FPS。
终端每100步显示`viewer_fps`与`simulation_speed`：前者是实际画面刷新次数/墙钟秒，后者是仿真时间/墙钟时间。负载和启动瞬态会影响结果。
`--show-collisions`可显示碰撞网格；默认使用显示模型。这不是垂直同步开关。
`stand.sh`内可看到完整的`play.py`调用。检查点保存了髋PD、每腿22.4649 N支撑前馈和原轮子策略；脚本现在用`--wheel-control heading --heading-kp 6 --heading-kd 0.6`覆盖原航向配置。
新模式保留RL的轮子差模平衡，由显式航向反馈取代RL共模转向；实际关节力矩仍限±9 Nm，转向只使用平衡力矩剩余的余量。
控制器内部航向参考与实际航向的差限制在±0.35 rad，减少转不动时的目标积压；这不会限制机器人的真实转角。
本次没有重训或修改权重；静止俯仰晃动尚无改善。伸腿仍由基础控制器完成，阶梯任务另有训练。测量与公式见[训练优化](TRAINING_OPTIMIZATION.md)。
`--wheel-balance-scale`仅供复现实验，默认1；1.25倍在60秒长测中虽将俯仰标准差从0.382°降至0.321°，最大XY漂移却从1.57 cm增至5.92 cm，因此未启用，`stand.sh`保持缩放1、额外俯仰阻尼0；静止俯仰尚未实质改善，见[长测报告](../validation/pitch_drive_scale_125_final60_heading6_20261022_summary.json)。

复测此前60秒零速站立基线（检查点原控制配置）：

```bash
./run.sh tools/evaluate_support.py --allow-approximate-loop \
  --checkpoint logs/standing_support_20261022_093823/input/standing_hold.pt \
  --seconds 60 --report validation/my_standing_check.json
```

`--leg-feedforward-force 0`可显式关闭前馈；`--heading-kp 0 --heading-kd 0`可显式关闭航向反馈，便于比较。
初始腿姿由`stand.sh`显式指定为5 cm；权重内保存的默认初态为8 cm，所以复测命令默认直接从8 cm开始。

固定10 cm目标启动：`./stand.sh --command 0 0 0 0.10`。
可用`--keyboard-leg-speed 0.01`把调腿长速度改为1 cm/仿真秒。
实际14秒变腿长序列（8→12→6→8 cm）无倒地、最大腿长误差约1.4 mm；0.6 m/s前后与0.6 rad/s自旋各做过10秒序列短测，无倒地。
这些是此前较小范围的短测，报告保存在`validation/keyboard_height_validation.json`、`keyboard_drive_06.json`和`keyboard_spin_06.json`。

新范围已完成8→20 cm（保持4秒）→8 cm的20秒测试，无倒地、最大腿长误差约1.42 mm。
20 cm腿长下±2 rad/s自旋、4 rad/s²加速的10秒序列也无倒地，实际转动段平均约±1.84 rad/s（含加速），机身倾斜峰约1.41°。
这不等于高腿姿下任意高速行驶都已验证。报告见`validation/higher_stance_20cm_validation.json`与`validation/spin_20_at_height20cm.json`。
20 cm腿姿最小关节余量约0.159 rad；保留这个余量，未把真实URDF关节硬限位放宽。

直接以15 cm目标起立可用`./stand.sh --command 0 0 0 0.15`；从默认8 cm开始按住E约6仿真秒可到20 cm。
手动调节上限与默认训练范围分开：`MAX_LEG_LENGTH=0.20`，默认随机腿长范围仍为0.06～0.15 m，旧权重未重新训练。

## 原地慢漂排查（尚未根治）

2026-10-23 的连续零速复测确认了慢漂。零速度目标不包含世界坐标位置保持；当前策略没有位置误差观测，发生位移后不会主动回到起点。左右轮也不能直接产生可控侧向驱动力，侧移还涉及轮胎接触和物理解算；这不是增加一个 yaw 反馈就能解决的问题。

匹配 `stand.sh` 的 5 cm 初态、8 cm 目标、CPU 单环境、种子 42、航向反馈 6/0.6，连续运行 300 仿真秒，没有周期重置：

| 结果 | 原 TGS、轮胎摩擦 1 | 可选 PGS、轮胎摩擦 2 |
| --- | --- | --- |
| 末端世界 X 位移 | 约 -16.76 mm | 约 -0.74 mm |
| 末端世界 Y 位移 | 约 -4.49 mm | 约 -2.43 mm |
| 全程世界 Y 位移绝对峰值 | 约 4.62 mm | 约 2.61 mm |
| 2 秒后俯仰波动标准差 | 约 0.344° | 约 0.398° |
| 倒地/超时重置 | 0/0 | 0/0 |

因此可选设置只是减轻此工况的慢漂，同时俯仰波动略增，不能称为稳定位置保持。默认 `stand.sh` 仍保留原设置。试用命令：

```bash
./stand.sh --physics-solver pgs --wheel-friction 2
```

这两个参数是本工程新增的参数：前者选择 PhysX 解算器，后者只设置左右轮胎的接触摩擦，不修改其他连杆材质。它们会写入新训练检查点的环境配置；推理恢复配置时，显式命令行参数优先。原检查点缺少这些字段时使用原 TGS、摩擦 1。

28 秒前后行驶/启停/左右转向复测无倒地，前后段末秒速度约 +0.580/-0.556 m/s，转向末秒约 +1.96/-1.97 rad/s。转向段航向误差 RMS 约 3.7～3.8°，高于原设置约 2.7～2.8°。这组可选参数尚未验证其他初态、腿长、GPU 或阶梯工况。

播放终端现在每 100 步打印第 0 个机器人相对最近复位位置的世界 X/Y 位移，以及机体坐标系 vx/vy。世界 Y 与屏幕左右不一定一致；有转向时，世界 Y 也不等于当前机体侧向。R 和物理失败重置会重新设定位移起点。

记录你实际键盘操作后的慢漂，可运行 `./stand.sh --event-log validation/my_drift.jsonl`。日志会保存每 100 步的位移、姿态、速度和指令，便于按实际操作复现。

支持评估报告新增分轴的位移峰值和平均速度，避免用总 XY 位移或“没倒地”掩盖侧移。精选测量见[慢漂排查报告](../validation/lateral_drift_work.json)。原始逐步轨迹保留在本机，未上传 Git。

## 平地驾驶复测

```bash
./run.sh tools/evaluate_drive_stability.py --allow-approximate-loop \
  --checkpoint logs/standing_support_20261022_093823/input/standing_hold.pt \
  --wheel-control heading --heading-kp 6 --heading-kd 0.6 \
  --report validation/my_drive_stability.json --record-trace
```

工具固定CPU单环境、物理线程0、种子42，连续运行28仿真秒：静止、0.6 m/s前进、停止、后退、短启停、左右2 rad/s转向并回正、末尾静止。
采用与`stand.sh`相同的5 cm初态、8 cm腿长目标、0.8 m/s²前后限加速度和4 rad/s²偏航限加速度。
去掉上述三个轮子控制参数可复现检查点原模式。`--drive-speed`与`--turn-speed`可改变测试目标，比较时应保持相同工况。

原模式与新`heading 6/0.6`的前进偏航误差RMS为10.78°→0.149°，后退为4.25°→0.087°；两组均无物理失败或超时。
新模式转向末1秒实际偏航速度约±1.98 rad/s，但启停与静止俯仰仍有晃动。这是有限工况复测，不代表所有腿长和任意操作均稳定。
原始及逐阶段数据见[对比报告](../logs/drive_stability_20261022/heading_gain_comparison_seq28.json)。

报告独立积分完整的转向要求角，不采用控制器被限制的内部参考来判断转向跟随；姿态和位移取终止前状态，保留所有失败，排除自动重置瞬移。
同时报告原始/限加速度目标的速度误差、实际有符号前进距离，以及各阶段的俯仰/横滚/偏航RMS和波动标准差。

## 阶梯演示与训练

```bash
./stairs.sh
```

默认加载已复测的第80轮策略，自动以目标0.25 m/s、腿长14 cm行驶，固定1 cm单阶的四上四下课程。
当前采用双轮中心进度与轮下地面扫描参考，配合航向反馈6/0.3，经新复测4个环境各26仿真秒均完整通过1 cm课程，无物理失败或超时。
同一策略在修正前的约2 cm历史复测中0次通过、22次俯仰失败，复杂阶梯尚未学会。
可用 `./stairs.sh --checkpoint 实际权重路径 --stair-level 1` 查看其他权重；默认演示不会自动切换到最新模型。

```bash
./train_stairs.sh
```

从`input/stage4_start.pt`继续512个GPU环境的400次新增PPO更新，初始/最低第1档、最高第4档，输出到新的`logs/stairs/日期_时间/`。
该起点使用髋/轮探索标准差0.25/0.02及每次2个epoch、4个minibatch；它不是已通过2 cm的模型。
五档地形、完整评估命令、代码学习入口和限制见[阶梯训练与复测](STAIRS.md)。平地键盘演示仍用`./stand.sh`。

## 环境与入口

本机已有 Conda 环境 `isaacgym`：Python 3.8、PyTorch 1.13.1+cu117、Isaac Gym Preview 4。
Isaac Lab 使用独立环境，不要把它的依赖装入旧 Gym 环境。

从仓库根目录：

```bash
conda activate isaacgym
cd IsaacGym
./run.sh train.py --help
```

`run.sh` 使用当前 Python，设置该环境的动态库路径，然后进入 `IsaacGym/` 启动程序。
也可明确指定解释器：

```bash
PYTHON_BIN=/home/gary/miniconda3/envs/isaacgym/bin/python ./run.sh play.py --help
```

换机器时先自行安装兼容的 Isaac Gym Preview 4，再安装 `requirements.txt` 中的补充依赖。
项目附带所需旧 RSL-RL 组件，无需安装最新 `rsl-rl-lib`。资产已导出，日常运行不需要 `pxr`。
入口必须先导入 `isaacgym`，再导入 `torch`。

| 入口 | 用途 | 默认物理设备 |
| --- | --- | --- |
| `train.py` | PPO 训练 / 续训 | `cuda:0` |
| `play.py` | 模型检查 / 权重播放 / 键盘 | `cpu` |
| `evaluate.py` | 无窗口评估 | `cuda:0`，单机器人可显式选 `cpu` |
| `smoke_test.py` | 有限动作下的闭环误差检查 | `cuda:0` |
| `tools/evaluate_stairs.py` | 固定档位、完整阶梯课程评估 | `cpu` |
| `tools/evaluate_drive_stability.py` | 固定28秒平地驾驶与转向复测 | `cpu`，固定单环境与物理线程0 |

`--allow-approximate-loop` 表示使用者接受当前的弹簧阻尼闭环近似，每次物理运行都需传入。
它不是刚性闭环求解开关。

## 先检查机构

固定机身、零动作、有窗口：

```bash
./run.sh play.py --allow-approximate-loop --zero-actions --fix-base --num-steps 500
```

加 `--headless` 可无窗口检查。零动作不具备自主平衡能力，固定机身只用于看模型。

有限周期动作、CPU 数值检查：

```bash
./run.sh smoke_test.py --allow-approximate-loop --sim-device cpu --num-envs 4 --num-steps 100 --action-amplitude 1 --max-error-mm 10 --report validation/closure_check_new.json
```

报告包括闭合误差峰值、95 分位数和失败次数。超过给定阈值会返回非零；
这是工况检查命令，不能预先保证该幅度/阈值一定通过。更详细的标定工具见 [工具索引](../tools/README.md)。

## 训练参数

默认值集中在 [config.py](../wheelleg_gym/config.py)，命令行参数由 `cli.py` 和 `train.py` 定义。
命令行会覆盖配置默认值。参数中的连字符和部分旧下划线别名均可使用，文档统一采用连字符。

| 参数 | 含义 |
| --- | --- |
| `--num-envs` | 并行仿真数量；本机历史 1024 环境吞吐较好，详见实验记录 |
| `--max-iterations` | 本次新增的 PPO 更新次数 |
| `--command VX VY WZ LENGTH` | 固定指令：m/s、m/s、rad/s、m |
| `--random-commands` | 训练时随机采样指令，替代固定运动目标 |
| `--forward-speed-range` / `--yaw-speed-range` | 随机前后/偏航指令范围 |
| `--leg-length-range` | 随机腿长目标范围，允许0.06–0.20 m，默认仍为0.06–0.15 m |
| `--save-interval` / `--log-dir` | 保存间隔和实验目录 |
| `--init-noise-std` | 初始探索噪声：一个数或按动作顺序六个数 |
| `--learning-rate` | 新训练的初始学习率 |
| `--max-learning-rate` | 自适应学习率上限，也可用于续训 |
| `--terrain plane` / `--terrain stairs` | 平地或阶梯地形；权重输入布局必须匹配 |
| `--stair-level` / `--stair-min-level` / `--stair-max-level` | 阶梯初始、最低、最高档位，范围0～4 |
| `--stair-curriculum` / `--no-stair-curriculum` | 开启或关闭回合间难度升降 |
| `--stair-episode-seconds` | 阶梯回合时间上限，默认24仿真秒 |
| `--observe-terrain` | 添加前方9个地形高度点；阶梯权重使用252维输入 |
| `--leg-control hip_target` / `--hip-target-scale` | 阶梯髋目标角动作及其增量系数，默认0.2 rad |
| `--wheel-control policy` / `--wheel-control heading` | 原轮子策略，或RL差模平衡配合显式航向；默认`policy`，`stand.sh`选择`heading` |
| `--heading-kp` / `--heading-kd` | 显式航向/偏航角速度反馈，`stand.sh`使用6/0.6 |
| `--wheel-pitch-damping` | `heading`模式下可选俯仰角速度反馈，当前默认0 |

稳定性功能已接入 `--profile stability`；可用 `./run.sh train.py --profile stability --help` 查看入口。
下面给出等价的主要显式参数，**执行后会开始新训练，参数本身不代表控制已经合格**：

```bash
./run.sh train.py --allow-approximate-loop --headless \
  --num-envs 1024 --max-iterations 1000 \
  --random-commands --forward-speed-range -0.3 0.3 --yaw-speed-range -0.8 0.8 \
  --leg-length-range 0.08 0.08 --command 0 0 0 0.08 \
  --observe-leg-lengths --tracking-penalties --stability-rewards \
  --command-resample-seconds 2 --zero-command-fraction 0.2 \
  --linear-acceleration 0.8 --yaw-acceleration 2 \
  --init-noise-std 1 1 1 1 0.1 0.1 --learning-rate 0.0003 --max-learning-rate 0.0003 \
  --save-interval 25 --log-dir logs/stability_example
```

这些开关彼此独立：腿长输入改变观测尺寸，稳定奖励改变训练目标，重采样改变指令分布，
限加速度改变指令变化速度。设置 1000 轮不表示届时一定收敛。

## 续训与观测兼容

`train.py` 恢复模型、优化器、训练轮次、训练配置及支持的环境字段。
参数顺序为默认值 → 检查点环境参数 → profile → 显式命令行。设备、窗口和环境数量不会被旧实验覆盖。
198维检查点恢复腿长观测，252维阶梯检查点还恢复地形观测及髋角动作模式；若显式配置与网络输入维度不一致，会在创建仿真前报错。

| 权重与模式 | 说明 |
| --- | --- |
| 现有 `model_794.pt` | 旧 186 维；仍有整机晃动，转向未训练 |
| 开启腿长输入的新训练 | 每帧 33 维，总共 198 维 |
| 186 维权重直接载入 198 维网络 | 输入层形状不同，不能直接严格加载 |
| 经升级工具转换的新权重 | 新增列和动量补零；初始预测保持，尚未学会使用新腿长输入 |
| 阶梯权重 | 42×6=252维，前四维动作改为髋目标角增量；准备与复测见[阶梯说明](STAIRS.md) |

如果要把旧策略作为新观测训练的起点，先运行 [权重升级工具](../tools/README.md#观测升级)，
再对转换后的新文件使用 `--profile stability` 或对应显式设置。保留源权重，给新实验单独指定 `--log-dir`。
新的 `tools/optimize_training.py` 已串起升级、有限训练和同条件前后复测，见 [训练优化](TRAINING_OPTIMIZATION.md)。
不要使用历史 `tools/train_and_evaluate.py` 自动续接新增观测与稳定性配置。

续训会恢复已训练的探索噪声和优化器，因此不能同时指定 `--init-noise-std` 或 `--learning-rate`。
网络结构与其他训练超参数以检查点保存的训练配置为准，不能只改 `config.py` 就期待续训全部更新。

## 键盘播放

下面打开本次优化后的第 994 轮实验权重。复制整个代码块到终端执行：

```bash
conda activate isaacgym
cd /home/gary/wheelleg_work/WheelLeg_RL_IsaacLab-main/IsaacGym

./run.sh play.py --allow-approximate-loop \
  --checkpoint logs/stability_optimization_20261021_215150/training/model_994.pt \
  --keyboard --num-steps 100000
```

该版本前后俯仰与一般跟踪有改善，但横滚、快速换向和低蹲仍有问题，尚未通过整体稳定性验收。
同条件实测见 [对比报告](../logs/stability_optimization_20261021_215150/reviewed_comparison.md)。

| 命令部分 | 含义 |
| --- | --- |
| `conda activate isaacgym` | 使用已经装好仿真依赖的 Python 环境 |
| `cd .../IsaacGym` | 进入当前训练项目，让相对权重路径可以找到文件 |
| `./run.sh play.py` | 设置运行环境并启动策略播放 |
| `--allow-approximate-loop` | 使用本项目的弹簧阻尼闭环近似 |
| `--checkpoint .../model_994.pt` | 加载本次训练保存的权重；换模型时修改这里 |
| `--keyboard` | 用方向键设置机器人运动目标 |
| `--num-steps 100000` | 运行最多 100000 个控制步，窗口可提前关闭 |

单击窗口取得焦点后：

| 操作 | 效果 |
| --- | --- |
| 按住 ↑ / ↓ | 前进 / 后退目标，默认 ±0.3 m/s |
| 按住 ← / → | 左转 / 右转目标，默认 ±0.8 rad/s |
| 按住 Q / E | 缩腿 / 伸腿，松开保持腿长目标 |
| 松开方向键 | 对应目标回到零 |
| 空格 | 清零目标，按住的方向需释放后重按 |
| R | 手动重置机器人 |

`--keyboard-speed` / `--keyboard-turn-speed` 修改按键目标。
上表运动默认值对应直接调用`play.py`；`stand.sh`显式设为0.6 m/s与2.0 rad/s。
`--keyboard-leg-speed`设置目标腿长变化率，单位m/s，默认0.02；`--keyboard-leg-range MIN MAX`设置范围，`stand.sh`使用0.06～0.20 m。
`--linear-acceleration` / `--yaw-acceleration` 修改目标变化速度，默认零表示不限制。
**播放和评估会恢复检查点中的这两个限制，显式命令行仍可覆盖。**
开启限制时，停车目标仍按限制渐变，空格不是直接锁住关节的机械制动。

播放/评估还会恢复物理步长、闭环刚度/阻尼及腿长目标；按检查点识别186、198或阶梯252维输入与对应动作模式。
键盘模式要求可见窗口、单机器人、自由机身和已加载权重，不与随机指令或零动作模式组合。
键盘播放不会每 16 秒自动回到原点，物理失败和 R 仍会重置。

相机使用原生 viewer 操作：右键拖动转动视角，右键配合 WASD 移动，Q/E 上下移动。

关闭仿真窗口即可结束播放；也可在启动它的终端按 `Ctrl+C`。
若需要记录按键与失败事件，在播放命令末尾添加 `--event-log validation/keyboard_events_994.jsonl`。
要复现旧基线，把 `--checkpoint` 改成 `logs/forward_backward_1024_recovered_20261021/model_794.pt`；旧基线没有接受转向训练。

## 查询参数与学习顺序

在当前项目目录、`isaacgym` 环境下，随时查看每个入口接受哪些参数：

```bash
./run.sh play.py --help
./run.sh train.py --help
./run.sh evaluate.py --help
./run.sh tools/evaluate_stairs.py --help
./run.sh tools/evaluate_drive_stability.py --help
```

这些帮助命令只显示用法，不会启动训练或仿真。终端命令末尾的 `\` 表示下一行继续同一条命令。
训练入口更新并保存权重；播放入口加载权重供观察与键盘操作；评估入口运行指定工况并写出数值报告。

建议先学本页的环境、键盘和参数，再读 [中文代码导读](CODE_GUIDE.md)，最后看 [训练优化](TRAINING_OPTIMIZATION.md)。
命令行参数的代码定义在 [cli.py](../wheelleg_gym/cli.py) 和各入口的 `parser.add_argument(...)`；
预设与检查点恢复规则在 [runtime_config.py](../wheelleg_gym/runtime_config.py)，基础默认值在 [config.py](../wheelleg_gym/config.py)。

## 策略评估

三组固定速度各运行 16 秒：

```bash
./run.sh evaluate.py --allow-approximate-loop --sim-device cpu \
  --checkpoint logs/forward_backward_1024_recovered_20261021/model_794.pt \
  --num-envs 3 --num-steps 800 --command 0 0 0 0.08 \
  --test-speeds -0.3 0 0.3 --report validation/speed_check_new.json
```

连续 60 秒静止检查，关闭正常时间上限重置：

```bash
./run.sh evaluate.py --allow-approximate-loop --sim-device cpu \
  --checkpoint logs/forward_backward_1024_recovered_20261021/model_794.pt \
  --num-envs 1 --num-steps 3000 --command 0 0 0 0.08 \
  --disable-time-limit --report validation/standing_60s_new.json
```

速度序列使用 `--speed-sequence 0.3 0 -0.3 0 --segment-steps 200`。
配套 `--yaw-sequence` 必须与速度序列等长；`--sequence-repeats` 可重复模式。
总步数由段长、段数和重复次数决定；若同时指定 `--num-steps`，必须与总数一致。
`--test-speeds` 与 `--speed-sequence` 互斥。

报告区分物理失败和正常超时，并包含速度误差、倾角/角速度、力矩变化和地面支撑。
启用指令平滑时，分别统计原始目标与实际发给策略的目标误差，不能只看后者。
短时无跌倒不等于长时稳定，评估程序正常退出也不代表策略质量合格。

## 测试

整理或修改代码后，在 Isaac Gym 环境内运行 CPU 回归：

```bash
WHEELLEG_TEST_DEVICE=cpu ./run.sh -m unittest discover -s tests -q
```

测试包括闭环数值、真实 CPU PhysX 环境时序、指令、按键、奖励、观测升级和评估统计。
它们不启动 PPO 训练，也不做控制稳定性验收。本次整理的实际结果见
[整理验证记录](../validation/project_cleanup.json)。
驾驶修正后的CPU回归共213项：212项通过、1项跳过、0失败，见[回归日志](../validation/drive_stability_tests.log)。

若要额外测试 GPU 环境接口，可单独运行（本次整理不执行）：

```bash
WHEELLEG_TEST_DEVICE=cuda:0 ./run.sh -m unittest discover -s tests -p test_environment.py -q
```

## 文件保留与历史

训练权重和日志保留在原 `logs/` 路径，评估与诊断保留在 `validation/`。
原 README 快照保存在 `docs/archive/README_20261021_before_cleanup.md`，其中旧命令和状态仅供追溯。
当前背景任务情况应检查实际进程，不能依据历史 PID 或旧状态 JSON 判断。
