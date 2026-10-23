<!-- 历史说明快照；当前入口为 ../../README.md，旧命令与状态不再作为当前建议。 -->

# WheelLeg Isaac Gym 迁移版

本机 RTX 4060 Laptop 已通过 37 项命令采样、键盘、CPU/CUDA 数值与 GPU 环境测试、64 环境两轮
PPO 更新及 100 步权重播放。详细结果见 [验证记录](validation/summary.json) 和
[周期动作测试](validation/free_base.json)。短训练只验证流程，不代表已学会平衡。

这里把本仓库的五连杆轮腿模型、6 维力矩动作、186 维历史观测和 PPO
训练入口迁移到 **Isaac Gym Preview 4**。运行这份代码不需要 Isaac Sim 或
Isaac Lab。原来的 `IsaacLab/` 项目仍可独立使用。

## 闭环实现与适用范围

**本版本使用柔性闭环近似，不是原 USD 刚性闭环约束的等价复现。**

旧 Isaac Gym 的公开 API 无法直接复用原 USD 的两个被动闭合铰链。转换后的
URDF 保留原模型的树形关节、质量、惯量和网格；另外两个闭合点保存在
`assets/wheelleg/loop_constraints.json` 中，运行时使用成对外力维持锚点接近，
使用成对力矩约束铰链轴的偏斜，并保留绕铰链轴的自由转动。
原始两个主动髋关节、两根被动杆和轮子构成的每侧机构仍在模型中。

近似闭环有有限刚度、有限误差，结果会受时间步长、外力上限、刚度、阻尼和
接触影响；不应用它直接验证真实机器人结构强度或宣称达到刚性约束的精度。
闭合力的幅值会在 **500 N** 饱和，轴对齐力矩也有上限；这与理想硬约束仍有区别。
日志中的 `loop_max` / `Loop/max_anchor_error_m` 用于检查闭合点误差。
先检查机构，再进行训练；训练完成仍需对闭环误差与运动行为做验证。

每条运行命令都包含 `--allow-approximate-loop`，使使用的模型假设明确。
迁移版需要重新训练；原 Isaac Lab 权重即使网络维度一致，也不代表能在变化后的
动力学中保持原来的性能。

## 环境

推荐 Linux + Python 3.8 + Isaac Gym Preview 4，使用与旧 Gym 兼容的 PyTorch。
本机已有的 `isaacgym` Conda 环境包含 PyTorch `1.13.1+cu117`，可先复用。
不要把这个环境与原项目的 Python 3.10 / Isaac Sim 环境混装。

激活已有环境后，在本目录运行：

```bash
conda activate isaacgym
cd IsaacGym
python -m pip install -r requirements.txt
```

如果 Isaac Gym 还没有装进当前环境，先从 NVIDIA 获得 Preview 4，进入它的
`python/` 目录执行 `python -m pip install -e .`，然后回到本目录。
项目内已附带所需的旧 RSL-RL 组件及 BSD 许可，无需另装最新版 `rsl-rl-lib`。

`run.sh` 使用当前环境的 Python，并把该环境的 `lib` 加到动态库搜索路径中。
它可避免旧 Gym 加载时找不到 `libpython3.8.so`。运行时应始终先导入
`isaacgym` 再导入 `torch`；本项目入口已处理这个顺序。

## 先检查模型

固定机身、零动作、有窗口查看机构：

```bash
./run.sh play.py --allow-approximate-loop --zero-actions --fix-base --num_envs 1 --num_steps 500
```

无窗口检查使用 `--headless`：

```bash
./run.sh play.py --allow-approximate-loop --zero-actions --fix-base --num_envs 1 --num_steps 500 --headless
```

终端每 100 步输出一次闭合点最大误差，单位为毫米。零动作是机构检查，
没有训练好的策略时机器人不具备自主平衡能力。

用周期动作检查闭环数值误差并输出报告：

```bash
./run.sh smoke_test.py --allow-approximate-loop --headless --num-envs 4 --num-steps 100 --action-amplitude 4 --max-error-mm 10 --report validation/closure_check.json
```

报告包含锚点误差的峰值、95 分位数和闭环失败次数；超过指定误差容限或出现
闭环失败时程序返回非零。它验证给定工况下的近似精度，不代表策略已经学会平衡。

## 在 RTX 4060 上开始训练

先用少量环境完成一次 PPO 更新：

```bash
./run.sh train.py --allow-approximate-loop --num_envs 8 --max_iterations 1 --random-commands --headless
```

确认能完成后再训练。默认 64 个环境，更大的数量应根据显存、内存和实际步速
逐步增加，不照搬原项目的 4096：

```bash
./run.sh train.py --allow-approximate-loop --num_envs 64 --max_iterations 20000 --random-commands --headless
```

日志和权重默认保存在本目录的 `logs/wheelleg/<运行时间>/`，每 100 次更新保存
一次并在本次训练结束时保存。`--max_iterations` 表示本次新增的更新次数。

查看训练曲线：

```bash
tensorboard --logdir logs/wheelleg
```

继续训练：

```bash
./run.sh train.py --allow-approximate-loop --headless --random-commands --checkpoint logs/wheelleg/<运行时间>/model_100.pt --max_iterations 1000
```

## 播放训练结果

### 先训练站稳

两轮短测权重不能用于平衡或行走。首次正式训练先固定零速度和 0.15 m 腿长，
使用分开的髋关节/轮子探索噪声，降低轮子在初期频繁饱和到最大力矩的情况：

```bash
./run.sh train.py --allow-approximate-loop --headless --num_envs 256 --max_iterations 500 --command 0 0 0 0.15 --init-noise-std 1 1 1 1 0.1 0.1 --learning-rate 0.0003 --save-interval 25 --log-dir logs/standing
```

这是站稳训练的起点，500 轮本身不保证收敛。每 25 轮保存权重；`progress.json`
和 TensorBoard 记录完成回合的持续时间、物理终止、超时及闭环失败次数。
六个噪声值依次对应四个主动髋关节和两个轮子。加载已有权重续训时会恢复已学到的
噪声及优化器状态，因此不要同时指定 `--init-noise-std` 或 `--learning-rate`。

用确定性策略评估站稳时间，区分跌倒和正常达到 16 秒时间上限：

```bash
./run.sh evaluate.py --allow-approximate-loop --checkpoint logs/standing/model_500.pt --num-envs 8 --num-steps 800 --command 0 0 0 0.15 --report validation/standing_evaluation.json
```

站稳后再加入较小的前后速度指令；使用此阶段的权重播放时保持同样的腿长目标，
例如 `--keyboard --command 0 0 0 0.15`。评估脚本不会把短测通过当作行走成功。

### 再训练小幅前后运动

在已能保持平衡的权重基础上，先只训练 ±0.3 m/s 的前后指令，暂不加入转向，
腿长目标固定为 0.08 m。`--tracking-penalties` 在原奖励之外加入持续有效的线速度
和腿长误差惩罚，减轻远离高斯奖励目标后反馈很弱的问题；默认关闭，原任务奖励保持可选。

```bash
./run.sh train.py --allow-approximate-loop --headless --num_envs 256 --max_iterations 500 --checkpoint logs/standing_20261021/model_500.pt --random-commands --forward-speed-range -0.3 0.3 --yaw-speed-range 0 0 --leg-length-range 0.08 0.08 --command 0 0 0 0.08 --tracking-penalties --save-interval 25 --log-dir logs/forward_backward
```

这里的 500 表示新增更新轮数；从第 500 轮权重继续后，最终文件编号为 `model_1000.pt`。
旧站稳权重保留。新范围参数仅控制随机命令采样，固定命令仍由 `--command` 指定。
阶段结束要分别评估零速、前进、后退的实际速度及失败次数，不能仅凭奖励或轮数判断成功。

### 本机并行数量测试

使用同一个权重和相同 PPO 配置，每组运行 3 次更新并剔除第一次热身后，实测如下：

| 并行环境 | 每次更新耗时 | 每秒环境步数 | 相对采样吞吐 |
| --- | --- | --- | --- |
| 256 | 4.29 s | 1491 | 1.00 倍 |
| 512 | 4.43 s | 2892 | 1.94 倍 |
| 1024 | 4.89 s | 5240 | 3.51 倍 |

记录见 `validation/parallelism_20261021/report.json`。每组只有两次有效计时，适合
判断明显的吞吐提升，不用于预测收敛倍数。1024 环境每轮收集的样本是 256 的四倍，
所以轮数不能直接比较。切换时可按原计划剩余样本量折算新轮数，并以最终控制测试验收。
权重和优化器可以续接；仿真状态与观测历史会在重启环境时重新初始化。

本次实际从完整的第 725 轮权重切换为 1024 环境，按剩余样本量折算追加 69 轮，
结束编号为 794。训练再次被终止后，已从完整的第 740 轮权重恢复，剩余 54 轮由
独立后台任务继续，每 5 轮保存一次。当前进度见
`logs/forward_backward_1024_recovered_20261021/progress.json`，训练及自动评估状态见
`validation/final_evaluation_pipeline.json`。
权重生成后，后台任务会并行测试后退、停止和前进三个目标；也可以手动重复测试：

```bash
./run.sh evaluate.py --allow-approximate-loop --checkpoint logs/forward_backward_1024_recovered_20261021/model_794.pt --num-envs 3 --num-steps 800 --command 0 0 0 0.08 --test-speeds -0.3 0 0.3 --report validation/final_speed_scenarios.json
```

报告的 `speed_scenarios` 按目标速度分别统计误差和跌倒，避免前后速度在整体均值中抵消。

本机第 794 轮权重的固定速度实测如下（各 16 秒、每个速度一个环境、无初始扰动）：

| 目标速度 | 实际平均速度 | 速度 RMSE | 摔倒次数 |
| --- | --- | --- | --- |
| -0.3 m/s | -0.2941 m/s | 0.0397 m/s | 0 |
| 0 m/s | 0.0061 m/s | 0.0397 m/s | 0 |
| 0.3 m/s | 0.2943 m/s | 0.0436 m/s | 0 |

三组均由轮子支撑，未记录到机身或其他杆件接触地面。当前是较低的蹲姿，
末段腿长约 0.050–0.055 m，尚未达到 0.08 m 的目标，不能据此宣称腿高控制或抗扰动训练完成。
闭环最大误差约 10.64 mm。

键盘切换可使用连续速度序列评估，每段 200 步（4 秒），切换时不主动重置机器人：

```bash
./run.sh evaluate.py --allow-approximate-loop --checkpoint logs/forward_backward_1024_recovered_20261021/model_794.pt --num-envs 1 --command 0 0 0 0.08 --speed-sequence 0.3 0 -0.3 0 --segment-steps 200 --report validation/final_keyboard_sequence.json
```

报告的 `speed_sequence` 按段统计实际速度、误差、失败与支撑，整体误差使用逐时变化的目标。
`--speed-sequence` 与 `--test-speeds` 互斥；序列总步数自动计算，显式指定 `--num-steps` 时必须与之相等。

本机第 794 轮的前进→停止→后退→停止测试完成 16 秒，0 次物理失败，
各段末 2 秒的平均速度依次为 0.2978、-0.0015、-0.3021、0.0024 m/s。
该测试只在最后一步发生一次正常的 16 秒超时重置，速度切换之间没有重置。

使用这份权重进行键盘控制：

```bash
./run.sh play.py --allow-approximate-loop --checkpoint logs/forward_backward_1024_recovered_20261021/model_794.pt --keyboard --command 0 0 0 0.08 --num-steps 100000
```

↑/↓ 前后行驶，松开归零速目标，空格停车，R 重置。键盘模式不按训练的 16 秒时限
自动重置，物理失败与手动重置仍有效。当前权重仅验证前后和停车，转向尚未验证。

### 查看策略

```bash
./run.sh play.py --allow-approximate-loop --checkpoint logs/wheelleg/<运行时间>/model_100.pt --command 0 0 0 0.08 --num_envs 1 --num_steps 1000
```

有窗口运行时会显示机器人；添加 `--headless` 可以无窗口验证。权重路径需替换成
实际保存的位置。训练入口和播放入口均支持 `--num-envs` / `--num_envs` 等别名。

`--command VX VY WZ LENGTH` 指定固定命令，单位依次为 m/s、m/s、rad/s、m。
例如 `--command 0.5 0 0 0.08` 要求向前 0.5 m/s；默认命令是静止、腿长 0.08 m。
腿长须在 0.06–0.15 m。添加 `--random-commands` 后，每次重置会重新采样该环境
的命令，因此不再使用固定命令。窗口关闭时，播放会退出；训练会保存最近一次
完整更新的权重并退出。

## 键盘控制前后与转向

加载权重后添加 `--keyboard`，先点击仿真画面使窗口获得焦点：

```bash
./run.sh play.py --allow-approximate-loop --checkpoint logs/smoke_64_final/model_2.pt --keyboard --num_envs 1 --num_steps 100000
```

| 按键 | 动作 |
| --- | --- |
| 按住 ↑ / ↓ | 目标前进 / 后退速度，默认 ±0.3 m/s |
| 按住 ← / → | 目标左转 / 右转速度，默认 ±0.8 rad/s |
| 松开方向键 | 对应速度指令回到零 |
| 空格 | 清零速度指令；松开并重新按方向键后恢复控制 |
| R | 重置机器人 |

`--keyboard-speed` 和 `--keyboard-turn-speed` 可调整速度。WASD 保留给原生相机控制。
键盘模式需要一个可见窗口、一个机器人和加载的策略，不与 `--headless`、
`--fix-base`、`--zero-actions`、`--random-commands` 同时使用。键盘改变的是策略
收到的速度指令；`model_2.pt` 仅经过短测试，不能据此认为机器人已经学会平衡或行走。
正式使用时将该路径替换为训练充分的权重。

## 与原任务的差异

除了仿真后端与闭环近似，迁移版还明确调整了以下行为：

- 动作变化惩罚在覆盖上一帧动作之前计算，修复原代码中动作时序使该项失效的问题。
- 航向目标和累计航向按真实控制周期 0.02 s 积分，处理跨越 ±π 的角度连续性。
- 默认命令为静止，不再在运行 3 秒后自动把命令清零；固定或随机命令由参数选择。
- 移除原 USD 被动关节的隐含位置 PD 驱动，使这些关节保持被动。
- 闭合点误差超过 30 mm、机构几何无效或姿态越界时终止并重置；非有限状态也会终止。

这些调整会改变训练轨迹，因此不应把迁移前后的奖励曲线直接当作完全相同的实验。

## 主要参数与文件

| 项目 | 设置或位置 |
| --- | --- |
| 主动动作 | 左前髋、左后髋、右前髋、右后髋、左轮、右轮，共 6 维 |
| 观测 | 当前 31 维观测加 5 帧历史，共 186 维 |
| Actor / Critic | `[512,256,128,64]` / `[256,128,64]`，ELU |
| PPO | 25 步收集、5 轮优化、5 个 minibatch，继承原配置 |
| 物理步长 / 控制周期 | 0.0005 s / 0.02 s；每次控制期间计算 40 次闭环外力 |
| 机器人 | `assets/wheelleg/urdf/Robot_Model.urdf` |
| 闭合点参数 | `assets/wheelleg/loop_constraints.json` |
| 仿真、观测与奖励 | `wheelleg_gym/env.py` |
| PPO 适配与保存加载 | `wheelleg_gym/training.py` |
| 旧 RSL-RL 及许可 | `third_party/rsl_rl/` |

`--closure-stiffness` 的单位为 N/m，默认 50000；`--closure-damping` 的单位为
N·s/m，默认 25。`--physics-dt` 默认 0.0005 s，需能整除 0.02 s 控制周期。
闭环外力每个物理步重新计算（默认 2000 Hz），并不是 PhysX 内部的刚性闭环求解。
提高刚度可能增加数值不稳定性，调参后须重新检查误差和运动。
`--fix-base` 仅供播放/机构检查使用。全部参数可用 `./run.sh train.py --help`
或 `./run.sh play.py --help` 查看。
