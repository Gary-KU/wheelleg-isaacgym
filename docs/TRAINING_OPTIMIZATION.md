# 训练优化与复测

这次优化针对实际操作中的整机晃动、起停换向和转向，不改变机器人资产或闭环物理参数。
旧权重保留；新实验单独记录。代码测试和短训练成功不等于机器人已经稳定。

## 已接入的改动

| 问题 | 处理 |
| --- | --- |
| 正常超时的收益估计使用了前一步价值 | 改为终止前观测的价值；物理失败不补值，不使用重置后的状态 |
| 旧观测缺少绝对腿长 | `--observe-leg-lengths` 追加左右腿长，186 → 198 维；旧权重通过补零转换保留初始预测 |
| 机身小角度晃动与轮子力矩突变的惩罚偏弱 | 稳定项按机体倾角、横滚/俯仰角速度和实际 Nm 力矩变化计算 |
| 旧训练只有前后速度、回合内不换目标 | 加入 ±0.8 rad/s 转向、每 2 秒重采样、20% 停车目标 |
| 键盘指令直接阶跃 | 训练和推理共用限加速度：平移 0.8 m/s²、偏航 2 rad/s² |
| 续训/播放参数需要手动记忆 | `runtime_config.py` 统一配置恢复与覆盖顺序，构建环境前检查网络输入维度 |
| 重置可能污染力矩变化统计 | 优先使用终止前实际力矩和对应上一力矩，避免把回合边界算成控制尖峰 |

参数优先级：**默认值 → 检查点环境参数 → profile → 显式命令行参数**。
推理只继承物理、观测、指令变化限制和腿长目标；不继承随机训练模式、环境数量、设备或窗口模式。

`config.py` 保留原默认值以兼容旧实验；优化功能通过 `--profile stability` 明确启用。
从零训练时，预设使用腿/轮分开的探索噪声 `[1,1,1,1,0.1,0.1]`，避免把轮子噪声大量裁剪到力矩上限。
续训保留已训练的噪声和优化器状态，自适应学习率上限设为 0.0003。

## 一条命令运行前后对比

```bash
conda activate isaacgym
cd IsaacGym
./run.sh tools/optimize_training.py \
  --checkpoint logs/forward_backward_1024_recovered_20261021/model_794.pt \
  --log-dir logs/my_stability_optimization \
  --iterations 200 --num-envs 1024
```

目录必须是新建或空目录，避免覆盖已有实验。该命令会实际启动有限训练，依次执行：

1. CPU 基线评估：60 秒静止、16 秒起停转向、12 秒快速切换；均关闭正常时限重置。
2. 如源策略是 186 维，生成独立的 198 维起点，不覆盖源文件。
3. GPU 上使用稳定性预设完成指定的新增更新，每 10 次保存。
4. 用完全相同的 CPU 物理与指令限制复测。
5. 写出前后指标、脚本检查及未覆盖范围。

基线也使用相同的 0.8 / 2 指令限制，因此报告比较的是同条件策略表现，不能把全部改善归因于某个单独奖励项。
测试没有初始扰动，不能据此宣称抗扰动通过。

## 已完成的有限实验

本次正式对比目录：

```text
logs/stability_optimization_20261021_215150/
```

已从第 794 轮增加 200 次更新，保存到第 994 轮，并完成三组前后复测。
程序状态为 `completed`，整体质量为 `needs_improvement`。

起停转向测试中，俯仰 RMS 下降 37.4%，前后速度误差下降 21.6%，偏航误差下降 56.3%（误差相对限加速度后目标）。
但横滚 RMS 从 0.65° 增至 1.71°；快速切换的力矩变化率 RMS 增加 7.1%，原始偏航目标误差增加 5.2%。
静止/停车末段腿长仍约 5 cm，未达到 8 cm 目标。新旧三组均为 0 次物理失败，不能声称减少了摔倒。

完整结论和启动/后续阶段比较见 [已复核报告](../logs/stability_optimization_20261021_215150/reviewed_comparison.md)。
新权重为独立实验版本，尚未经实际键盘及扰动验证。目录中的记录包括：

| 文件 | 内容 |
| --- | --- |
| `optimization_status.json` | 当前阶段、子进程、错误或程序完成状态 |
| `plan.json` | 完整的训练和评估计划 |
| `input/model_794_legs.pt` | 保留源策略预测的 198 维起点 |
| `training/config.json` | 实际训练参数 |
| `training/progress.json` | 最近完成的一次更新及姿态/力矩/跟踪指标 |
| `training/model_N.pt` | 中间与最终权重 |
| `evaluation/baseline/`、`evaluation/candidate/` | 同条件的原始评估报告与日志 |
| `actual_training_environment.json` | 从训练后检查点读取的真实环境配置 |
| `comparison.json` | 晃动、力矩变化、速度误差、失败与接地情况的前后对比 |
| `reviewed_comparison.md`、`reviewed_metrics.json` | 复核后的结论、峰值、启动/后续阶段及低蹲限制 |

启动记录索引见 [optimization_work.json](../validation/optimization_work.json)。
历史 PID 不能证明任务当前还活着，检查状态时间与实际进程。

## 低蹲补查：力矩与关节角限位

针对“髋关节是否无力、是否缺重力前馈”的观察，另用第 994 轮策略在 CPU 上运行 8 秒零速测试，
腿长目标为 0.08 m，统计最后 2 秒；无物理失败。原始逐步数据见
[actuation_994_20261021_224035.json](../validation/actuation_994_20261021_224035.json)。本次仅做诊断，没有改变控制器。

| 项目 | 测量或代码事实 |
| --- | --- |
| 髋关节力矩 | `clip(0.5 × action, -9, 9)` Nm；0.5 是缩放，不是最大值。Actor 末层为 Linear，输出不限定在 ±1 |
| 最后 2 秒四髋力矩 RMS | 按 LF、LH、RF、RH 顺序为 0.516、0.379、1.103、0.481 Nm；四髋绝对峰值最大为 1.337 Nm |
| 主动髋角度 | 距最近角限位均超过 0.36 rad，没有顶住主动角限位 |
| 被动膝角度 | 四个 Joint2 在最后 2 秒的所有采样都紧贴折叠方向限位 ±1.233774 rad，数值误差小于 5e-7 rad |
| 实际腿长 | 左右约 0.05002 / 0.04991 m，目标为 0.08 m |
| 重力与前馈 | 重力为 -9.81 m/s² 且启用；没有显式重力/支撑前馈，策略直接生成总关节力矩 |

独立 URDF 正运动学也表明：约 5 cm 腿长对应上述四个被动关节的折叠限位；
8 cm 时被动膝约 ±0.95462 rad，主动髋约 ±0.614 rad，目标几何上可达。
因此，目前不支持“9 Nm 硬限幅导致低蹲”，更支持策略收缩到被动关节止挡。
利用限位反力承重是与这些数据一致的解释，但本次未直接测量限位反力，不能量化其承重比例。

先前评估的“无非轮部件接地”不能排除内部关节止挡承重；零次摔倒也不能证明主动支撑已经学会。
后续应同时检查腿长误差、左右差异、被动关节限位停留时间和实际力矩。
若加入支撑前馈，应按机构姿态、关节方向和负载计算，并以“前馈 + RL 修正”的动作含义重新训练或受控续训。
现有 994 权重已经学输出总力矩，直接在播放时叠加补偿会改变控制含义，不能视为已验证的修复。

## 主动支撑课程与奖励修正

原奖励存在低蹲局部解：两腿 5 cm、目标 8 cm 时，旧腿长正奖仍为 +1.0976，
误差成本仅 -0.18，合计 +0.9176；正确 8 cm 为 +2，每步只差 1.0824。
直立、零速、偏航和原角度对称项仍可在低蹲时给奖，且没有被动角限位成本。
原角度对称项也不能区分左右分别对称的 5/5 cm 与 5/7 cm，需要单独约束左右腿长差。

新增 `--profile support`，保留原物理、力矩上限和重置姿态，不加重力前馈，先训练固定零速、8 cm 支撑。
只有启用 `support_rewards` 才替换旧宽腿长奖励与旧腿长成本，旧实验默认行为保持兼容。
令 `eL=L_left-target`、`eR=L_right-target`，新分项为：

| 分项 | 计算与目的 |
| --- | --- |
| `leg_tracking` | `2*exp(-0.5*((eL/0.01)^2+(eR/0.01)^2))`，明确奖励目标附近 |
| `leg_error` | 两腿各 `-0.25*min((e/0.01)^2,9)`；有界，避免初始高腿姿态出现无限负成本 |
| `leg_asymmetry` | `-0.5*min(((L_left-L_right)/0.01)^2,9)`，约束实际左右长度 |
| `passive_limit` | 四个被动膝到真实 URDF 两侧边界的距离，进入 0.10 rad 软区后平滑惩罚；完全贴边时合计 -4 |

5 cm 对称止挡状态的上述分项约 -8.50，8 cm 且远离止挡时约 +2，差约 10.5。
另外，支撑模式移除零速时不连续的方向 sign 项，动作大小成本按实际 Nm 计算。
没有增加一碰限位就立即终止的规则，以便策略有机会学习离开止挡。
各分项及 `Support/` 腿长、达标比例、贴限位比例会写入训练日志。

第 994 轮髋输出已倾向折叠，探索约 0.5 Nm 且学习率很低，因此新课程使用独立起点：
保留 actor 隐藏层和轮子输出，清零四髋末层输出，重新初始化 critic/优化器，髋探索标准差 1.5 Nm、轮子约 1 Nm。
源文件保留，训练编号从 0 重新计数，来源信息随权重保存在 `training_config.initialization_provenance`。
这是一组新学习设置，不是对“只修改某一个奖励系数”的因果验证。

本次目录 `logs/support_learning_20261021_230147/`，先执行 100 次更新，再用同条件支撑评估决定是否继续。
实际阶段、权重与启动记录见 [support_work.json](../validation/support_work.json)。
本课程完成支撑之前，不能把它当成已训练好的前后/转向策略。

实际首阶段 100 次更新的 20 秒复测仍为左右约 5.20 / 5.26 cm，两腿达标比例 0%、后段始终存在被动膝贴限位，未通过。
另用 8 cm 初态、CPU 128 环境训练 20 次更新，12 秒复测仍约 5.02 / 5.02 cm，亦未通过；
训练中的平均腿长混入探索与重置，不能据此声称主动支撑成功。两次训练已结束，尚未产生合格支撑权重。

可选 `--initial-leg-length .08` 会同时设置完整主动/被动关节、根高度及真实 SDK FK 缓存；
`--default-leg-pose` 可显式恢复原零关节起点。该设置不提供 PD 或前馈力矩，CPU/GPU 零动作测试都会塌下。
新增姿态通过 7 项有限 CPU/GPU 检查，GPU 8 环境初始化/reset 的闭环误差约 0.00401 mm。
以后比较不同权重时应显式使用相同初态，并同时检查是否能从原起点稳定到目标。

支撑检查示例（从 `IsaacGym/` 执行，先替换权重路径）：

```bash
./run.sh tools/evaluate_support.py --allow-approximate-loop \
  --checkpoint logs/your_support_run/model_100.pt \
  --seconds 60 --settle-seconds 2 --report validation/support_60s.json
```

报告包含全程与去掉首 2 秒的统计；物理失败全程计数，不因自动重置或去掉启动样本而忽略。
工程初筛要求后段两腿同时在目标 ±5 mm 内至少 95%、被动膝不进入 0.01 rad 边界区、无非轮接地、全程无物理失败，
并检查俯仰/横滚 RMS。通过固定初态站立仍不证明扰动恢复或任意键盘稳定。
旧 994 在同一检查下腿长达标比例为 0%、后段每个采样均存在被动膝贴限位；数值见本次目录的 `evaluation/baseline_994_20s.json`。

## 手动训练、续训与播放

### 可选混合支撑：位置 PD + RL

`--profile hybrid` 为零速、8 cm 初态的混合支撑课程，基础控制器是**位置 PD，不是重力前馈**。
`leg_control=hip_pd` 以目标腿长的对称逆解生成四髋目标，力矩为 `Kp*(q_target-q)-Kd*qdot + RL_residual`，
默认 Kp=80、Kd=4，总和仍限制在每关节 ±9 Nm。RL 每 0.02 秒更新，PD 每 0.0005 秒读取新关节状态并重算。
轮子继续使用策略力矩输出。旧权重默认 `leg_control=torque`，推理不会因预设名称而自动改变旧动作含义。

当前基础混合策略（髋 RL 输出清零，轮子保留旧初始化）12 秒零速短测：后 10 秒左右腿均约 7.73 cm，
两腿同时满足 8 cm ±5 mm 的比例为 100%，被动膝贴限位比例为 0%，无物理失败。
这验证了 PD 基础支撑，不能称为 RL 已独立学会支撑，也不能代替长时或运动验收。
记录见 `logs/hybrid_support_20261021_234236/evaluation/controller_initial_12s.json`，索引见
[hybrid_work.json](../validation/hybrid_work.json)。该历史版本尚未开始混合策略重训，也未加入重力前馈。

混合控制的 `last_applied_torques` 保存真实上一周期末力矩，避免把 RL 动作误当成总力矩。
支撑评估的力矩 RMS 是控制周期端点样本；峰值与饱和比例另采全部物理子步，报告注明两者采样区别。

### 2026-10-22：支撑前馈与航向反馈

参考 [Wheel-Legged-Gym 的 VMC](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/master/wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py)
中“反馈支撑力 + 常数支撑前馈，再映射为关节力矩”的结构。参考机器人的连杆、质量和动作定义不同，未复制其40 N、串联腿Jacobian或策略权重。

本机仍用髋位置PD，每条腿额外加入径向支撑力22.4649 N，通过**本机闭环五连杆**的解析Jacobian计算 `tau_ff = F*dL/dq`。
这是把4.58 kg非轮质量集中为支撑负载的近似，不是各杆件COM的精确重力补偿。8 cm时四髋约为 `[-1.64,+1.64,+1.64,-1.64] Nm`。
PD每个物理步（2 kHz）更新；支撑前馈根据周期开始的实测关节角在50 Hz更新并保持；RL也在50 Hz更新。
原来的腿部静态下垂因此缩小。前馈通过电机力矩实现，没有给机身额外向上的外力，总和仍限于每关节±9 Nm。

轮子增加可选航向/转速反馈：`b = clip(Kp*(yaw-yaw_ref)+Kd*(wz-wz_ref), -1, 1)`，两轮原策略力矩各减 `b`。
本模型左轮轴为世界−Y、右轮轴为+Y，所以同号正关节力矩产生正偏航；反馈符号经实际接地仿真检查。
`yaw_ref`由偏航速度命令积分，零命令时保持初始航向。选用Kp=2 Nm/rad、Kd=0.1 Nm·s/rad；单轮修正最多1 Nm，总力矩仍受9 Nm限制。

当时入口为 `./stand.sh`（当前已合并站立/跳跃，见 [运行说明](USAGE.md)），当时配置随 `logs/standing_support_20261022_093823/input/standing_hold.pt` 保存。
这个文件保留了原轮子策略，四髋RL输出为零，**本轮没有新做RL训练**；伸腿主要由PD与支撑前馈完成。
默认从5 cm低蹲姿态起步，目标8 cm，不是倒地翻身恢复。旧纯力矩权重的默认行为不变，所有新增反馈默认关闭。
`--profile hybrid`从新模型启动默认无前馈/航向反馈，续训时保留权重内已有的这几个参数；`--profile support`显式恢复纯力矩课程并关闭它们。

支撑报告现在额外检查位置、速度与航向：后段平面速度RMS≤0.05 m/s、最大XY位移≤5 cm、航向偏差峰值≤0.1 rad。
`tools/evaluate_support.py --kick-at-seconds 5 --kick-velocity 0.15 0 0`可测试一次世界X方向的速度增量（m/s）；不等于实测推力或任意扰动保证。
配置与最新验收索引见 [standing_work.json](../validation/standing_work.json)，结果见该实验的`evaluation/`。

从零使用预设：

```bash
./run.sh train.py --allow-approximate-loop --profile stability --headless --max-iterations 200 --log-dir logs/stability_from_scratch
```

手动续训 198 维检查点：

```bash
./run.sh train.py --allow-approximate-loop --profile stability --headless --checkpoint logs/my_stability_optimization/training/model_994.pt --max-iterations 200 --log-dir logs/stability_next
```

旧 186 维权重不能直接使用新增输入训练；先用观测升级工具，或使用上述自动对比流程。
`--num-envs`、速度范围和其他显式参数可以覆盖预设。

复测合格后再加载新权重进行人工检查：

```bash
./run.sh play.py --allow-approximate-loop --checkpoint logs/my_stability_optimization/training/model_994.pt --keyboard --num-steps 100000 --event-log validation/optimized_keyboard_events.jsonl
```

播放会从检查点恢复腿长观测和指令变化限制。显式传入的设置仍优先，例如 `--linear-acceleration 0` 可以关闭该限制。

## 验证范围

本次接入已通过 94 项 CPU 回归、1024 环境两次 GPU 更新，以及缩短时长的完整前后评估流程。
这些检查验证配置、维度、超时估值和流程是否正确，不用于判断控制已经收敛。

正式报告同时保留：

- 物理失败和正常超时数量；
- 横滚/俯仰 RMS、角速度 RMS；
- 实际力矩变化、饱和比例；
- 原始目标和限速后目标的速度/偏航误差；
- 腿长、轮子支撑、机身/杆件接地情况。

`optimization_status.json` 的 `state=completed` 仅表示程序完成；
`quality_status=needs_improvement` 表示脚本指标仍不合格；即使脚本检查通过，也需要人工键盘操作验证。
