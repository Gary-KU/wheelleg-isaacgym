# H723 / STM32Cube.AI 小网络

目标是缩小部署的 Actor，而不是把 PPO 的 Critic 和优化器放进单片机。
保留 132 个传感器历史输入、6 个动作输出和原外部控制器。
原 Actor 隐藏层为 512/256/128/64，240,966 参数，FP32 权重 941.27 KiB。
首选学生网络为 128/64，25,670 参数，FP32 权重 100.27 KiB、25,472 MAC/次。
以上只统计模型权重和乘累加，不是 STM32Cube.AI 的完整资源/耗时分析。

## 工作状态

当前仿真通过的发布候选位于`exports/h723_small`：128/64，100.27KiB原始FP32权重。
来源为小网络微调第10轮，现有航向外环`encoder_hold_heading_kp`由3改为6。
seed17/42/7/123的1cm跳台全部通过，最大倾角.193996rad、横偏.039755m；
8秒平地站立、1m/s行驶回归也通过。ONNX对实际传感器历史数值核对通过。
配套`controller_config.json`和`validation.json`必须与ONNX一起使用；
保持旧航向增益3只替换ONNX，不是这次通过的控制组合。
已停止旧增益3的自动导出监测，防止覆盖该候选；旧权重和完成的报告仍保留。
STM32Cube.AI分析和H723板上计时尚未执行。

原始教师：`logs/mujoco/ramp_v5/20261127_170720/model_200.pt`。
第一版 `logs/mujoco/distillation/student_128_64` 离线误差较小，但闭环出现漂移或过大倾角，**未通过**。
`exports/h723_student_candidate` 只是该失败候选的格式验证，不能作为通过验收的替代网络。
后续候选必须独立通过多种子 1cm 跳台和平地站立、行驶验证。
原教师策略不被覆盖。通过这些短仿真工况也不等于完成实机验证。

第二轮128/64学生为`student_128_64_dagger1`：四种子倾角均<.2rad，平地站立/行驶通过，
但三个种子横偏5.5–6.4cm，仍超5cm。小网络强化微调输出在
`logs/mujoco/h723_small_rl/20261128_093133`，200轮，学习率1e-6；
验证输出`logs/mujoco/h723_small_rl_validation`，不得以训练奖励替代验收。
另比较128/128/64（164.8KiB）及256/128（264.5KiB）容量方案，均需独立闭环验证。
小网络监测器带有`--resume --interval 10 --export-best exports/h723_small`：
只有多种子跳台和平地站立/行驶均通过，才自动生成该目录的ONNX，核对数值，
并写入`validation.json`及源权重哈希。失败候选不会自动导出到这个目录。
其中`simulation_stage_passed`仅表示仿真该台高的阶段通过，板上计时仍未完成。

## 蒸馏与复现

```bash
conda activate wheelleg_mujoco
python sim2sim/distill_policy.py \
  --checkpoint logs/mujoco/ramp_v5/20261127_170720/model_200.pt \
  --dataset logs/mujoco/distillation/teacher_data.pt \
  --output logs/mujoco/distillation/student_new \
  --hidden 128 64 --epochs 300
```

数据包括站立、前进、后退和正向 1cm 跳台。训练噪声种子42，留出数据种子123；
闭环评估另包含17/7。输入归一化折叠进第一层，动作归一化折叠进最后一层，
部署时无需新增预处理。左右轮动作转换为前进/转向分量分别归一化训练，再折叠回
原左右轮输出，防止小转向信号被大前进信号掩盖。
可用`--direct-actions`对比原始动作归一化；训练报告记录选项，后续训练使用余弦学习率衰减至1e-5。

若闭环偏离教师，使用 `--dagger-checkpoint <学生权重>` 采集学生访问的状态，
由教师提供标签，指定新 `--dataset` 路径，并用
`--extra-dataset logs/mujoco/distillation/teacher_data.pt` 混入原始轨迹。
缓存严格校验教师和驾驶网络哈希，不混用不同来源。

```bash
python sim2sim/watch_ramp.py --run <学生权重所在目录> \
  --height .01 --iterations 0 --flat-regression --seeds 17 42 7 123 \
  --output <独立验证目录>
```

目录中的 `model_0.pt` 是蒸馏后的网络。验证不改变原0.2rad/5cm门槛，也不改变电机力矩和速度限制。

## STM32Cube.AI 导出

```bash
python tools/deployment/export_policy.py --checkpoint <已验证的学生权重> \
  --output exports/h723_small \
  --validation-inputs logs/mujoco/distillation/teacher_data.pt --require-onnx
```

导入 `actor.onnx`，FP32，输入 `observations` 为 `[1,132]`，输出 `actions` 为 `[1,6]`。
不导入包含 Critic 的训练检查点。ONNX为opset11，仅使用Gemm/Elu；
导出会检查模型，并通过ONNX Runtime核对实际传感器历史上的PyTorch输出。
ST的[ONNX算子支持表](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_onnx.html)
列有FP32 Elu，最终以所安装的STM32Cube.AI版本分析结果为准。

在STM32Cube.AI里选择实际H723型号，执行Analyze，再执行Validate on target。
记录生成器版本、权重/激活内存、单次推理时间，确认推理和传感器/控制器一起能满足2ms控制周期。
当前环境没有ST编译器和板上计时结果；不能用Python耗时或仅权重大小替代这些结果。
暂不量化；INT8是独立候选，需要重新验证数值和闭环。

`export_actor_c.py` 为可选的便携FP32 C参考，非STM32Cube.AI生成代码。
它无堆分配、包含ELU，编译后与PyTorch比对；128/64网络显式临时数组1024字节，
不含调用方、数学库和编译器栈开销，因此不能当作完整RAM预算。
