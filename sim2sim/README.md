# MuJoCo：物理检查、策略迁移与微调

在 IsaacGym 项目根目录运行。使用独立环境，不需要导入 Isaac Gym：

```bash
conda create -n wheelleg_mujoco python=3.10
conda activate wheelleg_mujoco
python -m pip install -r sim2sim/requirements.txt
python sim2sim/check_model.py
python sim2sim/check_interface.py
python sim2sim/run.py --viewer --terrain ramp --ramp-height .01 --seconds 300
```

默认加载仓库内`exports/h723_small/checkpoint.pt`，与已导出的ONNX使用同一Actor及控制配置。
初始腿长9cm，方向键驾驶，Q/E变腿长，R重置。原始FP32权重约100KiB；
四种子1cm正向跳台和平地站立/行驶短回归通过，不代表任意操纵、更高台阶或实机通过。
该版本含航向增益6，不能只替换ONNX而保留旧增益3。
部署只导入`actor.onnx`到STM32Cube.AI；`checkpoint.pt`含训练网络状态，只用于Python仿真/续训。
以下带`logs/...`的路径记录历史本机实验，不随仓库发布，需换成自己的检查点。

```bash
python sim2sim/train.py --checkpoint exports/h723_small/checkpoint.pt --stage ramp --ramp-height .01 --iterations 100 --num-envs 8
python sim2sim/evaluate_flight.py --checkpoint exports/h723_small/checkpoint.pt --ramp-height .01 --policy-only --report logs/mujoco/replay.json
```

## 文件职责

- `build_model.py`：从源 URDF 和闭环元数据生成 MJCF，保留源质量/惯量/坐标精度及视觉碰撞区分。两个轴向点约束表示每处闭合铰链；禁止以改质量、惯量、摩擦来提高得分。
- `check_model.py`：核对初态闭环、质量/惯量/FK、重力与髋支撑、轮力矩方向、IMU 坐标。报告和模型 SHA256 是训练入口的前置条件。
- `environment.py`：批量 CPU 物理、原 132 维传感器观测/噪声/延迟和历史、原轮角/IMU外环、髋 PD/前馈、轮力矩与髋累计2秒峰值限幅。
- `train.py`：复用项目 PPO；从现有 Actor/探索参数初始化，Critic 和优化器重建。原 Actor 逐位校验后保存在新目录 `model_0.pt`；每10轮及结束保存。
- `evaluate.py`：确定性推理，记录失败次数、单回合连续时长、姿态、位移和力矩。10秒短检查不是完整验收。
- `run.py`：原网络或微调检查点的可视化入口，`--checkpoint <路径> --viewer`。

## 控制与测量

所有控制反馈只来自原13维IMU/虚拟腿/轮速和两个同步轮角。世界位置/线速度仅用于奖励与报告。机身速度使用 MuJoCo `mjOBJ_XBODY`，不能使用惯性主轴坐标的 `mjOBJ_BODY` 作为 IMU 输入。

训练当前先执行 `stand` 阶段，不随机驾驶。`drive` 阶段以原0.8m/s²斜坡采样零速及正负1m/s。驾驶、转向、变腿长全工况验收尚需完成。当前奖励是独立MuJoCo适配奖励（姿态、速度、转向、横向速度、腿长误差、倒地惩罚），不是Gym历史奖励的数值复刻，日志不得跨奖励配置直接比较。

生成MJCF带本机网格路径，换电脑重新生成。实验日志和候选权重不自动推送GitHub；精选H723检查点和部署包单独发布在`exports/h723_small`。

## 本次初始结果

已修复视觉网格碰撞和惯性坐标误作IMU的转换错误。物理检查12项、后端接口检查12项通过。完成25轮原Actor初始化的站立微调；两个种子各4环境的10秒站立和一组1m/s前进短检查均无倒地。还不是所有高度、转向、伸缩腿或实机的完整验收。

本机微调权重：`logs/mujoco_training/20261127_140335/model_25.pt`。

```bash
python sim2sim/run.py --checkpoint logs/mujoco_training/20261127_140335/model_25.pt --viewer --seconds 60
```

## 键盘遥控

```bash
python sim2sim/run.py --checkpoint logs/mujoco_training/20261127_140335/model_25.pt --viewer --seconds 300
```

点击窗口获得焦点。按住↑/↓前进后退（1m/s），←/→转向（1rad/s）；可组合。松键请求归零，按原斜坡减速。按住Q/E缩腿/伸腿（8–10cm，1cm/s）；松键保持腿长。空格发零速度请求；R重置。失去焦点自动零速。鼠标左键拖动旋转视角，滚轮缩放。退出关窗口。

默认开启键盘；固定指令播放可用 `--viewer --no-keyboard --speed 1`。遥控只作用于MuJoCo，不向实机发指令。键盘控制使用GLFW轮询按下/释放状态，不依赖passive viewer只有按键回调的接口。

## 波浪路

```bash
python sim2sim/run.py --checkpoint logs/mujoco_training/20261127_140335/model_25.pt --viewer --terrain wave --seconds 300
```

默认峰谷差5mm、波长40cm，出生点周围±40cm为平地。`--wave-height .01`改为1cm峰谷差；`--wave-length .4`设置波长（m）。地形为10m×4m有限区域。键盘驾驶时请保持在地形内。

原25轮微调权重在5mm/40cm波浪、seed42、4环境、1m/s、8秒检查中零倒地，实际稳定前进约.982–.987m/s，最大pitch .19252rad。只代表该条件短检查，不含停车/转向/其他高度的完整波浪验收。高度场没有输入网络，仍只用IMU与编码器反馈。

## 飞坡机械实验（未验收）

```bash
python sim2sim/check_flight.py
python sim2sim/evaluate_flight.py --ramp-height .01 --report logs/mujoco/flight.json
python sim2sim/run.py --checkpoint logs/mujoco_training/20261127_140335/model_25.pt --viewer --terrain ramp --ramp-height .01 --seconds 300
```

跳台从x=.4开始，坡长.3m，末端落差由`--ramp-height`指定，默认3cm。只在前进方向。原模型、摩擦、电机限幅不变。

`--flight-compliance`是默认关闭的实验选项；`--flight-mode support`只撤支撑前馈，`soft`还降低腿刚度。用额外3轴加速度计的延迟/带噪模长检测低支撑（不是MuJoCo接触真值），落地恢复支撑。加速度噪声/偏置.05m/s²是示例假设，非实测。未向原132维Actor添加任何输入。暂停腾空轮角里程并在状态切换重捕获，避免空转误作位移。

目前不建议启用：首轮3cm跳台两组均失稳；1cm跳台原控制未倒地但pitch达.416rad，仍未达到原.2rad标准。弱腿方案反而倒地。不能把“撤前馈/降刚度”视为已验证收益。

机械报告含各关节峰值速度、力矩和负机械功；负机械功不等于电池回收电能。没有Ke、减速比和驱动器回灌参数，反电动势/母线电压为null，不猜参数。

## 跳台训练

```bash
python sim2sim/train.py --stage ramp --ramp-height .01 --checkpoint logs/mujoco_training/20261127_140335/model_25.pt --iterations 100 --num-envs 8
python sim2sim/evaluate_flight.py --checkpoint <新权重> --ramp-height .01 --policy-only --report logs/mujoco/ramp_eval.json
```

先1cm，再2/3cm。命令目标1m/s，线加速度仍.8；不通过减速或降电机限制验收。训练起步时刻在1.5–2.3秒随机化，验收仍固定2秒。训练完整运行6秒，不在首次稳定落地后提前结束。稳定奖励要求越过跳台后连续.5秒姿态<.2rad、横偏<.05m、速度误差<.1m/s；另惩罚横偏、俯仰角速度和过大倾角。奖励统一乘.01以控制500Hz下Critic回报尺度，PPO优势仍按原实现归一化。`--rollout-steps 500`提供1秒采样段；ramp的gamma=.999、GAE lambda=.995。报告另严格核对离地、恢复行驶、姿态、横偏与闭环。不启用已失败的flight-compliance。训练只用原132维观测，位置/接触只用于奖励或验收。

`tune_landing.py --checkpoint <权重> --output <目录>`比较六组带噪/延迟IMU俯仰阻尼参数，每组保存独立权重配置并用seed17/42按原标准验证。改变控制参数的候选必须再测平地，不能直接当作通过验收的策略。历史v3训练最终未通过，较好的早期权重也未满足倾角标准。

2026-11-27 v4：在v3第350轮Actor上比较九组阻尼（额外三组用`--extended`）。选取rate kd=.05、guard kp=.2继续微调；参数保存在候选权重配置中，不改变原默认值。该候选1cm台阶seed17/42最大倾角.200174/.199237rad（只有42通过），横偏.00923/.02460m；8秒平地双环境1m/s检查无摔倒，稳定速度1.005/1.013m/s，最大倾角.20148rad，仍不满足.2rad门槛。不能把这些结果当作已验收。

v4训练输出：`logs/mujoco/ramp_v4/20261127_160405`。`watch_ramp.py --flat-regression`从第0轮起每50轮执行双种子跳台及8秒平地站立/1m/s回归；`best.json`记录验收表现较好的检查点，避免以最终轮次替代最佳表现。1cm阶段通过仍不代表2/3cm通过。

v5从v4第200轮开始，用`--fine`比较guard kp=.3/.4/.6（rate kd=.05不变）。选中kp=.4：seed17/42的1cm跳台完整通过，峰值倾角.196825/.195773rad；8秒平地1m/s双环境无摔倒，峰值倾角.193800rad，稳定速度1.00286/1.01009m/s。此结论仅覆盖这两个跳台种子和短平地行驶测试。

v5训练输出`logs/mujoco/ramp_v5/20261127_170720`，500轮、学习率3e-6、每段500步。训练倾角惩罚从.16rad开始、尺度.05rad，增加验收边界内的裕量；验收仍为.2rad。监测使用`--seeds 17 42 7 123 --flat-regression`，包括未参与参数选择的7/123种子及平地站立/行驶回归。原132维观测、速度斜坡和电机限幅保持不变。
