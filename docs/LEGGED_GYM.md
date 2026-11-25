# legged_gym 任务入口

本项目借用 [Wheel-Legged-Gym 固定版本](https://github.com/clearlab-sustech/Wheel-Legged-Gym/tree/c354431e5633eb98d9663fa3fbe444201d2de383) 的任务注册、嵌套配置、奖励方法和 runner 接口惯例，沿用 `legged_gym` 包名。物理、闭环近似与机器人控制使用本项目实现。

## 调用顺序

```text
manage.sh train → scripts/train.py
  → register_tasks() → task_registry.prepare_cfgs()
  → task_registry.make_env() → WheelLegged / 传感器任务
  → TaskRunner → PPOTrainer → PPO

manage.sh play / stairs / evaluate → scripts/play.py
  → 同一任务注册器 → 加载检查点 → 策略推理 → env.step()
```

先读 [config/base.py](../legged_gym/envs/wheelleg/config/base.py) 的参数，再读 [tasks/base.py](../legged_gym/envs/wheelleg/tasks/base.py) 的配置映射与奖励选择，最后读 [robot.py](../legged_gym/envs/wheelleg/robot.py) 的物理步进。[register.py](../legged_gym/envs/wheelleg/register.py) 是唯一任务注册清单。

## 注册任务

| 名称 | 观测 / 动作 | 用途 |
| --- | --- | --- |
| `wheelleg_flat` | 198 / 6 | 平地速度、转向和腿长训练 |
| `wheelleg_stand` | 198 / 6 | 零速站立 |
| `wheelleg_stairs` | 252 / 6 | 阶梯课程 |
| `wheelleg_sensor` | 132 / 6 | 实机可获取的 IMU / 编码器状态 |
| `wheelleg_sensor_tracking` | 132 / 6 | 停稳与直行课程至 1.0 m/s |
| `wheelleg_sensor_quiet` | 132 / 6 | 候选安静站立奖励/阻尼对照，尚未通过验收 |

所有任务使用当前 CAD 资产，**当前没有训练好的新模型权重**。旧模型权重已移除，不能用来证明新模型的站立或阶梯能力。

## 训练、续训和播放

```bash
conda activate isaacgym
./manage.sh tasks
./manage.sh train --task wheelleg_sensor_tracking --num_envs 128 --headless
# 有限接口检查可加 --max_iterations 3

# 从同一任务的完整检查点续训，新增 100 次更新
./manage.sh train --task wheelleg_sensor_tracking --num_envs 128 --headless \
  --checkpoint /绝对路径/model_100.pt --max_iterations 100

./manage.sh play --task wheelleg_sensor_tracking --checkpoint /绝对路径/model_200.pt
./manage.sh evaluate --task wheelleg_sensor_tracking \
  --checkpoint /绝对路径/model_200.pt --num_steps 3000 --report validation/result.json
```

训练默认使用 `cuda:0`；CPU 检查加 `--sim_device cpu --rl_device cpu`。仿真与策略设备必须相同。权重写入 `logs/<experiment_name>/<时间与名称>/model_<更新次数>.pt`；同目录保存配置快照、`progress.json` 和 TensorBoard 数据。完整参数用 `./manage.sh train --help`、`./manage.sh play --help` 查询。

播放默认使用单机器人。点击仿真窗口后用方向键前后/转向、Q/E 调腿长、空格停止、R 重置。驾驶与跳跃的专项演示使用 `./manage.sh stand`，交替抬轮使用 `./manage.sh lift`，两者需匹配的 198 维策略；具体按键见 [跳跃](JUMP.md) 和 [抬轮](WHEEL_LIFT.md)。

## 配置、奖励和权重

基础参数与 PPO 默认值在 `config/base.py`，传感器任务及速度课程在 `config/sensors.py`。非零的 `rewards.scales` 项注册 `_reward_<名称>()`，设为零可关闭；未知名称会报错。平地公式在 `rewards/flat.py`，阶梯公式在 `rewards/stairs.py`。奖励按每个 20 ms 控制步累计，没有额外乘 dt。

198 维为 33 × 6 帧，252 维为 42 × 6 帧，传感器任务为 22 × 6 帧。前馈策略的 `obs_history_length=1`，不会再次堆叠。任务返回六项，由 runner 适配；基础机器人保留五项返回接口。

播放和续训要求检查点包含 `legged_gym_env_config` / `legged_gym_train_config`，并检查模型指纹和观测尺寸。`--warm-start` 用于显式的权重初始化，不恢复优化器或训练进度。旧维度转换工具及重复训练入口已删除。

`compat/` 仅供抬轮、物理检查和专项评估共享参数解析与恢复；训练只走注册任务入口。更多说明见 [代码导读](CODE_GUIDE.md)、[项目结构](PROJECT_LAYOUT.md) 和 [扩展指南](EXTENDING.md)。
