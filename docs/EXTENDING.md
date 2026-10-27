# 添加任务、奖励和机器人

先读[目录职责](PROJECT_LAYOUT.md)和[当前接口](CODE_GUIDE.md)。现有轮腿任务在 `legged_gym/envs/wheelleg/`，共用训练在 `legged_gym/learning/`。以下路线保留已有任务行为，新增任务使用独立名字和实验目录。

## 同一机器人加一个任务

例如建立固定低速任务。先在 [wheelleg_config.py](../legged_gym/envs/wheelleg/wheelleg_config.py) 中继承配置：

```python
class WheelLeggedSlowCfg(WheelLeggedStandCfg):
    class commands(WheelLeggedStandCfg.commands):
        command = (0.1, 0.0, 0.0, 0.08)
        forward_speed_range = (0.1, 0.1)


class WheelLeggedSlowCfgPPO(WheelLeggedStandCfgPPO):
    class runner(WheelLeggedStandCfgPPO.runner):
        experiment_name = "wheelleg_slow"
        pretrained_checkpoint = None  # 新任务播放时明确指定自己的权重
```

仅修改配置时，直接复用 `WheelLegged`，无需新增任务文件或空子类。在 [register.py](../legged_gym/envs/wheelleg/register.py) 的 `register(registry)` 中导入新配置并追加注册：

```python
from .wheelleg import WheelLegged
from .wheelleg_config import WheelLeggedSlowCfg, WheelLeggedSlowCfgPPO

if "wheelleg_slow" not in registry.task_classes:
    registry.register(
        "wheelleg_slow", WheelLegged,
        WheelLeggedSlowCfg(), WheelLeggedSlowCfgPPO(),
    )
```

`register_tasks()` 已调用该机器人的注册函数。随后用任务名做短检查：

```bash
./run.sh legged_gym/scripts/list_tasks.py
./run.sh legged_gym/scripts/train.py --task wheelleg_slow \
  --sim_device cpu --rl_device cpu --num_envs 8 --headless --max_iterations 2
```

这是新增任务的示例，并未随项目注册或训练。相同观测尺寸只能保证输入形状兼容，换控制含义、奖励或指令范围仍需重新评估；`--checkpoint` 会恢复保存的环境和训练配置，先查看解析后的配置。

## 加一个奖励

只有新增奖励或行为时才需要任务子类。例如在新文件 `legged_gym/envs/wheelleg/wheelleg_slow.py` 中增加横滚角速度惩罚：

```python
from .wheelleg import WheelLegged


class WheelLeggedSlow(WheelLegged):
    def _reward_roll_rate(self):
        return -self.ang_vel[:, 0].square()
```

再在该任务配置中继承奖励结构，增加对应名称：

```python
class WheelLeggedSlowCfg(WheelLeggedStandCfg):
    # 保留前面的 commands 覆盖
    class rewards(WheelLeggedStandCfg.rewards):
        class scales(WheelLeggedStandCfg.rewards.scales):
            roll_rate = 0.02
```

方法返回形状 `[num_envs]`、与环境同设备的有限张量；此处方法已经返回负数，正系数控制惩罚强度。设为零移除该项。公式需要多个任务共用时放入 [rewards.py](../legged_gym/envs/wheelleg/rewards.py)，阶梯公式放入 [stair_rewards.py](../legged_gym/envs/wheelleg/stair_rewards.py)，任务方法负责调用。每个控制步累计奖励，不额外乘 `dt`；已有阶梯项内部包含原系数。

在 `register.py` 中导入 `WheelLeggedSlow`，把前面 `wheelleg_slow` 注册项的任务类从 `WheelLegged` 改为 `WheelLeggedSlow`，不要用同名重复注册。新增方法与 `rewards.scales.roll_rate` 名称对应后，训练仍走 `OnPolicyRunner → PPORunner`。检查零系数、返回形状/设备、实际符号和原任务数值，再做有限训练与工况复测。

## 加一个新机器人

建立 `legged_gym/envs/<robot>/`，独立放置 `<robot>.py`、`<robot>_config.py`、物理实现与 `register.py`；几何、控制、诊断可按当前轮腿目录的分组组织。环境配置和训练配置分别继承 [BaseConfig](../legged_gym/envs/base/base_config.py)，按注册器与适配器使用的结构提供 `seed`、`env` 和 `policy`、`algorithm`、`runner` 等参数。轮腿类的资产、闭链和动作约定专属于当前机器人，新机器人的物理实现自行定义。

当前注册任务与前馈训练适配要求如下：

| 部分 | 需要提供的内容 |
| --- | --- |
| 构造与配置 | `__init__(cfg, sim_device, headless)`；保存 `cfg`、`environment_config` |
| 基本属性 | `device`、`num_envs`、`num_obs`、`num_actions`、`num_privileged_obs=None`、`obs_history_length=1` |
| 观测 | 输入向量由环境完整准备；`reset()` 返回 `(obs, auxiliary)`，`get_observations()` 返回 `(obs, obs_history)` |
| 步进 | `step(actions)` 返回六项 `obs, None, rewards, dones, infos, obs_history`，结束环境自动重置 |
| 终止信息 | 区分正常超时和物理终止；提供 `time_outs` 及 PPO 超时估值需要的终止前观测信息 |
| 生命周期 | `close()`；需要回合长度统计或随机回合起点时，提供 `episode_length_buf`、`dt`、`max_episode_length` |
| 配置结构 | `env.num_observations` 等尺寸与真实输入一致；`runner` 提供实验名、保存周期、采样长度等现有字段 |

可选扩展点均由机器人提供：

- `create_training_metrics()` 返回 collector；其 `reset()`、`update(obs, rewards, dones, infos)`、`compute(num_steps)` 收集并返回日志指标字典。
- collector 的 `checkpoint_metadata()` / `restore_checkpoint(state)` 保存和恢复课程等状态，`configuration_metadata()` 补充运行配置；轮腿实例见 [training_metrics.py](../legged_gym/envs/wheelleg/diagnostics/training_metrics.py)。
- 任务静态方法 `restore_legacy_environment(cfg, values)` 导入该机器人的旧检查点参数；有旧平铺参数而未提供 importer 时，注册器会报错。
- `configure_execution(cfg, args, checkpoint_state)` 处理该机器人的播放模式、指令与地形覆盖；轮腿的实现见 [wheelleg.py](../legged_gym/envs/wheelleg/wheelleg.py)。

新机器人的 `register(registry)` 使用同样的 `registry.register(name, task_class, env_cfg, train_cfg)`。最后在 [envs/__init__.py](../legged_gym/envs/__init__.py) 的 `register_tasks()` 中追加它的注册调用。导入和任务加载保持显式，Gym 必须在 Torch 前导入。

满足这些接口后可复用现有 PPO 训练循环，无需把机器人常量或诊断写进 `learning/`。当前只实现并验证了轮腿机器人，训练适配支持同设备、无特权观测的前馈 `ActorCritic/PPO`。现有播放键盘、评估报告和任务列表展示仍有轮腿字段假设；接新机器人时需一起检查展示入口、资产、动作/观测布局、重置、超时估值和检查点恢复，再建立该机器人的物理与控制验证。
