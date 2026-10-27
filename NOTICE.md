# 来源与许可

本仓库是 WheelLeg 项目的 Isaac Gym 迁移与实验代码，供异地学习和开发。

- 原项目：[WheelLeg_RL_IsaacLab](https://github.com/2757657887/WheelLeg_RL_IsaacLab)。
- 任务组织与控制方法参考：[Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym)，固定版本为 `c354431e5633eb98d9663fa3fbe444201d2de383`。
- [third_party/rsl_rl/](third_party/rsl_rl/) 保留来源组件的 [BSD-3-Clause 许可](third_party/rsl_rl/LICENSE)与版权声明，具体本地适配见[组件说明](third_party/README.md)。
- [base_config.py](legged_gym/envs/base/base_config.py) 及 runner 接口最初参考上述固定版本；现有 [OnPolicyRunner](legged_gym/learning/on_policy_runner.py) 是 [PPORunner](legged_gym/learning/ppo_runner.py) 的薄适配层，现均位于 `legged_gym/learning/`，原版权头继续保留。相关 BSD-3-Clause 副本见 [LEGGED_GYM_LICENSE](third_party/LEGGED_GYM_LICENSE)。当前借用范围为任务、配置与前馈 PPO 接口；`third_party/rsl_rl/` 保存算法、网络、存储及相关来源声明。
- Isaac Lab 相关代码的许可副本见 [ISAAC_LAB_LICENSE](third_party/ISAAC_LAB_LICENSE)；源文件保留相应版权与来源注释。
- [assets/wheelleg/](assets/wheelleg/) 是从原项目机器人模型导出的资产，来源见[资产说明](assets/wheelleg/README.md)；这里没有为第三方资产授予额外使用权。
- NVIDIA Isaac Gym SDK、Conda 环境和 CUDA/PyTorch 安装包不包含在仓库中，通过各自官方渠道获取并遵守相应许可。

当前建议使用私人仓库；若之后公开或再分发，请分别核实机器人资产及其上游许可。

SCUT `wheeled-legged_RL` 的 `1a26356d6f591734d4f6f666f5cead1a6925da67` 只用于闭链方法及 USD 关节数据核对；运行不依赖其代码和资产。许可副本见 [SCUT_REFERENCE_LICENSE](third_party/SCUT_REFERENCE_LICENSE)，说明见 [SCUT_CLOSED_CHAIN.md](docs/SCUT_CLOSED_CHAIN.md)。
