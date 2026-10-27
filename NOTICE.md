# 来源与许可

本仓库是 WheelLeg 项目的 Isaac Gym 迁移与实验代码，供异地学习和开发。

- 原项目：https://github.com/2757657887/WheelLeg_RL_IsaacLab
- 控制方法参考：https://github.com/clearlab-sustech/Wheel-Legged-Gym
- `third_party/rsl_rl/` 保留原 BSD-3-Clause 许可与版权声明。
- Isaac Lab 相关代码的许可副本见 `third_party/ISAAC_LAB_LICENSE`；保留源文件中的版权与来源注释。
- `assets/wheelleg/` 是从原项目机器人模型导出的资产，来源见该目录说明；这里没有为第三方资产授予额外使用权。
- NVIDIA Isaac Gym SDK、Conda 环境和 CUDA/PyTorch 安装包不包含在仓库中，应通过各自官方渠道获取并遵守相应许可。

当前建议使用私人仓库；若之后公开或再分发，请分别核实机器人资产及其上游许可。

框架迁移参考固定版本：Wheel-Legged-Gym `c354431e5633eb98d9663fa3fbe444201d2de383`。`legged_gym/envs/base/base_config.py`、RSL-RL 的 `OnPolicyRunner` 和 `VecEnv` 来自该版本并作了本地适配，BSD-3-Clause 副本见 `third_party/LEGGED_GYM_LICENSE`，各源文件保留版权头。

SCUT `wheeled-legged_RL` 的 `1a26356d6f591734d4f6f666f5cead1a6925da67` 只用于闭链方法及 USD 关节数据核对；运行不依赖其代码和资产。许可副本见 `third_party/SCUT_REFERENCE_LICENSE`，说明见 `docs/SCUT_CLOSED_CHAIN.md`。
