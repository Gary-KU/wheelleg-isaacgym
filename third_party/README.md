# Vendored components and sources

The `rsl_rl` directory retains the feed-forward `ActorCritic`, PPO and rollout
storage used by this project from the local Wheel-Legged-Gym distribution,
which identifies its fork as version 2.0.1. Upstream authors are ETH Zurich,
Nikita Rudin and NVIDIA. Copyright headers and the
[BSD-3-Clause license](rsl_rl/LICENSE) are retained.

The task/config/runner conventions were adapted from
[Wheel-Legged-Gym commit c354431](https://github.com/clearlab-sustech/Wheel-Legged-Gym/tree/c354431e5633eb98d9663fa3fbe444201d2de383).
The corresponding license copy is [LEGGED_GYM_LICENSE](LEGGED_GYM_LICENSE).
This project implements the subset used by its five-bar wheelleg tasks.

The project-owned [OnPolicyRunner](../legged_gym/learning/on_policy_runner.py) is a thin local
adapter: it translates nested task configuration and the six-value task API to
the shared [PPORunner](../legged_gym/learning/ppo_runner.py). That engine owns rollout
collection, updates, checkpoint loading and saving for both entry styles.
Robot diagnostics and curriculum state enter through optional collector hooks
provided by the environment. The engine records legacy metadata, and the
adapter adds nested task metadata.

Local algorithm changes preserve scalar or six-motor exploration noise, cloned
rollout observations, terminal-observation timeout bootstrapping and the
configurable adaptive learning-rate ceiling. The supported policy is the
checkpoint-compatible feed-forward `ActorCritic/PPO`, with observations already
stacked by the environment. Simulation and PPO must use the same device.

Normal robot/task defaults live in
[wheelleg_config.py](../legged_gym/envs/wheelleg/wheelleg_config.py).
[constants.py](../legged_gym/envs/wheelleg/constants.py) retains robot units,
shared constants and legacy training defaults.
[learning/policy.py](../legged_gym/learning/policy.py) constructs policies from
caller-owned configuration. Project-owned environment, training and
compatibility code lives under `legged_gym`; root command-line entrypoints
remain available.

These local components avoid a runtime dependency on a neighbouring checkout or
on Isaac Lab's newer RSL-RL/Python versions. Robot assets are maintained once in
[assets/wheelleg](../assets/wheelleg/). Further source and asset notices are in
[NOTICE.md](../NOTICE.md).
