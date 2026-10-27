# Vendored RSL-RL components

The `rsl_rl` directory contains the feed-forward `ActorCritic`, PPO, rollout
storage and supporting utilities from the local Wheel-Legged-Gym distribution,
which identifies this fork as version 2.0.1. Upstream authors are ETH Zurich,
Nikita Rudin, and NVIDIA; BSD-3-Clause notices and the license are retained.

Local adaptations: imports use the standalone `rsl_rl` package name; only the
feed-forward policy is exported; the optional GitPython import is deferred until
repository snapshots are requested. The migration's small runner lives outside
this vendor directory in `wheelleg_gym/training.py`.

Additional local changes support scalar or six-motor exploration noise in
`modules/actor_critic.py`, cloned rollout observations, and a configurable
adaptive learning-rate ceiling in `algorithms/ppo.py`. Network/PPO defaults
are collected in `wheelleg_gym/config.py`; the runner still re-exports
`TRAIN_CONFIG` for existing imports.

The vendored files avoid coupling this port to the neighbouring checkout or to
Isaac Lab's newer RSL-RL/Python dependency versions.

The legged_gym migration adds `runners/OnPolicyRunner` and `env/VecEnv` from Wheel-Legged-Gym commit `c354431e5633eb98d9663fa3fbe444201d2de383`. The runner imports this same vendored package, supports the existing feed-forward policy, consumes the six-value task API, retains the timeout-bootstrap fix in PPO, and fixes completed-iteration/save counters on resume. It records both nested and legacy config metadata and restores curriculum/optimizer state. The legacy small runner remains available for old scripts. Upstream recurrent/sequence classes and unrelated tasks/assets were not copied.
