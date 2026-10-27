#!/usr/bin/env python3
"""legged_gym training: registered task -> nested configs -> OnPolicyRunner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym  # must precede torch
import torch
from legged_gym.envs import register_tasks
register_tasks()
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args("train")
    torch.set_num_threads(1)
    env_cfg, train_cfg = task_registry.prepare_cfgs(args.task, args)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    try:
        runner, train_cfg = task_registry.make_alg_runner(
            env, name=args.task, args=args, train_cfg=train_cfg
        )
        print(
            "Training task={} log_dir={}; closure remains spring/damper approximate".format(
                args.task, runner.log_dir
            ),
            flush=True,
        )
        runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()
