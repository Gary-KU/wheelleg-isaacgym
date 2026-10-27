#!/usr/bin/env python3
"""Train the Isaac Gym port using the vendored Python 3.8 compatible PPO."""

from datetime import datetime
from pathlib import Path
from copy import deepcopy
import json
import math
import sys

from legged_gym.compat.cli import common_parser, environment_kwargs, validate_common
from legged_gym.compat.runtime_config import explicit_argument_destinations, resolve_runtime_args


def main():
    parser = common_parser("WheelLeg Isaac Gym PPO (compliant loop approximation)", 64)
    parser.add_argument("--max-iterations", "--max_iterations", type=int, default=20000)
    parser.add_argument("--checkpoint", type=Path, help="Resume a matching Gym checkpoint")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--init-noise-std", nargs="+", type=float,
                        help="Initial exploration std: one scalar or six values (four hips, two wheels)")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-learning-rate", type=float,
                        help="Cap adaptive learning rate, including after checkpoint restore")
    parser.add_argument("--save-interval", type=int, help="Default 100, or preserve the checkpoint's interval when resuming")
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(args.checkpoint))
    state = None
    if args.checkpoint:
        import isaacgym  # noqa: F401; binary extension must load before Torch.
        import torch
        state = torch.load(str(args.checkpoint), map_location="cpu")
    try:
        args = resolve_runtime_args(args, explicit_argument_destinations(parser, argv), state, "train")
    except ValueError as error:
        parser.error(str(error))
    validate_common(parser, args)
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.init_noise_std is not None:
        if len(args.init_noise_std) not in (1, 6) or not all(math.isfinite(x) and x > 0 for x in args.init_noise_std):
            parser.error("--init-noise-std requires one or six positive finite values")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("--learning-rate must be positive and finite")
    if args.max_learning_rate is not None and (not math.isfinite(args.max_learning_rate) or args.max_learning_rate < 1e-5):
        parser.error("--max-learning-rate must be finite and at least 1e-5")
    if args.save_interval is not None and args.save_interval < 1:
        parser.error("--save-interval must be positive")
    if args.checkpoint and (args.init_noise_std is not None or args.learning_rate is not None):
        parser.error("A checkpoint restores noise and optimizer state; omit initial noise/learning-rate overrides when resuming")

    # Isaac Gym's binary extension must be loaded before importing torch.
    import isaacgym  # noqa: F401
    from legged_gym.envs.wheelleg.wheelleg_robot import WheelLegRobot
    from legged_gym.learning.ppo_runner import PPORunner
    from legged_gym.envs.wheelleg.constants import TRAIN_CONFIG

    log_dir = args.log_dir or (
        Path(__file__).resolve().parent / "logs" / "wheelleg"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    )
    kwargs = environment_kwargs(args)
    print("Loop closure: compliant external forces; exact rigid-loop equivalence is not assumed.")
    print("Training logs: {}".format(log_dir.resolve()))
    env = WheelLegRobot(**kwargs)
    try:
        config = deepcopy(TRAIN_CONFIG)
        if state is not None:
            config = deepcopy(state.get("training_config", TRAIN_CONFIG))
        if args.init_noise_std is not None:
            config["policy"]["init_noise_std"] = args.init_noise_std[0] if len(args.init_noise_std) == 1 else args.init_noise_std
        if args.learning_rate is not None:
            config["algorithm"]["learning_rate"] = args.learning_rate
        if args.max_learning_rate is not None:
            config["algorithm"]["max_learning_rate"] = args.max_learning_rate
        if args.save_interval is not None:
            config["save_interval"] = args.save_interval
        runner = PPORunner(env, log_dir, config=config, environment_config=kwargs)
        if args.checkpoint:
            runner.load(args.checkpoint)
        (runner.log_dir / "launch.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "start_iteration": runner.iteration,
            "planned_new_iterations": args.max_iterations,
            "planned_end_iteration": runner.iteration+args.max_iterations,
            "num_envs": env.num_envs,
            "samples_per_iteration": env.num_envs*config["num_steps_per_env"],
            "profile": args.profile,
        }, indent=2)+"\n")
        runner.learn(args.max_iterations)
    finally:
        env.close()


if __name__ == "__main__":
    main()
