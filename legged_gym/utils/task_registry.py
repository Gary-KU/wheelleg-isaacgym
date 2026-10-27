"""legged_gym task registry with independent configs and explicit checkpoint restore."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json

from legged_gym import LEGGED_GYM_ROOT_DIR
from .helpers import (
    class_to_dict,
    get_args,
    get_load_path,
    set_seed,
    update_class_from_dict,
)


class TaskRegistry:
    def __init__(self):
        self.task_classes, self.env_cfgs, self.train_cfgs = {}, {}, {}

    def register(self, name, task_class, env_cfg, train_cfg):
        if name in self.task_classes:
            raise ValueError("Duplicate task: " + name)
        self.task_classes[name] = task_class
        self.env_cfgs[name], self.train_cfgs[name] = env_cfg, train_cfg

    def get_task_class(self, name):
        if name not in self.task_classes:
            raise ValueError(
                "Unknown task {}. Available: {}".format(
                    name, ", ".join(self.task_classes)
                )
            )
        return self.task_classes[name]

    def get_cfgs(self, name):
        self.get_task_class(name)
        env, train = deepcopy(self.env_cfgs[name]), deepcopy(self.train_cfgs[name])
        env.seed = train.seed
        return env, train

    def prepare_cfgs(self, name, args):
        from legged_gym.envs.wheelleg.wheelleg import restore_environment_config
        import torch

        env_cfg, train_cfg = self.get_cfgs(name)
        root = (
            args.log_root
            or Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
        )
        checkpoint = None
        if args.checkpoint is not None and not args.checkpoint.lstrip("-").isdigit():
            checkpoint = Path(args.checkpoint).expanduser().resolve()
        elif args.resume or (args.mode == "play" and args.checkpoint is not None):
            checkpoint = Path(get_load_path(root, args.load_run, args.checkpoint or -1))
        elif args.mode == "play":
            # Explicit documented defaults only; do not silently choose an unrelated run.
            checkpoint = (
                Path(LEGGED_GYM_ROOT_DIR)
                / {
                    "wheelleg_stand": "logs/standing_support_20261022_093823/input/standing_hold.pt",
                    "wheelleg_flat": "logs/standing_support_20261022_093823/input/standing_hold.pt",
                    "wheelleg_stairs": "logs/stair_training_20261022_103417/evaluation/verified_level0.pt",
                }[name]
            )
        if checkpoint is not None:
            if not checkpoint.is_file():
                raise ValueError("Checkpoint does not exist: " + str(checkpoint))
            state = torch.load(str(checkpoint), map_location="cpu")
            if (
                state["model_state_dict"]["actor.0.weight"].shape[1]
                != env_cfg.env.num_observations
            ):
                raise ValueError(
                    "Checkpoint observation layout does not match task " + name
                )
            if "legged_gym_env_config" in state:
                update_class_from_dict(env_cfg, state["legged_gym_env_config"])
            else:
                restore_environment_config(env_cfg, state.get("environment_config", {}))
            # Legacy checkpoints retain the task defaults for execution-only controls
            # introduced after training (e.g. current heading feedback in stand play).
            if "legged_gym_train_config" in state:
                update_class_from_dict(train_cfg, state["legged_gym_train_config"])
            elif "training_config" in state:
                old = state["training_config"]
                update_class_from_dict(train_cfg.policy, old["policy"])
                update_class_from_dict(train_cfg.algorithm, old["algorithm"])
                train_cfg.runner.num_steps_per_env = old["num_steps_per_env"]
                train_cfg.runner.save_interval = old["save_interval"]
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        elif args.mode == "play":
            env_cfg.env.num_envs = 1
        if args.seed is not None:
            env_cfg.seed = train_cfg.seed = args.seed
        if args.max_iterations is not None:
            train_cfg.runner.max_iterations = args.max_iterations
        if args.run_name:
            train_cfg.runner.run_name = args.run_name
        if args.command is not None:
            env_cfg.commands.command = args.command
        if args.mode == "play":
            env_cfg.commands.random_commands = False
            env_cfg.commands.command_resample_seconds = 0.0
            env_cfg.terrain.stair_curriculum = False
            if (
                checkpoint is not None
                and "legged_gym_env_config" not in state
                and name != "wheelleg_stairs"
            ):
                env_cfg.control.wheel_control = "heading"
                env_cfg.control.heading_kp, env_cfg.control.heading_kd = 6.0, 0.6
        if args.stair_level is not None:
            if env_cfg.terrain.terrain != "stairs":
                raise ValueError("--stair_level requires a stair task")
            env_cfg.terrain.stair_level = args.stair_level
        args.checkpoint_path = checkpoint
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None):
        args = args or get_args()
        env_cfg = (
            deepcopy(env_cfg)
            if env_cfg is not None
            else self.prepare_cfgs(name, args)[0]
        )
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        set_seed(env_cfg.seed)
        env = self.get_task_class(name)(
            cfg=env_cfg, sim_device=args.sim_device, headless=args.headless
        )
        return env, env_cfg

    def make_alg_runner(
        self, env, name=None, args=None, train_cfg=None, log_root="default"
    ):
        from rsl_rl.runners import OnPolicyRunner

        args = args or get_args()
        if train_cfg is None:
            if name is None:
                raise ValueError("Provide a task name or train_cfg")
            train_cfg = self.prepare_cfgs(name, args)[1]
        train_cfg = deepcopy(train_cfg)
        if log_root == "default":
            root = (
                args.log_root
                or Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
            )
            log_dir = root / (
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                + "_"
                + train_cfg.runner.run_name
            )
        elif log_root is None:
            log_dir = None
        else:
            log_dir = Path(log_root)
        runner = OnPolicyRunner(
            env, class_to_dict(train_cfg), log_dir, device=args.rl_device
        )
        if getattr(args, "checkpoint_path", None) is not None:
            runner.load(args.checkpoint_path, load_optimizer=args.mode == "train")
        if log_dir is not None:
            config = dict(
                task=name,
                environment=class_to_dict(env.cfg),
                training=class_to_dict(train_cfg),
                checkpoint=(
                    str(args.checkpoint_path)
                    if getattr(args, "checkpoint_path", None)
                    else None
                ),
                start_iteration=runner.current_learning_iteration,
            )
            (Path(log_dir) / "config.json").write_text(
                json.dumps(config, indent=2) + "\n"
            )
        return runner, train_cfg


task_registry = TaskRegistry()
