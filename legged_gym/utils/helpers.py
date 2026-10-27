"""Small, explicit config and checkpoint helpers for the legged_gym entrypoints."""

from copy import deepcopy
from pathlib import Path
import argparse
import random

from legged_gym import LEGGED_GYM_ROOT_DIR


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [class_to_dict(value) for value in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    return {
        key: class_to_dict(getattr(obj, key))
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def update_class_from_dict(obj, values):
    for name, value in values.items():
        if not hasattr(obj, name):
            raise ValueError("Unknown config field: " + name)
        current = getattr(obj, name)
        if hasattr(current, "__dict__") and isinstance(value, dict):
            update_class_from_dict(current, value)
        else:
            setattr(obj, name, deepcopy(value))


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_load_path(root, load_run=-1, checkpoint=-1):
    """Select only actual run directories and numerically ordered model_N.pt."""
    root = Path(root)
    runs = sorted(
        (p for p in root.glob("*") if p.is_dir() and any(p.glob("model_*.pt"))),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
    )
    if str(load_run) == "-1":
        if not runs:
            raise ValueError("No saved runs under " + str(root))
        run = runs[-1]
    else:
        run = root / str(load_run)
    if str(checkpoint) == "-1":
        models = [p for p in run.glob("model_*.pt") if p.stem[6:].isdigit()]
        if not models:
            raise ValueError("No model_N.pt checkpoints under " + str(run))
        path = max(models, key=lambda p: int(p.stem[6:]))
    else:
        path = run / ("model_{}.pt".format(int(checkpoint)))
    if not path.is_file():
        raise ValueError("Checkpoint does not exist: " + str(path))
    return str(path.resolve())


def get_args(mode="train", argv=None):
    p = argparse.ArgumentParser(description="Five-bar wheelleg legged_gym " + mode)
    p.add_argument(
        "--task", default="wheelleg_flat" if mode == "train" else "wheelleg_stand"
    )
    for name, kind in (("num_envs", int), ("seed", int), ("max_iterations", int)):
        p.add_argument("--" + name, "--" + name.replace("_", "-"), type=kind)
    p.add_argument(
        "--sim_device", "--sim-device", default="cuda:0" if mode == "train" else "cpu"
    )
    p.add_argument("--rl_device", "--rl-device")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", help="model_N.pt path, or numeric checkpoint index")
    p.add_argument("--load_run", "--load-run", default="-1")
    p.add_argument("--run_name", "--run-name", default="")
    p.add_argument("--log_root", "--log-root", type=Path)
    p.add_argument("--num_steps", "--num-steps", type=int, default=3000)
    p.add_argument("--command", nargs=4, type=float)
    p.add_argument("--stair_level", "--stair-level", type=int)
    p.add_argument("--report", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--keyboard", dest="keyboard", action="store_true")
    group.add_argument(
        "--no_keyboard", "--no-keyboard", dest="keyboard", action="store_false"
    )
    p.set_defaults(keyboard=None)
    args = p.parse_args(argv)
    args.rl_device = args.rl_device or args.sim_device
    args.keyboard = (
        (mode == "play" and not args.headless)
        if args.keyboard is None
        else args.keyboard
    )
    if args.keyboard and (
        mode != "play" or args.headless or args.num_envs not in (None, 1)
    ):
        p.error("Keyboard requires play, a viewer, and one environment")
    for name in ("num_envs", "max_iterations", "num_steps"):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error("--" + name + " must be positive")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        p.error("--seed must be in [0, 2**32)")
    if args.stair_level is not None and not 0 <= args.stair_level <= 4:
        p.error("--stair_level must be in [0,4]")
    args.mode = mode
    return args
