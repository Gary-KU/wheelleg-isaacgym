#!/usr/bin/env python3
"""Create a separate pure-RL support warm start without old folding hip outputs.

Importing this module does not load Gym/Torch or modify Python search paths.
Runtime imports deliberately load Isaac Gym before Torch.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def _runtime():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import isaacgym  # noqa: F401; binary extension must precede Torch.
    import torch
    from wheelleg_gym.config import TRAIN_CONFIG
    from wheelleg_gym.runtime_config import SUPPORT_ENVIRONMENT
    from wheelleg_gym.training import make_policy
    return torch, TRAIN_CONFIG, SUPPORT_ENVIRONMENT, make_policy


def prepare_support_checkpoint(checkpoint, source_path=None, seed=42, hip_std=3.0):
    """Return an independent 198-D support warm start; never mutate the source.

    Actor hidden layers and wheel rows stay exact. Four hip output rows are
    zeroed, the critic is freshly initialized, and optimizer/counters reset.
    This is a new learning objective, not an equivalent-policy conversion.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(hip_std) or hip_std <= 0:
        raise ValueError("hip_std must be positive and finite")
    torch, defaults, support_environment, make_policy = _runtime()
    effective_std = float(torch.tensor(hip_std, dtype=torch.float32))
    if not math.isfinite(effective_std) or effective_std <= 0:
        raise ValueError("hip_std must remain positive and finite in float32")
    source_model = checkpoint.get("model_state_dict", {})
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = source_model.get(key)
        if weight is None or len(weight.shape) != 2 or weight.shape[1] != 198:
            raise ValueError("Support preparation requires a 198-D actor and critic")

    result = deepcopy(checkpoint)
    config = deepcopy(checkpoint.get("training_config", defaults))
    policy_config = config.setdefault("policy", deepcopy(defaults["policy"]))
    policy_config["init_noise_std"] = [float(hip_std)]*4+[0.1, 0.1]
    config.setdefault("algorithm", {}).update(learning_rate=0.0003, max_learning_rate=0.0003)
    config["save_interval"] = 10
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_policy(198, 6, "cpu", policy_config)
    fresh_critic = deepcopy(model.critic.state_dict())
    model.load_state_dict(source_model, strict=True)
    model.critic.load_state_dict(fresh_critic, strict=True)
    output_layers = [module for module in model.actor.modules() if isinstance(module, torch.nn.Linear)]
    output = output_layers[-1]
    if output.out_features != 6 or output.bias is None:
        raise ValueError("Support preparation requires a six-output actor with a bias")
    with torch.no_grad():
        output.weight[:4].zero_()
        output.bias[:4].zero_()
        model.std.copy_(model.std.new_tensor(policy_config["init_noise_std"]))
    result["model_state_dict"] = deepcopy(model.state_dict())
    result.pop("optimizer_state_dict", None)
    result["training_config"] = config
    environment = deepcopy(checkpoint.get("environment_config", {}))
    environment.update(deepcopy(support_environment))
    result["environment_config"] = environment
    result["iter"] = 0
    result["total_steps"] = 0
    result["support_bootstrap"] = {
        "source_checkpoint": str(Path(source_path).resolve()) if source_path is not None else None,
        "source_iteration": int(checkpoint.get("iter", 0)),
        "source_total_steps": int(checkpoint.get("total_steps", 0)),
        "seed": seed,
        "hip_std": float(hip_std),
        "description": "New pure-RL support objective: actor hidden layers and wheel outputs preserved; hip output rows zeroed; critic, optimizer and counters reset. No feedforward controller added.",
    }
    # PPORunner.save retains training_config while dropping arbitrary top-level
    # keys. Keep provenance there as well so later learned checkpoints trace back.
    config["initialization_provenance"] = deepcopy(result["support_bootstrap"])
    return result


def save_support_checkpoint(source, output, seed=42, hip_std=3.0):
    """Write a complete checkpoint atomically, refusing any existing output."""
    source, output = Path(source).resolve(), Path(output).absolute()
    if source == output.resolve():
        raise ValueError("Source and output must be different files")
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists: {}".format(output))
    if not source.is_file():
        raise ValueError("Source checkpoint does not exist: {}".format(source))
    torch, _, _, _ = _runtime()
    prepared = prepare_support_checkpoint(torch.load(str(source), map_location="cpu"),
                                          source, seed=seed, hip_std=hip_std)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(output.parent), prefix=".support_", suffix=".pt", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(prepared, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), an exclusive hard link cannot overwrite a racing writer.
        try:
            os.link(str(temporary), str(output))
        except FileExistsError:
            raise ValueError("Output already exists: {}".format(output))
    finally:
        if temporary is not None:
            temporary.unlink()
    return prepared["support_bootstrap"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hip-std", type=float, default=3.0)
    args = parser.parse_args()
    try:
        metadata = save_support_checkpoint(args.checkpoint, args.output, args.seed, args.hip_std)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output_checkpoint=str(args.output.resolve()), **metadata), indent=2))


if __name__ == "__main__":
    main()
