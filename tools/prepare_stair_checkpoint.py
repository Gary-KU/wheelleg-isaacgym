#!/usr/bin/env python3
"""Create a separate stair-learning warm start with terrain observations.

The source actor is preserved except that its four hip outputs start at zero
for the new target-angle residual meaning. Appended terrain input weights are
zero. This is not an already trained stair policy or an equivalent controller.
Importing this module alone never loads Isaac Gym or Torch.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile


FRAME_COUNT = 6
SOURCE_FRAME_SIZE = 33
TERRAIN_FEATURES = 9
TARGET_FRAME_SIZE = SOURCE_FRAME_SIZE+TERRAIN_FEATURES
SOURCE_OBSERVATIONS = FRAME_COUNT*SOURCE_FRAME_SIZE
TARGET_OBSERVATIONS = FRAME_COUNT*TARGET_FRAME_SIZE


def _runtime():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import isaacgym  # noqa: F401; binary extension must precede Torch.
    import torch
    from legged_gym.envs.wheelleg.constants import TRAIN_CONFIG
    from legged_gym.compat.runtime_config import STAIRS_ENVIRONMENT
    from legged_gym.learning.policy import make_policy
    return torch, TRAIN_CONFIG, STAIRS_ENVIRONMENT, make_policy


def expand_stair_input_columns(value):
    """Append nine zero-weight terrain features to each of the six frames."""
    torch, _, _, _ = _runtime()
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != SOURCE_OBSERVATIONS:
        raise ValueError("Stair input expansion requires a 198-D input matrix")
    expanded = value.new_zeros((value.shape[0], FRAME_COUNT, TARGET_FRAME_SIZE))
    expanded[:, :, :SOURCE_FRAME_SIZE] = value.reshape(value.shape[0], FRAME_COUNT, SOURCE_FRAME_SIZE)
    return expanded.reshape(value.shape[0], TARGET_OBSERVATIONS)


def prepare_stair_checkpoint(checkpoint, source_path=None, seed=42):
    """Return an independent new-task checkpoint, leaving its source untouched.

    Old actor features retain their per-frame positions. Terrain columns start
    at zero; actor hidden layers and wheel outputs retain their weights. Hip
    rows start at zero because their actions now mean target-angle residuals.
    Critic, optimizer, exploration noise and learning counters start afresh.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(checkpoint, dict):
        raise ValueError("Source checkpoint must contain a mapping")
    torch, defaults, stairs_environment, make_policy = _runtime()
    source_model = checkpoint.get("model_state_dict", {})
    for name in ("actor.0.weight", "critic.0.weight"):
        weight = source_model.get(name)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] != SOURCE_OBSERVATIONS:
            raise ValueError("Stair preparation requires a 198-D actor and critic")
    config = deepcopy(checkpoint.get("training_config", defaults))
    policy_config = config.setdefault("policy", deepcopy(defaults["policy"]))
    policy_config["init_noise_std"] = [.1]*4+[.02, .02]
    config.setdefault("algorithm", {}).update(learning_rate=.0003, max_learning_rate=.0003)
    config["save_interval"] = 10

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_policy(TARGET_OBSERVATIONS, 6, "cpu", policy_config)
    prepared_model = deepcopy(model.state_dict())
    actor_keys = {name for name in prepared_model if name.startswith("actor.")}
    source_actor_keys = {name for name in source_model if name.startswith("actor.")}
    if actor_keys != source_actor_keys:
        raise ValueError("Source actor layout does not match its saved policy configuration")
    for name in actor_keys:
        prepared_model[name] = (expand_stair_input_columns(source_model[name])
                                if name == "actor.0.weight" else source_model[name].clone())
    model.load_state_dict(prepared_model, strict=True)
    outputs = [module for module in model.actor.modules() if isinstance(module, torch.nn.Linear)]
    output = outputs[-1]
    if output.out_features != 6 or output.bias is None:
        raise ValueError("Stair preparation requires a six-output actor with a bias")
    with torch.no_grad():
        output.weight[:4].zero_()
        output.bias[:4].zero_()
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("Source actor contains non-finite parameters")

    result = deepcopy(checkpoint)
    # Restore from the updated flat metadata, not the source task's snapshots.
    for key in ("legged_gym_env_config", "legged_gym_train_config", "terrain_curriculum_state"):
        result.pop(key, None)
    result["model_state_dict"] = deepcopy(model.state_dict())
    result.pop("optimizer_state_dict", None)
    result.pop("support_bootstrap", None)
    result.pop("observation_upgrade", None)
    result["training_config"] = config
    environment = deepcopy(checkpoint.get("environment_config", {}))
    environment.update(deepcopy(stairs_environment))
    result["environment_config"] = environment
    result["iter"], result["total_steps"] = 0, 0
    metadata = {
        "source_checkpoint": str(Path(source_path).resolve()) if source_path is not None else None,
        "source_iteration": int(checkpoint.get("iter", 0)),
        "source_total_steps": int(checkpoint.get("total_steps", 0)),
        "seed": seed,
        "source_observations": SOURCE_OBSERVATIONS,
        "target_observations": TARGET_OBSERVATIONS,
        "frame_count": FRAME_COUNT,
        "source_features_per_frame": SOURCE_FRAME_SIZE,
        "target_features_per_frame": TARGET_FRAME_SIZE,
        "appended_terrain_features_per_frame": TERRAIN_FEATURES,
        "terrain_input_weights": "zero; old features preserve their within-frame order",
        "hip_action_meaning": "clipped normalized target-angle residual, scaled by hip_target_scale radians",
        "wheel_action_meaning": "original learned wheel torque outputs, unchanged",
        "hip_actor_output_rows": "zero",
        "hip_target_scale_rad": environment["hip_target_scale"],
        "new_rl_training_updates": 0,
        "description": "New stair task and new hip action meaning: 198-to-252-D actor input expansion preserves actor hidden/wheel initialization; hip target-residual outputs start at zero. Critic, optimizer, noise and counters reset. This is an untrained stair warm start, not an equivalent old controller or a trained stair policy.",
    }
    previous = checkpoint.get("training_config", {}).get("initialization_provenance")
    if previous is not None:
        metadata["source_initialization_provenance"] = deepcopy(previous)
    result["stair_bootstrap"] = metadata
    config["initialization_provenance"] = deepcopy(metadata)
    return result


def save_stair_checkpoint(source, output, seed=42):
    """Atomically create a distinct checkpoint, refusing existing destinations."""
    source, output = Path(source).resolve(), Path(output).absolute()
    if source == output.resolve():
        raise ValueError("Source and output must be different files")
    if os.path.lexists(str(output)):
        raise ValueError("Output already exists: {}".format(output))
    if not source.is_file():
        raise ValueError("Source checkpoint does not exist: {}".format(source))
    torch, _, _, _ = _runtime()
    prepared = prepare_stair_checkpoint(torch.load(str(source), map_location="cpu"), source, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(output.parent), prefix=".stairs_", suffix=".pt", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(prepared, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(temporary), str(output))
        except FileExistsError:
            raise ValueError("Output already exists: {}".format(output))
    finally:
        if temporary is not None:
            temporary.unlink()
    return prepared["stair_bootstrap"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        metadata = save_stair_checkpoint(args.checkpoint, args.output, args.seed)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output_checkpoint=str(args.output.resolve()), **metadata), indent=2))


if __name__ == "__main__":
    main()
