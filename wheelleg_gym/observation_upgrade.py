"""Expand legacy policy inputs while preserving predictions and Adam history."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile

import torch


FRAME_COUNT = 6
OLD_FRAME_SIZE = 31
NEW_FRAME_SIZE = 33
OLD_OBSERVATIONS = FRAME_COUNT*OLD_FRAME_SIZE
NEW_OBSERVATIONS = FRAME_COUNT*NEW_FRAME_SIZE
INPUT_WEIGHTS = ("actor.0.weight", "critic.0.weight")
ADAM_MOMENTS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def expand_input_columns(value):
    """Append two zero columns to each of the six legacy observation frames."""
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != OLD_OBSERVATIONS:
        raise ValueError("Input matrix must have shape [outputs,{}]".format(OLD_OBSERVATIONS))
    expanded = value.new_zeros((value.shape[0], FRAME_COUNT, NEW_FRAME_SIZE))
    expanded[:, :, :OLD_FRAME_SIZE] = value.reshape(value.shape[0], FRAME_COUNT, OLD_FRAME_SIZE)
    return expanded.reshape(value.shape[0], NEW_OBSERVATIONS)


def _optimizer_parameter_mapping(optimizer, model):
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        raise ValueError("Unsupported optimizer state: missing parameter groups or states")
    group_ids = [group.get("params", []) for group in groups]
    ids = [parameter_id for group in group_ids for parameter_id in group]
    if len(set(ids)) != len(ids):
        raise ValueError("Optimizer parameter IDs must be unique")
    if any("param_names" in group for group in groups):
        if not all("param_names" in group for group in groups):
            raise ValueError("Optimizer param_names must be present for every group")
        name_groups = [group["param_names"] for group in groups]
    else:
        # PPORunner/PPO constructs Adam with these three explicit groups. Using
        # their names avoids confusing equally sized actor/critic input layers.
        name_groups = [[name for name in model if name.startswith("actor.")],
                       [name for name in model if name.startswith("critic.")], ["std"]]
        if len(groups) != len(name_groups):
            raise ValueError("Unsupported optimizer layout: expected actor, critic, std groups or explicit param_names")
    names = [name for group in name_groups for name in group]
    if len(set(names)) != len(names) or set(names) != set(model):
        raise ValueError("Optimizer parameter names must identify each model parameter exactly once")
    if any(len(parameters) != len(names_for_group)
           for parameters, names_for_group in zip(group_ids, name_groups)):
        raise ValueError("Optimizer parameter group sizes do not match the model")
    mapping = {}
    for parameters, names_for_group in zip(group_ids, name_groups):
        for parameter_id, name in zip(parameters, names_for_group):
            parameter = model[name]
            if not isinstance(parameter, torch.Tensor):
                raise ValueError("Unsupported non-tensor model parameter: {}".format(name))
            state = states.get(parameter_id, {})
            for field, value in state.items():
                if field in ADAM_MOMENTS:
                    if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                        raise ValueError("Optimizer {} shape does not match {}".format(field, name))
                elif isinstance(value, torch.Tensor) and value.numel() != 1:
                    raise ValueError("Unsupported optimizer tensor state {} for {}".format(field, name))
            if state and not all(field in state for field in ("exp_avg", "exp_avg_sq")):
                raise ValueError("Incomplete Adam momentum state for {}".format(name))
            mapping[name] = parameter_id
    if set(states)-set(ids):
        raise ValueError("Optimizer contains states for unknown parameter IDs")
    return mapping


def upgrade_checkpoint_observations(checkpoint, source_path=None):
    """Return a new 198-input checkpoint without changing the supplied mapping.

    The two added features follow each frame's original 31 features. Their
    initial weights and Adam moments are zero, so nonzero new observations do
    not change the old actor or critic prediction at the conversion boundary.
    Only the current feed-forward actor/critic and Adam layouts are supported.
    """
    if checkpoint.get("observation_upgrade") is not None:
        raise ValueError("Checkpoint already contains an observation upgrade")
    environment = checkpoint.get("environment_config", {})
    if not isinstance(environment, dict) or environment.get("observe_leg_lengths", False):
        raise ValueError("Checkpoint is not a legacy observation configuration")
    model = checkpoint.get("model_state_dict")
    if not isinstance(model, dict):
        raise ValueError("Checkpoint has no model_state_dict")
    for name in INPUT_WEIGHTS:
        value = model.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != OLD_OBSERVATIONS:
            raise ValueError("{} must be a legacy {}-input matrix".format(name, OLD_OBSERVATIONS))
    optimizer = checkpoint.get("optimizer_state_dict")
    mapping = None
    if optimizer is not None:
        if not isinstance(optimizer, dict):
            raise ValueError("optimizer_state_dict must be a mapping")
        mapping = _optimizer_parameter_mapping(optimizer, model)
    upgraded = deepcopy(checkpoint)
    for name in INPUT_WEIGHTS:
        upgraded["model_state_dict"][name] = expand_input_columns(model[name])
        if mapping is not None:
            state = upgraded["optimizer_state_dict"]["state"].get(mapping[name], {})
            for field in ADAM_MOMENTS:
                if field in state:
                    state[field] = expand_input_columns(state[field])
    upgraded.setdefault("environment_config", {})["observe_leg_lengths"] = True
    upgraded["observation_upgrade"] = {
        "schema_version": 1,
        "source_checkpoint": str(Path(source_path).resolve()) if source_path is not None else None,
        "source_observations": OLD_OBSERVATIONS,
        "target_observations": NEW_OBSERVATIONS,
        "frame_count": FRAME_COUNT,
        "source_features_per_frame": OLD_FRAME_SIZE,
        "target_features_per_frame": NEW_FRAME_SIZE,
        "appended_features": ["actual_leg_length_left", "actual_leg_length_right"],
        "new_input_weights_and_adam_moments": "zero",
        "preserves_actor_and_critic_predictions_at_upgrade": True,
    }
    return upgraded


def save_upgraded_checkpoint(source, target):
    """Load on CPU and atomically create a distinct output without overwriting."""
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise ValueError("Source checkpoint does not exist: {}".format(source))
    if os.path.lexists(str(target)):
        raise ValueError("Target already exists: {}".format(target))
    checkpoint = torch.load(str(source), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Source checkpoint must contain a mapping")
    upgraded = upgrade_checkpoint_observations(checkpoint, source_path=source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".observation-upgrade-", suffix=".pt",
                                         dir=str(target.parent), delete=False) as stream:
            temporary = Path(stream.name)
        torch.save(upgraded, str(temporary))
        # Hard-linking a completed file is atomic and fails if a target appears
        # concurrently; replace/rename could overwrite an existing checkpoint.
        os.link(str(temporary), str(target))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return upgraded["observation_upgrade"]
