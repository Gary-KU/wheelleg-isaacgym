"""Shared feed-forward policy construction and small runtime helpers."""

from copy import deepcopy
import torch

from rsl_rl.modules import ActorCritic


def scalar(value):
    """Reduce an optional scalar/vector diagnostic for terminal/log output."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().max().item()
    return float(value)


def viewer_is_closed(env):
    viewer = getattr(env, "viewer", None)
    return viewer is not None and env.gym.query_viewer_has_closed(viewer)


def make_policy(num_obs, num_actions, device, policy_config):
    """Construct a policy from caller-owned architecture and exploration settings."""
    if policy_config is None:
        raise ValueError("Provide an explicit policy configuration")
    config = deepcopy(policy_config)
    return ActorCritic(num_obs, num_obs, num_actions, **config).to(device)
