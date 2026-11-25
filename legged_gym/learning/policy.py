"""策略网络工厂：创建 ActorCritic 并按配置初始化输出层。"""

from copy import deepcopy
import math
import torch

from rsl_rl.modules import ActorCritic


def make_policy(num_obs, num_actions, device, policy_config):
    """Construct a policy from caller-owned architecture and exploration settings."""
    if policy_config is None:
        raise ValueError("Provide an explicit policy configuration")
    config = deepcopy(policy_config)
    output_scale = config.pop("actor_output_scale", None)
    if output_scale is not None and (
        not math.isfinite(output_scale) or not 0 < output_scale <= 1
    ):
        raise ValueError("actor_output_scale must be finite and in (0, 1]")
    policy = ActorCritic(num_obs, num_obs, num_actions, **config).to(device)
    if output_scale is not None:
        # Opt-in only: preserve legacy initialization and loaded checkpoints.
        # A small initial mean avoids a random torque bias at the first reset.
        with torch.no_grad():
            policy.actor[-1].weight.mul_(output_scale)
            policy.actor[-1].bias.zero_()
    return policy
