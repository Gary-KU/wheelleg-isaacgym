"""Frozen pre-cleanup rewards guard both environment interfaces.

These vectors were recorded from the original LeggedRobot._rewards at
aee6fa3 before moving its equations to the shared reward helpers. They are
deliberately independent of the current scales and helper implementation.
The fixture covers support/stability/tracking switches, failures, asymmetric
legs, passive-limit proximity and nonzero motion/torque errors.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import isaacgym  # noqa: F401; extension must precede PyTorch in combined tests.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.envs.wheelleg.wheelleg_robot import WheelLegRobot
from legged_gym.envs.wheelleg.wheelleg import WheelLegged
from legged_gym.envs.wheelleg.wheelleg_config import WheelLeggedCfg


# Key order: support_rewards, stability_rewards, tracking_penalties.
PRE_CLEANUP_REWARDS = {
    "000": [
        6.946565628051758,
        -93.05292510986328,
        8.145373344421387,
        -93.8896255493164,
    ],
    "001": [
        6.793757438659668,
        -93.69886779785156,
        7.644414901733398,
        -94.35813903808594,
    ],
    "010": [
        6.795547962188721,
        -93.47187042236328,
        7.516438007354736,
        -94.18277740478516,
    ],
    "011": [
        6.642739772796631,
        -94.11781311035156,
        7.015479564666748,
        -94.65129089355469,
    ],
    "100": [
        5.900451183319092,
        -94.40702819824219,
        5.534780979156494,
        -95.36461639404297,
    ],
    "101": [
        5.747842788696289,
        -95.04566955566406,
        5.046822547912598,
        -95.82313537597656,
    ],
    "110": [
        5.749433517456055,
        -94.82597351074219,
        4.905844688415527,
        -95.65776824951172,
    ],
    "111": [
        5.596825122833252,
        -95.46461486816406,
        4.417886257171631,
        -96.11628723144531,
    ],
}


def measured_state():
    """Reproduce the inputs used to capture the pre-cleanup output vectors."""
    generator = torch.Generator().manual_seed(742)

    def values(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    env = SimpleNamespace(num_envs=4, device="cpu", terrain_model=None)
    env.commands = torch.tensor(
        [[0, 0, 0, 0.08], [0.3, 0, 0.2, 0.12], [-0.2, 0, -0.1, 0.1], [0, 0, 0, 0.15]],
        dtype=torch.float64,
    )
    env.lin_vel = values(4, 3) * 0.15
    env.ang_vel = values(4, 3) * 0.2
    env.actions = values(4, 6) * 0.3
    env.last_actions = values(4, 6) * 0.3
    env.slice_obs = values(4, 33) * 0.2
    env.slice_obs[:, 8] = -1 + values(4) * 0.03
    env.length_l = torch.tensor([0.081, 0.112, 0.089, 0.15], dtype=torch.float64)
    env.length_r = torch.tensor([0.079, 0.123, 0.103, 0.14], dtype=torch.float64)
    env.torques = values(4, 6) * 2
    env.action_indices = torch.arange(6)
    env.last_applied_torques = values(4, 6) * 2
    env.dof_state = torch.zeros(4, 4, 2, dtype=torch.float64)
    env.dof_state[:, :, 0] = values(4, 4) * 0.8
    env.passive_indices = torch.arange(4)
    env.passive_lower = torch.full((4,), -1.0, dtype=torch.float64)
    env.passive_upper = torch.full((4,), 1.0, dtype=torch.float64)
    env.passive_limit_margin = torch.minimum(
        env.dof_state[:, :, 0] - env.passive_lower,
        env.passive_upper - env.dof_state[:, :, 0],
    )
    env.projected_gravity = values(4, 3) * 0.1
    env.projected_gravity[:, 2] = -1
    for name in ("phi_l4", "phi_r1", "phi_r4", "phi_l1"):
        setattr(env, name, values(4) * 0.2 + 1.5)
    env.theta_l = values(4) * 0.1
    env.theta_r = values(4) * 0.1
    env.total_yaw = values(4) * 0.2
    env.target_yaw = values(4) * 0.2
    return env


class RewardRegressionTests(unittest.TestCase):
    def check_interface(self, registered):
        state = measured_state()
        if registered:
            # Exercise real reward registration without creating a simulator.
            env = WheelLegged.__new__(WheelLegged)
            env.__dict__.update(vars(state))
            env.cfg = WheelLeggedCfg()
            env._prepare_reward_function()
        else:
            env = state
        terminated = torch.tensor([False, True, False, True])
        for flags, expected in PRE_CLEANUP_REWARDS.items():
            with self.subTest(flags=flags):
                (env.support_rewards, env.stability_rewards, env.tracking_penalties) = (
                    flag == "1" for flag in flags
                )
                actual = (
                    env._rewards(terminated)
                    if registered
                    else WheelLegRobot._rewards(env, terminated)
                )
                # The original sums into float32; changing grouping can move
                # the final result by a few ULPs without changing its math.
                torch.testing.assert_close(
                    actual,
                    torch.tensor(expected, dtype=actual.dtype),
                    rtol=0,
                    atol=2e-5,
                )

    def test_legacy_rewards_match_pre_cleanup_outputs_for_all_switches(self):
        self.check_interface(registered=False)

    def test_registered_rewards_match_pre_cleanup_outputs_for_all_switches(self):
        self.check_interface(registered=True)


if __name__ == "__main__":
    unittest.main()
