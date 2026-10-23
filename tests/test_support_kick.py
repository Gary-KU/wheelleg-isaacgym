"""A velocity kick must preserve the closure cache and be visible to policy."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_support import apply_velocity_kick, quaternion_yaw


class SupportKickTests(unittest.TestCase):
    def test_additive_world_velocity_updates_root_all_body_caches_and_current_observation(self):
        root = torch.zeros(2, 13)
        root[:, 6] = 1
        root[:, 7:10] = torch.tensor([[1., 2., 3.], [-1., -2., -3.]])
        bodies = torch.zeros(2, 3, 13)
        bodies[:, :, 7:10] = root[:, None, 7:10]
        before_root, before_bodies = root.clone(), bodies.clone()
        events = []
        descriptor = object()
        env = SimpleNamespace(root_states=root, rb_states=bodies, sim=object(), _root_tensor=descriptor)
        env.gym = SimpleNamespace(set_actor_root_state_tensor=lambda sim, tensor: events.append((sim, tensor)))
        env._kinematics = lambda: events.append("kinematics")
        env._observations = lambda advance_history: events.append(("observations", advance_history))
        result = apply_velocity_kick(env, [.1, -.2, .3])
        delta = root.new_tensor([.1, -.2, .3])
        torch.testing.assert_close(root[:, 7:10], before_root[:, 7:10]+delta)
        torch.testing.assert_close(bodies[:, :, 7:10], before_bodies[:, :, 7:10]+delta)
        torch.testing.assert_close(root[:, :7], before_root[:, :7])
        torch.testing.assert_close(root[:, 10:], before_root[:, 10:])
        torch.testing.assert_close(bodies[:, :, :7], before_bodies[:, :, :7])
        self.assertEqual(events, [(env.sim, descriptor), "kinematics", ("observations", False)])
        self.assertEqual(result["base_linear_velocity_before_world_m_per_s"], before_root[:, 7:10].tolist())
        self.assertEqual(result["base_linear_velocity_after_world_m_per_s"], root[:, 7:10].tolist())

    def test_root_quaternion_yaw_sign_and_wrap_boundary(self):
        for angle in (0., .3, -.7, math.pi-.01, -math.pi+.01):
            quaternion = [0., 0., math.sin(angle/2), math.cos(angle/2)]
            self.assertAlmostEqual(quaternion_yaw(quaternion), angle)


if __name__ == "__main__":
    unittest.main()
