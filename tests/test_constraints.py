"""Physical invariants for the compliant closure, without running a simulator."""

import math
from pathlib import Path
import sys
import unittest

# Other tests may load a simulation in this process; Gym requires this order.
from isaacgym import gymapi  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.constraints import ClosureForces, quat_rotate, quat_rotate_inverse


def metadata(anchor0=(1.0, 0.0, 0.0)):
    return {"constraints": [{
        "name": "loop", "body0": "a", "body1": "b",
        "local_pos0": list(anchor0), "local_pos1": [0.0, 0.0, 0.0],
        "local_axis0": [0.0, 0.0, 1.0], "local_axis1": [0.0, 0.0, 1.0],
    }]}


def state(device="cpu"):
    result = torch.zeros(1, 2, 13, device=device)
    result[..., 6] = 1.0
    result[0, 1, 0] = 1.0
    return result


class ClosureTests(unittest.TestCase):
    def assertTensorEqual(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def closure(self, **kwargs):
        defaults = dict(stiffness=10.0, damping=1.0, angular_stiffness=2.0,
                        angular_damping=0.5, max_force=None, max_torque=None)
        defaults.update(kwargs)
        return ClosureForces(metadata(), ["a", "b"], [[0.2, 0, 0], [0, 0, 0]],
                             1, "cpu", **defaults)

    def test_quaternion_rotation_and_inverse(self):
        q = torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
        vector = torch.tensor([[1.0, 0.0, 0.0]])
        self.assertTensorEqual(quat_rotate(q, vector), torch.tensor([[0.0, 1.0, 0.0]]))
        self.assertTensorEqual(quat_rotate_inverse(q, quat_rotate(q, vector)), vector)

    def test_coincident_stationary_anchors_have_zero_wrench(self):
        forces, torques = self.closure().compute(state())
        self.assertTensorEqual(forces, torch.zeros_like(forces))
        self.assertTensorEqual(torques, torch.zeros_like(torques))

    def test_restoring_force_and_com_moment(self):
        initial = state()
        initial[0, 1, 1] = 0.1
        forces, torques = self.closure().compute(initial)
        self.assertTensorEqual(forces, torch.tensor([[[0., 1., 0.], [0., -1., 0.]]]))
        # Anchor is one meter from body origin but 0.8 meters from COM.
        self.assertTensorEqual(torques, torch.tensor([[[0., 0., 0.8], [0., 0., 0.]]]))

    def test_positional_spring_conserves_total_linear_and_angular_momentum(self):
        initial = state()
        initial[0, 1, :3] = torch.tensor([1.2, 0.1, -0.05])
        closure = self.closure(damping=0)
        forces, torques = closure.compute(initial)
        world_com = initial[..., :3] + quat_rotate(initial[..., 3:7], closure.com_local)
        self.assertTensorEqual(forces.sum(dim=1), torch.zeros(1, 3))
        world_moments = torch.cross(world_com, forces, dim=-1) + torques
        self.assertTensorEqual(world_moments.sum(dim=1), torch.zeros(1, 3))

    def test_anchor_velocity_is_measured_from_com(self):
        initial = state()
        # COM moves downward as rotation makes the off-center anchor move up.
        initial[0, 0, 8] = -0.8
        initial[0, 0, 12] = 1.0
        forces, torques = self.closure().compute(initial)
        self.assertTensorEqual(forces, torch.zeros_like(forces))
        self.assertTensorEqual(torques, torch.zeros_like(torques))

    def test_axis_spring_opposes_misalignment(self):
        initial = state()
        angle = 0.1
        initial[0, 1, 3] = math.sin(angle / 2)
        initial[0, 1, 6] = math.cos(angle / 2)
        closure = self.closure()
        forces, torques = closure.compute(initial)
        self.assertTensorEqual(forces, torch.zeros_like(forces))
        self.assertTensorEqual(torques[0, :, 0], torch.tensor([0.2, -0.2]))
        self.assertTensorEqual(closure.axis_errors(initial), torch.tensor([[angle]]))

    def test_hinge_axis_remains_free_and_other_rotation_is_damped(self):
        closure = ClosureForces(metadata((0, 0, 0)), ["a", "b"],
                                 [[0, 0, 0], [0, 0, 0]], 1, "cpu",
                                 angular_damping=0.5)
        initial = state()
        initial[0, 1, 0] = 0
        initial[0, 0, 12] = 3
        initial[0, 1, 12] = -4
        forces, torques = closure.compute(initial)
        self.assertTensorEqual(forces, torch.zeros_like(forces))
        self.assertTensorEqual(torques, torch.zeros_like(torques))
        initial[0, 1, 10] = 2
        _, torques = closure.compute(initial)
        self.assertTensorEqual(torques[0, :, 0], torch.tensor([1., -1.]))

    def test_force_cap_preserves_equal_and_opposite_reaction(self):
        initial = state()
        initial[0, 1, 1] = 100
        forces, _ = self.closure(max_force=3.0).compute(initial)
        self.assertTensorEqual(torch.linalg.vector_norm(forces, dim=-1), torch.tensor([[3., 3.]]))
        self.assertTensorEqual(forces.sum(dim=1), torch.zeros(1, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu(self):
        initial = state()
        initial[0, 1, 1] = 0.1
        cpu = self.closure()
        gpu = ClosureForces(metadata(), ["a", "b"], [[0.2, 0, 0], [0, 0, 0]],
                            1, "cuda", stiffness=10, damping=1,
                            angular_stiffness=2, angular_damping=0.5,
                            max_force=None, max_torque=None)
        cpu_wrench = cpu.compute(initial)
        gpu_wrench = gpu.compute(initial.cuda())
        for expected, actual in zip(cpu_wrench, gpu_wrench):
            self.assertTensorEqual(actual.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
