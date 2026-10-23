"""CPU regressions for scalar/vector exploration and saved-policy compatibility."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

# Match the runtime's extension import ordering when discovered with Gym tests.
import isaacgym  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))
from rsl_rl.modules import ActorCritic


def policy(std):
    with contextlib.redirect_stdout(io.StringIO()):
        return ActorCritic(
            186, 186, 6, actor_hidden_dims=[16, 8],
            critic_hidden_dims=[16, 8], init_noise_std=std,
        ).cpu()


class ExplorationPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.observations = torch.randn(4, 186)

    def test_scalar_initialization_samples_finite_actions(self):
        model = policy(0.5)
        torch.testing.assert_close(model.std, torch.full((6,), 0.5))
        actions = model.act(self.observations)
        self.assertEqual(actions.shape, (4, 6))
        self.assertTrue(torch.isfinite(actions).all().item())
        self.assertTrue(torch.isfinite(model.get_actions_log_prob(actions)).all().item())

    def test_vector_initialization_preserves_each_motor_scale(self):
        scales = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1]
        model = policy(scales)
        torch.testing.assert_close(model.std, torch.tensor(scales))
        actions = model.act(self.observations)
        self.assertTrue(torch.isfinite(actions).all().item())
        torch.testing.assert_close(model.action_std, torch.tensor(scales).expand(4, -1))

    def test_vector_noise_remains_trainable(self):
        model = policy([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])
        self.assertIsInstance(model.std, torch.nn.Parameter)
        model.act(self.observations)
        # Entropy is an actual PPO objective term and must update every scale.
        loss = -model.entropy.mean()
        loss.backward()
        self.assertIsNotNone(model.std.grad)
        self.assertTrue(torch.isfinite(model.std.grad).all().item())
        self.assertTrue((model.std.grad.abs() > 0).all().item())

    def test_invalid_noise_shape_or_values_are_rejected(self):
        for invalid in ([1.0, 2.0], [[1.0]*6], 0.0, -1.0,
                        [1.0]*5+[0.0], float("nan"), float("inf")):
            with self.subTest(noise=invalid):
                with self.assertRaises(ValueError):
                    policy(invalid)

    def test_legacy_scalar_checkpoint_strictly_loads_into_vector_policy(self):
        legacy = policy(4.0)
        state = legacy.state_dict()
        # The previous implementation stored scalar initialization as a 6-value
        # trainable parameter, not a scalar tensor.
        state["std"] = 4.0*torch.ones(6)
        checkpoint = io.BytesIO()
        torch.save({"model_state_dict": state,
                    "training_config": {"policy": {"init_noise_std": 4.0}}}, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, map_location="cpu")
        target = policy([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])
        target.load_state_dict(restored["model_state_dict"], strict=True)
        torch.testing.assert_close(target.std, legacy.std)
        torch.testing.assert_close(target.act_inference(self.observations),
                                   legacy.act_inference(self.observations))

    def test_vector_checkpoint_strictly_loads_into_scalar_config_policy(self):
        trained = policy([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])
        with torch.no_grad():
            trained.std.add_(0.02)
        restored = policy(4.0)
        restored.load_state_dict(trained.state_dict(), strict=True)
        torch.testing.assert_close(restored.std, trained.std)
        torch.testing.assert_close(restored.act_inference(self.observations),
                                   trained.act_inference(self.observations))


if __name__ == "__main__":
    unittest.main()
