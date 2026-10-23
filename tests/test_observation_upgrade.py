"""CPU regressions for preserving a trained policy when adding leg sensors."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import isaacgym  # noqa: F401; extension must precede torch in combined discovery.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"third_party"))
from rsl_rl.modules import ActorCritic
from wheelleg_gym.observation_upgrade import (
    expand_input_columns, save_upgraded_checkpoint, upgrade_checkpoint_observations,
)


def policy(observations):
    with contextlib.redirect_stdout(io.StringIO()):
        return ActorCritic(observations, observations, 6, actor_hidden_dims=[16, 8],
                           critic_hidden_dims=[16, 8], init_noise_std=[1]*4+[0.1]*2).cpu()


def optimizer(model):
    return torch.optim.Adam([{"params": model.actor.parameters()},
                             {"params": model.critic.parameters()},
                             {"params": [model.std]}], lr=0.0003, amsgrad=True)


class ObservationUpgradeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = policy(186)
        self.optimizer = optimizer(self.model)
        self.observations = torch.randn(5, 186)
        loss = (self.model.act_inference(self.observations).square().mean()
                + self.model.evaluate(self.observations).square().mean()
                + self.model.std.square().mean())
        loss.backward()
        self.optimizer.step()
        self.checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter": 794, "total_steps": 6406400,
            "training_config": {"policy": {"actor_hidden_dims": [16, 8], "critic_hidden_dims": [16, 8]}},
            "environment_config": {"command": [0, 0, 0, 0.08]},
        }

    def test_nonzero_new_features_preserve_actor_and_critic_predictions(self):
        upgraded = upgrade_checkpoint_observations(self.checkpoint)
        restored = policy(198)
        restored.load_state_dict(upgraded["model_state_dict"], strict=True)
        extended = torch.randn(5, 6, 33)
        extended[:, :, :31] = self.observations.reshape(5, 6, 31)
        # The added features are deliberately nonzero, as real leg sensors are.
        extended[:, :, 31:] += 5
        extended = extended.reshape(5, 198)
        torch.testing.assert_close(restored.act_inference(extended), self.model.act_inference(self.observations))
        torch.testing.assert_close(restored.evaluate(extended), self.model.evaluate(self.observations))

    def test_frame_padding_preserves_every_old_column_in_order(self):
        original = torch.arange(2*186).reshape(2, 186)
        expanded = expand_input_columns(original).reshape(2, 6, 33)
        torch.testing.assert_close(expanded[:, :, :31], original.reshape(2, 6, 31), rtol=0, atol=0)
        torch.testing.assert_close(expanded[:, :, 31:], torch.zeros(2, 6, 2, dtype=original.dtype), rtol=0, atol=0)

    def test_adam_moments_load_and_allow_learning_the_new_features(self):
        upgraded = upgrade_checkpoint_observations(self.checkpoint)
        restored = policy(198)
        restored.load_state_dict(upgraded["model_state_dict"], strict=True)
        resumed_optimizer = optimizer(restored)
        resumed_optimizer.load_state_dict(upgraded["optimizer_state_dict"])
        for old_parameter, new_parameter in ((self.model.actor[0].weight, restored.actor[0].weight),
                                             (self.model.critic[0].weight, restored.critic[0].weight)):
            old_state = self.optimizer.state[old_parameter]
            new_state = resumed_optimizer.state[new_parameter]
            torch.testing.assert_close(new_state["step"], old_state["step"], rtol=0, atol=0)
            for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                torch.testing.assert_close(new_state[field], expand_input_columns(old_state[field]), rtol=0, atol=0)
        # A real update verifies that all loaded optimizer tensors match their
        # new parameters, and that zero initialization does not freeze sensors.
        observations = torch.randn(5, 198)
        resumed_optimizer.zero_grad()
        loss = restored.act_inference(observations).square().mean()+restored.evaluate(observations).square().mean()
        loss.backward()
        resumed_optimizer.step()
        self.assertTrue(restored.actor[0].weight.reshape(16, 6, 33)[:, :, 31:].abs().sum().item() > 0)

    def test_source_other_parameters_counters_and_std_are_preserved(self):
        snapshot = deepcopy(self.checkpoint)
        upgraded = upgrade_checkpoint_observations(self.checkpoint, source_path="legacy.pt")
        self.assertEqual(upgraded["iter"], 794)
        self.assertEqual(upgraded["total_steps"], 6406400)
        self.assertTrue(upgraded["environment_config"]["observe_leg_lengths"])
        self.assertNotIn("observe_leg_lengths", self.checkpoint["environment_config"])
        for name, value in snapshot["model_state_dict"].items():
            torch.testing.assert_close(self.checkpoint["model_state_dict"][name], value, rtol=0, atol=0)
            if name not in ("actor.0.weight", "critic.0.weight"):
                torch.testing.assert_close(upgraded["model_state_dict"][name], value, rtol=0, atol=0)
        self.assertEqual(upgraded["observation_upgrade"]["source_checkpoint"], str(Path("legacy.pt").resolve()))

    def test_reupgrade_wrong_shapes_or_ambiguous_optimizer_layout_are_rejected(self):
        upgraded = upgrade_checkpoint_observations(self.checkpoint)
        with self.assertRaises(ValueError):
            upgrade_checkpoint_observations(upgraded)
        invalid = deepcopy(self.checkpoint)
        invalid["model_state_dict"]["critic.0.weight"] = torch.zeros(16, 187)
        with self.assertRaises(ValueError):
            upgrade_checkpoint_observations(invalid)
        invalid = deepcopy(self.checkpoint)
        groups = invalid["optimizer_state_dict"]["param_groups"]
        groups[0]["params"] += groups.pop()["params"]
        with self.assertRaises(ValueError):
            upgrade_checkpoint_observations(invalid)
        invalid = deepcopy(self.checkpoint)
        invalid["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.zeros(16, 185)
        with self.assertRaises(ValueError):
            upgrade_checkpoint_observations(invalid)

    def test_save_and_cli_refuse_overwriting_any_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"old.pt"
            target = Path(directory)/"new.pt"
            torch.save(self.checkpoint, str(source))
            original_bytes = source.read_bytes()
            save_upgraded_checkpoint(source, target)
            self.assertEqual(source.read_bytes(), original_bytes)
            restored = torch.load(str(target), map_location="cpu")
            self.assertEqual(restored["model_state_dict"]["actor.0.weight"].shape, (16, 198))
            target_bytes = target.read_bytes()
            result = subprocess.run([sys.executable, str(ROOT/"tools/upgrade_observations.py"),
                                     "--source", str(source), "--target", str(target)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Target already exists", result.stderr)
            self.assertEqual(target.read_bytes(), target_bytes)
            with self.assertRaises(ValueError):
                save_upgraded_checkpoint(source, source)


if __name__ == "__main__":
    unittest.main()
