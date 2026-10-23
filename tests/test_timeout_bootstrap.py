"""CPU Bellman targets must use terminal states rather than current/reset states."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
import warnings

import isaacgym  # noqa: F401; extension must precede torch in combined discovery.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"third_party"))
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


class TimeoutBootstrapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        with contextlib.redirect_stdout(io.StringIO()):
            model = ActorCritic(3, 3, 1, actor_hidden_dims=[2], critic_hidden_dims=[2])
        # A known value function makes V(current), V(terminal), and V(reset)
        # distinguishable without any optimization or simulator rollout.
        model.critic = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.critic.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        self.algorithm = PPO(model, gamma=0.9, device="cpu")
        self.algorithm.init_storage(4, 1, [3], [3], [3], [1])
        self.current = torch.tensor([[1., 0, 0], [2., 0, 0], [3., 0, 0], [4., 0, 0]])
        self.algorithm.act(self.current, self.current, self.current)
        self.rewards = torch.tensor([1., 2., 3., 4.])
        self.dones = torch.tensor([True, False, True, True])
        self.reset_observations = torch.full((4, 3), 99.0)

    def stored_rewards(self):
        return self.algorithm.storage.rewards[0, :, 0]

    def test_sparse_mixed_timeouts_failures_and_survivors_use_terminal_values(self):
        ids = torch.tensor([3, 0, 2])
        terminal = torch.tensor([[10., 0, 0], [20., 0, 0], [30., 0, 0]])
        timeouts = torch.tensor([True, False, False, True])
        snapshots = [value.clone() for value in (self.rewards, self.dones, ids, terminal, timeouts)]
        self.algorithm.process_env_step(self.rewards, self.dones, {
            "time_outs": timeouts, "terminal_env_ids": ids,
            "terminal_observation": terminal,
        }, self.reset_observations)
        # Only environments 0 and 3 time out. Environment 2 physically fails.
        torch.testing.assert_close(self.stored_rewards(), torch.tensor([19., 2., 3., 13.]))
        for original, snapshot in zip((self.rewards, self.dones, ids, terminal, timeouts), snapshots):
            torch.testing.assert_close(original, snapshot, rtol=0, atol=0)

    def test_full_terminal_batch_without_ids_uses_environment_order(self):
        terminal = torch.tensor([[10., 0, 0], [20., 0, 0], [30., 0, 0], [40., 0, 0]])
        self.algorithm.process_env_step(self.rewards, self.dones, {
            "time_outs": torch.tensor([True, False, False, True]),
            "terminal_observation": terminal,
        }, self.reset_observations)
        torch.testing.assert_close(self.stored_rewards(), torch.tensor([10., 2., 3., 40.]))

    def test_no_normal_timeouts_does_not_require_terminal_observations(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            self.algorithm.process_env_step(self.rewards, self.dones, {
                "time_outs": torch.zeros(4, dtype=torch.bool),
            }, self.reset_observations)
        self.assertEqual(recorded, [])
        torch.testing.assert_close(self.stored_rewards(), self.rewards)

    def test_legacy_missing_terminal_data_warns_and_never_uses_reset_observation(self):
        with self.assertWarnsRegex(RuntimeWarning, r"falling back to V\(s_t\)"):
            self.algorithm.process_env_step(self.rewards, self.dones, {
                "time_outs": torch.tensor([True, False, False, True]),
            }, self.reset_observations)
        torch.testing.assert_close(self.stored_rewards(), torch.tensor([1.9, 2., 3., 7.6]))

    def test_incomplete_terminal_mapping_cannot_silently_bootstrap_wrong_state(self):
        with self.assertRaisesRegex(ValueError, "every timed-out environment"):
            self.algorithm.process_env_step(self.rewards, self.dones, {
                "time_outs": torch.tensor([True, False, False, True]),
                "terminal_env_ids": torch.tensor([0]),
                "terminal_observation": torch.tensor([[10., 0, 0]]),
            }, self.reset_observations)
        self.assertEqual(self.algorithm.storage.step, 0)


if __name__ == "__main__":
    unittest.main()
