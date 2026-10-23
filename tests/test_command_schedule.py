"""CPU regressions for command transitions and observable leg lengths."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest

import isaacgym  # noqa: F401; extension must precede Torch.
from isaacgym import gymapi
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.cli import common_parser, environment_kwargs, validate_common
from wheelleg_gym.commands import CommandGovernor, sample_command_targets, slew_commands
from wheelleg_gym.env import WheelLegGymEnv


class CommandGovernorTests(unittest.TestCase):
    def test_disabled_limits_preserve_legacy_targets_without_mutation(self):
        current = torch.tensor([[0.1, -0.2, 0.3, 0.08]])
        target = torch.tensor([[-0.3, 0.4, -0.8, 0.12]])
        before = current.clone()
        actual = slew_commands(current, target, 0.02)
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
        torch.testing.assert_close(current, before, rtol=0, atol=0)
        self.assertNotEqual(actual.data_ptr(), target.data_ptr())

    def test_planar_norm_and_yaw_changes_follow_simulation_time(self):
        current = torch.tensor([[0., 0., 0., 0.08]])
        target = torch.tensor([[0.3, 0.4, -0.8, 0.12]])
        actual = slew_commands(current, target, 0.1, 0.5, 1.0)
        torch.testing.assert_close(actual, torch.tensor([[0.03, 0.04, -0.1, 0.12]]))
        self.assertAlmostEqual(torch.linalg.vector_norm(actual[0, :2]).item(), 0.05)

    def test_small_target_is_not_overshot_and_axes_can_be_disabled_independently(self):
        current = torch.tensor([[0., 0., 0., 0.08]])
        target = torch.tensor([[0.01, 0., 0.1, 0.08]])
        torch.testing.assert_close(slew_commands(current, target, 0.1, 1., 2.), target)
        actual = slew_commands(current, target, 0.01, 0., 2.)
        torch.testing.assert_close(actual[0, :2], target[0, :2])
        self.assertAlmostEqual(actual[0, 2].item(), 0.02)

    def test_governor_keeps_target_separate_and_reset_clears_transition(self):
        governor = CommandGovernor([[0., 0., 0., 0.08]], 0.8, 2.)
        target = torch.tensor([[0.3, 0., 0.8, 0.08]])
        first = governor.step(target, 0.02)
        torch.testing.assert_close(first, torch.tensor([[0.016, 0., 0.04, 0.08]]))
        torch.testing.assert_close(governor.target_commands, target)
        governor.reset([[0., 0., 0., 0.08]])
        torch.testing.assert_close(governor.commands, torch.tensor([[0., 0., 0., 0.08]]))
        torch.testing.assert_close(governor.target_commands, governor.commands)

    def test_invalid_limits_values_and_shapes_are_rejected(self):
        initial = torch.zeros(1, 4)
        for dt, linear, yaw in ((0., 0., 0.), (.02, -1., 0.), (.02, 0., float("nan"))):
            with self.assertRaises(ValueError):
                slew_commands(initial, initial, dt, linear, yaw)
        for target in (torch.zeros(2, 4), torch.zeros(1, 3), torch.full((1, 4), float("inf"))):
            with self.assertRaises(ValueError):
                slew_commands(initial, target, .02)

    def test_stationary_mixture_preserves_leg_targets(self):
        actual = sample_command_targets(8, "cpu", (.3, .3), (.8, .8), (.1, .1), 1.)
        torch.testing.assert_close(actual[:, :3], torch.zeros(8, 3))
        torch.testing.assert_close(actual[:, 3], torch.full((8,), .1))
        moving = sample_command_targets(8, "cpu", (-.3, .3), (-.8, .8), (.08, .08))
        self.assertTrue((moving[:, 0].abs() <= .3).all())
        self.assertTrue((moving[:, 2].abs() <= .8).all())

    def test_cli_round_trip_and_invalid_limits(self):
        parser = common_parser("commands", 1)
        args = parser.parse_args(["--allow-approximate-loop", "--command-resample-seconds", "2",
            "--linear-acceleration", ".8", "--yaw-acceleration", "2", "--zero-command-fraction", ".25",
            "--observe-leg-lengths", "--stability-rewards"])
        validate_common(parser, args)
        kwargs = environment_kwargs(args)
        self.assertEqual(kwargs["command_resample_seconds"], 2.)
        self.assertEqual(kwargs["linear_acceleration"], .8)
        self.assertEqual(kwargs["yaw_acceleration"], 2.)
        self.assertEqual(kwargs["zero_command_fraction"], .25)
        self.assertTrue(kwargs["observe_leg_lengths"])
        self.assertTrue(kwargs["stability_rewards"])
        for option, value in (("--linear-acceleration", "-1"), ("--yaw-acceleration", "nan"),
                              ("--command-resample-seconds", "-2"), ("--zero-command-fraction", "1.1")):
            bad = parser.parse_args(["--allow-approximate-loop", option, value])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_common(parser, bad)

    def test_manual_height_extension_keeps_old_training_range_and_supports_cpu_thread_choice(self):
        parser = common_parser("commands", 1)
        args = parser.parse_args(["--allow-approximate-loop", "--command", "0", "0", "0", ".20",
                                 "--initial-leg-length", ".20", "--physics-threads", "0"])
        validate_common(parser, args)
        kwargs = environment_kwargs(args)
        self.assertEqual(kwargs["command"][3], .20)
        self.assertEqual(kwargs["initial_leg_length"], .20)
        self.assertEqual(kwargs["physics_threads"], 0)
        self.assertEqual(kwargs["leg_length_range"], (.06, .15))
        for options in (["--command", "0", "0", "0", ".201"], ["--physics-threads", "-1"]):
            bad = parser.parse_args(["--allow-approximate-loop"]+options)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_common(parser, bad)


class TimedCommandEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = WheelLegGymEnv(num_envs=2, sim_device="cpu", headless=True,
            random_commands=True, forward_speed_range=(.3, .3), yaw_speed_range=(.8, .8),
            leg_length_range=(.08, .08), command_resample_seconds=.04,
            linear_acceleration=.8, yaw_acceleration=2., zero_command_fraction=0.,
            observe_leg_lengths=True, allow_approximate_loop=True)
        params = cls.env.gym.get_sim_params(cls.env.sim)
        params.gravity = gymapi.Vec3(0, 0, 0)
        cls.env.gym.set_sim_params(cls.env.sim, params)
        shift = 1.0-cls.env.default_root_states[:, 2]
        cls.env.default_root_states[:, 2] += shift
        cls.env.default_rb_states[:, :, 2] += shift.unsqueeze(1)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        env = self.env
        env.random_commands = True
        env.forward_speed_range = (.3, .3)
        env.yaw_speed_range = (.8, .8)
        env.zero_command_fraction = 0.
        env.reset()
        self.zero_actions = torch.zeros(2, 6)

    def test_reward_command_precedes_next_observation_command_and_timed_reversal(self):
        env = self.env
        observations, _, _, dones, first = env.step(self.zero_actions)
        self.assertFalse(dones.any())
        torch.testing.assert_close(first["pre_reset_commands"][:, :3], torch.zeros(2, 3))
        torch.testing.assert_close(env.commands[:, [0, 2]], torch.tensor([[.016, .04], [.016, .04]]))
        torch.testing.assert_close(observations[:, -33+9], env.commands[:, 0]*1.5)
        torch.testing.assert_close(observations[:, -33+11], env.commands[:, 2])
        self.assertEqual(first["command_resamples"], 0)
        env.forward_speed_range = (-.3, -.3)
        env.yaw_speed_range = (-.8, -.8)
        used = env.commands.clone()
        _, _, _, dones, second = env.step(self.zero_actions)
        self.assertFalse(dones.any())
        self.assertEqual(second["command_resamples"], 2)
        torch.testing.assert_close(second["pre_reset_commands"], used)
        torch.testing.assert_close(env.command_targets[:, [0, 2]], torch.tensor([[-.3, -.8], [-.3, -.8]]))
        torch.testing.assert_close(env.commands[:, :3], torch.zeros(2, 3), atol=1e-7, rtol=0)

    def test_partial_reset_preserves_survivor_schedule_and_new_episode_starts_at_rest(self):
        env = self.env
        env.step(self.zero_actions)
        before = env.commands[1].clone()
        env._reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.commands[0, :3], torch.zeros(3))
        torch.testing.assert_close(env.commands[1], before)
        self.assertEqual(env.command_age_steps.tolist(), [0, 1])
        _, _, _, _, infos = env.step(self.zero_actions)
        self.assertEqual(infos["command_resamples"], 1)
        self.assertEqual(env.command_age_steps.tolist(), [1, 0])

    def test_fixed_commands_are_not_filtered_twice(self):
        env = self.env
        env.random_commands = False
        env.set_commands([.3, 0., .8, .08])
        _, _, _, _, infos = env.step(self.zero_actions)
        torch.testing.assert_close(env.commands, torch.tensor([[.3, 0., .8, .08]]).repeat(2, 1))
        self.assertEqual(infos["command_resamples"], 0)

    def test_appended_leg_lengths_keep_existing_indices_and_command_update(self):
        env = self.env
        self.assertEqual(env.num_obs, 198)
        self.assertEqual(env.observation_frame_size, 33)
        self.assertEqual(env.history.shape, (2, 5, 33))
        expected = 5*torch.stack((env.length_l, env.length_r), dim=-1)
        torch.testing.assert_close(env.obs_buf[:, -2:], expected)
        previous = env.obs_buf[:, :-33].clone()
        observations = env.set_commands([.1, 0., -.2, .1])
        torch.testing.assert_close(observations[:, :-33], previous, rtol=0, atol=0)
        torch.testing.assert_close(observations[:, -2:], expected)
        torch.testing.assert_close(observations[:, -33+9:-33+13], torch.tensor([[.15, 0., -.2, .5]]).repeat(2, 1))


if __name__ == "__main__":
    unittest.main()
