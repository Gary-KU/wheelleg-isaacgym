"""Check optional steering against actuator limits and real ground traction."""

from pathlib import Path
import sys
import unittest

import isaacgym  # noqa: F401; the extension must load before Torch.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.cli import common_parser, environment_kwargs
from wheelleg_gym.env import WheelLegGymEnv
from wheelleg_gym.runtime_config import explicit_argument_destinations, resolve_runtime_args
from wheelleg_gym.wheel_control import bounded_heading_target


class WheelControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Construct without either new option to also exercise legacy defaults.
        cls.env = WheelLegGymEnv(
            num_envs=4, sim_device="cpu", headless=True, seed=19,
            allow_approximate_loop=True, initial_leg_length=.08,
            leg_control="hip_pd", leg_feedforward_force=22.465,
        )
        cls.default_mode = cls.env.wheel_control
        cls.default_pitch_damping = cls.env.wheel_pitch_damping

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.wheel_control = "heading"
        self.env.heading_kp = self.env.heading_kd = 0.
        self.env.wheel_pitch_damping = 0.
        self.env.wheel_balance_scale = 1.
        self.env.reset()
        self.actions = torch.zeros(self.env.num_envs, 6)

    def test_policy_common_effort_is_rejected_without_losing_raw_balance_effort(self):
        # The first two would lose their differential if each wheel were
        # clipped to +/-9 Nm before extracting the balance effort.
        self.actions[:, 4:] = torch.tensor([[20., 21.], [-21., -20.],
                                            [-2., 2.], [8., 12.]])/10.
        _, _, _, _, info = self.env.step(self.actions)
        torch.testing.assert_close(
            info["pre_reset_applied_torques"][:, 4:],
            torch.tensor([[-.5, .5], [-.5, .5], [-2., 2.], [-2., 2.]]))

    def test_steering_uses_remaining_headroom_without_disturbing_balance(self):
        env = self.env
        env.heading_kp = 1.
        env.target_yaw[:] = torch.tensor([1., -1., 1., -1.])
        self.actions[:, 4:] = torch.tensor([[-8.5, 8.5], [8.5, -8.5],
                                            [-20., 20.], [20., -20.]])/10.
        _, _, _, _, info = env.step(self.actions)
        # At 8.5 Nm balance only .5 Nm steering fits. At saturated balance
        # steering yields entirely. Independent wheel clipping would change
        # the first two differential efforts from 8.5 to 8.25 Nm.
        applied = info["pre_reset_applied_torques"][:, 4:]
        torch.testing.assert_close(applied, torch.tensor([[-8., 9.], [8., -9.],
                                                          [-9., 9.], [9., -9.]]))
        self.assertLessEqual(float(applied.abs().max()), 9.)

    def test_optional_balance_scale_changes_drive_without_scaling_steering(self):
        env = self.env
        env.wheel_balance_scale = 1.25
        env.heading_kp = 3.
        env.target_yaw[:] = .2
        self.actions[:, 4:] = torch.tensor([-.2, .2])
        _, _, _, _, info = env.step(self.actions)
        torque = info["pre_reset_applied_torques"][:, 4:]
        torch.testing.assert_close(.5*(torque[:, 1]-torque[:, 0]), torch.full((4,), 2.5))
        torch.testing.assert_close(.5*(torque[:, 1]+torque[:, 0]), torch.full((4,), .6))

    def test_real_ground_traction_turns_in_both_heading_directions(self):
        env = self.env
        env.heading_kp = .5
        env.target_yaw[:] = torch.tensor([.2, -.2, 0., 0.])
        # Large policy steering in conflicting directions must not determine
        # the turn. Only explicit steering should reach the physical wheels.
        self.actions[:, 4:] = torch.tensor([[-6., -6.], [6., 6.],
                                            [6., 6.], [-6., -6.]])/10.
        for _ in range(15):  # 0.3 s, before a learned pitch policy is needed.
            _, _, _, dones, _ = env.step(self.actions)
            self.assertFalse(bool(dones.any()))
        env.gym.refresh_net_contact_force_tensor(env.sim)
        wheel_load = env.contact_forces[:, env.wheel_indices, 2].sum(-1)
        self.assertTrue(bool((wheel_load[:2] > 5.).all()), wheel_load.tolist())
        self.assertGreater(float(env.total_yaw[0]), .003)
        self.assertLess(float(env.total_yaw[1]), -.003)
        self.assertLess(float(env.total_yaw[2:].abs().max()), .003)
        self.assertLess(abs(float(env.total_yaw[0])-.2), .2)
        self.assertLess(abs(float(env.total_yaw[1])+.2), .2)

    def test_heading_target_does_not_store_a_long_blocked_turn(self):
        measured = torch.zeros(2)
        target = measured.clone()
        command = torch.tensor([2., -2.])
        for _ in range(500):  # Ten seconds requested while physically blocked.
            target = bounded_heading_target(measured, target, command, .02)
        torch.testing.assert_close(target, torch.tensor([.35, -.35]))
        # Releasing the key leaves at most a small correction, not 20 radians
        # of accumulated turning. Once that correction is reached it is done.
        target = bounded_heading_target(measured, target, torch.zeros(2), .02)
        reached = target.clone()
        target = bounded_heading_target(reached, target, torch.zeros(2), .02)
        torch.testing.assert_close(target, reached)
        torch.testing.assert_close(
            bounded_heading_target(torch.zeros(2), torch.zeros(2), command, .02),
            torch.tensor([.04, -.04]))

    def test_legacy_default_keeps_original_independent_wheel_actions(self):
        self.assertEqual(self.default_mode, "policy")
        self.assertEqual(self.default_pitch_damping, 0.)
        self.env.wheel_control = "policy"
        self.actions[:, 4:] = torch.tensor([[2., 3.], [-2., -3.],
                                            [20., 21.], [-20., -21.]])/10.
        _, _, _, _, info = self.env.step(self.actions)
        torch.testing.assert_close(
            info["pre_reset_applied_torques"][:, 4:],
            torch.tensor([[2., 3.], [-2., -3.], [9., 9.], [-9., -9.]]))

    def test_checkpoint_restores_new_options_and_explicit_legacy_mode_wins(self):
        parser = common_parser("test", 4)
        saved = {"model_state_dict": {"actor.0.weight": torch.empty(1, 186)},
                 "environment_config": {"wheel_control": "heading", "wheel_pitch_damping": .4,
                                        "wheel_balance_scale": 1.25}}
        for mode in ("train", "play", "evaluate"):
            with self.subTest(mode=mode):
                argv = []
                actual = resolve_runtime_args(parser.parse_args(argv),
                    explicit_argument_destinations(parser, argv), saved, mode)
                kwargs = environment_kwargs(actual)
                self.assertEqual((kwargs["wheel_control"], kwargs["wheel_pitch_damping"]),
                                 ("heading", .4))
                self.assertEqual(kwargs["wheel_balance_scale"], 1.25)
                argv = ["--wheel-control", "policy", "--wheel-pitch-damping", "0",
                        "--wheel-balance-scale", "1"]
                actual = resolve_runtime_args(parser.parse_args(argv),
                    explicit_argument_destinations(parser, argv), saved, mode)
                self.assertEqual((actual.wheel_control, actual.wheel_pitch_damping), ("policy", 0.))
        legacy = dict(saved, environment_config={})
        actual = resolve_runtime_args(parser.parse_args([]), set(), legacy, "play")
        self.assertEqual((actual.wheel_control, actual.wheel_pitch_damping), ("policy", 0.))


if __name__ == "__main__":
    unittest.main()
