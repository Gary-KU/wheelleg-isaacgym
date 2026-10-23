"""Profiles and checkpoint restores must preserve explicit choices and model shape."""

import argparse
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.cli import common_parser, environment_kwargs, validate_common
from wheelleg_gym.runtime_config import explicit_argument_destinations, resolve_runtime_args


class Weight:
    def __init__(self, width):
        self.shape = (16, width)


def checkpoint(width=198):
    return {"model_state_dict": {"actor.0.weight": Weight(width)},
            "environment_config": {
                "num_envs": 256, "sim_device": "cuda:7", "headless": True,
                "random_commands": True, "command": [0.2, 0.0, -0.4, 0.12],
                "physics_dt": 0.00025, "closure_stiffness": 75000.0,
                "closure_damping": 40.0, "seed": 9,
                "observe_leg_lengths": width == 198, "linear_acceleration": 0.6,
                "yaw_acceleration": 1.3, "stability_rewards": True,
                "tracking_penalties": True, "command_resample_seconds": 3.0}}


def arguments(argv=(), mode="train"):
    parser = common_parser("test", 64 if mode == "train" else 1)
    if mode == "train":
        parser.add_argument("--init-noise-std", nargs="+", type=float)
        parser.add_argument("--learning-rate", type=float)
        parser.add_argument("--max-learning-rate", type=float)
        parser.add_argument("--save-interval", type=int)
        parser.add_argument("--max-iterations", "--max_iterations", type=int, default=20000)
    args = parser.parse_args(argv)
    return parser, args, explicit_argument_destinations(parser, argv)


class RuntimeConfigurationTests(unittest.TestCase):
    def test_no_profile_or_checkpoint_retains_legacy_defaults(self):
        _, args, explicit = arguments()
        actual = resolve_runtime_args(args, explicit)
        self.assertEqual(vars(actual), vars(args))
        self.assertFalse(actual.random_commands)
        self.assertFalse(actual.observe_leg_lengths)
        self.assertFalse(actual.stability_rewards)

    def test_stability_profile_covers_motion_and_appropriate_noise(self):
        _, args, explicit = arguments(["--profile", "stability"])
        actual = resolve_runtime_args(args, explicit)
        self.assertEqual(actual.num_envs, 1024)
        self.assertEqual(actual.yaw_speed_range, [-0.8, 0.8])
        self.assertTrue(actual.observe_leg_lengths and actual.stability_rewards)
        self.assertTrue(actual.random_commands)
        self.assertEqual(actual.command_resample_seconds, 2)
        self.assertEqual(actual.init_noise_std, [1, 1, 1, 1, 0.1, 0.1])
        self.assertEqual(actual.learning_rate, 0.0003)
        self.assertEqual(actual.max_learning_rate, 0.0003)

    def test_support_profile_starts_with_stationary_length_task_and_separate_exploration(self):
        _, args, explicit = arguments(["--profile", "support"])
        actual = resolve_runtime_args(args, explicit)
        self.assertTrue(actual.support_rewards and actual.observe_leg_lengths)
        self.assertFalse(actual.random_commands)
        self.assertEqual(actual.command, [0, 0, 0, .08])
        self.assertEqual(actual.forward_speed_range, [0, 0])
        self.assertEqual(actual.yaw_speed_range, [0, 0])
        self.assertEqual(actual.init_noise_std, [3, 3, 3, 3, .1, .1])
        self.assertEqual(actual.leg_control, "torque")

    def test_hybrid_profile_has_stationary_pd_stance_and_residual_noise(self):
        _, args, explicit = arguments(["--profile", "hybrid"])
        actual = resolve_runtime_args(args, explicit)
        self.assertEqual(actual.leg_control, "hip_pd")
        self.assertEqual((actual.leg_kp, actual.leg_kd), (80, 4))
        self.assertEqual(actual.initial_leg_length, .08)
        self.assertFalse(actual.random_commands)
        self.assertEqual(actual.command, [0, 0, 0, .08])
        self.assertEqual(actual.init_noise_std, [1, 1, 1, 1, .1, .1])
        self.assertEqual(environment_kwargs(actual)["leg_control"], "hip_pd")

    def test_hybrid_controller_restores_in_all_modes_and_explicit_flags_win(self):
        saved = checkpoint()
        saved["environment_config"].update(leg_control="hip_pd", leg_kp=35, leg_kd=1.5)
        for mode in ("train", "play", "evaluate"):
            _, args, explicit = arguments([], mode)
            actual = resolve_runtime_args(args, explicit, saved, mode)
            self.assertEqual((actual.leg_control, actual.leg_kp, actual.leg_kd), ("hip_pd", 35, 1.5))
            _, args, explicit = arguments(["--leg-control", "torque", "--leg-kp=20", "--leg-kd", "0"], mode)
            actual = resolve_runtime_args(args, explicit, saved, mode)
            self.assertEqual((actual.leg_control, actual.leg_kp, actual.leg_kd), ("torque", 20, 0))

    def test_inference_hybrid_profile_never_reinterprets_legacy_torque_actions(self):
        for mode in ("play", "evaluate"):
            _, args, explicit = arguments(["--profile", "hybrid"], mode)
            actual = resolve_runtime_args(args, explicit, checkpoint(), mode)
            self.assertEqual(actual.leg_control, "torque")
            _, args, explicit = arguments(["--profile", "hybrid", "--leg-control", "hip_pd"], mode)
            self.assertEqual(resolve_runtime_args(args, explicit, checkpoint(), mode).leg_control, "hip_pd")

    def test_support_profile_explicitly_selects_pure_torque_when_switching_from_hybrid(self):
        saved = checkpoint()
        saved["environment_config"]["leg_control"] = "hip_pd"
        _, args, explicit = arguments(["--profile", "support"])
        self.assertEqual(resolve_runtime_args(args, explicit, saved).leg_control, "torque")

    def test_leg_gains_validate_finite_positive_kp_and_nonnegative_kd(self):
        for name, value in (("--leg-kp", "0"), ("--leg-kp", "-1"), ("--leg-kp", "nan"),
                            ("--leg-kd", "-1"), ("--leg-kd", "inf")):
            parser, args, _ = arguments(["--allow-approximate-loop", name, value])
            with self.subTest(name=name, value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_common(parser, args)
        parser, args, _ = arguments(["--allow-approximate-loop", "--leg-kd", "0"])
        validate_common(parser, args)

    def test_support_feedforward_restores_and_rejects_silently_unused_force(self):
        saved = checkpoint()
        saved["environment_config"].update(leg_control="hip_pd", leg_feedforward_force=22.465)
        for mode in ("train", "play", "evaluate"):
            parser, args, explicit = arguments([], mode)
            actual = resolve_runtime_args(args, explicit, saved, mode)
            self.assertEqual(environment_kwargs(actual)["leg_feedforward_force"], 22.465)
        parser, args, explicit = arguments(["--profile", "support"])
        actual = resolve_runtime_args(args, explicit, saved)
        self.assertEqual(actual.leg_feedforward_force, 0)
        parser, args, explicit = arguments(["--profile", "hybrid"])
        self.assertEqual(resolve_runtime_args(args, explicit, saved).leg_feedforward_force, 22.465)
        for options in (["--leg-feedforward-force", "1"],
                        ["--leg-control", "hip_pd", "--leg-feedforward-force", "nan"],
                        ["--leg-control", "hip_pd", "--leg-feedforward-force", "-1"]):
            parser, args, _ = arguments(["--allow-approximate-loop"]+options)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_common(parser, args)

    def test_heading_feedback_restores_and_explicit_zero_disables_it(self):
        saved = checkpoint()
        saved["environment_config"].update(heading_kp=.5, heading_kd=.05)
        parser, args, explicit = arguments(["--profile", "support"])
        actual = resolve_runtime_args(args, explicit, saved)
        self.assertEqual((actual.heading_kp, actual.heading_kd), (0, 0))
        parser, args, explicit = arguments(["--profile", "hybrid"])
        actual = resolve_runtime_args(args, explicit, saved)
        self.assertEqual((actual.heading_kp, actual.heading_kd), (.5, .05))
        for mode in ("train", "play", "evaluate"):
            parser, args, explicit = arguments([], mode)
            actual = resolve_runtime_args(args, explicit, saved, mode)
            self.assertEqual((actual.heading_kp, actual.heading_kd), (.5, .05))
            parser, args, explicit = arguments(["--heading-kp", "0", "--heading-kd", "0"], mode)
            actual = resolve_runtime_args(args, explicit, saved, mode)
            self.assertEqual((actual.heading_kp, actual.heading_kd), (0, 0))
        for name, value in (("--heading-kp", "-1"), ("--heading-kd", "nan")):
            parser, args, _ = arguments(["--allow-approximate-loop", name, value])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_common(parser, args)

    def test_support_reward_flag_is_restored_for_training_and_can_be_overridden(self):
        saved = checkpoint()
        saved["environment_config"]["support_rewards"] = True
        _, args, explicit = arguments()
        self.assertTrue(resolve_runtime_args(args, explicit, saved).support_rewards)
        _, args, explicit = arguments(["--no-support-rewards"])
        self.assertFalse(resolve_runtime_args(args, explicit, saved).support_rewards)

    def test_initial_support_pose_can_be_restored_and_original_pose_explicitly_selected(self):
        saved = checkpoint()
        saved["environment_config"]["initial_leg_length"] = .08
        for mode in ("train", "play", "evaluate"):
            _, args, explicit = arguments([], mode)
            self.assertEqual(resolve_runtime_args(args, explicit, saved, mode).initial_leg_length, .08)
            _, args, explicit = arguments(["--default-leg-pose"], mode)
            self.assertIsNone(resolve_runtime_args(args, explicit, saved, mode).initial_leg_length)

    def test_explicit_alias_equal_zero_and_negative_flags_override_profile(self):
        _, args, explicit = arguments([
            "--profile=stability", "--num_envs=8", "--sim_device=cpu",
            "--linear-acceleration=0", "--yaw-speed-range", "-0.2", "0.2",
            "--no-random-commands", "--no-stability-rewards",
            "--max_iterations=2", "--save-interval", "5"])
        actual = resolve_runtime_args(args, explicit)
        self.assertEqual(actual.num_envs, 8)
        self.assertEqual(actual.sim_device, "cpu")
        self.assertEqual(actual.linear_acceleration, 0)
        self.assertEqual(actual.yaw_speed_range, [-0.2, 0.2])
        self.assertFalse(actual.random_commands)
        self.assertFalse(actual.stability_rewards)
        self.assertEqual(actual.max_iterations, 2)
        self.assertEqual(actual.save_interval, 5)

    def test_training_resumes_environment_without_changing_execution_choices(self):
        saved = checkpoint()
        old_command = deepcopy(saved["environment_config"]["command"])
        _, args, explicit = arguments(["--num-envs", "32", "--sim-device", "cpu"])
        actual = resolve_runtime_args(args, explicit, saved)
        self.assertEqual(actual.num_envs, 32)
        self.assertEqual(actual.sim_device, "cpu")
        self.assertFalse(actual.headless)
        self.assertTrue(actual.observe_leg_lengths)
        self.assertEqual(actual.command, old_command)
        self.assertEqual(actual.physics_dt, 0.00025)
        actual.command[3] = 0.09
        self.assertEqual(saved["environment_config"]["command"], old_command)

    def test_resume_profile_preserves_optimizer_noise_initialization_contract(self):
        _, args, explicit = arguments(["--profile", "stability"])
        actual = resolve_runtime_args(args, explicit, checkpoint())
        self.assertIsNone(actual.init_noise_std)
        self.assertIsNone(actual.learning_rate)
        self.assertEqual(actual.max_learning_rate, 0.0003)
        self.assertEqual(actual.command, [0, 0, 0, 0.08])

    def test_inference_restores_physics_and_limiter_but_not_training_motion(self):
        for mode in ("play", "evaluate"):
            _, args, explicit = arguments(["--sim-device", "cpu"], mode)
            actual = resolve_runtime_args(args, explicit, checkpoint(), mode)
            self.assertEqual(actual.physics_dt, 0.00025)
            self.assertEqual(actual.closure_stiffness, 75000)
            self.assertEqual(actual.linear_acceleration, 0.6)
            self.assertEqual(actual.yaw_acceleration, 1.3)
            self.assertTrue(actual.observe_leg_lengths)
            self.assertFalse(actual.random_commands)
            self.assertEqual(actual.command, [0, 0, 0, 0.12])
            self.assertEqual(actual.num_envs, 1)
            self.assertEqual(actual.sim_device, "cpu")

    def test_inference_explicit_overrides_including_limiter_disabled(self):
        _, args, explicit = arguments([
            "--linear-acceleration=0", "--physics-dt", "0.0005",
            "--command", "-0.3", "0", "0.8", "0.08"], "play")
        actual = resolve_runtime_args(args, explicit, checkpoint(), "play")
        self.assertEqual(actual.linear_acceleration, 0)
        self.assertEqual(actual.physics_dt, 0.0005)
        self.assertEqual(actual.command, [-0.3, 0, 0.8, 0.08])

    def test_legacy_checkpoint_requires_explicit_conversion_before_198_profile(self):
        _, args, explicit = arguments(["--profile", "stability"])
        with self.assertRaisesRegex(ValueError, "upgrade_observations.py"):
            resolve_runtime_args(args, explicit, checkpoint(186))

    def test_inference_profile_does_not_force_new_input_shape_into_old_policy(self):
        _, args, explicit = arguments(["--profile", "stability"], "play")
        actual = resolve_runtime_args(args, explicit, checkpoint(186), "play")
        self.assertFalse(actual.observe_leg_lengths)
        self.assertEqual(actual.linear_acceleration, 0.8)
        self.assertEqual(actual.yaw_acceleration, 2)

    def test_stair_curriculum_lower_bound_restores_validates_and_explicit_override_wins(self):
        saved = checkpoint(252)
        saved["environment_config"].update(terrain="stairs", observe_terrain=True,
            observe_leg_lengths=True, stair_min_level=1, stair_level=1, stair_max_level=4,
            stair_curriculum=True)
        for mode in ("train", "play", "evaluate"):
            parser, args, explicit = arguments(["--allow-approximate-loop"], mode)
            actual = resolve_runtime_args(args, explicit, saved, mode)
            validate_common(parser, actual)
            self.assertEqual(environment_kwargs(actual)["stair_min_level"], 1)
            parser, args, explicit = arguments(["--allow-approximate-loop", "--stair-min-level", "0", "--stair-level", "0"], mode)
            actual = resolve_runtime_args(args, explicit, saved, mode)
            validate_common(parser, actual)
            self.assertEqual((actual.stair_min_level, actual.stair_level), (0, 0))
        for options in (["--stair-min-level", "-1"], ["--stair-min-level", "1"],
                        ["--stair-min-level", "2", "--stair-level", "1"],
                        ["--stair-level", "4", "--stair-max-level", "3"]):
            parser, args, _ = arguments(["--allow-approximate-loop"]+options)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_common(parser, args)

    def test_flat_profiles_switch_saved_stair_terrain_without_breaking_252_inputs(self):
        saved = checkpoint(252)
        saved["environment_config"].update(terrain="stairs", observe_terrain=True,
            observe_leg_lengths=True, stair_min_level=1, stair_level=1, stair_max_level=4,
            stair_curriculum=True)
        for mode in ("train", "play", "evaluate"):
            for profile in ("stability", "support", "hybrid"):
                parser, args, explicit = arguments(["--profile", profile], mode)
                actual = resolve_runtime_args(args, explicit, saved, mode)
                self.assertEqual(actual.terrain, "plane")
                self.assertFalse(actual.stair_curriculum)
                self.assertTrue(actual.observe_terrain)
                self.assertTrue(actual.observe_leg_lengths)

    def test_stair_profile_has_new_heading_gains_and_default_zero_lower_bound(self):
        _, args, explicit = arguments(["--profile", "stairs"])
        actual = resolve_runtime_args(args, explicit)
        self.assertEqual((actual.heading_kp, actual.heading_kd), (6., .3))
        self.assertEqual((actual.stair_min_level, actual.stair_level, actual.stair_max_level), (0, 0, 4))
        self.assertEqual(actual.terrain, "stairs")
        self.assertTrue(actual.stair_curriculum)


if __name__ == "__main__":
    unittest.main()
