"""Truth-based motion evaluation, warm start isolation and curriculum gates."""

import math
from pathlib import Path
import tempfile
import unittest

import isaacgym
import torch

from legged_gym.envs import register_tasks
from legged_gym.envs.wheelleg.diagnostics.motion_evaluation import MotionEvaluation
from legged_gym.envs.wheelleg.sensor_config import WheelLeggedTrackingCfg
from legged_gym.envs.wheelleg.tracking_task import TrackingWheelLegged
from legged_gym.utils import get_args, task_registry


class MotionEvaluationTests(unittest.TestCase):
    def test_body_velocity_cross_track_and_reset_teleport(self):
        roots = torch.zeros(1, 13)
        roots[:, 5:7] = math.sqrt(0.5)  # yaw90: body forward = world Y
        metric = MotionEvaluation(roots, 0.02, settle_seconds=0)
        terminal = roots.clone()
        terminal[:, :2] = torch.tensor([-0.2, 1.0])
        terminal[:, 8] = 1.0
        commands = torch.tensor([[1.0, 0.0, 0.0, 0.08]])
        reset = roots.clone()
        reset[:, :2] = 100.0
        metric.update(terminal, commands, torch.tensor([True]), reset)
        metric.update(reset, commands * 0, torch.tensor([False]), reset)
        report = metric.report()
        self.assertAlmostEqual(report["segment_peak_cross_track_m"], 0.2, places=5)
        self.assertAlmostEqual(
            report["segment_peak_forward_displacement_m"], 1.0, places=5
        )
        self.assertLess(report["rms_forward_speed_error_m_s"], 1e-5)
        self.assertLess(report["rms_lateral_speed_m_s"], 1e-5)

    def test_warm_start_rejects_resume_and_play(self):
        for mode, extra in [("train", ["--resume"]), ("play", [])]:
            with self.assertRaises(SystemExit):
                get_args(mode, ["--warm-start", "/tmp/not-a-checkpoint"] + extra)


class TrackingTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        register_tasks()
        cfg = WheelLeggedTrackingCfg()
        cfg.env.num_envs = 2
        cfg.env.fix_base = True
        cfg.sim.physics_threads = 0
        cls.env = TrackingWheelLegged(cfg, sim_device="cpu", headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.assessment_running = False
        self.env.assessment_active[:] = False
        self.env.speed_levels[:] = 0
        self.env.tracking_failed[:] = False
        self.env.reset()

    def test_speed_stopping_and_yaw_costs_have_correct_targets(self):
        e = self.env
        e.commands[:] = torch.tensor([1.0, 0.0, 0.0, 0.08])
        e.lin_vel[:] = torch.tensor([1.0, 0.0, 0.0])
        e.ang_vel[:] = 0
        t = e._compute_reward_terms(torch.zeros(2, dtype=torch.bool))
        for k in (
            "speed_error",
            "lateral_speed",
            "yaw_rate_error",
            "stopped_wheel_speed",
        ):
            torch.testing.assert_close(t[k], torch.zeros(2))
        e.lin_vel[:, 0] = 0.5
        e.lin_vel[:, 1] = 0.05
        e.ang_vel[:, 2] = 0.1
        t = e._compute_reward_terms(torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(t["speed_error"], torch.full((2,), 6.25))
        torch.testing.assert_close(t["lateral_speed"], torch.ones(2))
        torch.testing.assert_close(t["yaw_rate_error"], torch.ones(2))
        # Stopping is a soft wheel-speed cost and never changes the actor's input.
        e.commands[:, :3] = 0
        e.dof_state[:, e.action_indices[4:], 1] = 2
        t = e._compute_reward_terms(torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(t["stopped_wheel_speed"], torch.ones(2))

    def test_completed_episode_needs_tracking_quality_before_promotion(self):
        e = self.env
        e.assessment_running = True
        e.assessment_active[:] = True
        e.episode_length_buf[:] = e.max_episode_length
        e.tracking_counts[:] = 400
        e.tracking_sums[:] = 0
        e.tracking_sums[1, 2] = 400 * 0.1**2  # Excess yaw blocks promotion.
        e.pitch[:] = e.roll[:] = 0
        e.geometry_valid[:] = True
        e._reset_idx(torch.arange(2))
        self.assertEqual(e.speed_levels.tolist(), [1, 0])
        self.assertEqual(
            float(e.commands[:, 0].abs().max()), 0.0
        )  # Ramp starts from rest.
        self.assertLessEqual(float(e.command_targets[:, 0].max()), 0.2 + 1e-6)

    def test_stationary_drift_and_heading_block_promotion(self):
        e = self.env
        e.assessment_running = True
        e.assessment_active[:] = True
        e.episode_length_buf[:] = e.max_episode_length
        e.tracking_counts[:] = 400
        e.tracking_sums[:] = 0
        e.tracking_sums[0, 0] = 400 * 0.03**2
        e.total_yaw[1] = 0.2
        e.pitch[:] = e.roll[:] = 0
        e.geometry_valid[:] = True
        e._reset_idx(torch.arange(2))
        self.assertEqual(e.speed_levels.tolist(), [0, 0])

    def test_stationary_trial_cannot_skip_motion_stage_and_failure_demotes(self):
        e = self.env
        e.assessment_running = True
        e.assessment_active[:] = True
        e.speed_levels[:] = 3
        e.command_targets[:, 0] = 0
        e.episode_length_buf[:] = e.max_episode_length
        e.tracking_counts[:] = 400
        e.tracking_sums[:] = 0
        e.pitch[:] = e.roll[:] = 0
        e.geometry_valid[:] = True
        e.tracking_failed[1] = True
        e._reset_idx(torch.arange(2))
        self.assertEqual(e.speed_levels.tolist(), [3, 2])

    def test_curriculum_restores_and_play_disables_advancement(self):
        e = self.env
        metrics = e.create_training_metrics()
        metrics.restore_checkpoint({"speed_curriculum_levels": [2, 5]})
        self.assertEqual(e.speed_levels.tolist(), [2, 5])
        self.assertEqual(
            metrics.checkpoint_metadata()["speed_curriculum_levels"], [2, 5]
        )
        cfg, _ = task_registry.get_cfgs("wheelleg_sensor_tracking")
        TrackingWheelLegged.configure_execution(
            cfg, get_args("play", ["--headless"]), None
        )
        self.assertFalse(cfg.curriculum.enabled)
        self.assertFalse(cfg.commands.random_commands)

    def test_frozen_playback_uses_same_acceleration_ramp(self):
        e = self.env
        enabled, random = e.cfg.curriculum.enabled, e.random_commands
        command = e.default_command.clone()
        try:
            e.cfg.curriculum.enabled = False
            e.random_commands = False
            e.default_command[0] = 1.0
            e.reset()
            self.assertEqual(float(e.commands[:, 0].max()), 0.0)
            e._advance_random_commands(torch.zeros(2, dtype=torch.bool))
            torch.testing.assert_close(e.commands[:, 0], torch.full((2,), 0.4 * e.dt))
            torch.testing.assert_close(e.command_targets[:, 0], torch.ones(2))
        finally:
            e.cfg.curriculum.enabled, e.random_commands = enabled, random
            e.default_command.copy_(command)
            e.reset()

    def test_cross_task_resume_cannot_restore_stationary_configuration(self):
        cfg, _ = task_registry.get_cfgs("wheelleg_sensor_tracking")
        args = get_args("train", ["--task", "wheelleg_sensor_tracking", "--headless"])
        with self.assertRaisesRegex(ValueError, "warm-start"):
            TrackingWheelLegged.configure_execution(
                cfg, args, {"legged_gym_env_config": {}}
            )

    def test_noisy_training_episode_does_not_grade_curriculum(self):
        e = self.env
        e.speed_levels[:] = 2
        e.episode_length_buf[:] = e.max_episode_length
        e.tracking_counts[:] = 400
        e.tracking_sums[:] = 0
        e.tracking_failed[0] = True
        e._reset_idx(torch.arange(2))
        self.assertEqual(e.speed_levels.tolist(), [2, 2])

    def test_deterministic_assessment_promotes_and_restores_training_state(self):
        class ZeroPolicy(torch.nn.Module):
            def act_inference(self, obs):
                return torch.zeros(len(obs), 6)

        e = self.env
        policy = ZeroPolicy()
        old_length = e.max_episode_length
        old_min, old_settle = (
            e.cfg.curriculum.minimum_samples,
            e.cfg.curriculum.settle_seconds,
        )
        try:
            e.max_episode_length = 10
            e.cfg.curriculum.minimum_samples = 1
            e.cfg.curriculum.settle_seconds = 0
            e.last_assessment_iteration = 0
            result = e.assess_curriculum(policy, 1)
            self.assertEqual(result["Assessment/promotions"], 2)
            self.assertEqual(result["Assessment/steps"], 10)
            self.assertTrue(policy.training)
            self.assertFalse(e.assessment_running)
            self.assertFalse(bool(e.assessment_active.any()))
            self.assertEqual(e.episode_length_buf.tolist(), [0, 0])
            self.assertIsNone(e.assess_curriculum(policy, 1))
            self.assertIsNone(e.assess_curriculum(policy, 2))
        finally:
            e.max_episode_length = old_length
            e.cfg.curriculum.minimum_samples, e.cfg.curriculum.settle_seconds = (
                old_min,
                old_settle,
            )

    def test_runner_assessment_is_outside_ppo_and_next_rollout_uses_reset_observation(
        self,
    ):
        from legged_gym.learning.on_policy_runner import OnPolicyRunner
        from legged_gym.utils import class_to_dict

        e = self.env
        old_length = e.max_episode_length
        cc = e.cfg.curriculum
        old_min, old_settle = cc.minimum_samples, cc.settle_seconds
        try:
            e.max_episode_length = 10
            cc.minimum_samples, cc.settle_seconds = 1, 0
            e.last_assessment_iteration = 0
            _, cfg = task_registry.get_cfgs("wheelleg_sensor_tracking")
            cfg.policy.actor_hidden_dims = cfg.policy.critic_hidden_dims = [16, 8]
            cfg.runner.num_steps_per_env = 2
            cfg.algorithm.num_learning_epochs = cfg.algorithm.num_mini_batches = 1
            with tempfile.TemporaryDirectory() as folder:
                runner = OnPolicyRunner(e, class_to_dict(cfg), folder, device="cpu")
                runner.learn(1)
                self.assertEqual(runner.tot_timesteps, 4)
                self.assertEqual(e.episode_length_buf.tolist(), [0, 0])
                lines = (
                    (Path(folder) / "curriculum_evaluations.jsonl")
                    .read_text()
                    .splitlines()
                )
                self.assertEqual(len(lines), 1)
                after_assessment = e.get_observations()[0].clone()
                runner.learn(1)
                torch.testing.assert_close(
                    runner.alg.storage.observations[0], after_assessment
                )
                self.assertEqual(runner.tot_timesteps, 8)
                self.assertEqual(
                    len(
                        (Path(folder) / "curriculum_evaluations.jsonl")
                        .read_text()
                        .splitlines()
                    ),
                    1,
                )
        finally:
            e.max_episode_length = old_length
            cc.minimum_samples, cc.settle_seconds = old_min, old_settle

    def test_completed_update_is_saved_before_interrupted_assessment(self):
        from unittest.mock import patch
        from legged_gym.learning.on_policy_runner import OnPolicyRunner
        from legged_gym.utils import class_to_dict

        _, cfg = task_registry.get_cfgs("wheelleg_sensor_tracking")
        cfg.policy.actor_hidden_dims = cfg.policy.critic_hidden_dims = [16, 8]
        cfg.runner.num_steps_per_env = 2
        cfg.runner.save_interval = 1
        cfg.algorithm.num_learning_epochs = cfg.algorithm.num_mini_batches = 1
        with tempfile.TemporaryDirectory() as folder:
            runner = OnPolicyRunner(self.env, class_to_dict(cfg), folder, device="cpu")
            with patch.object(
                self.env,
                "assess_curriculum",
                side_effect=RuntimeError("interrupted assessment"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted assessment"):
                    runner.learn(1)
            saved = torch.load(Path(folder) / "model_1.pt", map_location="cpu")
            self.assertEqual(saved["iter"], 1)
            torch.testing.assert_close(
                saved["model_state_dict"]["actor.0.weight"],
                runner.alg.actor_critic.actor[0].weight,
            )

    def test_weights_only_keeps_new_config_and_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "source.pt"
            cfg, train = task_registry.get_cfgs("wheelleg_sensor_tracking")
            # Real source policy and its deliberately incompatible saved commands.
            from legged_gym.learning.policy import make_policy
            from legged_gym.utils import class_to_dict

            policy = make_policy(132, 6, "cpu", class_to_dict(train.policy))
            with torch.no_grad():
                policy.critic[0].weight.fill_(123.0)
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "iter": 600,
                    "legged_gym_env_config": {"commands": {"random_commands": False}},
                },
                p,
            )
            args = get_args(
                "train",
                [
                    "--task",
                    "wheelleg_sensor_tracking",
                    "--warm-start",
                    str(p),
                    "--sim_device",
                    "cpu",
                    "--headless",
                ],
            )
            cfg, train = task_registry.prepare_cfgs(args.task, args)
            self.assertTrue(cfg.commands.random_commands)
            runner, _ = task_registry.make_alg_runner(
                self.env, name=args.task, args=args, train_cfg=train, log_root=None
            )
            self.assertEqual(runner.current_learning_iteration, 0)
            self.assertFalse(runner.alg.optimizer.state)
            torch.testing.assert_close(
                runner.alg.actor_critic.actor[-1].weight, policy.actor[-1].weight
            )
            self.assertEqual(train.initialization_provenance["mode"], "actor_only")
            self.assertFalse(
                bool((runner.alg.actor_critic.critic[0].weight == 123.0).all())
            )
