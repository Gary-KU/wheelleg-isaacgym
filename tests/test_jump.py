"""Jump triggering, contact evidence, landing recovery and input isolation."""

from types import SimpleNamespace
import unittest

import isaacgym  # must precede torch when combined with physics tests
import torch

from legged_gym.envs.wheelleg.control.jump import JumpController, JumpKeyboard
from legged_gym.envs.wheelleg.jump_config import WheelLeggedJumpCfg
from legged_gym.envs.wheelleg.keyboard import KeyboardDrive


class FakeEnv:
    dt, num_envs, device = 0.02, 1, "cpu"
    leg_feedforward_force = 22.4649

    def __init__(self):
        self.commands = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
        self.root_states = torch.zeros(1, 13)
        self.root_states[:, 2] = 0.162
        self.pitch, self.roll, self.yaw = [torch.zeros(1) for _ in range(3)]
        self.ang_vel, self.lin_vel = [torch.zeros(1, 3) for _ in range(2)]
        self.length_l = self.length_r = torch.tensor([0.1])
        self.dof_state = torch.zeros(1, 10, 2)
        self.action_indices = torch.tensor([0, 3, 6, 9, 2, 7])
        self.normal = torch.full((1, 2), 28.0)

    def lift_measurements(self):
        return dict(
            wheel_normal_force_n=self.normal,
            nonwheel_contact=torch.tensor([False]),
        )

    def set_commands(self, value):
        self.commands.copy_(value)
        return torch.zeros(1, 198)

    def info(self, clearance=(0.0, 0.0), contact=(28.0, 28.0)):
        return dict(
            pre_reset_root_states=self.root_states.clone(),
            pre_reset_pitch=self.pitch.clone(),
            pre_reset_roll=self.roll.clone(),
            lift_clearance_m=torch.tensor([clearance]),
            lift_wheel_contact_norm_n=torch.tensor([contact]),
            lift_nonwheel_contact=torch.tensor([False]),
        )


class JumpTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.controller = JumpController(self.env, WheelLeggedJumpCfg())

    def tick(self, info=None, done=False):
        self.controller.before_step()
        self.controller.after_step(info or self.env.info(), torch.tensor([done]))

    def ready(self):
        for _ in range(65):
            self.tick()

    def test_space_is_a_press_edge_and_does_not_change_existing_keyboard(self):
        keyboard = JumpKeyboard(leg_length=0.1)
        event = lambda value: SimpleNamespace(action="jump", value=value)
        keyboard.process_events([event(1)])
        self.assertTrue(keyboard.consume_jump())
        keyboard.process_events([event(1), event(2)])
        self.assertFalse(keyboard.consume_jump())
        keyboard.process_events([event(0), event(1)])
        self.assertTrue(keyboard.consume_jump())
        self.assertEqual(JumpKeyboard.KEYS["drive_stop"], "KEY_X")
        self.assertEqual(KeyboardDrive.KEYS["drive_stop"], "KEY_SPACE")

    def test_manual_jump_does_not_wait_for_startup_or_acceleration(self):
        self.assertEqual(self.controller.states[0]["stable"], 0.0)
        self.controller.before_step([0.6, 0, 0, 0.1])
        self.assertTrue(self.controller.request_jump())
        self.assertFalse(self.controller.request_jump())
        self.assertEqual(self.controller.states[0]["mode"], "crouch")

    def test_accelerating_jump_holds_applied_velocity_instead_of_ramping_to_key_target(
        self,
    ):
        for _ in range(5):
            self.controller.before_step([0.6, 0, 0, 0.1])
        applied = float(self.env.commands[0, 0])
        self.assertLess(applied, 0.1)
        self.assertTrue(self.controller.request_jump())
        for _ in range(35):
            self.tick()
            self.assertAlmostEqual(float(self.env.commands[0, 0]), applied, places=6)
        self.controller.states[0]["mode"] = "land"
        self.controller.before_step()
        self.assertGreater(float(self.env.commands[0, 0]), applied)

    def test_steady_forward_and_backward_roll_can_jump_without_braking(self):
        for speed in (-0.6, 0.6):
            with self.subTest(speed=speed):
                self.setUp()
                self.env.lin_vel[0, 0] = speed
                self.env.root_states[0, 7] = speed
                self.env.dof_state[0, self.env.action_indices[4:], 1] = (
                    torch.tensor([-speed, speed]) / 0.0625
                )
                self.controller.before_step([speed, 0, 0, 0.1])
                for _ in range(110):
                    self.tick()
                self.assertTrue(self.controller.request_jump())
                for mode in ("crouch", "push", "flight", "land"):
                    self.controller.states[0]["mode"] = mode
                    self.controller.before_step()
                    self.assertAlmostEqual(
                        float(self.env.commands[0, 0]), speed, places=6
                    )
                self.env.root_states[0, 0] += speed * 2
                self.controller.after_step(self.env.info(), torch.tensor([False]))
                attempt = self.controller.states[0]["attempt"]
                self.assertGreater(attempt["peak_xy_displacement_m"], 1.0)
                self.assertAlmostEqual(attempt["peak_lateral_displacement_m"], 0.0)
                # Intended travel is not stationary drift; acceptance still
                # requires real body rise, flight and landing recovery.
                attempt.update(
                    peak_z=0.22, longest_airborne_s=0.16, max_upward_velocity=1.0
                )
                self.controller._finish(0)
                self.assertTrue(self.controller.attempts[0]["success"])

    def test_airborne_key_release_is_applied_at_touchdown_without_teleporting(self):
        speed = 0.6
        self.env.lin_vel[0, 0] = speed
        self.env.root_states[0, 7] = speed
        self.controller.before_step([speed, 0, 0, 0.1])
        for _ in range(110):
            self.tick()
        self.assertTrue(self.controller.request_jump())
        self.controller.states[0]["mode"] = "flight"
        root = self.env.root_states.clone()
        self.controller.before_step([0, 0, 0, 0.1])
        self.assertAlmostEqual(float(self.env.commands[0, 0]), speed, places=6)
        torch.testing.assert_close(self.env.root_states, root, rtol=0, atol=0)
        self.controller.states[0]["mode"] = "land"
        self.controller.before_step()
        self.assertLess(float(self.env.commands[0, 0]), speed)
        self.assertGreater(float(self.env.commands[0, 0]), 0.0)

    def test_turning_slipping_and_tilt_do_not_cancel_manual_launch(self):
        self.env.lin_vel[0] = torch.tensor([0.2, 0.1, 0.0])
        self.env.pitch[0] = 0.15  # Exceeds the former prelaunch abort threshold.
        self.env.roll[0] = 0.11
        self.controller.before_step([0.6, 0, 2.0, 0.1])
        self.assertTrue(self.controller.request_jump())
        for _ in range(45):
            self.tick()
        self.assertEqual(self.controller.states[0]["mode"], "flight")
        self.assertEqual(self.controller.attempts, [])

    def test_touchdown_allows_next_jump_before_recovery_without_erasing_evidence(self):
        self.assertTrue(self.controller.request_jump())
        self.controller.states[0]["mode"] = "flight"
        self.tick(self.env.info(clearance=(0.02, 0.02), contact=(0, 0)))
        self.assertFalse(self.controller.request_jump())
        self.tick(self.env.info())
        self.assertEqual(self.controller.states[0]["mode"], "land")
        self.assertTrue(self.controller.request_jump())
        self.assertEqual(self.controller.states[0]["mode"], "crouch")
        self.assertEqual(len(self.controller.attempts), 1)
        previous = self.controller.attempts[0]
        self.assertEqual(previous["failure"], "restarted_before_recovery")
        self.assertFalse(previous["recovered"])
        self.assertFalse(previous["success"])
        self.assertTrue(previous["airborne_seen"])

    def test_push_duration_then_flight_has_no_support_force(self):
        self.ready()
        self.controller.request_jump()
        forces = []
        for _ in range(50):
            self.tick()
            forces.append(float(self.env.lift_forces[0, 0]))
            if self.controller.states[0]["mode"] == "flight":
                break
        self.assertEqual(
            forces.count(self.controller.jump.push_force),
            round(self.controller.jump.push_seconds / self.env.dt),
        )
        self.assertEqual(forces[-1], 0.0)
        self.assertGreaterEqual(float(self.env.lift_lengths.min()), 0.06)
        self.assertLessEqual(float(self.env.lift_lengths.max()), 0.20)

    def test_one_wheel_or_contacting_wheels_do_not_count_as_airborne(self):
        self.ready()
        self.controller.request_jump()
        self.controller.states[0]["mode"] = "flight"
        for _ in range(3):
            self.tick(self.env.info(clearance=(0.02, 0), contact=(0, 0)))
        for _ in range(3):
            self.tick(self.env.info(clearance=(0.02, 0.02), contact=(2, 2)))
        attempt = self.controller.states[0]["attempt"]
        self.assertFalse(attempt["airborne_seen"])
        self.assertEqual(attempt["longest_airborne_s"], 0.0)

    def test_clearance_without_body_rise_cannot_pass_jump_acceptance(self):
        self.ready()
        self.controller.request_jump()
        self.controller.states[0]["mode"] = "flight"
        for _ in range(3):
            self.tick(self.env.info(clearance=(0.02, 0.02), contact=(0, 0)))
        self.tick(self.env.info())
        self.assertEqual(self.controller.states[0]["mode"], "land")
        for _ in range(70):
            self.tick()
        self.assertEqual(len(self.controller.attempts), 1)
        self.assertFalse(self.controller.attempts[0]["success"])
        self.assertTrue(self.controller.attempts[0]["recovered"])

    def test_landing_completes_promptly_without_falsely_certifying_stability(self):
        self.controller.request_jump()
        self.controller.states[0]["mode"] = "flight"
        for _ in range(3):
            self.tick(self.env.info(clearance=(0.03, 0.03), contact=(0, 0)))
        self.tick()
        touchdown = self.controller.time
        self.env.pitch[0] = 0.15  # Fails the slow recovery check.
        self.controller.before_step([0, 0, 0, 0.14])
        self.assertAlmostEqual(float(self.env.commands[0, 3]), 0.14, places=6)
        torch.testing.assert_close(self.env.lift_lengths, torch.full((1, 2), 0.14))
        for _ in range(6):
            self.tick()
        self.assertEqual(self.controller.states[0]["mode"], "idle")
        events = [e for e in self.controller.events if e["event"] == "jump_completed"]
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0]["time_s"] - touchdown, 0.12 + 1e-9)
        self.assertEqual(self.controller.attempts, [])
        self.assertIsNotNone(self.controller.states[0]["attempt"])
        self.assertTrue(self.controller.can_jump)

    def test_airborne_height_selection_is_applied_on_touchdown_not_late_recovery(self):
        self.controller.request_jump()
        self.controller.states[0]["mode"] = "flight"
        self.controller.before_step([0, 0, 0, 0.14])
        self.assertAlmostEqual(float(self.env.commands[0, 3]), 0.1, places=6)
        self.controller.after_step(
            self.env.info(clearance=(0.03, 0.03), contact=(0, 0)), torch.tensor([False])
        )
        self.controller.after_step(self.env.info(), torch.tensor([False]))
        self.controller.before_step()
        self.assertEqual(self.controller.states[0]["mode"], "land")
        self.assertAlmostEqual(float(self.env.commands[0, 3]), 0.14, places=6)
        self.assertGreater(float(self.env.lift_forces.min()), 0.0)

    def test_one_wheel_contact_does_not_complete_landing(self):
        self.controller.request_jump()
        self.controller.states[0]["mode"] = "flight"
        self.tick(self.env.info(clearance=(0.03, 0.03), contact=(0, 0)))
        self.tick()
        self.env.normal[0, 1] = 0.0
        for _ in range(10):
            self.tick()
        self.assertEqual(self.controller.states[0]["mode"], "land")
        self.assertFalse(
            any(e["event"] == "jump_completed" for e in self.controller.events)
        )
        self.env.normal[:] = 28.0
        for _ in range(6):
            self.tick()
        self.assertEqual(self.controller.states[0]["mode"], "idle")

    def test_selected_extended_height_does_not_shrink_when_no_jump_is_requested(self):
        keyboard = JumpKeyboard(leg_length=0.1, leg_range=(0.08, 0.14))
        keyboard.process_events([SimpleNamespace(action="leg_extend", value=1)])
        for _ in range(110):
            keyboard.advance_leg_length(self.env.dt)
            self.controller.before_step(keyboard.command())
        keyboard.process_events([SimpleNamespace(action="leg_extend", value=0)])
        for _ in range(300):
            keyboard.advance_leg_length(self.env.dt)
            self.controller.before_step(keyboard.command())
            self.assertAlmostEqual(float(self.env.commands[0, 3]), 0.14, places=6)
            torch.testing.assert_close(self.env.lift_lengths, torch.full((1, 2), 0.14))
        self.assertEqual(self.controller.events, [])

    def test_physical_reset_retains_failure_evidence_and_allows_manual_retry(self):
        self.controller.request_jump()
        self.tick(done=True)
        self.assertEqual(self.controller.states[0]["mode"], "blocked")
        self.assertFalse(self.controller.attempts[0]["success"])
        self.assertTrue(self.controller.request_jump())
        self.assertEqual(self.controller.states[0]["mode"], "crouch")
        self.controller.reset()
        self.assertTrue(self.controller.request_jump())

    def test_airborne_policy_does_not_spin_wheels_with_ground_policy(self):
        self.controller.states[0]["mode"] = "flight"
        self.env.pitch = torch.tensor([0.3])
        self.env.ang_vel = torch.tensor([[0.0, 10.0, 0.0]])
        policy = SimpleNamespace(act_inference=lambda obs: torch.full((1, 6), 100.0))
        actions = self.controller.policy_actions(policy, torch.zeros(1, 198))
        torch.testing.assert_close(actions[:, :4], torch.zeros(1, 4))
        self.assertLessEqual(
            float(actions[:, 4:].abs().max()) * 10,
            self.controller.jump.air_wheel_torque_limit + 1e-6,
        )

    def test_parking_is_bounded_velocity_feedback_without_root_state_writes(self):
        self.env.root_states[0, 0] = 1.0
        before = self.env.root_states.clone()
        for _ in range(100):
            self.controller.before_step()
        self.assertLess(float(self.env.commands[0, 0]), 0)
        self.assertLessEqual(abs(float(self.env.commands[0, 0])), 0.080001)
        torch.testing.assert_close(self.env.root_states, before, rtol=0, atol=0)

    def test_released_drive_holds_stopping_position_instead_of_release_point(self):
        self.controller.before_step([0.6, 0, 0, 0.1])
        self.env.root_states[0, 7] = 0.10
        self.env.root_states[0, 0] = 0.2
        self.controller.before_step([0, 0, 0, 0.1])
        self.assertTrue(self.controller.braking[0])
        self.assertAlmostEqual(float(self.controller.anchor_xy[0, 0]), 0.2, places=6)
        self.env.root_states[0, 0] = 0.3
        self.controller.before_step()
        self.assertAlmostEqual(float(self.controller.anchor_xy[0, 0]), 0.3, places=6)
        self.env.root_states[0, 7] = 0.0
        for _ in range(50):
            self.controller.before_step()
        self.assertFalse(self.controller.braking[0])
        anchor = self.controller.anchor_xy.clone()
        self.env.root_states[0, 0] += 0.01
        self.controller.before_step()
        torch.testing.assert_close(self.controller.anchor_xy, anchor, rtol=0, atol=0)

    def test_twenty_cm_standing_target_reaches_controller_and_excess_is_rejected(self):
        self.controller.before_step([0, 0, 0, 0.20])
        torch.testing.assert_close(self.env.lift_lengths, torch.full((1, 2), 0.20))
        self.assertAlmostEqual(float(self.env.commands[0, 3]), 0.20, places=6)
        for length in (0.201, 0.079, float("nan")):
            with self.assertRaises(ValueError):
                self.controller.before_step([0, 0, 0, length])

    def test_jump_targets_follow_requested_height_without_exceeding_joint_workspace(
        self,
    ):
        for nominal in (0.08, 0.10, 0.14, 0.18, 0.20):
            with self.subTest(length=nominal):
                self.setUp()
                self.env.commands[:, 3] = nominal
                self.env.length_l = self.env.length_r = torch.tensor([nominal])
                self.controller = JumpController(self.env, WheelLeggedJumpCfg())
                self.ready()
                self.assertTrue(self.controller.request_jump())
                state = self.controller.states[0]
                self.assertGreaterEqual(state["crouch_length"], 0.06)
                self.assertLess(state["crouch_length"], nominal)
                self.assertGreaterEqual(state["extension_length"] + 1e-8, nominal)
                self.assertLessEqual(state["extension_length"], 0.20)


if __name__ == "__main__":
    unittest.main()
