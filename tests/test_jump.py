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
        self.pitch = self.roll = self.yaw = torch.zeros(1)
        self.ang_vel = self.lin_vel = torch.zeros(1, 3)
        self.length_l = self.length_r = torch.tensor([0.1])
        self.dof_state = torch.zeros(1, 10, 2)
        self.action_indices = torch.tensor([0, 3, 6, 9, 2, 7])

    def lift_measurements(self):
        return dict(
            wheel_normal_force_n=torch.full((1, 2), 28.0),
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

    def test_jump_requires_stable_contact_and_stationary_user_commands(self):
        self.assertFalse(self.controller.request_jump())
        self.ready()
        self.controller.before_step([0.2, 0, 0, 0.1])
        self.assertFalse(self.controller.request_jump())
        self.controller.before_step([0, 0, 0, 0.1])
        self.assertTrue(self.controller.request_jump())
        self.assertFalse(self.controller.request_jump())
        self.assertEqual(self.controller.states[0]["mode"], "crouch")

    def test_push_duration_then_flight_has_no_support_force(self):
        self.ready()
        self.controller.request_jump()
        forces = []
        for _ in range(50):
            self.tick()
            forces.append(float(self.env.lift_forces[0, 0]))
            if self.controller.states[0]["mode"] == "flight":
                break
        self.assertEqual(forces.count(45.0), 5)
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
        for _ in range(61):
            self.tick()
        self.assertEqual(len(self.controller.attempts), 1)
        self.assertFalse(self.controller.attempts[0]["success"])
        self.assertTrue(self.controller.attempts[0]["recovered"])

    def test_physical_reset_blocks_repeated_jumps_until_explicit_reset(self):
        self.ready()
        self.controller.request_jump()
        self.tick(done=True)
        self.assertEqual(self.controller.states[0]["mode"], "blocked")
        self.assertFalse(self.controller.attempts[0]["success"])
        self.assertFalse(self.controller.request_jump())
        self.controller.reset()
        self.assertEqual(self.controller.states[0]["mode"], "idle")
        self.assertFalse(self.controller.request_jump())

    def test_airborne_policy_does_not_spin_wheels_with_ground_policy(self):
        self.controller.states[0]["mode"] = "flight"
        self.env.pitch = torch.tensor([0.3])
        self.env.ang_vel = torch.tensor([[0.0, 10.0, 0.0]])
        policy = SimpleNamespace(act_inference=lambda obs: torch.full((1, 6), 100.0))
        actions = self.controller.policy_actions(policy, torch.zeros(1, 198))
        torch.testing.assert_close(actions[:, :4], torch.zeros(1, 4))
        self.assertLessEqual(float(actions[:, 4:].abs().max()) * 10, 0.250001)

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

    def test_jump_targets_follow_requested_height_without_exceeding_joint_workspace(
        self,
    ):
        for nominal in (0.08, 0.10, 0.14):
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
                self.assertGreater(state["extension_length"], nominal)
                self.assertLessEqual(state["extension_length"], 0.20)


if __name__ == "__main__":
    unittest.main()
