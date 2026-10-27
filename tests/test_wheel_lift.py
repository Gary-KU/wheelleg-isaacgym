"""Independent support geometry and measured-contact gates for wheel lifting."""

import math
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import unittest
from types import SimpleNamespace

import isaacgym  # extension before torch
import torch

from legged_gym.envs.wheelleg.control.leg_control import hip_targets, hip_support_torques
from legged_gym.envs.wheelleg.model.kinematics import fivebar
from legged_gym.envs.wheelleg.control.wheel_lift import (
    AlternatingWheelLift,
    WheelLiftEnv,
    validate_spherical_tread,
    sagittal_wheel_observations,
    WheelLiftKeyboard,
)
from legged_gym.envs.wheelleg.wheelleg_robot import ASSET_ROOT


class IndependentLegTests(unittest.TestCase):
    def test_independent_targets_reach_each_requested_length_with_correct_mirrors(self):
        lengths = torch.tensor([[0.08, 0.12], [0.15, 0.07]], dtype=torch.float64)
        q = hip_targets(lengths)
        left = fivebar(3 * math.pi / 4 + q[:, 0], math.pi / 4 + q[:, 1], 0.0)[0]
        right = fivebar(3 * math.pi / 4 + q[:, 3], math.pi / 4 + q[:, 2], 0.0)[0]
        torch.testing.assert_close(
            torch.stack((left, right), -1), lengths, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            hip_targets(lengths[:, 0]), hip_targets(lengths[:, 0, None].expand(-1, 2))
        )

    def test_independent_loads_match_virtual_work_and_zero_swing_load(self):
        q = hip_targets(
            torch.tensor([[0.08, 0.12], [0.15, 0.07]], dtype=torch.float64)
        ).requires_grad_(True)
        left = fivebar(3 * math.pi / 4 + q[:, 0], math.pi / 4 + q[:, 1], 0.0)[0]
        right = fivebar(3 * math.pi / 4 + q[:, 3], math.pi / 4 + q[:, 2], 0.0)[0]
        loads = torch.tensor([[0.0, 44.9], [15.0, 29.9]], dtype=q.dtype)
        expected = torch.autograd.grad(
            (torch.stack((left, right), -1) * loads).sum(), q
        )[0]
        actual = hip_support_torques(q.detach(), loads)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(actual[0, :2], torch.zeros(2, dtype=q.dtype))
        torch.testing.assert_close(
            hip_support_torques(q.detach(), torch.full_like(loads, 22.4)),
            hip_support_torques(q.detach(), 22.4),
        )

    def test_invalid_support_tensors_cannot_apply_nonfinite_or_misaligned_loads(self):
        for loads in (
            torch.zeros(1, 2),
            torch.tensor([[1.0, float("nan")], [1.0, 2.0]]),
            -torch.ones(2, 2),
        ):
            with self.assertRaises(ValueError):
                hip_support_torques(torch.zeros(2, 4), loads)


class TreadAndPolicyAdapterTests(unittest.TestCase):
    def test_source_tread_matches_proxy_and_operating_roll_stays_inside_band(self):
        for name in ("L_lun", "R_lun"):
            validate_spherical_tread(ASSET_ROOT / "meshes" / (name + "_visual.obj"))
        self.assertLess(0.0625 * math.sin(0.22), 0.02)

    def test_proxy_retains_all_joint_frames_mass_inertia_and_visual_geometry(self):
        class AssetReader:
            def load_asset(self, sim, root, name, options):
                self.tree = ET.parse(Path(root) / name)
                return "asset"

        env = object.__new__(WheelLiftEnv)
        env.wheel_collision = "sphere_tread"
        env._tread_asset_dir = None
        env.metadata = json.loads((ASSET_ROOT / "loop_constraints.json").read_text())
        env.sim = None
        env.gym = AssetReader()
        try:
            self.assertEqual(env._create_robot_asset(object()), "asset")
            original = ET.parse(ASSET_ROOT / env.metadata["urdf"]).getroot()
            proxy = env.gym.tree.getroot()
            for a, b in zip(original.findall("joint"), proxy.findall("joint")):
                self.assertEqual(ET.tostring(a), ET.tostring(b))
            for a, b in zip(original.findall("link"), proxy.findall("link")):
                self.assertEqual(
                    ET.tostring(a.find("inertial")), ET.tostring(b.find("inertial"))
                )
                self.assertEqual(
                    a.find("visual/origin").attrib, b.find("visual/origin").attrib
                )
                original_mesh = a.find("visual/geometry/mesh").get("filename")
                expected = (ASSET_ROOT / "urdf" / original_mesh).resolve()
                self.assertEqual(
                    Path(b.find("visual/geometry/mesh").get("filename")), expected
                )
                if a.get("name") in ("L_lun", "R_lun"):
                    self.assertEqual(
                        b.find("collision/geometry/sphere").get("radius"), "0.0625"
                    )
                else:
                    self.assertIsNone(b.find("collision/geometry/sphere"))
        finally:
            if env._tread_asset_dir is not None:
                env._tread_asset_dir.cleanup()

    def test_sagittal_adapter_preserves_measurements_and_correct_mirror_pairs(self):
        original = torch.zeros(2, 198)
        frames = original.reshape(2, 6, 33)
        frames[..., 0] = 0.3
        frames[..., 1] = 0.2
        frames[..., 3] = 0.4
        frames[..., 6] = 0.1
        frames[..., 7] = 0.2
        frames[..., 8] = -0.97
        frames[..., 13] = 1.0
        frames[..., 15] = 7.0
        frames[..., 14] = 2.0
        frames[..., 16] = 4.0
        frames[..., 17] = 1.0
        frames[..., 20] = 11.0
        frames[..., 18] = 2.0
        frames[..., 19] = 6.0
        frames[..., 31] = 0.08
        frames[..., 32] = 0.12
        before = original.clone()
        adapted = sagittal_wheel_observations(original).reshape(2, 6, 33)
        torch.testing.assert_close(original, before)
        for index, expected in (
            (0, 0.3),
            (1, 0.0),
            (3, 0.0),
            (6, 0.1),
            (7, 0.0),
            (13, 4.0),
            (15, 4.0),
            (14, 3.0),
            (16, 3.0),
            (17, 6.0),
            (20, 6.0),
            (18, 4.0),
            (19, 4.0),
            (31, 0.1),
            (32, 0.1),
        ):
            torch.testing.assert_close(
                adapted[..., index], torch.full((2, 6), expected)
            )
        torch.testing.assert_close(
            adapted[..., 8], torch.full((2, 6), -math.sqrt(0.99))
        )
        with self.assertRaises(ValueError):
            sagittal_wheel_observations(torch.zeros(1, 186))


class SensorFixture:
    num_envs = 1
    device = "cpu"
    dt = 0.02
    leg_kp = 80.0
    leg_kd = 4.0
    leg_feedforward_force = 22.4649

    def __init__(self):
        self.root_states = torch.zeros(1, 13)
        self.pitch = torch.zeros(1)
        self.roll = torch.zeros(1)
        self.yaw = torch.zeros(1)
        self.length_l = torch.full((1,), 0.1)
        self.length_r = torch.full((1,), 0.1)
        self.passive_limit_margin = torch.full((1, 4), 0.1)
        self.lin_vel = torch.zeros(1, 3)
        self.ang_vel = torch.zeros(1, 3)
        self.commands = torch.tensor([[0.0, 0.0, 0.0, 0.1]])

    def lift_measurements(self):
        return dict(
            wheel_normal_force_n=torch.full((1, 2), 30.0),
            nonwheel_contact=torch.zeros(1, dtype=torch.bool),
        )

    def set_commands(self, value):
        self.commands = value
        return value

    def info(self):
        return dict(
            pre_reset_root_states=self.root_states.clone(),
            lift_clearance_m=torch.zeros(1, 2),
            lift_wheel_contact_norm_n=torch.full((1, 2), 30.0),
            lift_wheel_normal_force_n=torch.full((1, 2), 30.0),
            lift_nonwheel_contact=torch.zeros(1, dtype=torch.bool),
        )


class LiftGateTests(unittest.TestCase):
    def test_manual_wait_does_not_time_out_and_one_press_starts_only_one_side(self):
        env = SensorFixture()
        c = AlternatingWheelLift(env, interactive=True)
        for _ in range(1300):
            c.before_step()
            c.after_step(env.info(), torch.zeros(1, dtype=torch.bool))
        self.assertEqual(c.states[0]["mode"], "settle")
        self.assertFalse(c.events)
        c.request_once()
        c.before_step()
        self.assertEqual(c.states[0]["mode"], "preload")
        self.assertFalse(c.pending_once)
        c.request_once()  # an in-progress action cannot queue another lift
        self.assertFalse(c.pending_once)

    def test_pause_finishes_landing_and_recovery_without_starting_the_next_side(self):
        env, c = self.start_lift()
        c.interactive = True
        c.set_running(False)
        env.roll.zero_()
        for _ in range(200):
            c.before_step()
            c.after_step(env.info(), torch.zeros(1, dtype=torch.bool))
        self.assertEqual(c.states[0]["mode"], "settle")
        self.assertEqual(len(c.attempts), 1)
        self.assertEqual(c.states[0]["side"], 1)
        torch.testing.assert_close(env.lift_lengths, torch.full((1, 2), 0.1))

    def test_keyboard_binding_and_press_edges_do_not_repeat_held_keys(self):
        class Receiver:
            running = False
            requests = 0

            def request_once(self):
                self.requests += 1

            def set_running(self, running):
                self.running = running

        receiver = Receiver()
        keyboard = WheelLiftKeyboard(receiver)
        bindings = []
        gym = SimpleNamespace(
            subscribe_viewer_keyboard_event=lambda *x: bindings.append(x)
        )
        api = SimpleNamespace(KEY_N=1, KEY_SPACE=2)
        keyboard.bind(gym, "viewer", api)
        self.assertEqual(
            bindings, [("viewer", 1, "lift_next"), ("viewer", 2, "lift_toggle")]
        )

        def event(action, value):
            return SimpleNamespace(action=action, value=value)

        keyboard.process_events([event("lift_next", 1), event("lift_next", 1)])
        self.assertEqual(receiver.requests, 1)
        keyboard.process_events([event("lift_next", 0), event("lift_next", 1)])
        self.assertEqual(receiver.requests, 1)  # X11 held-key repeat pair
        keyboard.process_events([event("lift_next", 0)])
        keyboard.process_events([])
        keyboard.process_events([])
        keyboard.process_events([event("lift_next", 1)])
        self.assertEqual(receiver.requests, 2)
        keyboard.process_events([event("lift_toggle", 1), event("lift_toggle", 1)])
        self.assertTrue(receiver.running)
        keyboard.process_events([event("lift_toggle", 0)])
        keyboard.process_events([])
        keyboard.process_events([])
        keyboard.process_events([event("lift_toggle", 1)])
        self.assertFalse(receiver.running)

    def start_lift(self):
        env = SensorFixture()
        controller = AlternatingWheelLift(env, cycles=1)
        for _ in range(400):
            controller.before_step()
            if controller.states[0]["mode"] == "lift":
                return env, controller
            # Feed a settled measured roll compatible with the requested lean.
            env.roll[0] = math.atan2(
                float(env.lift_lengths[0, 0] - env.lift_lengths[0, 1]), 0.23
            )
            controller.after_step(env.info(), torch.zeros(1, dtype=torch.bool))
        self.fail("Stable sensors should admit a lift")

    def test_unstable_robot_never_starts_preload_or_retracts_a_wheel(self):
        env = SensorFixture()
        env.pitch[:] = 0.1
        controller = AlternatingWheelLift(env)
        for _ in range(250):
            controller.before_step()
            controller.after_step(env.info(), torch.zeros(1, dtype=torch.bool))
        self.assertFalse(controller.events)
        torch.testing.assert_close(env.lift_lengths, torch.full((1, 2), 0.1))

    def test_clearance_unloading_and_continuous_support_are_all_required(self):
        env, c = self.start_lift()
        info = env.info()
        done = torch.zeros(1, dtype=torch.bool)
        info["lift_wheel_contact_norm_n"][0, 0] = 0.0
        c.after_step(info, done)  # no geometric gap, merely unloaded
        self.assertEqual(c.states[0]["attempt"]["longest_airborne_s"], 0.0)
        info["lift_clearance_m"][0, 0] = 0.005
        info["lift_wheel_contact_norm_n"][0, 0] = 2.0
        c.after_step(info, done)  # contact peak from any physics substep rejects air
        self.assertEqual(c.states[0]["attempt"]["longest_airborne_s"], 0.0)
        info["lift_wheel_contact_norm_n"][0, 0] = 0.0
        c.after_step(info, done)
        self.assertAlmostEqual(c.states[0]["attempt"]["airborne_run_s"], 0.02)
        info["lift_wheel_contact_norm_n"][0, 0] = 2.0
        c.after_step(info, done)
        self.assertEqual(c.states[0]["attempt"]["airborne_run_s"], 0.0)
        info["lift_wheel_contact_norm_n"][0, 0] = 0.0
        c.after_step(info, done)
        c.after_step(info, done)
        self.assertAlmostEqual(c.states[0]["attempt"]["longest_airborne_s"], 0.04)

    def test_rebound_after_lowering_and_falling_do_not_count_as_success(self):
        env, c = self.start_lift()
        info = env.info()
        c.states[0]["mode"] = "recover"
        info["lift_clearance_m"][0, 0] = 0.02
        info["lift_wheel_contact_norm_n"][0, 0] = 0.0
        c.after_step(info, torch.zeros(1, dtype=torch.bool))
        c.after_step(info, torch.ones(1, dtype=torch.bool))
        self.assertFalse(c.attempts[0]["success"])
        self.assertEqual(c.summary(1)["confirmed_lifts"], dict(left=0, right=0))
        self.assertFalse(c.summary(1)["engineering_screen_passed"])

    def test_safety_return_is_ramped_instead_of_snapping_to_stand_pose(self):
        env, c = self.start_lift()
        info = env.info()
        done = torch.zeros(1, dtype=torch.bool)
        c.after_step(info, done)
        c.before_step()  # reach full retraction target
        before = env.lift_lengths.clone()
        env.roll[:] = 0.24
        c.before_step()
        self.assertEqual(c.states[0]["mode"], "abort_lower")
        self.assertTrue(c.states[0]["attempt"]["safety_abort"])
        torch.testing.assert_close(before, env.lift_lengths)

    def test_fractional_control_ticks_are_rejected(self):
        with self.assertRaises(ValueError):
            AlternatingWheelLift(SensorFixture(), retract=0.03)


if __name__ == "__main__":
    unittest.main()
