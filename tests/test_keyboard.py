"""Keyboard drive state: releases, emergency stop and episode resets."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.envs.wheelleg.keyboard import KeyboardDrive


def event(action, value=1.0):
    return SimpleNamespace(action=action, value=value)


class FakeGym:
    def __init__(self):
        self.bindings = []

    def subscribe_viewer_keyboard_event(self, viewer, key, action):
        self.bindings.append((viewer, key, action))


class KeyboardDriveTests(unittest.TestCase):
    def setUp(self):
        self.drive = KeyboardDrive(speed=0.3, turn_speed=0.8)

    def test_binds_directions_stop_and_reset_without_camera_keys(self):
        gym = FakeGym()
        api = SimpleNamespace(KEY_UP=57, KEY_DOWN=56, KEY_LEFT=55,
                              KEY_RIGHT=54, KEY_SPACE=0, KEY_R=35, KEY_Q=34, KEY_E=18)
        self.drive.bind(gym, "viewer", api)
        self.assertEqual(len(gym.bindings), 8)
        self.assertEqual({item[1] for item in gym.bindings}, {57, 56, 55, 54, 0, 35, 34, 18})
        self.assertEqual({item[2] for item in gym.bindings}, {
            "drive_forward", "drive_backward", "drive_left", "drive_right",
            "drive_stop", "drive_reset", "leg_shorten", "leg_extend",
        })

    def test_forward_and_backward_stop_on_release(self):
        for action, velocity in (("drive_forward", 0.3), ("drive_backward", -0.3)):
            self.assertFalse(self.drive.process_events([event(action)]))
            self.assertEqual(self.drive.command(), [velocity, 0.0, 0.0, 0.08])
            self.assertFalse(self.drive.process_events([event(action, 0)]))
            self.assertEqual(self.drive.command(), [0.0, 0.0, 0.0, 0.08])

    def test_opposite_keys_cancel_and_remaining_held_key_resumes(self):
        self.drive.process_events([event("drive_forward"), event("drive_backward")])
        self.assertEqual(self.drive.command()[0], 0.0)
        self.drive.process_events([event("drive_backward", 0)])
        self.assertEqual(self.drive.command()[0], 0.3)
        self.drive.process_events([event("drive_left"), event("drive_right")])
        self.assertEqual(self.drive.command()[2], 0.0)
        self.drive.process_events([event("drive_left", 0)])
        self.assertEqual(self.drive.command()[2], -0.8)

    def test_space_blocks_held_key_repeats_until_release_and_repress(self):
        self.drive.process_events([event("drive_forward"), event("drive_left")])
        self.drive.process_events([event("drive_stop")])
        self.assertEqual(self.drive.command(), [0.0, 0.0, 0.0, 0.08])
        self.drive.process_events([event("drive_forward"), event("drive_left")])
        self.assertEqual(self.drive.command(), [0.0, 0.0, 0.0, 0.08])
        self.drive.process_events([event("drive_stop", 0), event("drive_forward", 0)])
        self.assertEqual(self.drive.command()[0], 0.0)
        self.drive.process_events([event("drive_forward")])
        self.assertEqual(self.drive.command(), [0.3, 0.0, 0.0, 0.08])

    def test_reset_request_does_not_lose_held_direction(self):
        self.drive.process_events([event("drive_backward")])
        self.assertTrue(self.drive.process_events([event("drive_reset")]))
        # The environment reset can clear its own commands. The next loop can
        # reapply this held direction without requiring a second key-down event.
        self.assertEqual(self.drive.command(), [-0.3, 0.0, 0.0, 0.08])
        self.assertFalse(self.drive.process_events([]))
        self.assertFalse(self.drive.process_events([event("drive_reset", 0)]))
        self.assertEqual(self.drive.command()[0], -0.3)

    def test_turn_and_leg_length_and_unrelated_camera_event(self):
        self.drive.process_events([event("drive_left"), event("camera_move_forward")])
        self.assertEqual(self.drive.command(0.12), [0.0, 0.0, 0.8, 0.12])
        self.drive.process_events([event("drive_left", 0)])
        self.assertEqual(self.drive.command(0.12), [0.0, 0.0, 0.0, 0.12])

    def test_height_changes_with_simulation_time_and_holds_on_release(self):
        drive = KeyboardDrive(leg_length=.08, leg_speed=.02, leg_range=(.06, .12))
        drive.process_events([event("leg_extend")])
        for _ in range(50):
            drive.advance_leg_length(.02)
        self.assertAlmostEqual(drive.command()[3], .10)
        # Reading/logging a command must never advance the target twice.
        self.assertEqual(drive.command(), drive.command())
        drive.process_events([event("leg_extend", 0)])
        drive.advance_leg_length(.1)
        self.assertAlmostEqual(drive.command()[3], .10)
        drive.process_events([event("leg_shorten")])
        drive.advance_leg_length(.5)
        self.assertAlmostEqual(drive.command()[3], .09)

    def test_height_limits_opposite_keys_and_variable_step_size(self):
        drive = KeyboardDrive(leg_length=.08, leg_range=(.06, .12))
        drive.process_events([event("leg_extend"), event("leg_shorten")])
        drive.advance_leg_length(.2)
        self.assertEqual(drive.command()[3], .08)
        drive.process_events([event("leg_shorten", 0)])
        drive.advance_leg_length(10)
        self.assertEqual(drive.command()[3], .12)
        drive.process_events([event("leg_extend", 0), event("leg_shorten")])
        drive.advance_leg_length(10)
        self.assertEqual(drive.command()[3], .06)

    def test_space_stops_height_change_until_release_and_repress(self):
        drive = KeyboardDrive(leg_length=.08)
        drive.process_events([event("leg_extend"), event("drive_forward")])
        drive.advance_leg_length(.5)
        drive.process_events([event("drive_stop"), event("leg_extend")])
        drive.advance_leg_length(.5)
        self.assertAlmostEqual(drive.command()[3], .09)
        self.assertEqual(drive.command()[0], 0)
        drive.process_events([event("leg_extend", 0), event("leg_extend")])
        drive.advance_leg_length(.5)
        self.assertAlmostEqual(drive.command()[3], .10)

    def test_reset_restores_height_and_requires_a_new_height_press(self):
        drive = KeyboardDrive(leg_length=.08)
        drive.process_events([event("leg_extend"), event("drive_backward")])
        drive.advance_leg_length(.5)
        drive.reset_leg_length(.08)
        drive.advance_leg_length(.5)
        self.assertEqual(drive.command(), [-.3, 0., 0., .08])
        drive.process_events([event("leg_extend", 0), event("leg_extend")])
        drive.advance_leg_length(.5)
        self.assertAlmostEqual(drive.command()[3], .09)

    def test_invalid_height_settings_and_time_step_are_rejected(self):
        for kwargs in ({"leg_length": .04}, {"leg_length": .13, "leg_range": (.06, .12)},
                       {"leg_speed": 0}, {"leg_speed": float("nan")},
                       {"leg_range": (.12, .06)}, {"leg_range": (.05, .12)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KeyboardDrive(**kwargs)
        drive = KeyboardDrive(leg_length=.08)
        for dt in (0, -.02, float("nan")):
            with self.assertRaises(ValueError):
                drive.advance_leg_length(dt)


if __name__ == "__main__":
    unittest.main()
