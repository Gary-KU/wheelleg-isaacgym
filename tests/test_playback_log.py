"""Failure telemetry must describe the terminal state and remain valid JSON."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.playback_log import PlaybackEventLog, reset_diagnostics


class PlaybackLogTests(unittest.TestCase):
    def test_failure_selects_the_terminal_environment_and_all_reason_flags(self):
        infos = {
            "pre_reset_commands": [[0.3, 0, 0, 0.08], [-0.3, 0, 0, 0.08]],
            "pre_reset_pitch": [0.1, 0.4], "pre_reset_roll": [0.0, -0.2],
            "loop_error": [[0.001, 0.002], [0.003, 0.005]],
            "termination_reasons": {"pitch": [False, True], "loop": [False, False]},
        }
        report = reset_diagnostics(infos, 1)
        self.assertEqual(report["active_command"], [-0.3, 0, 0, 0.08])
        self.assertEqual(report["pitch_rad"], 0.4)
        self.assertEqual(report["roll_rad"], -0.2)
        self.assertEqual(report["anchor_error_max_m"], 0.005)
        self.assertEqual(report["termination_reasons"], {"pitch": True, "loop": False})

    def test_missing_terminal_data_is_unknown_rather_than_a_reset_pose(self):
        report = reset_diagnostics({}, 0, [0.3, 0, 0, 0.08])
        self.assertEqual(report["active_command"], [0.3, 0, 0, 0.08])
        self.assertIsNone(report["pitch_rad"])
        self.assertIsNone(report["roll_rad"])
        self.assertIsNone(report["anchor_error_max_m"])
        self.assertEqual(report["termination_reasons"], {})

    def test_nonfinite_terminal_values_do_not_break_failure_logging(self):
        report = reset_diagnostics({
            "pre_reset_pitch": [float("nan")],
            "loop_error": [[float("inf"), 0.003]],
            "termination_reasons": {"nonfinite": [True]},
        }, 0)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = Path(folder)/"events.jsonl"
            log = PlaybackEventLog(path)
            try:
                log.emit("physical_failure", step=12, sim_time_s=0.24, **report)
            finally:
                log.close()
            record = json.loads(path.read_text())
        self.assertIsNone(record["pitch_rad"])
        self.assertEqual(record["anchor_error_m"], [None, 0.003])
        self.assertEqual(record["anchor_error_max_m"], 0.003)
        self.assertTrue(record["termination_reasons"]["nonfinite"])

    def test_jsonl_preserves_command_timing_and_manual_reset_order(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = Path(folder)/"nested"/"events.jsonl"
            log = PlaybackEventLog(path)
            try:
                log.emit("playback_started", working_directory=Path(folder))
                log.emit("command_change", step=0, sim_time_s=0.0,
                         active_command=[0.3, 0, 0, 0.08])
                log.emit("manual_reset", step=7, sim_time_s=0.14)
                log.emit("command_change", step=8, sim_time_s=0.16,
                         active_command=[-0.3, 0, 0, 0.08])
            finally:
                log.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["event"] for record in records],
                         ["playback_started", "command_change", "manual_reset", "command_change"])
        self.assertEqual(records[-1]["step"], 8)
        self.assertEqual(records[-1]["active_command"][0], -0.3)
        self.assertTrue(all("timestamp_utc" in record for record in records))


if __name__ == "__main__":
    unittest.main()
