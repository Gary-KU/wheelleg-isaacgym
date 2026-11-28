"""Compare sensor pitch damping candidates without changing actuator limits.

Every candidate is a separate checkpoint carrying its actual control settings.
The original policy, geometry, observations, command slew and acceptance test
remain unchanged. This search is diagnostic; it does not declare deployment safe.
"""

import argparse
import contextlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.evaluate_flight import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Test higher rate damping after the initial comparison",
    )
    parser.add_argument(
        "--fine",
        action="store_true",
        help="Refine the pitch guard around the selected rate damping",
    )
    parser.add_argument(
        "--heading",
        action="store_true",
        help="Compare existing sensor heading-loop gain on a compressed actor",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    results = []
    candidates = (
        (0.006, 0.2),
        (0.012, 0.2),
        (0.02, 0.2),
        (0.035, 0.2),
        (0.012, 0.4),
        (0.02, 0.4),
    )
    if args.extended:
        candidates = ((0.05, 0.2), (0.07, 0.2), (0.1, 0.2))
    if args.fine:
        candidates = ((0.05, 0.3), (0.05, 0.4), (0.05, 0.6))
    if args.heading:
        candidates = ((3.0, None), (6.0, None), (9.0, None), (12.0, None))
    for index, (kd, kp) in enumerate(candidates):
        control = source["legged_gym_env_config"]["control"]
        if args.heading:
            control["encoder_hold_heading_kp"] = kd
            settings = dict(encoder_hold_heading_kp=kd)
        else:
            control["sensor_pitch_rate_kd"] = kd
            control["sensor_pitch_guard_kp"] = kp
            settings = dict(sensor_pitch_rate_kd=kd, sensor_pitch_guard_kp=kp)
        source["control_tuning"] = dict(source=str(args.checkpoint), settings=settings)
        checkpoint = args.output / f"candidate_{index}.pt"
        torch.save(source, checkpoint)
        reports = []
        for seed in (17, 42):
            with (args.output / f"candidate_{index}_seed{seed}.log").open(
                "w"
            ) as stream:
                with contextlib.redirect_stdout(stream):
                    report = run(checkpoint, False, seed, 6, 0.01, "support")
            (args.output / f"candidate_{index}_seed{seed}.json").write_text(
                json.dumps(report, indent=2)
            )
            reports.append(
                {key: value for key, value in report.items() if key != "trace"}
            )
            print(
                index,
                kd,
                kp,
                seed,
                report["tilt_peak_rad"],
                report["cross_track_peak_m"],
                report["passed"],
                flush=True,
            )
        results.append(
            dict(checkpoint=str(checkpoint), settings=settings, reports=reports)
        )
        (args.output / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
