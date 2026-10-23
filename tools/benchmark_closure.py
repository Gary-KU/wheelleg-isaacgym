#!/usr/bin/env python3
"""Measure compliant-closure load response; this does not validate rigidity.

The diagnostic suspends a fixed-base robot above ground and disables gravity.
Static mode separates the two closing-anchor pairs with equal/opposite forces.
Driven mode applies matched sinusoidal hip torques without a balancing policy.
Neither mode modifies production defaults, assets, or saved policy weights.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time


def parse_case(value):
    try:
        stiffness, damping, dt = map(float, value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Expected stiffness_N_per_m,damping_Ns_per_m,physics_dt_s")
    if not all(math.isfinite(x) for x in (stiffness, damping, dt)) or stiffness <= 0 or damping < 0 or dt <= 0:
        raise argparse.ArgumentTypeError("Require finite stiffness>0, damping>=0, dt>0")
    return stiffness, damping, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case,
                        help="Repeat K,D,DT; default compares 50k/25/.0005, 100k/35/.0005, 100k/35/.00025")
    parser.add_argument("--mode", choices=("static", "driven", "both"), default="both")
    parser.add_argument("--static-duration", type=float, default=0.75)
    parser.add_argument("--driven-duration", type=float, default=2.0)
    parser.add_argument("--load-force", type=float, default=20.0, help="N per closing-anchor pair")
    parser.add_argument("--load-axis", choices=("x", "z"), default="x")
    parser.add_argument("--torque-amplitude", type=float, default=0.2, help="Nm per active hip")
    parser.add_argument("--frequency", type=float, default=1.0, help="Hz; cosine phase avoids secular sine-input drift")
    parser.add_argument("--force-cap", type=float, default=500.0, help="N; held constant across cases")
    parser.add_argument("--report", type=Path, default=Path("validation/closure_benchmark.json"))
    args = parser.parse_args()
    for key in ("static_duration", "driven_duration", "load_force", "torque_amplitude", "frequency", "force_cap"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error("{} must be finite and positive".format(key))

    # CPU only: deliberately leaves a user's interactive GPU viewer untouched.
    from isaacgym import gymapi, gymtorch
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wheelleg_gym.env import WheelLegGymEnv

    class DiagnosticEnv(WheelLegGymEnv):
        def _load_robot(self, stiffness, damping):
            self.metadata = deepcopy(self.metadata)
            self.metadata["default_root_pose"]["pos"][2] = 1.0
            super()._load_robot(stiffness, damping)

    def run_case(mode, stiffness, damping, physics_dt):
        env = DiagnosticEnv(num_envs=1, sim_device="cpu", headless=True,
                            fix_base=True, physics_dt=physics_dt,
                            closure_stiffness=stiffness, closure_damping=damping,
                            allow_approximate_loop=True)
        try:
            params = env.gym.get_sim_params(env.sim)
            params.gravity = gymapi.Vec3(0, 0, 0)
            env.gym.set_sim_params(env.sim, params)
            env.closure.max_force = args.force_cap
            env.reset()
            duration = args.static_duration if mode == "static" else args.driven_duration
            steps = int(round(duration / physics_dt))
            if steps < 5:
                raise ValueError("Duration must contain at least five physics steps")
            direction = torch.tensor([1., 0., 0.] if args.load_axis == "x" else [0., 0., 1.])
            hip_pattern = torch.tensor([1., -1., -1., 1.])
            warmup = min(0.1, duration * 0.2)
            records = []
            stopped_reason = None
            started = time.perf_counter()
            initial_error = env.closure.errors(env.rb_states).max().item()
            with torch.inference_mode():
                for i in range(steps):
                    t = i * physics_dt
                    data = env.closure._geometry(env.rb_states)
                    raw = stiffness * (data[3] - data[2]) + damping * (data[7] - data[6])
                    raw_norm = torch.linalg.vector_norm(raw, dim=-1)
                    forces, moments = env.closure.compute(env.rb_states)
                    env.torques.zero_()
                    if mode == "static" and t >= warmup:
                        # Pull endpoints apart; closure must carry this known load.
                        force0 = (-args.load_force * direction).view(1, 1, 3).expand_as(data[2])
                        force1 = -force0
                        forces.index_add_(1, env.closure.body0, force0)
                        forces.index_add_(1, env.closure.body1, force1)
                        moments.index_add_(1, env.closure.body0, torch.cross(data[4], force0, dim=-1))
                        moments.index_add_(1, env.closure.body1, torch.cross(data[5], force1, dim=-1))
                    elif mode == "driven":
                        env.torques[:, env.action_indices[:4]] = (
                            args.torque_amplitude * math.cos(2 * math.pi * args.frequency * t) * hip_pattern)
                    env.gym.apply_rigid_body_force_tensors(
                        env.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(moments), gymapi.ENV_SPACE)
                    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
                    env.gym.simulate(env.sim)
                    env.gym.fetch_results(env.sim, True)
                    env._refresh()
                    if not torch.isfinite(env.rb_states).all():
                        stopped_reason = "nonfinite_rigid_body_state"
                        break
                    geometry = env.closure._geometry(env.rb_states)
                    error = env.closure.errors(env.rb_states)
                    axis_error = env.closure.axis_errors(env.rb_states)
                    projected = ((geometry[3] - geometry[2]) * direction).sum(dim=-1)
                    records.append((t + physics_dt, error[0].tolist(), axis_error[0].tolist(),
                                    raw_norm[0].tolist(), projected[0].tolist()))
                    if error.max().item() > 0.03:
                        stopped_reason = "anchor_separation_exceeded_production_30mm_limit"
                        break
            times = np.array([r[0] for r in records])
            gaps = np.array([r[1] for r in records])
            angles = np.array([r[2] for r in records])
            raw_forces = np.array([r[3] for r in records])
            projections = np.array([r[4] for r in records])
            if not records:
                return {"mode": mode, "stiffness_N_per_m": stiffness, "damping_Ns_per_m": damping,
                        "physics_dt_s": physics_dt, "stopped_reason": stopped_reason, "completed_physics_steps": 0}
            settled = times >= 0.8 * duration
            tail_gap = gaps[settled] if settled.any() else gaps[-1:]
            tail_projection = projections[settled] if settled.any() else projections[-1:]
            result = {
                "mode": mode, "stiffness_N_per_m": stiffness, "damping_Ns_per_m": damping,
                "physics_dt_s": physics_dt, "force_cap_N": args.force_cap,
                "simulated_seconds": float(times[-1]), "wall_seconds": time.perf_counter() - started,
                "completed_physics_steps": len(records), "stopped_reason": stopped_reason,
                "initial_max_anchor_error_mm": initial_error * 1000,
                "peak_anchor_error_mm": float(gaps.max() * 1000),
                "p95_anchor_error_mm": float(np.percentile(gaps, 95) * 1000),
                "tail_mean_anchor_error_mm_per_loop": (tail_gap.mean(axis=0) * 1000).tolist(),
                "tail_gap_peak_to_peak_mm_per_loop": (np.ptp(tail_gap, axis=0) * 1000).tolist(),
                "peak_axis_error_rad": float(angles.max()),
                "peak_raw_closure_force_N": float(raw_forces.max()),
                "force_cap_fraction_of_loop_samples": float((raw_forces > args.force_cap).mean()),
            }
            if mode == "static":
                projected_mean = tail_projection.mean(axis=0)
                result.update(load_force_N=args.load_force, load_axis=args.load_axis,
                              expected_linear_gap_mm=1000 * args.load_force / stiffness,
                              tail_mean_gap_along_load_mm_per_loop=(projected_mean * 1000).tolist(),
                              apparent_stiffness_N_per_m_per_loop=(args.load_force / np.maximum(np.abs(projected_mean), 1e-12)).tolist())
            else:
                result.update(torque_amplitude_Nm=args.torque_amplitude, frequency_Hz=args.frequency,
                              hip_torque_pattern=hip_pattern.tolist(), torque_phase="cosine")
            print("{} K={:.0f} D={:.1f} dt={:.6f}: peak={:.3f}mm tail={}mm rawF={:.2f}N cap={:.3%} stop={}".format(
                mode, stiffness, damping, physics_dt, result["peak_anchor_error_mm"],
                [round(x, 4) for x in result["tail_mean_anchor_error_mm_per_loop"]],
                result["peak_raw_closure_force_N"], result["force_cap_fraction_of_loop_samples"], stopped_reason), flush=True)
            return result
        finally:
            env.close()

    cases = args.case or [(50000., 25., .0005), (100000., 35., .0005), (100000., 35., .00025)]
    modes = ("static", "driven") if args.mode == "both" else (args.mode,)
    results = [run_case(mode, *case) for mode in modes for case in cases]
    report = {"purpose": "CPU compliant-closure calibration; not a rigid-joint validation or policy score",
              "setup": "One fixed-base robot at 1m, zero gravity, no ground contact, no policy, no automatic resets",
              "interpretation": "Compare the same load and force cap. Static F/K and dt convergence distinguish compliant load deflection from integration artifacts. Dynamic trajectories may differ across K/D.",
              "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Report: {}".format(args.report.resolve()), flush=True)


if __name__ == "__main__":
    main()
