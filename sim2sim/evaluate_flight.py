"""Paired ramp diagnostics; contact truth is logging only, not control input."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.environment import MujocoWheelLeg
from legged_gym.learning.policy import make_policy


def run(cp, enabled, seed, seconds, height, mode):
    source = torch.load(cp, map_location="cpu", weights_only=False)
    env = MujocoWheelLeg(
        source["legged_gym_env_config"],
        1,
        seed,
        seconds + 1,
        terrain="ramp",
        flight_compliance=enabled,
        ramp_height=height,
        flight_mode=mode,
    )
    policy = make_policy(132, 6, "cpu", source["training_config"]["policy"])
    policy.load_state_dict(source["model_state_dict"])
    policy.eval()
    stable_landing_s = 0.0
    longest_stable_landing_s = 0.0
    reached_ramp = False
    air_after_lip_s = 0.0
    max_cross = 0.0
    closure_peak = 0.0
    speed = torch.zeros(6)
    energy = torch.zeros(6)
    peak_torque = torch.zeros(6)
    peak_force = 0.0
    tilt = 0.0
    failures = 0
    airtime = 0.0
    detected = 0.0
    max_air = 0.0
    air_run = 0.0
    landings = 0
    previous = False
    trace = []
    wheel_ids = {env.model.body("L_lun").id, env.model.body("R_lun").id}
    with torch.inference_mode():
        for step in range(round(seconds / env.dt)):
            if step * env.dt >= 2:
                env.targets[0, 0] = 1
            _, _, _, done, info = env.step(policy.act_inference(env.obs))
            # Truth is used solely to score the accelerometer-based detector.
            d = env.data[0]
            contact = False
            normal_force = 0.0
            for n in range(d.ncon):
                g = d.contact[n].geom
                bodies = {
                    int(env.model.geom_bodyid[g[0]]),
                    int(env.model.geom_bodyid[g[1]]),
                }
                if bodies & wheel_ids:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(env.model, d, n, force)
                    normal_force += max(0.0, float(force[0]))
                    contact |= bool(force[0] > 0.01)
            peak_force = max(peak_force, normal_force)
            airborne = not contact
            reached_ramp |= bool(d.qpos[0] > 0.7)
            if d.qpos[0] > 0.65 and step * env.dt > 1:
                air_after_lip_s += airborne * env.dt
            max_cross = max(max_cross, abs(float(d.qpos[1])))
            for constraint in env.metadata["constraints"]:
                for n in (0, 1):
                    closure_peak = max(
                        closure_peak,
                        float(
                            np.linalg.norm(
                                d.site(constraint["name"] + "_0_" + str(n)).xpos
                                - d.site(constraint["name"] + "_1_" + str(n)).xpos
                            )
                        ),
                    )
            stable = (
                contact
                and d.qpos[0] > 1.1
                and info["angles"].abs().max() < 0.2
                and abs(float(info["velocity"][0, 3]) - 1) < 0.1
            )
            stable_landing_s = stable_landing_s + env.dt if stable else 0.0
            longest_stable_landing_s = max(longest_stable_landing_s, stable_landing_s)
            if airborne:
                air_run += env.dt
            else:
                if previous and step * env.dt > 1:
                    landings += 1
                air_run = 0.0
            max_air = max(max_air, air_run)
            previous = airborne
            if step * env.dt > 1:
                airtime += airborne * env.dt
            if env.flight is not None:
                detected += bool(env.flight.air[0]) * env.dt
            speed = torch.maximum(speed, env.last_speed_peak[0])
            energy += env.last_regen_j[0]
            peak_torque = torch.maximum(peak_torque, env.last_torque_peak[0])
            tilt = max(tilt, info["angles"].abs().max().item())
            failures += int(info["physical_failures"].sum())
            if step % 25 == 0:
                trace.append(
                    dict(
                        t=step * env.dt,
                        x=float(d.qpos[0]),
                        air_truth=airborne,
                        air_detected=bool(env.flight.air[0]) if env.flight else None,
                        blend=float(env.flight.blend[0]) if env.flight else 1.0,
                        pitch=float(info["angles"][0, 0]),
                        y=float(d.qpos[1]),
                        yaw=float(
                            np.arctan2(
                                d.xmat[env.base].reshape(3, 3)[1, 0],
                                d.xmat[env.base].reshape(3, 3)[0, 0],
                            )
                        ),
                        measured_yaw_rate=float(env.sensor.sample[0, 2]),
                        estimated_heading=float(env.hold.heading[0]),
                        heading_error=float(
                            env.hold.heading[0] - env.hold.heading_reference[0]
                        ),
                        heading_correction=float(
                            env.hold.corrections(env.commands, env.targets)[1][0]
                        ),
                    )
                )
            if bool(done.any()):
                break
    return dict(
        passed=bool(
            failures == 0
            and tilt <= 0.2
            and max_cross <= 0.05
            and closure_peak <= 0.001
            and reached_ramp
            and air_after_lip_s >= 0.004
            and longest_stable_landing_s >= 0.5
        ),
        crossed_lip=reached_ramp,
        air_after_lip_s=air_after_lip_s,
        stable_landing_s=longest_stable_landing_s,
        cross_track_peak_m=max_cross,
        closure_peak_m=closure_peak,
        enabled=enabled,
        mode=mode,
        seed=seed,
        seconds=(step + 1) * env.dt,
        failures=failures,
        tilt_peak_rad=tilt,
        sampled_wheel_normal_force_peak_n=peak_force,
        wheel_air_s_after1=airtime,
        max_continuous_air_s=max_air,
        detector_air_s=detected,
        landings=landings,
        joint_speed_peak_rad_s=speed.tolist(),
        mechanical_negative_work_j=energy.tolist(),
        torque_peak_nm=peak_torque.tolist(),
        back_emf_v=None,
        bus_voltage_v=None,
        electrical_note="No calibrated Ke/gear/drive model. Negative mechanical work is not battery recovered energy.",
        trace=trace,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "exports/h723_small/checkpoint.pt",
    )
    p.add_argument("--mode", choices=("support", "soft"), default="support")
    p.add_argument("--ramp-height", type=float, default=0.03)
    p.add_argument("--policy-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument(
        "--report", type=Path, default=ROOT / "logs/mujoco/flight_comparison.json"
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    result = dict(
        ramp_rise_m=a.ramp_height,
        ramp_run_m=0.3,
        requested_speed_m_s=1,
        baseline=run(a.checkpoint, False, a.seed, a.seconds, a.ramp_height, a.mode),
        compliant=(
            run(a.checkpoint, True, a.seed, a.seconds, a.ramp_height, a.mode)
            if not a.policy_only
            else None
        ),
    )
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: (
                    {n: v for n, v in d.items() if n != "trace"}
                    if isinstance(d, dict)
                    else d
                )
                for k, d in result.items()
            },
            indent=2,
        )
    )
