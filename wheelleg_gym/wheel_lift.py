"""Contact-gated, in-place wheel-lift practice; the old wheel policy balances.

No root pose/velocity is written and no external support is supplied. A short
single-support pulse is an experiment, not quasi-static one-legged standing.
"""

import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import isaacgym  # binary extension before torch
import torch

from .constraints import quat_rotate_inverse
from .env import ASSET_ROOT, WheelLegGymEnv
from .leg_control import hip_targets, hip_support_torques


def validate_spherical_tread(path):
    """Reject a proxy when the source tread geometry changes."""
    rings = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("v "):
            continue
        x, y, z = map(float, line.split()[1:4])
        if -0.042001 <= z <= -0.001999:
            key = round(z, 6)
            if key not in rings or math.hypot(x, y) > math.hypot(*rings[key][:2]):
                rings[key] = (x, y, z)
    if len(rings) != 7 or any(
        abs(math.sqrt(x * x + y * y + (z + 0.022) ** 2) - 0.0625) > 1e-7
        for x, y, z in rings.values()
    ):
        raise ValueError("Source tire no longer matches the spherical tread proxy")


class WheelLiftEnv(WheelLegGymEnv):
    def __init__(self, wheel_collision="mesh", **kwargs):
        if wheel_collision not in ("mesh", "sphere_tread"):
            raise ValueError("wheel_collision must be mesh or sphere_tread")
        self.wheel_collision = wheel_collision
        self._tread_asset_dir = None
        self.lift_lengths = self.lift_forces = self.lift_kp = self.lift_kd = None
        super().__init__(**kwargs)
        self._tire_vertices = []
        for name in ("L_lun", "R_lun"):
            path = ASSET_ROOT / "meshes" / (name + "_visual.obj")
            vertices = {
                tuple(map(float, line.split()[1:4]))
                for line in path.read_text().splitlines()
                if line.startswith("v ")
            }
            self._tire_vertices.append(
                torch.tensor(sorted(vertices), device=self.device)
            )
        self._normal_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self._contact_max = torch.zeros_like(self._normal_sum)
        self._physics_step_callback = self._collect_contacts
        self._nonwheel = [
            i for i in range(self.num_bodies) if i not in self.wheel_indices.tolist()
        ]
        self._nonwheel_contact_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _create_robot_asset(self, options):
        if self.wheel_collision == "mesh":
            return super()._create_robot_asset(options)
        # The original tread vertices are a spherical band: radius62.5mm,
        # local centerZ=-22mm, cut at +/-20mm along the wheel axis. A native
        # sphere has the same rolling surface on flat ground below18deg roll.
        # Keep original visual meshes, masses, COMs, inertias and joint frames.
        self._tread_asset_dir = tempfile.TemporaryDirectory(prefix="wheelleg-tread-")
        tree = ET.parse(ASSET_ROOT / self.metadata["urdf"])
        for mesh in tree.getroot().iter("mesh"):
            mesh.set(
                "filename",
                str(
                    (
                        ASSET_ROOT
                        / Path(self.metadata["urdf"]).parent
                        / mesh.get("filename")
                    ).resolve()
                ),
            )
        for name in ("L_lun", "R_lun"):
            validate_spherical_tread(ASSET_ROOT / "meshes" / (name + "_visual.obj"))
            collision = tree.find("link[@name='{}']/collision".format(name))
            mesh = collision.find("geometry/mesh")
            origin = collision.find("origin")
            if (
                mesh is None
                or any(float(x) != 1 for x in mesh.get("scale", "1 1 1").split())
                or any(
                    float(x) != 0
                    for key in ("xyz", "rpy")
                    for x in origin.get(key, "0 0 0").split()
                )
            ):
                raise ValueError(
                    "Source tire collision frame or scale changed; revalidate tread proxy"
                )
            collision.find("origin").set("xyz", "0 0 -0.022")
            collision.find("origin").set("rpy", "0 0 0")
            geometry = collision.find("geometry")
            geometry.clear()
            ET.SubElement(geometry, "sphere", radius="0.0625")
        tree.write(Path(self._tread_asset_dir.name) / "tread.urdf")
        return self.gym.load_asset(
            self.sim, self._tread_asset_dir.name, "tread.urdf", options
        )

    def close(self):
        super().close()
        if self._tread_asset_dir is not None:
            self._tread_asset_dir.cleanup()
            self._tread_asset_dir = None

    def _hip_targets(self):
        return (
            super()._hip_targets()
            if self.lift_lengths is None
            else hip_targets(self.lift_lengths)
        )

    def _leg_support_bias(self):
        return (
            super()._leg_support_bias()
            if self.lift_forces is None
            else hip_support_torques(
                self.dof_state[:, self.action_indices[:4], 0], self.lift_forces
            )
        )

    def _hip_pd_torques(self, targets):
        if self.lift_kp is None:
            return super()._hip_pd_torques(targets)
        return (
            self.lift_kp * (targets - self.dof_state[:, self.action_indices[:4], 0])
            - self.lift_kd * self.dof_state[:, self.action_indices[:4], 1]
        )

    def _collect_contacts(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        forces = self.contact_forces[:, self.wheel_indices]
        self._normal_sum += forces[..., 2]
        self._contact_max.copy_(
            torch.maximum(self._contact_max, torch.linalg.vector_norm(forces, dim=-1))
        )
        self._nonwheel_contact_seen |= (
            torch.linalg.vector_norm(self.contact_forces[:, self._nonwheel], dim=-1)
            > 1.0
        ).any(-1)

    def step(self, actions):
        self._normal_sum.zero_()
        self._contact_max.zero_()
        self._nonwheel_contact_seen.zero_()
        return super().step(actions)

    def lift_measurements(self):
        states = self.rb_states[:, self.wheel_indices]
        up = torch.zeros_like(states[..., :3])
        up[..., 2] = 1.0
        local_up = quat_rotate_inverse(states[..., 3:7], up)
        # min vertex world Z gives a conservative original-mesh clearance.
        clearance = torch.stack(
            [
                states[:, j, 2] + (local_up[:, j, None, :] * v).sum(-1).min(-1).values
                for j, v in enumerate(self._tire_vertices)
            ],
            dim=-1,
        )
        if self.wheel_collision == "sphere_tread":
            clearance = states[..., 2] - 0.022 * local_up[..., 2] - 0.0625
        return {
            "clearance_m": clearance,
            "tread_axis_tilt_rad": torch.asin(local_up[..., 2].abs().clamp(max=1.0)),
            "wheel_contact_norm_n": self._contact_max,
            "wheel_normal_force_n": self._normal_sum / self.decimation,
            "nonwheel_contact": self._nonwheel_contact_seen,
        }

    def _extra_step_info(self):
        values = self.lift_measurements()
        values["wheel_normal_force_n"] = self._normal_sum / self.decimation
        values["wheel_contact_norm_n"] = self._contact_max
        return {"lift_" + key: value.clone() for key, value in values.items()}


def sagittal_wheel_observations(observations):
    """Reuse the standing wheel actor on left/right averaged sagittal features.

    A clone is transformed; real physics, environment observations, contacts,
    roll safety and independent leg targets are untouched. This is an explicit
    policy-input adapter, not a newly trained 3D gait policy.
    """
    if observations.ndim != 2 or observations.shape[1] != 198:
        raise ValueError("Sagittal wheel adapter requires six 33-feature frames")
    result = observations.clone()
    frames = result.reshape(-1, 6, 33)
    frames[..., 1] = 0.0  # intentional lateral weight-transfer velocity
    frames[..., 3] = 0.0  # roll rate is handled by the two-leg posture controller
    frames[..., 7] = 0.0
    frames[..., 8] = -torch.sqrt((1.0 - frames[..., 6].square()).clamp_min(0.0))
    for left, right in ((13, 15), (14, 16), (17, 20), (18, 19), (31, 32)):
        mean = 0.5 * (frames[..., left] + frames[..., right])
        frames[..., left] = mean
        frames[..., right] = mean
    return result


def smoothstep(value):
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


class AlternatingWheelLift:
    """Settle -> short retract/hold/lower -> recover; alternate only after recovery."""

    def __init__(
        self,
        env,
        *,
        length=0.10,
        shortening=0.035,
        retract=0.02,
        hold=0.06,
        lower=0.12,
        settle=3.0,
        recovery=0.8,
        cycles=3,
        lean=0.012,
        clearance=0.003,
        airborne_seconds=0.04
    ):
        values = (
            length,
            shortening,
            retract,
            hold,
            lower,
            settle,
            recovery,
            clearance,
            airborne_seconds,
            lean,
        )
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("Lift dimensions and times must be positive and finite")
        if (
            length + lean > 0.20
            or min(length - lean, length + lean - shortening) < 0.06 - 1e-9
        ):
            raise ValueError("Lift targets must stay in the 6..20cm software range")
        if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 1:
            raise ValueError("cycles must be a positive integer")
        self.env, self.length, self.shortening = env, length, shortening
        self.retract, self.hold, self.lower = retract, hold, lower
        self.settle, self.recovery, self.cycles, self.lean = (
            settle,
            recovery,
            cycles,
            lean,
        )
        self.clearance, self.airborne_seconds = clearance, airborne_seconds
        self.time = 0.0
        self.steps = 0
        for value in (retract, hold, lower, settle, recovery, airborne_seconds):
            if not math.isclose(
                round(value / env.dt) * env.dt, value, abs_tol=1e-9, rel_tol=0
            ):
                raise ValueError(
                    "Lift durations must be multiples of the control period"
                )
        self.up_steps, self.hold_steps, self.down_steps = [
            round(x / env.dt) for x in (retract, hold, lower)
        ]
        self.states = [
            {
                "mode": "settle",
                "stable": 0.0,
                "start": 0.0,
                "side": 0,
                "count": 0,
                "attempt": None,
                "wait_start": 0.0,
            }
            for _ in range(env.num_envs)
        ]
        self.events, self.attempts = [], []
        self.origin_xy = env.root_states[:, :2].clone()
        self.max_displacement = 0.0
        self.nonwheel_contact_steps = 0
        self.parking_velocity = torch.zeros(env.num_envs, device=env.device)

    def _stable(self, measurements):
        e = self.env
        delta = e.root_states[:, :2] - self.origin_xy
        forward_error = delta[:, 0] * torch.cos(e.yaw) + delta[:, 1] * torch.sin(e.yaw)
        return (
            (
                (forward_error.abs() < 0.025)
                & (e.pitch.abs() < 0.05)
                & (e.roll.abs() < 0.025)
                & (torch.stack((e.length_l, e.length_r), dim=-1) - self.length)
                .abs()
                .lt(0.005)
                .all(-1)
                & (e.passive_limit_margin > 0.01).all(-1)
                & (torch.linalg.vector_norm(e.root_states[:, 7:9], dim=-1) < 0.04)
                & (e.ang_vel[:, 2].abs() < 0.1)
                & (measurements["wheel_normal_force_n"] > 5.0).all(-1)
                & ~measurements["nonwheel_contact"]
            )
            .cpu()
            .tolist()
        )

    def _finish_attempt(self, i, aborted=False):
        state = self.states[i]
        attempt = state["attempt"]
        if attempt is None:
            return
        attempt.update(
            recovered=not aborted,
            aborted=aborted,
            success=not aborted
            and not attempt["support_lost"]
            and attempt["longest_airborne_s"] + 1e-9 >= self.airborne_seconds,
        )
        self.attempts.append(attempt)
        self.events.append(dict(event="attempt_finished", **attempt))
        state["attempt"] = None
        state["count"] += 1
        state["side"] = 1 - state["side"]
        state.update(
            mode="done" if state["count"] >= 2 * self.cycles else "settle",
            stable=0.0,
            wait_start=self.time,
        )

    def before_step(self):
        e = self.env
        measurements = e.lift_measurements()
        stable = self._stable(measurements)
        lengths = torch.full((e.num_envs, 2), self.length, device=e.device)
        forces = torch.full_like(lengths, e.leg_feedforward_force)
        kp = torch.full((e.num_envs, 4), e.leg_kp, device=e.device)
        kd = torch.full_like(kp, e.leg_kd)
        for i, state in enumerate(self.states):
            mode = state["mode"]
            side = state["side"]
            if mode == "done":
                state["stable"] = state["stable"] + e.dt if stable[i] else 0.0
                continue
            if mode in ("failed", "blocked"):
                continue
            safe = abs(float(e.pitch[i])) < 0.12 and abs(float(e.roll[i])) < 0.22
            if mode == "lift" and not safe:
                state.update(
                    mode="abort_lower",
                    start=self.time,
                    abort_amount=state.get("amount", 0.0),
                )
                state["attempt"]["safety_abort"] = True
                self.events.append(dict(event="safety_return", env=i, time_s=self.time))
            if state["mode"] in ("settle", "recover"):
                state["stable"] = state["stable"] + e.dt if stable[i] else 0.0
                if (
                    state["mode"] == "recover"
                    and state["stable"] + 1e-9 >= self.recovery
                ):
                    self._finish_attempt(i, state["attempt"].get("safety_abort", False))
                elif (
                    state["mode"] == "settle"
                    and self.time >= self.settle
                    and state["stable"] + 1e-9 >= self.recovery
                ):
                    state.update(mode="preload", start=self.time, stable=0.0)
                    self.events.append(
                        dict(
                            event="weight_transfer_started",
                            env=i,
                            side=side,
                            time_s=self.time,
                        )
                    )
                elif self.time - state["wait_start"] > 20.0:
                    if state["attempt"] is not None:
                        self._finish_attempt(i, True)
                    state["mode"] = "blocked"
                    self.events.append(
                        dict(event="stability_gate_timeout", env=i, time_s=self.time)
                    )
            lean_amount = amount = 0.0
            mode = state["mode"]
            if mode == "preload":
                elapsed = self.time - state["start"]
                lean_amount = smoothstep(elapsed / 0.4)
                # Longer swing-side leg leans the body towards the support side.
                expected = (1.0 if side == 0 else -1.0) * math.atan2(
                    2 * self.lean, 0.23
                )
                ready = (
                    abs(float(e.roll[i]) - expected) < 0.05
                    and abs(float(e.pitch[i])) < 0.06
                    and abs(float(e.ang_vel[i, 0])) < 0.10
                    and abs(float(e.ang_vel[i, 1])) < 0.15
                    and float(torch.linalg.vector_norm(e.root_states[i, 7:9])) < 0.05
                    and bool((measurements["wheel_normal_force_n"][i] > 5.0).all())
                    and not bool(measurements["nonwheel_contact"][i])
                )
                state["stable"] = (
                    state["stable"] + e.dt if elapsed >= 0.4 and ready else 0.0
                )
                if state["stable"] + 1e-9 >= 0.2:
                    state.update(mode="lift", start=self.time, stable=0.0)
                    state["attempt"] = {
                        "env": i,
                        "side": "left" if side == 0 else "right",
                        "start_s": self.time,
                        "max_clearance_m": 0.0,
                        "active_clearance_peak_m": 0.0,
                        "post_lift_rebound_peak_m": 0.0,
                        "longest_airborne_s": 0.0,
                        "airborne_run_s": 0.0,
                        "support_lost": False,
                        "safety_abort": False,
                    }
                    self.events.append(
                        dict(
                            event="lift_started",
                            env=i,
                            side=state["attempt"]["side"],
                            time_s=self.time,
                        )
                    )
                elif elapsed > 2.5:
                    state.update(
                        mode="unlean", start=self.time, blocked_after_unlean=True
                    )
                    self.events.append(
                        dict(
                            event="weight_transfer_not_stable", env=i, time_s=self.time
                        )
                    )
            mode = state["mode"]
            if mode in ("lift", "abort_lower"):
                lean_amount = 1.0
                tick = round((self.time - state["start"]) / e.dt)
                if mode == "abort_lower":
                    amount = state["abort_amount"] * (
                        1.0 - smoothstep(tick / self.down_steps)
                    )
                    if tick >= self.down_steps:
                        state.update(mode="unlean", start=self.time)
                elif tick < self.up_steps:
                    amount = smoothstep(tick / self.up_steps)
                elif tick < self.up_steps + self.hold_steps:
                    amount = 1.0
                elif tick < self.up_steps + self.hold_steps + self.down_steps:
                    amount = 1.0 - smoothstep(
                        (tick - self.up_steps - self.hold_steps) / self.down_steps
                    )
                else:
                    state.update(mode="unlean", start=self.time)
                state["amount"] = amount
                swing = slice(2 * side, 2 * side + 2)
                if mode == "lift" and tick < self.up_steps + self.hold_steps:
                    kp[i, swing] = 160.0
                    kd[i, swing] = 1.5
                else:
                    kd[i, swing] = 6.0
            if state["mode"] == "unlean":
                t = self.time - state["start"]
                lean_amount = 1.0 - smoothstep(t / 0.5)
                kd[i, 2 * side : 2 * side + 2] = 6.0
                if t + 1e-9 >= 0.5:
                    state.update(
                        mode=(
                            "blocked"
                            if state.get("blocked_after_unlean", False)
                            else "recover"
                        ),
                        stable=0.0,
                        wait_start=self.time,
                    )
            lengths[i, side] += self.lean * lean_amount - self.shortening * amount
            lengths[i, 1 - side] -= self.lean * lean_amount
            swing_force = (
                e.leg_feedforward_force * (1.0 - 0.4 * lean_amount) * (1.0 - amount)
            )
            forces[i, side] = swing_force
            forces[i, 1 - side] = 2 * e.leg_feedforward_force - swing_force
        e.lift_lengths, e.lift_forces, e.lift_kp, e.lift_kd = lengths, forces, kp, kd
        # An outer position loop asks the unchanged wheel policy for small
        # forward/backward corrections. It never moves the root state directly.
        delta = e.root_states[:, :2] - self.origin_xy
        forward_error = delta[:, 0] * torch.cos(e.yaw) + delta[:, 1] * torch.sin(e.yaw)
        correction = (-0.8 * forward_error - 0.2 * e.lin_vel[:, 0]).clamp(-0.04, 0.04)
        self.parking_velocity += (e.dt / (0.1 + e.dt)) * (
            correction - self.parking_velocity
        )
        commands = e.commands.clone()
        commands[:, :3] = 0.0
        commands[:, 0] = self.parking_velocity
        commands[:, 3] = self.length
        return e.set_commands(commands)

    def after_step(self, info, dones):
        dt = self.env.dt
        self.nonwheel_contact_steps += int(info["lift_nonwheel_contact"].sum())
        displacement = torch.linalg.vector_norm(
            info["pre_reset_root_states"][:, :2] - self.origin_xy, dim=-1
        )
        self.max_displacement = max(self.max_displacement, float(displacement.max()))
        for i, state in enumerate(self.states):
            attempt = state["attempt"]
            if attempt is not None:
                side = state["side"]
                gap = float(info["lift_clearance_m"][i, side])
                contact = float(info["lift_wheel_contact_norm_n"][i, side])
                supported = float(info["lift_wheel_normal_force_n"][i, 1 - side]) > 5.0
                air = (
                    state["mode"] == "lift"
                    and not attempt["safety_abort"]
                    and gap >= self.clearance
                    and contact < 1.0
                    and supported
                    and not bool(info["lift_nonwheel_contact"][i])
                )
                attempt["max_clearance_m"] = max(attempt["max_clearance_m"], gap)
                key = (
                    "active_clearance_peak_m"
                    if state["mode"] == "lift"
                    else "post_lift_rebound_peak_m"
                )
                attempt[key] = max(attempt[key], gap)
                attempt["airborne_run_s"] = (
                    attempt["airborne_run_s"] + dt if air else 0.0
                )
                attempt["longest_airborne_s"] = max(
                    attempt["longest_airborne_s"], attempt["airborne_run_s"]
                )
                attempt["support_lost"] |= (
                    float(info["lift_clearance_m"][i, 1 - side]) >= self.clearance
                    and float(info["lift_wheel_contact_norm_n"][i, 1 - side]) < 1.0
                )
            if bool(dones[i]):
                self._finish_attempt(i, True)
                state["mode"] = "failed"
                self.events.append(
                    dict(event="physical_or_time_reset", env=i, time_s=self.time + dt)
                )
        self.steps += 1
        self.time = self.steps * dt

    def summary(self, physical_failures):
        confirmed = {
            side: sum(a["success"] and a["side"] == side for a in self.attempts)
            for side in ("left", "right")
        }
        completed = all(
            s["mode"] == "done" and s["stable"] + 1e-9 >= 2.0 for s in self.states
        )
        passed = (
            completed
            and physical_failures == 0
            and self.nonwheel_contact_steps == 0
            and self.max_displacement <= 0.03
            and all(a["success"] and not a["support_lost"] for a in self.attempts)
            and len(self.attempts) == 2 * self.cycles * self.env.num_envs
        )
        return {
            "engineering_screen_passed": passed,
            "status": (
                "short_alternating_lifts_verified" if passed else "needs_improvement"
            ),
            "confirmed_lifts": confirmed,
            "attempts": self.attempts,
            "final_stages": [s["mode"] for s in self.states],
            "physical_failure_resets": physical_failures,
            "peak_xy_displacement_m": self.max_displacement,
            "nonwheel_contact_steps": self.nonwheel_contact_steps,
            "final_settled_seconds_per_env": [s["stable"] for s in self.states],
            "required_final_settled_seconds": 2.0,
            "clearance_threshold_m": self.clearance,
            "required_continuous_airborne_s": self.airborne_seconds,
            "position_hold": {
                "method": "small proportional forward velocity correction through wheel policy",
                "forward_kp_per_s": 0.8,
                "velocity_damping": 0.2,
                "velocity_limit_m_per_s": 0.04,
                "velocity_filter_time_constant_s": 0.1,
                "settle_forward_position_tolerance_m": 0.025,
            },
            "interpretation": "Short in-place lift/recover pulses only; not sustained one-wheel standing or forward walking.",
        }
