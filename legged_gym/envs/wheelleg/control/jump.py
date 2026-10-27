"""Manual standing and rolling jumps; all motion comes from capped joint torques.

The wheel actor is an existing standing policy. Jumping is a leg trajectory,
not a newly trained jumping policy. No root states are changed by this module.
"""

import math

import torch

from ..keyboard import KeyboardDrive
from ..constants import ACTION_SCALES, MAX_LEG_LENGTH


class JumpKeyboard(KeyboardDrive):
    KEYS = dict(KeyboardDrive.KEYS, drive_stop="KEY_X", jump="KEY_SPACE")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jump_down = False
        self.jump_requested = False

    def process_events(self, events):
        other = []
        for event in events:
            if event.action == "jump":
                down = event.value > 0
                if down and not self.jump_down:
                    self.jump_requested = True
                self.jump_down = down
            else:
                other.append(event)
        return super().process_events(other)

    def consume_jump(self):
        requested, self.jump_requested = self.jump_requested, False
        return requested


class JumpController:
    """Idle -> crouch -> hold -> push -> flight -> land -> idle.

    Landing completes the action quickly; recovery is verified in idle without
    holding driver/height commands or claiming that contact proves stability.

    Manual requests do not require readiness. An executing launch cannot
    restart in midair; after touchdown a new press can skip recovery. Both
    clearance and contact loss are required to count a real airborne interval.
    """

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.jump, self.parking = cfg.jump, cfg.parking
        j = self.jump
        if not 0.06 <= j.crouch_length < j.extension_length <= 0.20:
            raise ValueError("Jump leg targets must stay inside 6..20 cm")
        for name in (
            "crouch_seconds",
            "crouch_hold_seconds",
            "push_seconds",
            "stable_seconds",
            "landing_contact_seconds",
        ):
            value = getattr(j, name)
            if (
                not math.isfinite(value)
                or value <= 0
                or not math.isclose(round(value / env.dt) * env.dt, value, abs_tol=1e-9)
            ):
                raise ValueError(name + " must be a positive multiple of control dt")
        self.events, self.attempts = [], []
        self.time = 0.0
        self.steps = 0
        self.reset()

    def reset(self):
        e = self.env
        self.anchor_xy = e.root_states[:, :2].clone()
        self.parking_velocity = torch.zeros(e.num_envs, device=e.device)
        self.command_velocity = torch.zeros(e.num_envs, 3, device=e.device)
        self.was_driving = torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
        self.braking = torch.zeros_like(self.was_driving)
        self.requested = e.commands.clone()
        self.requested[:, :3] = 0.0
        self.states = [
            dict(mode="idle", stable=0.0, start=self.time, attempt=None)
            for _ in range(e.num_envs)
        ]
        e.lift_lengths = e.lift_forces = None

    @property
    def can_jump(self):
        """Allow a manual launch without speed, posture or settling gates."""
        return all(s["mode"] in ("idle", "land", "blocked") for s in self.states)

    def request_jump(self):
        if not self.can_jump:
            self.events.append(
                dict(
                    event="jump_ignored",
                    time_s=self.time,
                    reason="Launch already executing; requests are not queued.",
                )
            )
            return False
        for i, state in enumerate(self.states):
            if state["attempt"] is not None:
                # Keep the previous flight evidence; a manual restart does
                # not certify that its landing recovery passed.
                self._finish(i, "restarted_before_recovery")
            nominal = float(self.requested[i, 3])
            offset = nominal - self.jump.reference_length
            state.update(
                mode="crouch",
                start=self.time,
                landed_for=0.0,
                stable=0.0,
                length=nominal,
                rolling=bool(self.requested[i, :3].abs().max() > 1e-4),
                origin_xy=self.env.root_states[i, :2].clone(),
                origin_yaw=float(self.env.yaw[i]),
                # Capture the already applied ramp, not the eventual key target:
                # a jump during acceleration must not keep accelerating while crouched.
                launch_command=self.command_velocity[i].clone(),
                crouch_length=max(0.06, self.jump.crouch_length + offset),
                extension_length=min(0.20, self.jump.extension_length + offset),
            )
            state["attempt"] = dict(
                env=i,
                start_s=self.time,
                standing_z=float(self.env.root_states[i, 2]),
                peak_z=float(self.env.root_states[i, 2]),
                peak_clearance_m=0.0,
                airborne_run_s=0.0,
                longest_airborne_s=0.0,
                airborne_seen=False,
                peak_pitch_rad=0.0,
                peak_roll_rad=0.0,
                nonwheel_contact=False,
                peak_xy_displacement_m=0.0,
                peak_lateral_displacement_m=0.0,
                requested_velocity=self.requested[i, :3].tolist(),
                rolling=state["rolling"],
                max_upward_velocity=0.0,
                airborne_travel_m=0.0,
                takeoff_velocity_m_s=None,
                touchdown_velocity_m_s=None,
                touchdown_s=None,
                action_complete_s=None,
            )
        self.events.append(dict(event="jump_started", time_s=self.time))
        return True

    def _stable(self, lengths, normal, nonwheel):
        e = self.env
        desired = self.requested[:, 3:4].clone()
        for i, state in enumerate(self.states):
            if state["mode"] in ("crouch", "hold", "push", "flight"):
                desired[i] = state["length"]
        moving = self.requested[:, :3].abs().max(-1).values > 1e-4
        stationary = (
            (torch.linalg.vector_norm(e.root_states[:, 7:9], dim=-1) < 0.025)
            & (e.lin_vel[:, 1].abs() < 0.002)
            & (
                torch.linalg.vector_norm(e.root_states[:, :2] - self.anchor_xy, dim=-1)
                < 0.03
            )
            & (e.dof_state[:, e.action_indices[4:], 1].abs().max(-1).values < 0.35)
        )
        # Readiness for the automatic test and measured landing recovery.
        # Manual space presses do not depend on these thresholds.
        rolling = (
            ((e.lin_vel[:, 0] - self.requested[:, 0]).abs() < 0.08)
            & (e.lin_vel[:, 1].abs() < 0.025)
            & (self.requested[:, 0].abs() <= 0.60001)
            & (self.requested[:, 1].abs() < 1e-4)
            & (self.requested[:, 2].abs() < 1e-4)
            & (
                (self.command_velocity - self.requested[:, :3]).abs().max(-1).values
                < 0.03
            )
        )
        return (
            (e.pitch.abs() < torch.where(moving, 0.06, 0.03))
            & (e.roll.abs() < 0.025)
            & (e.ang_vel[:, :2].abs().max(-1).values < 0.20)
            & (e.ang_vel[:, 2].abs() < 0.10)
            & torch.where(moving, rolling, stationary)
            & ((lengths - desired).abs().max(-1).values < 0.005)
            & (normal > 5.0).all(-1)
            & ~nonwheel
        )

    def _finish(self, i, failure=None):
        state = self.states[i]
        attempt = state["attempt"]
        if attempt is not None:
            attempt = dict(attempt)
            attempt.update(
                finish_s=self.time,
                body_rise_m=attempt["peak_z"] - attempt["standing_z"],
                recovered=failure is None,
                failure=failure,
            )
            attempt["success"] = (
                failure is None
                and not attempt["nonwheel_contact"]
                and attempt["longest_airborne_s"] + 1e-9
                >= self.jump.minimum_airborne_seconds
                and attempt["body_rise_m"] >= self.jump.minimum_body_rise
                and attempt["max_upward_velocity"] > 0.15
                and (
                    attempt["peak_lateral_displacement_m"]
                    if attempt["rolling"]
                    else attempt["peak_xy_displacement_m"]
                )
                <= 0.06
                and attempt["peak_pitch_rad"] < 0.20
                and attempt["peak_roll_rad"] < 0.10
            )
            self.attempts.append(attempt)
            self.events.append(dict(event="jump_finished", **attempt))
        state.update(
            attempt=None,
            mode="idle" if failure is None else "blocked",
            stable=0.0,
            start=self.time,
        )

    def before_step(self, requested=None):
        e, j = self.env, self.jump
        if requested is not None:
            value = torch.as_tensor(requested, device=e.device, dtype=torch.float32)
            if value.shape == (4,):
                value = value[None].expand(e.num_envs, -1)
            if value.shape != self.requested.shape or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError("Expected finite [N,4] or [4] user commands")
            if bool(((value[:, 3] < 0.08) | (value[:, 3] > MAX_LEG_LENGTH)).any()):
                raise ValueError("Jump demo standing targets must be within 8..20 cm")
            self.requested.copy_(value)
        m = e.lift_measurements()
        lengths = torch.stack((e.length_l, e.length_r), -1)
        stable = self._stable(lengths, m["wheel_normal_force_n"], m["nonwheel_contact"])
        targets = self.requested[:, 3:4].expand(-1, 2).clone()
        forces = torch.full_like(targets, e.leg_feedforward_force)
        for i, state in enumerate(self.states):
            mode = state["mode"]
            if mode == "idle":
                state["stable"] = state["stable"] + e.dt if bool(stable[i]) else 0.0
                if state["attempt"] is not None:
                    if state["stable"] + 1e-9 >= j.stable_seconds:
                        self._finish(i)
                    elif self.time - state["start"] > j.recovery_timeout:
                        self._finish(i, "landing_recovery_timeout")
                continue
            if mode == "blocked":
                continue
            age = self.time - state["start"]
            nominal = state["length"]
            if mode != "land":
                targets[i] = nominal
            durations = dict(
                crouch=j.crouch_seconds, hold=j.crouch_hold_seconds, push=j.push_seconds
            )
            next_mode = dict(crouch="hold", hold="push", push="flight")
            if mode in durations and age + 1e-9 >= durations[mode]:
                mode = next_mode[mode]
                state.update(mode=mode, start=self.time)
                age = 0.0
            if mode == "crouch":
                x = min(1.0, age / j.crouch_seconds)
                blend = x * x * (3 - 2 * x)
                targets[i] = nominal + (state["crouch_length"] - nominal) * blend
            elif mode == "hold":
                targets[i] = state["crouch_length"]
            elif mode == "push":
                x = min(1.0, age / j.push_seconds)
                targets[i] = (
                    state["crouch_length"]
                    + (state["extension_length"] - state["crouch_length"]) * x
                )
                forces[i] = j.push_force
            elif mode == "flight":
                forces[i] = 0.0
                if age > j.flight_timeout:
                    state.update(mode="land", start=self.time)
                    self.events.append(
                        dict(event="flight_timeout", env=i, time_s=self.time)
                    )
            elif mode == "land":
                # Height commands are live from first touchdown. Completing
                # the action requires sustained two-wheel support, not zero
                # speed, small pitch or a long stability verification period.
                supported = bool(
                    (m["wheel_normal_force_n"][i] > 5.0).all()
                ) and not bool(m["nonwheel_contact"][i])
                state["landed_for"] = (
                    state.get("landed_for", 0.0) + e.dt if supported else 0.0
                )
                if (
                    state["landed_for"] + 1e-9 >= j.landing_contact_seconds
                    and age + 1e-9 >= j.landing_contact_seconds
                ):
                    state["mode"] = "idle"
                    state["stable"] = 0.0
                    state["attempt"]["action_complete_s"] = self.time
                    self.events.append(
                        dict(event="jump_completed", env=i, time_s=self.time)
                    )
                elif age > j.recovery_timeout:
                    self._finish(i, "landing_contact_timeout")
        # Position keeping changes velocity targets, never body states.
        moving = self.requested[:, :3].abs().max(-1).values > 1e-4
        # A released key initiates a real braking interval. Hold the resulting
        # stopping position rather than pulling back to the key-release point.
        self.braking |= self.was_driving & ~moving
        still_moving = (
            torch.linalg.vector_norm(e.root_states[:, 7:9], dim=-1) > 0.03
        ) | (self.command_velocity.abs().max(-1).values > 0.03)
        self.braking &= still_moving
        rolling_attempt = torch.tensor(
            [s.get("rolling", False) and s["mode"] == "land" for s in self.states],
            device=e.device,
        )
        follow = moving | self.braking | rolling_attempt
        self.anchor_xy[follow] = e.root_states[follow, :2]
        self.was_driving.copy_(moving)
        delta = e.root_states[:, :2] - self.anchor_xy
        error = delta[:, 0] * torch.cos(e.yaw) + delta[:, 1] * torch.sin(e.yaw)
        p = self.parking
        correction = (-p.position_kp * error - p.velocity_kd * e.lin_vel[:, 0]).clamp(
            -p.maximum_correction, p.maximum_correction
        )
        self.parking_velocity += (
            e.dt / (p.filter_seconds + e.dt) * (correction - self.parking_velocity)
        )
        goal = self.requested[:, :3].clone()
        goal[~moving] = 0.0
        goal[~moving, 0] = self.parking_velocity[~moving]
        # The launch lasts less than a second. Keep its velocity target
        # continuous if a key changes mid-crouch or in flight; apply the newest
        # driving/stop request as soon as a wheel touches down.
        for i, state in enumerate(self.states):
            if state["attempt"] is not None:
                state["attempt"]["rolling"] |= bool(
                    self.requested[i, :3].abs().max() > 1e-4
                )
                if state["mode"] in ("crouch", "hold", "push", "flight"):
                    goal[i] = state["launch_command"]
        limit = (
            goal.new_tensor(
                [p.linear_acceleration, p.linear_acceleration, p.yaw_acceleration]
            )
            * e.dt
        )
        self.command_velocity += torch.maximum(
            -limit, torch.minimum(limit, goal - self.command_velocity)
        )
        commands = self.requested.clone()
        commands[:, :3] = self.command_velocity
        for i, state in enumerate(self.states):
            if state["mode"] in ("crouch", "hold", "push", "flight"):
                commands[i, 3] = state["length"]
        e.lift_lengths, e.lift_forces = targets, forces
        return e.set_commands(commands)

    def policy_actions(self, policy, observations):
        actions = policy.act_inference(observations).clone()
        actions[:, :4] = 0.0
        # Bounded wheel reaction torques control pitch in flight; do not run a
        # ground-trained wheel policy against freely spinning airborne wheels.
        for i, state in enumerate(self.states):
            if state["mode"] == "flight":
                j = self.jump
                drive = (
                    j.air_pitch_kp * self.env.pitch[i]
                    + j.air_pitch_kd * self.env.ang_vel[i, 1]
                ).clamp(-j.air_wheel_torque_limit, j.air_wheel_torque_limit)
                actions[i, 4] = -drive / ACTION_SCALES[4]
                actions[i, 5] = drive / ACTION_SCALES[5]
        return actions

    def after_step(self, info, dones):
        e, j = self.env, self.jump
        self.steps += 1
        self.time = self.steps * e.dt
        for i, state in enumerate(self.states):
            attempt = state["attempt"]
            if attempt is not None:
                root = info["pre_reset_root_states"][i]
                gap = float(info["lift_clearance_m"][i].min())
                free = bool((info["lift_wheel_contact_norm_n"][i] < 1.0).all())
                air = (
                    gap >= j.clearance
                    and free
                    and not bool(info["lift_nonwheel_contact"][i])
                )
                attempt["airborne_run_s"] = (
                    attempt["airborne_run_s"] + e.dt if air else 0.0
                )
                attempt["longest_airborne_s"] = max(
                    attempt["longest_airborne_s"], attempt["airborne_run_s"]
                )
                if air:
                    if not attempt["airborne_seen"]:
                        attempt["takeoff_velocity_m_s"] = root[7:10].tolist()
                    # Integrate only measured simultaneous contact-free flight.
                    attempt["airborne_travel_m"] += (
                        float(torch.linalg.vector_norm(root[7:9])) * e.dt
                    )
                attempt["airborne_seen"] |= air
                attempt["peak_z"] = max(attempt["peak_z"], float(root[2]))
                attempt["peak_clearance_m"] = max(attempt["peak_clearance_m"], gap)
                attempt["peak_pitch_rad"] = max(
                    attempt["peak_pitch_rad"], abs(float(info["pre_reset_pitch"][i]))
                )
                attempt["peak_roll_rad"] = max(
                    attempt["peak_roll_rad"], abs(float(info["pre_reset_roll"][i]))
                )
                attempt["max_upward_velocity"] = max(
                    attempt["max_upward_velocity"], float(root[9])
                )
                attempt["nonwheel_contact"] |= bool(info["lift_nonwheel_contact"][i])
                attempt["peak_xy_displacement_m"] = max(
                    attempt["peak_xy_displacement_m"],
                    float(torch.linalg.vector_norm(root[:2] - state["origin_xy"])),
                )
                delta = root[:2] - state["origin_xy"]
                lateral = -delta[0] * math.sin(state["origin_yaw"]) + delta[
                    1
                ] * math.cos(state["origin_yaw"])
                attempt["peak_lateral_displacement_m"] = max(
                    attempt["peak_lateral_displacement_m"], abs(float(lateral))
                )
                contact = bool((info["lift_wheel_contact_norm_n"][i] > 5.0).any())
                if (
                    state["mode"] in ("flight", "push")
                    and attempt["airborne_seen"]
                    and contact
                ):
                    attempt["touchdown_s"] = self.time
                    attempt["touchdown_velocity_m_s"] = root[7:10].tolist()
                    state.update(mode="land", start=self.time, stable=0.0)
                    self.events.append(dict(event="touchdown", env=i, time_s=self.time))
            if bool(dones[i]):
                self._finish(i, "physical_or_time_reset")
