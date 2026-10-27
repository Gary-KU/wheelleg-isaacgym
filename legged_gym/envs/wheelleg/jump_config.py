"""Standing and rolling physical jumps with the existing wheel policy and explicit leg targets."""

from .wheelleg_config import WheelLeggedStandCfg


class WheelLeggedJumpCfg(WheelLeggedStandCfg):
    class env(WheelLeggedStandCfg.env):
        initial_leg_length = 0.10

    class sim(WheelLeggedStandCfg.sim):
        physics_threads = 0
        physics_solver = "pgs"

    class asset(WheelLeggedStandCfg.asset):
        wheel_friction = 2.0

    class commands(WheelLeggedStandCfg.commands):
        command = (0.0, 0.0, 0.0, 0.10)

    class jump:
        # Targets are metres; support/thrust are radial forces per leg in N.
        reference_length = 0.10
        crouch_length = 0.065
        extension_length = 0.19
        crouch_seconds = 0.40
        crouch_hold_seconds = 0.20
        push_seconds = 0.10
        push_force = 65.0
        landing_contact_seconds = 0.10
        stable_seconds = 1.2  # Report verification only; does not lock controls.
        recovery_timeout = 10.0
        flight_timeout = 0.8
        clearance = 0.003
        minimum_airborne_seconds = 0.04
        minimum_body_rise = 0.01
        air_pitch_kp = 20.0
        air_pitch_kd = 2.0
        air_wheel_torque_limit = 3.0

    class parking:
        position_kp = 1.2
        velocity_kd = 0.35
        maximum_correction = 0.08
        filter_seconds = 0.10
        linear_acceleration = 0.8
        yaw_acceleration = 4.0
