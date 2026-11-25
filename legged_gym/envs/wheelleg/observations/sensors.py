"""MCU-obtainable planar state contract, independent of Isaac Gym.

Angles are radians; lengths metres; rates per second. Positive pitch is nose
down, positive yaw is left, and both virtual-leg angles are forward of down.
This is a new layout, not padding for a legacy 198-input checkpoint.
"""

import math

import torch
from legged_gym.envs.wheelleg.model.geometry import UPPER, LOWER, HIP_SPAN, HIP_DOWN


SENSOR_SCHEMA = "mylegwheel_planar_v1"
SENSOR_FIELDS = (
    "pitch",
    "pitch_rate",
    "yaw_rate",
    "leg_angle_left",
    "leg_angle_right",
    "leg_length_left",
    "leg_length_right",
    "leg_angle_rate_left",
    "leg_angle_rate_right",
    "leg_length_rate_left",
    "leg_length_rate_right",
    "wheel_forward_rate_left",
    "wheel_forward_rate_right",
)
FRAME_FIELDS = SENSOR_FIELDS + (
    "command_vx",
    "command_yaw_rate",
    "command_leg_length",
    "previous_action_lf",
    "previous_action_lh",
    "previous_action_rf",
    "previous_action_rh",
    "previous_action_lwheel",
    "previous_action_rwheel",
)
FRAME_SIZE = len(FRAME_FIELDS)
HISTORY_FRAMES = 6
OBSERVATION_SIZE = FRAME_SIZE * HISTORY_FRAMES
SENSOR_SCALES = (1.0, 0.5, 0.5, 1.0, 1.0, 5.0, 5.0, 0.05, 0.05, 5.0, 5.0, 0.05, 0.05)
COMMAND_SLICE = slice(13, 16)


def build_sensor_frame(sensors, commands, previous_actions):
    """The same normalization is usable by simulation and a future MCU adapter."""
    n = sensors.shape[0]
    if (
        sensors.shape != (n, 13)
        or commands.shape != (n, 4)
        or previous_actions.shape != (n, 6)
    ):
        raise ValueError("Expected sensors[N,13], commands[N,4], actions[N,6]")
    return torch.cat(
        (
            sensors * sensors.new_tensor(SENSOR_SCALES),
            commands[:, [0, 2, 3]] * commands.new_tensor([1.5, 1.0, 5.0]),
            previous_actions,
        ),
        dim=-1,
    )


def from_mylegwheel(
    snapshot, *, wheel_directions=(-1.0, 1.0), yaw_right_direction=-1.0
):
    """Convert a coherent WheelLegDebug-style snapshot to the 13 SI fields.

    Tensor fields: body_pitch_rad[N], body_pitch_rate_rad_s[N], gyro_rad_s[N,3],
    leg_angle_rad/leg_length_m/angle_speed_rad_s/length_speed_m_s[N,2],
    wheel_speed_rad_s[N,2], imu_valid[N], leg_valid/wheel_valid[N,2]. Joint
    calibration and freshness checks belong to the producer. This does not decode the incomplete robot24 wire
    frame, and never substitutes zero for an unavailable field.
    """
    if (
        any(x not in (-1.0, 1.0) for x in (*wheel_directions, yaw_right_direction))
        or len(wheel_directions) != 2
    ):
        raise ValueError("Calibrated directions must be +/-1")
    p = snapshot["body_pitch_rad"]
    n = len(p)
    for name, shape in (
        ("imu_valid", (n,)),
        ("leg_valid", (n, 2)),
        ("wheel_valid", (n, 2)),
    ):
        if snapshot[name].shape != shape or not bool((snapshot[name] == 1).all()):
            raise ValueError("Invalid MCU feedback: " + name)
    for name, shape in (
        ("body_pitch_rad", (n,)),
        ("body_pitch_rate_rad_s", (n,)),
        ("gyro_rad_s", (n, 3)),
        ("leg_angle_rad", (n, 2)),
        ("leg_length_m", (n, 2)),
        ("angle_speed_rad_s", (n, 2)),
        ("length_speed_m_s", (n, 2)),
        ("wheel_speed_rad_s", (n, 2)),
    ):
        value = snapshot[name]
        if value.shape != shape or not bool(torch.isfinite(value).all()):
            raise ValueError("Missing, invalid or nonfinite MCU state: " + name)
    return torch.cat(
        (
            p[:, None],
            snapshot["body_pitch_rate_rad_s"][:, None],
            -yaw_right_direction * snapshot["gyro_rad_s"][:, 2:3],
            math.pi / 2 - snapshot["leg_angle_rad"],
            snapshot["leg_length_m"],
            -snapshot["angle_speed_rad_s"],
            snapshot["length_speed_m_s"],
            snapshot["wheel_speed_rad_s"] * p.new_tensor(wheel_directions),
        ),
        dim=-1,
    )


def encoder_leg_state(joint_position, joint_velocity):
    """URDF encoder FK and analytic Jacobian, with both sides forward-positive.

    Simulation math X points backward on the left and forward on the right;
    MCU leg_model uses X forward, Y down on both sides. No body/world pose or
    velocity is used here. Geometry is the CURRENT simulation asset's geometry.
    """
    phi1 = 3 * math.pi / 4 + joint_position[:, [0, 3]]
    phi4 = math.pi / 4 + joint_position[:, [1, 2]]
    rate1, rate4 = joint_velocity[:, [0, 3]], joint_velocity[:, [1, 2]]
    bx, by = UPPER * torch.cos(phi1), UPPER * torch.sin(phi1)
    dx, dy = HIP_SPAN + UPPER * torch.cos(phi4), HIP_DOWN + UPPER * torch.sin(phi4)
    ux, uy = dx - bx, dy - by
    d2 = ux.square() + uy.square()
    h = (LOWER**2 - 0.25 * d2).clamp_min(1e-12).sqrt()
    d = d2.clamp_min(1e-12).sqrt()
    cx, cy = 0.5 * (bx + dx) - h * uy / d, 0.5 * (by + dy) + h * ux / d
    x = cx - (HIP_SPAN/2)
    cy_centered = cy-HIP_DOWN/2
    length2 = x.square() + cy_centered.square()
    length = length2.sqrt()
    r1x, r1y, r4x, r4y = cx - bx, cy - by, cx - dx, cy - dy
    det = r1x * r4y - r1y * r4x
    valid = (d2 > 1e-10) & (d2 < 4 * LOWER**2) & (det.abs() > 1e-8) & (length2 > 1e-10)
    den = torch.where(valid, det, torch.ones_like(det))
    q1 = UPPER * (-r1x * torch.sin(phi1) + r1y * torch.cos(phi1))
    q4 = UPPER * (-r4x * torch.sin(phi4) + r4y * torch.cos(phi4))
    vx = (r4y * q1 * rate1 - r1y * q4 * rate4) / den
    vy = (-r4x * q1 * rate1 + r1x * q4 * rate4) / den
    sign = joint_position.new_tensor([1.0, -1.0])
    angle = (torch.atan2(cy_centered, x) - math.pi / 2) * sign
    length_rate = (x * vx + cy_centered * vy) / length.clamp_min(1e-8)
    angle_rate = (x * vy - cy_centered * vx) / length2.clamp_min(1e-10) * sign
    values = torch.stack((angle, length, angle_rate, length_rate), -1)
    return torch.where(valid[..., None], values, torch.full_like(values, float("nan")))


class SensorSampler:
    """Per-episode bias, white noise and a fixed latency; reset only selected rows."""

    def __init__(self, n, device, delay_steps, noise_std, bias_std, seed):
        if (
            isinstance(delay_steps, bool)
            or not isinstance(delay_steps, int)
            or not 0 <= delay_steps <= 10
        ):
            raise ValueError("Sensor delay_steps must be an integer in 0..10")
        for name, values in (("noise_std", noise_std), ("bias_std", bias_std)):
            if len(values) != 13 or any(not math.isfinite(x) or x < 0 for x in values):
                raise ValueError(name + " must contain 13 finite nonnegative values")
        self.noise = torch.tensor(noise_std, device=device)
        self.bias_std = torch.tensor(bias_std, device=device)
        self.queue = torch.zeros(n, delay_steps + 1, 13, device=device)
        self.bias = torch.zeros(n, 13, device=device)
        self.sample = torch.zeros_like(self.bias)
        self.generator = torch.Generator(device=device).manual_seed(seed + 193)

    def _normal(self, shape):
        return torch.randn(shape, device=self.sample.device, generator=self.generator)

    def reset(self, ids, truth):
        self.bias[ids] = self._normal((len(ids), 13)) * self.bias_std
        value = truth[ids] + self.bias[ids] + self._normal((len(ids), 13)) * self.noise
        self.queue[ids] = value[:, None]
        self.sample[ids] = value

    def advance(self, truth):
        self.queue[:, :-1] = self.queue[:, 1:].clone()
        self.queue[:, -1] = truth + self.bias + self._normal(truth.shape) * self.noise
        self.sample.copy_(self.queue[:, 0])
