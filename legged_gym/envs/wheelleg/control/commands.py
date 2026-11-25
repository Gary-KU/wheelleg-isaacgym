"""Command sampling and acceleration limits shared by training and playback."""

import math

import torch


def validate_command_limits(dt, linear_acceleration, yaw_acceleration):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Command dt must be finite and positive")
    for name, value in (("linear_acceleration", linear_acceleration),
                        ("yaw_acceleration", yaw_acceleration)):
        if not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(name))


def slew_commands(current, target, dt, linear_acceleration=0.0, yaw_acceleration=0.0):
    """Limit velocity changes per simulation second; zero disables a limit.

    Commands have shape [..., 4] and order [vx, vy, yaw_rate, leg_length].
    The planar limit applies to the norm of the velocity change, while yaw
    is limited independently. Leg length passes through immediately.
    """
    validate_command_limits(dt, linear_acceleration, yaw_acceleration)
    if (current.shape != target.shape or current.ndim < 1 or current.shape[-1] != 4
            or not current.is_floating_point() or not target.is_floating_point()):
        raise ValueError("Current and target commands must be floating tensors of equal shape [...,4]")
    if not bool(torch.isfinite(current).all() & torch.isfinite(target).all()):
        raise ValueError("Command values must be finite")
    result = target.clone()
    if linear_acceleration > 0:
        delta = target[..., :2] - current[..., :2]
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(linear_acceleration * dt / norm.clamp_min(1e-12), max=1.0)
        result[..., :2] = current[..., :2] + delta * scale
    if yaw_acceleration > 0:
        delta = (target[..., 2] - current[..., 2]).clamp(
            -yaw_acceleration * dt, yaw_acceleration * dt)
        result[..., 2] = current[..., 2] + delta
    return result


class CommandGovernor:
    """Stateful wrapper for the same velocity limits used by the environment."""

    def __init__(self, initial_commands, linear_acceleration=0.0, yaw_acceleration=0.0):
        validate_command_limits(1.0, linear_acceleration, yaw_acceleration)
        self.linear_acceleration = float(linear_acceleration)
        self.yaw_acceleration = float(yaw_acceleration)
        self.commands = None
        self.target_commands = None
        self.reset(initial_commands)

    def reset(self, commands):
        if self.commands is None:
            value = torch.as_tensor(commands)
            if not value.is_floating_point():
                value = value.to(dtype=torch.float32)
        else:
            value = torch.as_tensor(commands, dtype=self.commands.dtype, device=self.commands.device)
            if value.shape != self.commands.shape:
                raise ValueError("Reset commands must preserve the governor's shape")
        if value.ndim < 1 or value.shape[-1] != 4 or not bool(torch.isfinite(value).all()):
            raise ValueError("Commands must be finite with shape [...,4]")
        self.commands = value.clone()
        self.target_commands = value.clone()
        return self.commands

    def step(self, target_commands, dt):
        target = torch.as_tensor(target_commands, dtype=self.commands.dtype, device=self.commands.device)
        updated = slew_commands(self.commands, target, dt,
                                self.linear_acceleration, self.yaw_acceleration)
        self.target_commands = target.clone()
        self.commands = updated
        return self.commands


def sample_command_targets(count, device, forward_range, yaw_range, leg_range,
                           zero_fraction=0.0):
    """Sample motion targets, optionally mixing in exact stationary commands."""
    if not 0 <= zero_fraction <= 1 or not math.isfinite(zero_fraction):
        raise ValueError("zero_fraction must be in [0,1]")
    if count < 0:
        raise ValueError("Command count must be nonnegative")
    result = torch.zeros(count, 4, device=device)
    for column, bounds in ((0, forward_range), (2, yaw_range), (3, leg_range)):
        low, high = bounds
        result[:, column] = low + (high-low)*torch.rand(count, device=device)
    if zero_fraction > 0:
        stopped = torch.rand(count, device=device) < zero_fraction
        result[stopped, :3] = 0.0
    return result
