"""Stair-course rewards: locomotion progress instead of stationary support.

Call this only for the stair task. Geometry, course-success detection and
termination remain the environment's responsibility; the flat task is unchanged.
"""

import math

import torch


def stair_reward_terms(delta_x, lin_vel, commands, projected_gravity, ang_vel,
                       applied_torques, previous_torques, passive_margin,
                       base_contact, stalled, success, terminated, dt=0.02):
    """Return separate reward vectors of shape ``[N]``.

    ``delta_x`` is signed course-axis displacement in metres, measured before
    any automatic reset. It must exclude reset teleports. Its progress reward
    is linear, so back-and-forth motion with zero net displacement earns zero
    net progress reward at a fixed control period. Clipping velocity would
    break that property when the two directions have different speeds.

    ``lin_vel`` is the wheel-center mean world-XY displacement per control
    period divided by dt (zero Z), aligned with the straight course and its
    planar commands. Angular velocity still uses body coordinates. Commands
    follow ``[vx, vy, yaw_rate, leg_length]``, and torque inputs are actual applied
    Nm after clipping. ``passive_margin`` is the distance in radians to the
    nearest passive-joint hard limit. Contact, stall, success and termination
    flags are boolean ``[N]`` tensors. The caller must supply disjoint success
    and physical-termination flags and make success a one-time episode event.

    There is deliberately no strict leg-length or left/right symmetry cost:
    the robot may change leg shape while crossing a stair edge.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Control dt must be positive and finite")
    if not isinstance(lin_vel, torch.Tensor) or lin_vel.ndim != 2 or lin_vel.shape[1] != 3 or lin_vel.shape[0] < 1:
        raise ValueError("lin_vel must have shape [N,3] with N >= 1")
    size = lin_vel.shape[0]
    inputs = {
        "delta_x": (delta_x, (size,)),
        "lin_vel": (lin_vel, (size, 3)),
        "commands": (commands, (size, 4)),
        "projected_gravity": (projected_gravity, (size, 3)),
        "ang_vel": (ang_vel, (size, 3)),
    }
    if (not isinstance(applied_torques, torch.Tensor) or applied_torques.ndim != 2
            or applied_torques.shape[0] != size or applied_torques.shape[1] < 1):
        raise ValueError("applied_torques must have shape [N,J] with J >= 1")
    inputs["applied_torques"] = (applied_torques, applied_torques.shape)
    inputs["previous_torques"] = (previous_torques, applied_torques.shape)
    if (not isinstance(passive_margin, torch.Tensor) or passive_margin.ndim != 2
            or passive_margin.shape[0] != size or passive_margin.shape[1] < 1):
        raise ValueError("passive_margin must have shape [N,P] with P >= 1")
    inputs["passive_margin"] = (passive_margin, passive_margin.shape)
    for name, (value, shape) in inputs.items():
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or not value.is_floating_point() or value.device != lin_vel.device):
            raise ValueError("{} must be a floating tensor of shape {} on the velocity device".format(name, tuple(shape)))
    flags = {"base_contact": base_contact, "stalled": stalled,
             "success": success, "terminated": terminated}
    for name, value in flags.items():
        if (not isinstance(value, torch.Tensor) or value.shape != (size,)
                or value.dtype != torch.bool or value.device != lin_vel.device):
            raise ValueError("{} must be a boolean tensor of shape [N] on the velocity device".format(name))

    return {
        "progress": (9.0/dt)*delta_x,
        "velocity": 4.0*torch.exp(-(lin_vel[:, :2]-commands[:, :2]).square().sum(-1)/0.04),
        "yaw": torch.exp(-(ang_vel[:, 2]-commands[:, 2]).square()/0.2),
        "posture": 0.5*torch.exp(-projected_gravity[:, :2].square().sum(-1)/0.09),
        "angular_damping": -0.1*ang_vel[:, :2].square().sum(-1),
        "stall": -2.0*stalled.to(lin_vel.dtype),
        "base_contact": -5.0*base_contact.to(lin_vel.dtype),
        "passive_limit": -4.0*(1.0-passive_margin/0.1).clamp(0.0, 1.0).square().mean(-1),
        "effort": -0.0001*applied_torques.square().sum(-1),
        "torque_smoothness": -0.05*((applied_torques-previous_torques)/9.0).square().sum(-1),
        "failure": -100.0*terminated.to(lin_vel.dtype),
        "success": 25.0*success.to(lin_vel.dtype),
    }
