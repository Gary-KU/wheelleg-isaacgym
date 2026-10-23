# SPDX-License-Identifier: BSD-3-Clause
"""Optional tracking and stability penalties for the driving training stage."""

import math

import torch


def _length_vector(length, batch_size, name):
    if length.shape == (batch_size, 1):
        return length[:, 0]
    if length.shape != (batch_size,):
        raise ValueError("{} must have shape [N] or [N,1]".format(name))
    return length


def tracking_error_penalty(lin_vel, commands, length_l, length_r):
    """Return a dense nonpositive per-environment tracking term.

    Penalize planar linear-velocity error by 4 and the sum of squared left/right
    leg-length errors, normalized by 0.05 m, by 0.25. ``commands`` follows the
    task's [vx, vy, yaw_rate, leg_length] layout. Vertical velocity and yaw-rate
    terms stay in the task's existing reward rather than entering this helper.

    Inputs are tensors with batch-first shapes [N,3], [N,4], and [N] (or [N,1]
    for the lengths). The result has shape [N], including for a single robot.
    """
    if lin_vel.ndim != 2 or lin_vel.shape[1] != 3:
        raise ValueError("lin_vel must have shape [N,3]")
    batch_size = lin_vel.shape[0]
    if commands.shape != (batch_size, 4):
        raise ValueError("commands must have shape [N,4]")
    length_l = _length_vector(length_l, batch_size, "length_l")
    length_r = _length_vector(length_r, batch_size, "length_r")
    velocity_error = (lin_vel[:, :2]-commands[:, :2]).square().sum(dim=-1)
    target_length = commands[:, 3]
    leg_error = ((length_l-target_length)/0.05).square()
    leg_error = leg_error+((length_r-target_length)/0.05).square()
    return -4.0*velocity_error-0.25*leg_error


def stability_penalty(projected_gravity, angular_velocity, applied_torques,
                      previous_applied_torques, torque_scale=9.0):
    """Penalize body tilt, roll/pitch motion and physical torque changes.

    Gravity is the unit gravity direction in the body frame, shape [N,3].
    Angular velocity is in body-frame rad/s with the same shape. The two
    torque tensors must have matching [N,J] shapes and contain the actual
    clipped joint torques in Nm, not the policy's differently scaled actions.

    The per-environment term is -4*|g_xy|^2 -0.1*|omega_xy|^2
    -0.2*|(tau-tau_previous)/torque_scale|^2. Yaw rotation is intentionally
    excluded so commanded turning remains possible. A constant nonzero
    holding torque receives no torque-change cost. With the default 9 Nm
    scale, a +9 to -9 Nm step costs 0.8 for any joint, including a wheel.

    Read the previous torques before updating that buffer; reset its entries
    to zero whenever the corresponding environment resets. Returns [N].
    """
    if projected_gravity.ndim != 2 or projected_gravity.shape[1] != 3:
        raise ValueError("projected_gravity must have shape [N,3]")
    batch_size = projected_gravity.shape[0]
    if angular_velocity.shape != (batch_size, 3):
        raise ValueError("angular_velocity must have shape [N,3]")
    if (applied_torques.ndim != 2 or applied_torques.shape[0] != batch_size
            or applied_torques.shape[1] < 1):
        raise ValueError("applied_torques must have shape [N,J] with J >= 1")
    if previous_applied_torques.shape != applied_torques.shape:
        raise ValueError("previous_applied_torques must match applied_torques")
    if not math.isfinite(torque_scale) or torque_scale <= 0:
        raise ValueError("torque_scale must be finite and positive")
    tilt = projected_gravity[:, :2].square().sum(dim=-1)
    rotation = angular_velocity[:, :2].square().sum(dim=-1)
    torque_change = ((applied_torques-previous_applied_torques)/torque_scale).square().sum(dim=-1)
    return -4.0*tilt-0.1*rotation-0.2*torque_change


def support_reward_terms(length_l, length_r, target_length, passive_positions,
                         passive_lower, passive_upper, length_scale=0.01,
                         soft_limit_margin=0.10):
    """Return separate bounded terms for learning active leg support.

    Length tensors have shape [N] (or [N,1]) in metres. Passive joint angles
    have shape [N,P], with lower/upper limits [P] or [N,P] in radians. Every
    returned term has shape [N]; environments are never mixed.

    A 1 cm error scale makes the commanded length meaningful. Dense length
    and left/right costs cap squared normalized errors at 9, so a far-away
    initial pose cannot produce an unbounded negative reward. A passive
    joint within 0.10 rad of either boundary receives a soft penalty, with
    the maximum retained outside the bounds. No torque or termination is
    added here: these are rewards for a policy that produces total torque.
    """
    for name, value in (("length_scale", length_scale),
                        ("soft_limit_margin", soft_limit_margin)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("{} must be positive and finite".format(name))
    if length_l.ndim not in (1, 2) or length_l.shape[0] < 1:
        raise ValueError("length_l must have shape [N] or [N,1] with N >= 1")
    batch_size = length_l.shape[0]
    left = _length_vector(length_l, batch_size, "length_l")
    right = _length_vector(length_r, batch_size, "length_r")
    target = _length_vector(target_length, batch_size, "target_length")
    if (passive_positions.ndim != 2 or passive_positions.shape[0] != batch_size
            or passive_positions.shape[1] < 1):
        raise ValueError("passive_positions must have shape [N,P] with P >= 1")
    joint_count = passive_positions.shape[1]
    for name, value in (("passive_lower", passive_lower),
                        ("passive_upper", passive_upper)):
        if value.shape not in ((joint_count,), passive_positions.shape):
            raise ValueError("{} must have shape [P] or [N,P]".format(name))
    if bool((passive_lower >= passive_upper).any()):
        raise ValueError("Passive joint lower limits must be below upper limits")

    squared_errors = torch.stack(((left-target)/length_scale,
                                  (right-target)/length_scale), dim=-1).square()
    margin = torch.minimum(passive_positions-passive_lower,
                           passive_upper-passive_positions)
    return {
        "leg_tracking": 2.0*torch.exp(-0.5*squared_errors.sum(dim=-1)),
        "leg_error": -0.25*squared_errors.clamp(max=9.0).sum(dim=-1),
        "leg_asymmetry": -0.5*((left-right)/length_scale).square().clamp(max=9.0),
        "passive_limit": -4.0*(1.0-margin/soft_limit_margin).clamp(0.0, 1.0).square().mean(dim=-1),
    }
