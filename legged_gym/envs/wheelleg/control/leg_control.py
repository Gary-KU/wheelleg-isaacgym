"""可选髋关节 PD 与径向支撑前馈；策略动作作为额外力矩修正。"""

import math

import torch

from legged_gym.envs.wheelleg.model.kinematics import inverse_symmetric


def hip_targets(leg_lengths):
    """LF,LH,RF,RH targets from shared [N] or independent [N,2] lengths."""
    if leg_lengths.ndim == 1:
        phi1, _ = inverse_symmetric(leg_lengths)
        angle = phi1-3*math.pi/4
        return torch.stack((angle, -angle, -angle, angle), dim=-1)
    if leg_lengths.ndim != 2 or leg_lengths.shape[1] != 2:
        raise ValueError("leg_lengths must have shape [N] or [N,2] (left,right)")
    phi1, _ = inverse_symmetric(leg_lengths)
    left, right = (phi1-3*math.pi/4).unbind(-1)
    return torch.stack((left, -left, -right, right), dim=-1)


def hip_pd_torques(positions, velocities, targets, kp=80.0, kd=4.0):
    """Unclipped Nm; add residual first, then apply the physical torque limit."""
    if positions.ndim != 2 or positions.shape[1] != 4:
        raise ValueError("Hip positions must have shape [N,4]")
    if velocities.shape != positions.shape or targets.shape != positions.shape:
        raise ValueError("Hip positions, velocities and targets must have matching shapes")
    if not math.isfinite(kp) or kp <= 0 or not math.isfinite(kd) or kd < 0:
        raise ValueError("Require finite kp > 0 and kd >= 0")
    return kp*(targets-positions)-kd*velocities


def _radial_leg_jacobian(phi1, phi4):
    """d(length)/d(phi1, phi4), from the two lower-link circle constraints.

    The upper circle intersection is the assembly branch in kinematics.fivebar.
    Invalid or near-singular geometry contributes no feedforward. The derivative
    cap (m/rad) is only a singularity safeguard, not an actuator torque limit.
    """
    finite = torch.isfinite(phi1) & torch.isfinite(phi4)
    phi1 = torch.where(finite, phi1, torch.zeros_like(phi1))
    phi4 = torch.where(finite, phi4, torch.zeros_like(phi4))
    bx, by = .105*torch.cos(phi1), .105*torch.sin(phi1)
    dx, dy = .088+.105*torch.cos(phi4), .105*torch.sin(phi4)
    ux, uy = dx-bx, dy-by
    distance2 = ux.square()+uy.square()
    distance = torch.sqrt(distance2.clamp_min(1e-12))
    height2 = .16**2-.25*distance2
    height = torch.sqrt(height2.clamp_min(0))
    cx = .5*(bx+dx)-height*uy/distance
    cy = .5*(by+dy)+height*ux/distance
    r1x, r1y, r4x, r4y = cx-bx, cy-by, cx-dx, cy-dy
    determinant = r1x*r4y-r1y*r4x
    safe_determinant = torch.where(determinant.abs() > 1e-6,
                                   determinant, torch.ones_like(determinant))
    length = torch.sqrt((cx-.044).square()+cy.square()).clamp_min(1e-6)
    # r1 dot dC = r1 dot dB; r4 dot dC = r4 dot dD.
    rhs1 = .105*(-r1x*torch.sin(phi1)+r1y*torch.cos(phi1))
    rhs4 = .105*(-r4x*torch.sin(phi4)+r4y*torch.cos(phi4))
    gradient1 = rhs1*((cx-.044)*r4y-cy*r4x)/(safe_determinant*length)
    gradient4 = rhs4*(-(cx-.044)*r1y+cy*r1x)/(safe_determinant*length)
    valid = (finite & (distance2 > 1e-10) & (height2 > 1e-10)
             & (determinant.abs() > 1e-6) & (length > 1e-6))
    gradient = torch.stack((gradient1, gradient4), dim=-1).clamp(-.5, .5)
    return torch.where(valid.unsqueeze(-1), gradient, torch.zeros_like(gradient))


def hip_support_torques(positions, force_per_leg):
    """Return LF,LH,RF,RH torques for scalar or independent [N,2] leg loads.

    Virtual work gives tau = (dL/dq)^T F, so positive force extends each leg.
    This approximates the supported mass as a load along each leg. It is not
    exact gravity compensation for the distributed link masses, and applies
    no external force to the body. Add PD and RL residuals before actuator
    clipping in the environment. A zero force exactly disables this term.
    """
    if positions.ndim != 2 or positions.shape[1] != 4:
        raise ValueError("Hip positions must have shape [N,4]")
    if isinstance(force_per_leg, torch.Tensor):
        force = force_per_leg.to(device=positions.device, dtype=positions.dtype)
        if force.shape != (positions.shape[0], 2) or not bool(torch.isfinite(force).all()) or bool((force < 0).any()):
            raise ValueError("Support force tensor must be finite nonnegative [N,2] (left,right)")
        force = torch.repeat_interleave(force, 2, dim=-1)
    else:
        force = float(force_per_leg)
        if not math.isfinite(force) or force < 0:
            raise ValueError("force_per_leg must be a finite scalar >= 0")
        if force == 0:
            return torch.zeros_like(positions)
    left = _radial_leg_jacobian(3*math.pi/4+positions[:, 0],
                                math.pi/4+positions[:, 1])
    # Right phi1 is RH and phi4 is RF; return the actual action joint order.
    right = _radial_leg_jacobian(3*math.pi/4+positions[:, 3],
                                 math.pi/4+positions[:, 2])
    return force*torch.cat((left, right.flip(-1)), dim=-1)
