"""Compliant loop closure for Isaac Gym Preview 4.

This is an external spring/damper approximation, not a native PhysX joint.
State quaternions use Gym's XYZW order. Rigid-body poses refer to body
origins, whereas their linear velocities refer to centers of mass.
"""

import json
import math
from pathlib import Path

import torch


def quat_rotate(q, v):
    """Rotate vectors by unit XYZW quaternions, with broadcasting."""
    xyz, v = torch.broadcast_tensors(q[..., :3], v)
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2.0 * (q[..., 3:4] * uv + torch.cross(xyz, uv, dim=-1))


def quat_rotate_inverse(q, v):
    """Apply the inverse of a unit XYZW quaternion to vectors."""
    xyz, v = torch.broadcast_tensors(-q[..., :3], v)
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2.0 * (q[..., 3:4] * uv + torch.cross(xyz, uv, dim=-1))


def _limit_norm(v, limit):
    if limit is None:
        return v
    return v * torch.clamp(limit / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)


class ClosureForces:
    """Apply equal-and-opposite compliant closure wrenches to a body tree.

    ``metadata`` is the exported JSON mapping (or its file path), containing
    ``constraints`` entries with body0/body1, local_pos0/local_pos1 and
    local_axis0/local_axis1. These vectors use each final URDF body frame.
    ``com_local`` must come from Gym's imported rigid-body properties, with
    shape [bodies, 3] or [environments, bodies, 3].

    ``compute`` returns world/environment-aligned forces and COM torques for
    ``apply_rigid_body_force_tensors(..., ENV_SPACE)``. The positional spring
    closes the anchors. The angular spring aligns their hinge axes while
    leaving relative rotation about the common axis free. ``max_torque``
    limits the angular alignment couple; moments induced by anchor forces
    are preserved. Evaluate at every *external* physics step, not only at
    the policy/control rate. Internal simulator substeps cannot refresh it.
    """

    def __init__(self, metadata, body_names, com_local, num_envs, device,
                 stiffness=2000.0, damping=15.0, angular_stiffness=20.0,
                 angular_damping=0.2, max_force=100.0, max_torque=10.0):
        if isinstance(metadata, (str, Path)):
            with open(str(metadata), encoding="utf-8") as stream:
                metadata = json.load(stream)
        entries = metadata.get("constraints", metadata.get("closures"))
        if not entries:
            raise ValueError("Closure metadata contains no constraints")
        if len(set(body_names)) != len(body_names):
            raise ValueError("Rigid-body names must be unique")
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.num_bodies = len(body_names)
        self.names = [entry["name"] for entry in entries]
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.angular_stiffness = float(angular_stiffness)
        self.angular_damping = float(angular_damping)
        for name in ("stiffness", "damping", "angular_stiffness", "angular_damping"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and nonnegative".format(name))
        for name, value in (("max_force", max_force), ("max_torque", max_torque)):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("{} must be positive or None".format(name))
            setattr(self, name, value)
        indices = {name: i for i, name in enumerate(body_names)}
        self.body0 = torch.tensor([indices[x["body0"]] for x in entries], device=self.device, dtype=torch.long)
        self.body1 = torch.tensor([indices[x["body1"]] for x in entries], device=self.device, dtype=torch.long)
        for key in ("local_pos0", "local_pos1", "local_axis0", "local_axis1"):
            values = torch.tensor([x[key] for x in entries], device=self.device, dtype=torch.float32)
            if values.shape != (len(entries), 3) or not torch.isfinite(values).all():
                raise ValueError("Invalid {} vectors in closure metadata".format(key))
            if "axis" in key:
                norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
                if (norm < 1e-8).any():
                    raise ValueError("Closure axes must be nonzero")
                values = values / norm
            setattr(self, key, values)
        self.com_local = torch.as_tensor(com_local, dtype=torch.float32, device=self.device)
        if self.com_local.shape == (self.num_bodies, 3):
            self.com_local = self.com_local.unsqueeze(0).expand(self.num_envs, -1, -1)
        if self.com_local.shape != (self.num_envs, self.num_bodies, 3):
            raise ValueError("com_local must have shape [B,3] or [N,B,3]")
        if not torch.isfinite(self.com_local).all():
            raise ValueError("COM offsets must be finite")
        # Rotate three constant vectors together for each endpoint. On a
        # single-robot CPU workload, fewer tensor dispatches matter much more
        # than the tiny additional vector dimension; equations stay identical.
        self._geometry_local0 = torch.stack((
            self.local_pos0.expand(self.num_envs, -1, -1),
            self.local_pos0-self.com_local[:, self.body0],
            self.local_axis0.expand(self.num_envs, -1, -1)), dim=-2)
        self._geometry_local1 = torch.stack((
            self.local_pos1.expand(self.num_envs, -1, -1),
            self.local_pos1-self.com_local[:, self.body1],
            self.local_axis1.expand(self.num_envs, -1, -1)), dim=-2)

    def _geometry(self, rb_state):
        if rb_state.shape != (self.num_envs, self.num_bodies, 13):
            raise ValueError("rb_state must have shape [N,B,13]")
        s0, s1 = rb_state[:, self.body0], rb_state[:, self.body1]
        q0, q1 = s0[..., 3:7], s1[..., 3:7]
        rotated0 = quat_rotate(q0.unsqueeze(-2), self._geometry_local0)
        rotated1 = quat_rotate(q1.unsqueeze(-2), self._geometry_local1)
        anchor0 = s0[..., :3] + rotated0[..., 0, :]
        anchor1 = s1[..., :3] + rotated1[..., 0, :]
        # Lever arms start at COM, not the link origin in rb_state[..., :3].
        lever0, lever1 = rotated0[..., 1, :], rotated1[..., 1, :]
        velocity0 = s0[..., 7:10] + torch.cross(s0[..., 10:13], lever0, dim=-1)
        velocity1 = s1[..., 7:10] + torch.cross(s1[..., 10:13], lever1, dim=-1)
        axis0, axis1 = rotated0[..., 2, :], rotated1[..., 2, :]
        return s0, s1, anchor0, anchor1, lever0, lever1, velocity0, velocity1, axis0, axis1

    def errors(self, rb_state):
        """Return anchor separation in meters, shape [N, closures]."""
        if rb_state.shape != (self.num_envs, self.num_bodies, 13):
            raise ValueError("rb_state must have shape [N,B,13]")
        s0, s1 = rb_state[:, self.body0], rb_state[:, self.body1]
        anchor0 = s0[..., :3]+quat_rotate(s0[..., 3:7], self.local_pos0)
        anchor1 = s1[..., :3]+quat_rotate(s1[..., 3:7], self.local_pos1)
        return torch.linalg.vector_norm(anchor1-anchor0, dim=-1)

    def axis_errors(self, rb_state):
        """Return the signed-axis alignment angle magnitude in radians."""
        data = self._geometry(rb_state)
        a0, a1 = data[-2:]
        sine = torch.linalg.vector_norm(torch.cross(a0, a1, dim=-1), dim=-1)
        cosine = (a0 * a1).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.atan2(sine, cosine)

    def compute(self, rb_state):
        """Return [N,B,3] forces and torques about each body's COM."""
        s0, s1, p0, p1, r0, r1, v0, v1, a0, a1 = self._geometry(rb_state)
        force = _limit_norm(self.stiffness * (p1 - p0) + self.damping * (v1 - v0), self.max_force)
        cross = torch.cross(a0, a1, dim=-1)
        sine = torch.linalg.vector_norm(cross, dim=-1, keepdim=True)
        cosine = (a0 * a1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        angle = torch.atan2(sine, cosine)
        # At 180 degrees choose a deterministic perpendicular correction axis.
        basis = torch.nn.functional.one_hot(a0.abs().argmin(dim=-1), 3).to(a0.dtype)
        fallback = torch.cross(a0, basis, dim=-1)
        fallback = fallback / torch.linalg.vector_norm(fallback, dim=-1, keepdim=True).clamp_min(1e-12)
        direction = torch.where(sine > 1e-7, cross / sine.clamp_min(1e-12), fallback)
        average_axis = a0 + a1
        average_norm = torch.linalg.vector_norm(average_axis, dim=-1, keepdim=True)
        hinge_axis = torch.where(average_norm > 1e-7, average_axis / average_norm.clamp_min(1e-12), a0)
        delta_omega = s1[..., 10:13] - s0[..., 10:13]
        perpendicular_omega = delta_omega - (delta_omega * hinge_axis).sum(dim=-1, keepdim=True) * hinge_axis
        couple = _limit_norm(self.angular_stiffness * angle * direction + self.angular_damping * perpendicular_omega, self.max_torque)
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=rb_state.dtype, device=rb_state.device)
        torques = torch.zeros_like(forces)
        forces.index_add_(1, self.body0, force)
        forces.index_add_(1, self.body1, -force)
        torques.index_add_(1, self.body0, torch.cross(r0, force, dim=-1) + couple)
        torques.index_add_(1, self.body1, torch.cross(r1, -force, dim=-1) - couple)
        return forces, torques
