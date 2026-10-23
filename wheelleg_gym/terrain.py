"""Deterministic stair courses, without adding actors to the Gym simulation.

Each environment owns a patch containing parallel difficulty lanes. All lanes
start at z=0; their closed stair prisms sit on the existing infinite plane.
Robot environments must have zero Gym layout bounds and use ``origins`` as
explicit spawn coordinates, so mesh positions and robot state tensors agree.
"""

import math

import numpy as np


_BOX_TRIANGLES = np.array([
    [4, 5, 6], [4, 6, 7],       # top, +Z
    [0, 2, 1], [0, 3, 2],       # bottom, -Z
    [0, 1, 5], [0, 5, 4],       # -Y
    [1, 2, 6], [1, 6, 5],       # +X
    [2, 3, 7], [2, 7, 6],       # +Y
    [3, 0, 4], [3, 4, 7],       # -X
], dtype=np.uint32)


class StairTerrain:
    """Five progressively harder four-up/four-down stair courses per patch.

    Public NumPy arrays:
      origins[N, levels, 3]: flat spawn locations in mesh/world coordinates.
      vertices[V, 3], triangles[T, 3]: one combined static collision mesh.
      x_edges[N, levels, 12], section_heights[N, levels, 2, 11]: the exact
        piecewise-constant surfaces, including the initial/final ground.

    Side 0 is negative local Y and side 1 positive Y. The hardest two lanes
    have different left/right riser increments; easier lanes remain symmetric.
    Geometry is fixed after construction; a curriculum changes the selected
    spawn lane, never the simulated mesh underneath a moving robot.
    """

    start_x = 0.45
    finish_x = 3.80
    lane_half_width = 0.65
    lane_spacing = 1.80
    course_start_x = -0.80
    course_end_x = 4.50
    sample_offsets = np.array([
        [x, y] for x in (0.10, 0.25, 0.45) for y in (-0.12, 0.0, 0.12)
    ], dtype=np.float32)

    def __init__(self, num_envs, num_levels=5, seed=42):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(num_levels, int) or isinstance(num_levels, bool) or not 1 <= num_levels <= 5:
            raise ValueError("num_levels must be in [1,5]")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        self.num_envs, self.num_levels, self.seed = num_envs, num_levels, seed
        self.level_step_heights = np.array([.01, .02, .035, .05, .07][:num_levels], dtype=np.float32)
        self.origins = np.zeros((num_envs, num_levels, 3), dtype=np.float32)
        self.x_edges = np.zeros((num_envs, num_levels, 12), dtype=np.float32)
        self.section_heights = np.zeros((num_envs, num_levels, 2, 11), dtype=np.float32)
        self.patch_spacing = (self.finish_x+2.0, num_levels*self.lane_spacing+2.0)
        side = int(math.ceil(math.sqrt(num_envs)))
        vertices, triangles = [], []
        vertex_count = 0
        rng = np.random.RandomState(seed)
        for env_index in range(num_envs):
            patch_x = (env_index % side)*self.patch_spacing[0]
            patch_y = (env_index // side)*self.patch_spacing[1]
            for level in range(num_levels):
                origin = np.array([patch_x, patch_y+level*self.lane_spacing, 0.0])
                self.origins[env_index, level] = origin
                widths = np.full(8, .35)
                if level > 0:
                    variation = rng.uniform(-.025, .025, 8)
                    widths += variation-variation.mean()  # Sum stays 2.8m.
                intervals = np.r_[self.start_x-self.course_start_x,
                                  widths[:4], .55, widths[4:],
                                  self.course_end_x-self.finish_x]
                edges = self.course_start_x+np.r_[0.0, np.cumsum(intervals)]
                # Avoid accumulating a tiny floating discrepancy at the finish.
                edges[-2:] = (self.finish_x, self.course_end_x)
                self.x_edges[env_index, level] = edges
                base_height = float(self.level_step_heights[level])
                increments = np.full(4, base_height)
                if level > 0:
                    variation = rng.uniform(-.12, .12, 4)
                    increments *= 1.0+variation-variation.mean()
                split = level >= 3
                side_variation = rng.uniform(-.10, .10, 4) if split else np.zeros(4)
                for lane_side in range(2):
                    multiplier = 1.0+side_variation*(1 if lane_side == 0 else -1)
                    up = np.cumsum(increments*multiplier)
                    self.section_heights[env_index, level, lane_side] = np.r_[
                        0.0, up, up[-1], up[-2::-1], 0.0, 0.0]
                for lane_side in range(2 if split else 1):
                    y0, y1 = ((-self.lane_half_width, 0.0) if lane_side == 0
                              else (0.0, self.lane_half_width)) if split else (
                                  -self.lane_half_width, self.lane_half_width)
                    for section, height in enumerate(self.section_heights[env_index, level, lane_side]):
                        # z=0 sections use the existing ground plane. Duplicate
                        # coplanar ground triangles would add unnecessary contact.
                        if height <= 0.0:
                            continue
                        x0, x1 = map(float, self.x_edges[env_index, level, section:section+2])
                        bottom = -.025
                        box = np.array([
                            [x0, y0, bottom], [x1, y0, bottom],
                            [x1, y1, bottom], [x0, y1, bottom],
                            [x0, y0, height], [x1, y0, height],
                            [x1, y1, height], [x0, y1, height],
                        ], dtype=np.float32)
                        vertices.append(box+self.origins[env_index, level])
                        triangles.append(_BOX_TRIANGLES+vertex_count)
                        vertex_count += 8
        self.vertices = np.ascontiguousarray(np.concatenate(vertices), dtype=np.float32)
        self.triangles = np.ascontiguousarray(np.concatenate(triangles), dtype=np.uint32)
        self._tensor_cache = {}
        self.spec = {
            "kind": "stairs", "num_envs": num_envs, "num_levels": num_levels,
            "seed": seed, "start_x_m": self.start_x, "finish_x_m": self.finish_x,
            "lane_half_width_m": self.lane_half_width, "lane_spacing_m": self.lane_spacing,
            "level_step_heights_m": self.level_step_heights.tolist(),
            "up_risers": 4, "down_risers": 4, "extra_top_platform_m": .55,
            "tread_width_nominal_m": .35, "tread_width_range_m": [.30, .40],
            "split_left_right_from_level": 3, "ground_plane_required": True,
            "geometry": "Closed rectangular prisms; exact vertical risers, no actor obstacles",
            "coordinate_system": "Explicit global spawn origins; create_env layout bounds must be zero",
        }

    def add_to_sim(self, gym, sim):
        """Add one static triangle mesh, leaving the actor/body tensors intact."""
        from isaacgym import gymapi
        params = gymapi.TriangleMeshParams()
        params.nb_vertices, params.nb_triangles = len(self.vertices), len(self.triangles)
        params.static_friction = params.dynamic_friction = 1.0
        params.restitution = 0.0
        gym.add_triangle_mesh(sim, self.vertices.flatten(), self.triangles.flatten(), params)

    def heights_at(self, xy_tensor, env_ids_tensor, levels_tensor):
        """Query mesh top heights at [N,2] or [N,K,2] world points.

        Environment/level IDs may be scalars, [N], or broadcastable point IDs.
        At a vertical X boundary the section to its right is used; at the
        left/right seam the higher adjacent top is used. Outside the selected
        lane or course the result is zero, matching the underlying plane.
        """
        import torch
        if not isinstance(xy_tensor, torch.Tensor) or not xy_tensor.is_floating_point():
            raise ValueError("xy_tensor must be a floating Torch tensor")
        if xy_tensor.ndim < 2 or xy_tensor.shape[-1] != 2:
            raise ValueError("World points must have shape [N,2] or [N,K,2]")
        ids = torch.as_tensor(env_ids_tensor, dtype=torch.long, device=xy_tensor.device)
        levels = torch.as_tensor(levels_tensor, dtype=torch.long, device=xy_tensor.device)
        if xy_tensor.ndim > 2:
            if ids.ndim == 1 and ids.shape[0] == xy_tensor.shape[0]:
                ids = ids.reshape((-1,)+(1,)*(xy_tensor.ndim-2))
            if levels.ndim == 1 and levels.shape[0] == xy_tensor.shape[0]:
                levels = levels.reshape((-1,)+(1,)*(xy_tensor.ndim-2))
        try:
            ids, levels, x, y = torch.broadcast_tensors(ids, levels, xy_tensor[..., 0], xy_tensor[..., 1])
        except RuntimeError as error:
            raise ValueError("Environment/level IDs must broadcast over point coordinates") from error
        if bool(((ids < 0) | (ids >= self.num_envs) | (levels < 0) | (levels >= self.num_levels)).any()):
            raise ValueError("Environment or terrain level index out of range")
        key = (str(xy_tensor.device), xy_tensor.dtype)
        if key not in self._tensor_cache:
            self._tensor_cache[key] = tuple(torch.as_tensor(value, device=xy_tensor.device, dtype=xy_tensor.dtype)
                                             for value in (self.origins, self.x_edges, self.section_heights))
        origins, all_edges, all_heights = self._tensor_cache[key]
        origin, edges, heights = origins[ids, levels], all_edges[ids, levels], all_heights[ids, levels]
        local_x, local_y = x-origin[..., 0], y-origin[..., 1]
        section = (local_x.unsqueeze(-1) >= edges[..., 1:]).sum(-1).clamp(max=self.section_heights.shape[-1]-1)
        pair = heights.gather(-1, section[..., None, None].expand(section.shape+(2, 1))).squeeze(-1)
        height = torch.where(local_y < 0, pair[..., 0], pair[..., 1])
        height = torch.where(local_y == 0, pair.max(-1).values, height)
        inside = ((local_x >= self.course_start_x) & (local_x <= self.course_end_x)
                  & (local_y.abs() <= self.lane_half_width))
        return torch.where(inside, height, torch.zeros_like(height))
