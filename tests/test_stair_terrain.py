"""Stair geometry, world-coordinate height queries and deterministic courses."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.envs.wheelleg.terrain import StairTerrain


class StairTerrainTests(unittest.TestCase):
    def setUp(self):
        self.terrain = StairTerrain(3, seed=17)

    def test_geometry_is_deterministic_valid_and_has_closed_nonzero_faces(self):
        terrain = self.terrain
        other = StairTerrain(3, seed=17)
        np.testing.assert_array_equal(terrain.vertices, other.vertices)
        np.testing.assert_array_equal(terrain.triangles, other.triangles)
        self.assertEqual(terrain.vertices.dtype, np.float32)
        self.assertEqual(terrain.triangles.dtype, np.uint32)
        self.assertTrue(np.isfinite(terrain.vertices).all())
        self.assertLess(int(terrain.triangles.max()), len(terrain.vertices))
        faces = terrain.vertices[terrain.triangles]
        areas2 = np.linalg.norm(np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]), axis=-1)
        self.assertTrue((areas2 > 1e-8).all())
        for start in range(0, len(terrain.triangles), 12):
            faces = terrain.triangles[start:start+12]
            edges = {}
            for face in faces:
                for a, b in zip(face, np.roll(face, -1)):
                    key = tuple(sorted((int(a), int(b))))
                    edges[key] = edges.get(key, 0)+1
            self.assertTrue(all(count == 2 for count in edges.values()))

    def test_patch_lanes_and_flat_spawn_have_clearance(self):
        terrain = self.terrain
        self.assertEqual(terrain.origins.shape, (3, 5, 3))
        self.assertTrue((np.diff(terrain.origins[:, :, 1], axis=1) > 2*terrain.lane_half_width).all())
        np.testing.assert_allclose(terrain.x_edges[:, :, -2], terrain.finish_x)
        np.testing.assert_allclose(terrain.section_heights[:, :, :, 0], 0)
        points = torch.from_numpy(terrain.origins[:, :, :2].reshape(-1, 2))
        env_ids = torch.arange(3).repeat_interleave(5)
        levels = torch.arange(5).repeat(3)
        self.assertTrue(bool((terrain.heights_at(points, env_ids, levels) == 0).all()))

    def test_height_query_matches_real_mesh_box_tops_all_sections_and_sides(self):
        terrain = self.terrain
        boxes = terrain.vertices.reshape(-1, 8, 3)
        mins, maxs = boxes.min(1), boxes.max(1)
        points, env_ids, levels, expected = [], [], [], []
        for env_id in range(3):
            for level in range(5):
                origin = terrain.origins[env_id, level]
                for local_x in .5*(terrain.x_edges[env_id, level, :-1]+terrain.x_edges[env_id, level, 1:]):
                    for local_y in (-.12, .12):
                        point = origin[:2]+[local_x, local_y]
                        on = ((point[0] >= mins[:, 0]) & (point[0] <= maxs[:, 0])
                              & (point[1] >= mins[:, 1]) & (point[1] <= maxs[:, 1]))
                        top = max(float(maxs[on, 2].max()) if on.any() else 0.0, 0.0)
                        points.append(point); env_ids.append(env_id); levels.append(level); expected.append(top)
        actual = terrain.heights_at(torch.tensor(np.array(points)), torch.tensor(env_ids), torch.tensor(levels))
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), atol=1e-7, rtol=0)

    def test_multi_sample_queries_outside_lane_and_ground_are_zero(self):
        terrain = self.terrain
        env_ids, levels = torch.tensor([0, 1, 2]), torch.tensor([1, 3, 4])
        origins = torch.from_numpy(terrain.origins[env_ids, levels, :2])
        offsets = torch.tensor([[.60, -.12], [.60, .12], [.60, .80], [4.8, 0.], [-1., 0.]])
        points = origins[:, None]+offsets
        heights = terrain.heights_at(points, env_ids, levels)
        self.assertEqual(tuple(heights.shape), (3, 5))
        self.assertTrue(bool((heights[:, :2] > 0).all()))
        self.assertTrue(bool((heights[:, 2:] == 0).all()))
        torch.testing.assert_close(heights[:, 0], terrain.heights_at(points[:, 0], env_ids, levels))

    def test_risers_ascend_and_descend_and_hard_levels_have_left_right_variation(self):
        heights = self.terrain.section_heights
        self.assertTrue((np.diff(heights[..., 1:5], axis=-1) > 0).all())
        self.assertTrue((np.diff(heights[..., 5:10], axis=-1) < 0).all())
        np.testing.assert_array_equal(heights[:, :3, 0], heights[:, :3, 1])
        self.assertGreater(float(np.max(abs(heights[:, 3:, 0]-heights[:, 3:, 1]))), 1e-4)
        np.testing.assert_allclose(self.terrain.level_step_heights, [.01, .02, .035, .05, .07])

    def test_invalid_counts_and_indices_are_rejected(self):
        for args in ((0,), (1, 0), (1, 6), (True,), (1, 5, -1)):
            with self.assertRaises(ValueError):
                StairTerrain(*args)
        for env_ids, levels in (([-1], [0]), ([3], [0]), ([0], [5])):
            with self.assertRaises(ValueError):
                self.terrain.heights_at(torch.zeros(1, 2), env_ids, levels)


if __name__ == "__main__":
    unittest.main()
