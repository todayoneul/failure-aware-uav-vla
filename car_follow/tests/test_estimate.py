"""Demo 1 step 5: target position from a box (src/car_follow/estimate.py). No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import math
import unittest

import numpy as np

from src.car_follow import clearance, estimate, screen
from test_clearance import render

CAM = {"width": 960, "height": 540, "fov": 90.0}


class RayTests(unittest.TestCase):
    def test_the_ray_through_a_projected_pixel_points_at_the_point(self):
        eye, yaw, pitch = np.array([1.0, 2.0, 16.0]), 35.0, -38.0
        point = np.array([17.0, 14.0, 0.7])
        uv, _ = screen.Camera(eye, yaw, pitch, CAM["width"], CAM["height"], CAM["fov"]).project(point)
        ray = estimate.pixel_ray(eye, yaw, pitch, CAM["width"], CAM["height"], CAM["fov"], uv[0])
        want = (point - eye) / np.linalg.norm(point - eye)
        self.assertLess(np.linalg.norm(ray - want), 1e-9)

    def test_flat_ground_and_depth_agree_on_flat_ground(self):
        eye, yaw, pitch = np.array([0.0, 0.0, 15.0]), 10.0, -37.0
        dcam = clearance.DepthCamera()
        depth = render(eye, yaw)
        box = (450.0, 250.0, 510.0, 290.0)
        m = estimate.measure(box, eye, yaw, pitch, CAM, depth, dcam)
        self.assertAlmostEqual(m["flat"][2], 0.0)
        t = estimate.along_depth(eye, m["ray_bottom"], depth, dcam, yaw)
        self.assertLess(np.linalg.norm(eye + t * m["ray_bottom"] - m["flat"]), 0.3)   # one depth pixel
        self.assertLess(abs(m["depth"][2]), 0.3)                                       # the centre ray hits ground

    def test_push_moves_horizontally_along_the_ray(self):
        p = estimate.push(np.zeros(3), np.array([0.6, 0.0, -0.8]), 1.5)
        self.assertTrue(np.allclose(p, [1.5, 0.0, 0.0]))


class FilterTests(unittest.TestCase):
    def test_constant_velocity_track_with_noise(self):
        rng = np.random.default_rng(0)
        f = estimate.TargetFilter()
        dt, v = 0.05, np.array([8.0, 6.0])
        for k in range(200):
            truth = np.r_[v * k * dt, 0.0]
            f.predict(dt)
            f.update(truth + np.r_[rng.normal(0, 0.7, 2), 0.0], dt)
        self.assertLess(np.linalg.norm(f.position[:2] - v * 199 * dt), 0.6)
        self.assertLess(np.linalg.norm(f.velocity[:2] - v), 0.8)
        self.assertAlmostEqual(f.yaw, math.degrees(math.atan2(6, 8)), delta=3.0)

    def test_coasting_predicts_and_counts_time(self):
        f = estimate.TargetFilter()
        for k in range(100):
            f.predict(0.05)
            f.update(np.array([10.0 * k * 0.05, 0.0, 0.0]), 0.05)
        last = f.position.copy()
        for _ in range(20):
            f.predict(0.05)
        self.assertAlmostEqual(f.since_update, 1.0)
        self.assertAlmostEqual(f.position[0] - last[0], 10.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
