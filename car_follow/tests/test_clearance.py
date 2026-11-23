"""Demo 1: the altitude guard (src/car_follow/clearance.py) on depth maps ray-cast from simple scenes (ground plane
and boxes). No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import math
import unittest

import numpy as np

from src.car_follow import clearance, follow, drone

CAM = clearance.DepthCamera()
P = clearance.Params()


def render(eye, yaw, boxes=(), cam=CAM):
    """Depth map (along the optical axis) of the ground z = 0 and axis-aligned boxes [(lo xyz, hi xyz)]."""
    f, r, u = clearance.axes(yaw, cam.mount_pitch)
    rays = clearance.pixel_rays(cam)
    d = f + rays[..., :1] * r + rays[..., 1:] * u          # H x W x 3, unit depth along the axis
    eye = np.asarray(eye, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d[..., 2] < 0, -eye[2] / d[..., 2], np.inf)
        for lo, hi in boxes:
            t1, t2 = (np.asarray(lo) - eye) / d, (np.asarray(hi) - eye) / d
            near = np.nanmax(np.minimum(t1, t2), axis=-1)
            far = np.nanmin(np.maximum(t1, t2), axis=-1)
            hit = (far >= near) & (near > 0)
            t = np.where(hit & (near < t), near, t)
    return np.minimum(t, 1000.0)


def line_clear(eye, target, boxes, margin=0.0, steps=2000):
    """Whether the segment eye-target passes at least `margin` over every box (sampled)."""
    for a in np.linspace(0.0, 1.0, steps)[1:-1]:
        q = (1 - a) * np.asarray(eye) + a * np.asarray(target)
        for lo, hi in boxes:
            if lo[0] <= q[0] <= hi[0] and lo[1] <= q[1] <= hi[1] and q[2] < hi[2] + margin and q[2] > lo[2] - 1e-6:
                return False
    return True


class GeometryTests(unittest.TestCase):
    def test_back_projection_of_the_ground_lies_on_it(self):
        eye = np.array([3.0, -2.0, 15.0])
        depth = render(eye, 30.0)
        pts = clearance.back_project(depth, CAM, eye, 30.0)
        self.assertGreater(len(pts), 1000)
        self.assertLess(np.abs(pts[:, 2]).max(), 1e-6)

    def test_project_inverts_back_projection(self):
        eye, yaw = np.array([0.0, 0.0, 15.0]), -45.0
        point = np.array([14.0, -15.0, 0.75])
        (x, y), z = clearance.project(point, CAM, eye, yaw)
        depth = np.full((CAM.height, CAM.width), 1000.0)
        depth[int(y), int(x)] = z
        back = clearance.back_project(depth, CAM, eye, yaw)
        self.assertLess(np.linalg.norm(back[0] - point), 0.25)  # within one pixel of the point


class PathTests(unittest.TestCase):
    eye = np.array([0.0, 0.0, 15.0])

    def need(self, boxes, velocity=(10.0, 0.0, 0.0)):
        pts = clearance.back_project(render(self.eye, 0.0, boxes), CAM, self.eye, 0.0)
        return clearance.path_need(pts, self.eye, np.array(velocity), 0.0, 15.0, P)

    def test_flat_ground_needs_nothing(self):
        self.assertEqual(self.need(()), 0.0)

    def test_a_wall_ahead_near_the_flying_height_is_cleared_by_the_clearance(self):
        wall = ((20.0, -10.0, 0.0), (21.0, 10.0, 13.0))
        self.assertAlmostEqual(self.need([wall]), 13.0 + P.clearance_m - 15.0, places=3)

    def test_low_far_or_side_obstacles_need_nothing(self):
        self.assertEqual(self.need([((20.0, -10.0, 0.0), (21.0, 10.0, 11.5))]), 0.0)    # below the band
        self.assertEqual(self.need([((40.0, -10.0, 0.0), (41.0, 10.0, 14.0))]), 0.0)    # beyond 3 s at 10 m/s
        self.assertEqual(self.need([((10.0, 5.0, 0.0), (15.0, 8.0, 14.0))]), 0.0)       # beside the corridor

    def test_a_hovering_drone_looks_ahead_along_its_heading(self):
        wall = ((10.0, -10.0, 0.0), (11.0, 10.0, 14.0))
        self.assertAlmostEqual(self.need([wall], velocity=(0.0, 0.0, 0.0)), 2.0, places=3)


class LineOfSightTests(unittest.TestCase):
    target = np.array([20.0, 0.0, 0.75])
    base = 15.75

    def test_open_view_is_clear(self):
        eye = np.array([0.0, 0.0, self.base])
        need, state = clearance.los_need(render(eye, 0.0), CAM, eye, 0.0, self.target, self.base, P)
        self.assertEqual((need, state), (0.0, "clear"))

    def test_climbing_clears_a_railing_between_drone_and_target(self):
        rail = ((4.0, -10.0, 0.0), (4.3, 10.0, 13.5))
        eye = np.array([0.0, 0.0, self.base])
        self.assertFalse(line_clear(eye, self.target, [rail]))
        need, state, steps = 0.0, "blocked", 0
        while state == "blocked" and steps < 20:
            eye = np.array([0.0, 0.0, self.base + need])
            depth = render(eye, 0.0, [rail])
            pts = clearance.back_project(depth, CAM, eye, 0.0)
            new, state = clearance.los_need(depth, CAM, eye, 0.0, self.target, self.base, P, points=pts)
            need = max(need, new)
            steps += 1
        self.assertEqual(state, "clear")
        self.assertLess(steps, 10)
        self.assertLessEqual(need, P.max_offset_m)
        self.assertTrue(line_clear(eye, self.target, [rail]))

    def test_a_target_under_a_deck_is_unreachable(self):
        deck = ((10.0, -10.0, 5.0), (30.0, 10.0, 6.0))
        eye = np.array([0.0, 0.0, self.base])
        depth = render(eye, 0.0, [deck])
        pts = clearance.back_project(depth, CAM, eye, 0.0)
        self.assertEqual(clearance.los_need(depth, CAM, eye, 0.0, self.target, self.base, P, points=pts),
                         (0.0, "unreachable"))

    def test_target_outside_the_depth_image_is_unseen(self):
        eye = np.array([0.0, 0.0, self.base])
        behind = np.array([-20.0, 0.0, 0.75])
        self.assertEqual(clearance.los_need(render(eye, 0.0), CAM, eye, 0.0, behind, self.base, P)[1], "unseen")


class GuardTests(unittest.TestCase):
    def test_climb_hold_and_smooth_return(self):
        guard = clearance.AltitudeGuard()
        eye, dt = np.array([0.0, 0.0, 15.0]), 0.05
        wall = ((10.0, -10.0, 0.0), (11.0, 10.0, 15.0))      # needs +3 m while it is in view
        offsets = []
        for k in range(400):
            boxes = [wall] if k < 100 else []
            d = guard.update(render(eye, 0.0, boxes), eye, 0.0, np.zeros(3), 15.0, None, dt)
            offsets.append(d.offset)
        o = np.array(offsets)
        rate = np.diff(o) / dt
        self.assertAlmostEqual(o[99], 3.0, delta=0.15)                      # climbed to the need
        self.assertLessEqual(rate.max(), P.climb_rate + 1e-9)
        self.assertGreaterEqual(rate.min(), -P.return_rate - 1e-9)
        hold = int(P.hold_s / dt)
        self.assertLess(np.ptp(o[100:100 + hold]), 0.15)                    # held after the wall was gone
        self.assertLess(o[100 + hold + 10], o[100])                         # then returned
        self.assertLess(o[-1], 0.05)
        self.assertLessEqual(np.abs(np.diff(rate)).max() / dt, P.climb_accel + 1e-6)  # no jumps in rate

    def test_the_follower_raises_its_goal_by_the_offset(self):
        f = follow.Follower(drone.Limits())
        state = follow.start_state([0, 0, 15, 0, 0, 0])
        _, goal = f.command(state, np.array([20.0, 0, 0]), 0.0, np.array([20.0, 0, 0.75]), 0.05, offset=2.5)
        self.assertAlmostEqual(goal[2], follow.ABOVE_M + 2.5)


if __name__ == "__main__":
    unittest.main()
