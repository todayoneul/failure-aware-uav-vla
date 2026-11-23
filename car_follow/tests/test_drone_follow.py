"""Demo 1 step 2: the drone model's limits and lag, the follow controller on synthetic targets, and the camera angle
of the metrics. No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import math
import unittest

import numpy as np

from src.car_follow import drone, follow, metrics

DT = 0.05


def straight_track(n=1200, top=10.0, accel=2.0):
    pos, v, x = [], 0.0, 0.0
    for _ in range(n):
        pos.append([x, 0.0, 0.0])
        v = min(top, v + accel * DT)
        x += v * DT
    pos = np.array(pos)
    return {"position": pos, "yaw": np.zeros(n), "aim": pos + [0.0, 0.0, 0.8]}


def turn_track(n=1200, top=8.0, radius=15.0):
    """60 m straight, a 90 degree right turn of `radius`, then straight on."""
    pos, yaw, x, y, h, s = [], [], 0.0, 0.0, 0.0, 0.0
    for k in range(n):
        pos.append([x, y, 0.0])
        yaw.append(math.degrees(h))
        v = min(top, 2.0 * k * DT)
        s += v * DT
        if 60.0 <= s <= 60.0 + radius * math.pi / 2:
            h += v * DT / radius
        x += v * DT * math.cos(h)
        y += v * DT * math.sin(h)
    pos = np.array(pos)
    return {"position": pos, "yaw": np.array(yaw), "aim": pos + [0.0, 0.0, 0.8]}


def fly(track, lag, gains=follow.Gains()):
    goal = follow.goal_point(track["position"][0], track["yaw"][0])
    yaw, pitch = follow.aim_angles(goal, track["aim"][0])
    limits = drone.Limits(lag_s=lag)
    log = follow.simulate(track, drone.State(goal.copy(), yaw=yaw, pitch=pitch), limits, gains, DT)
    return metrics.follow_quality(log, track, limits, DT)


class DroneTests(unittest.TestCase):
    def test_limits_hold_under_any_command(self):
        limits = drone.Limits(lag_s=0.0)
        state = drone.State(np.zeros(3))
        cmd = drone.Command(np.array([100.0, 50.0, 30.0]), 0.0, 0.0)
        for _ in range(200):
            new = drone.step(state, cmd, limits, DT)
            self.assertLessEqual(np.linalg.norm(new.acceleration[:2]), limits.accel_h + 1e-9)
            self.assertLessEqual(abs(new.acceleration[2]), limits.accel_v + 1e-9)
            state = new
        self.assertAlmostEqual(np.linalg.norm(state.velocity[:2]), limits.speed_h, places=6)
        self.assertAlmostEqual(state.velocity[2], limits.speed_v, places=6)

    def test_first_order_lag(self):
        limits = drone.Limits(lag_s=1.0, accel_h=1e6)
        state = drone.State(np.zeros(3))
        for _ in range(20):  # one time constant
            state = drone.step(state, drone.Command(np.array([10.0, 0.0, 0.0]), 0.0, 0.0), limits, DT)
        self.assertAlmostEqual(state.velocity[0], 10.0 * (1.0 - math.exp(-1.0)), places=6)

    def test_turns_the_short_way_at_the_rate_limit(self):
        limits = drone.Limits()
        state = drone.State(np.zeros(3), yaw=170.0, pitch=-10.0)
        state = drone.step(state, drone.Command(np.zeros(3), -170.0, -120.0), limits, DT)
        self.assertAlmostEqual(state.yaw, 173.0)          # through 180, 60 deg/s x 0.05 s
        self.assertAlmostEqual(state.pitch, -14.5)        # 90 deg/s x 0.05 s
        for _ in range(40):
            state = drone.step(state, drone.Command(np.zeros(3), -170.0, -120.0), limits, DT)
        self.assertAlmostEqual(state.yaw, -170.0)
        self.assertAlmostEqual(state.pitch, limits.pitch_min)  # the gimbal stops at straight down


class FollowTests(unittest.TestCase):
    def test_sqrt_correction_is_continuous_and_can_stop(self):
        kp, accel = 0.5, 2.0
        linear = accel / kp ** 2
        below = np.linalg.norm(follow.sqrt_correction(np.array([linear - 1e-6, 0, 0]), kp, accel))
        above = np.linalg.norm(follow.sqrt_correction(np.array([linear + 1e-6, 0, 0]), kp, accel))
        self.assertAlmostEqual(below, above, places=4)
        for d in (10.0, 50.0, 200.0):
            speed = np.linalg.norm(follow.sqrt_correction(np.array([0, d, 0]), kp, accel))
            self.assertLessEqual(speed, math.sqrt(2 * accel * d))

    def test_holds_the_goal_behind_a_straight_target(self):
        for lag in (0.0, 0.5, 1.0, 2.0):
            q = fly(straight_track(), lag)
            self.assertLess(q["distance_error_m"]["mean"], 0.3, lag)
            self.assertLess(q["aim_error_deg"]["max"], 1.0, lag)
            self.assertEqual(q["target_in_view"], 1.0, lag)

    def test_swings_round_a_turning_target_at_the_right_range(self):
        # A tight synthetic turn (radius 15 m); with the 1 s heading filter the range stays within 2.5 m (p95).
        for lag in (0.0, 0.5, 1.0):
            q = fly(turn_track(), lag)
            self.assertLess(q["range_error_m"]["p95"], 2.5, lag)
            self.assertLess(q["aim_error_deg"]["p95"], 2.0, lag)
            self.assertEqual(q["target_in_view"], 1.0, lag)

    def test_heading_filter_keeps_the_drone_from_overshooting_a_turn(self):
        jumpy = fly(turn_track(), 0.5, follow.Gains(heading_s=0.0))
        smooth = fly(turn_track(), 0.5)
        self.assertGreater(jumpy["range_error_m"]["max"], 10.0)
        self.assertLess(smooth["range_error_m"]["max"], 3.0)
        self.assertLess(smooth["jerk_rms"], jumpy["jerk_rms"])


class CameraAngleTests(unittest.TestCase):
    def test_angles_off_the_optical_axis(self):
        eye = np.zeros(3)
        h, v = metrics.camera_angles(np.array([0.0]), np.array([0.0]), eye, np.array([[10.0, 0.0, 0.0]]))
        self.assertAlmostEqual(h[0], 0.0)
        self.assertAlmostEqual(v[0], 0.0)
        h, v = metrics.camera_angles(np.array([0.0]), np.array([0.0]), eye, np.array([[10.0, 10.0, 10.0]]))
        self.assertAlmostEqual(h[0], 45.0)                 # CARLA y is to the right
        self.assertGreater(v[0], 0.0)                      # z up is up in the image
        h, v = metrics.camera_angles(np.array([90.0]), np.array([-90.0]), eye, np.array([[0.0, 0.0, -5.0]]))
        self.assertAlmostEqual(float(np.hypot(h[0], v[0])), 0.0, places=5)  # looking straight down at it


if __name__ == "__main__":
    unittest.main()
