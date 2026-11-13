"""Observation framing must never change physics or model input cameras."""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DemoViewConfigTests(unittest.TestCase):
    def test_only_chase_capture_changes_and_source_is_untouched(self):
        from scripts.demo_view_config import build_demo_robot
        original = json.loads((ROOT / "configs/robot_quadrotor_fastphysics.jsonc").read_text())
        saved = copy.deepcopy(original)
        demo = build_demo_robot(original)
        self.assertEqual(original, saved)
        without_chase = lambda robot: {**robot, "sensors": [
            s for s in robot["sensors"] if s["id"] != "Chase"]}
        self.assertEqual(without_chase(demo), without_chase(original))
        for camera in ("FrontCamera", "DownCamera"):
            self.assertEqual(next(s for s in demo["sensors"] if s["id"] == camera),
                             next(s for s in original["sensors"] if s["id"] == camera))

    def test_relative_camera_optical_axis_points_to_drone(self):
        from scripts.demo_view_config import VIEW_PROFILES, camera_pose
        for name, profile in VIEW_PROFILES.items():
            with self.subTest(name=name):
                pose = camera_pose(profile)
                t, q = pose["translation"], pose["rotation"]
                length = math.hypot(t["x"], t["z"])
                target = [-t["x"] / length, 0, -t["z"] / length]
                optical_axis = [1 - 2*q["y"]**2, 0, -2*q["w"]*q["y"]]
                for actual, expected in zip(optical_axis, target):
                    self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
