"""A fresh checkout must prepare the same camera configuration without audit files."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectAirSimConfigTests(unittest.TestCase):
    def test_prepare_config_in_checkout_without_outputs_or_upstream_source(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            (checkout / "scripts").mkdir()
            (checkout / "configs").mkdir()
            script = checkout / "scripts/projectairsim_probe.py"
            shutil.copyfile(ROOT / "scripts/projectairsim_probe.py", script)
            scene = {"id": "SceneBasicDrone", "actors": [{"name": "Drone1",
                     "robot-config": "robot_quadrotor_fastphysics.jsonc"}]}
            robot = {"physics-type": "fast-physics", "sensors": [
                {"id": camera, "type": "camera", "origin": {"rpy-deg": rotation},
                 "capture-interval": 0.0333333, "capture-settings": [
                     {"image-type": 0, "width": 256, "height": 256,
                      "capture-enabled": True, "streaming-enabled": camera == "Chase"}]}
                for camera, rotation in (("FrontCamera", "0 0 0"),
                                         ("DownCamera", "0 -90 0"), ("Chase", "0 0 0"))]}
            expected = {"scene_basic_drone.jsonc": scene,
                        "robot_quadrotor_fastphysics.jsonc": robot}
            for name, content in expected.items():
                (checkout / "configs" / name).write_text(json.dumps(content))

            spec = importlib.util.spec_from_file_location("checkout_probe", script)
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
            probe.prepare_config()

            for name, content in expected.items():
                self.assertEqual(json.loads((probe.CONFIG / name).read_text()), content)
            self.assertFalse((checkout / "outputs/platform_final/source").exists())


if __name__ == "__main__":
    unittest.main()
