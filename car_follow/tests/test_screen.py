"""Demo 1 step 3: the screen's geometry (box corners, camera projection, chase camera placement) and drawing.
No simulator (the carla package is used only for its client-side Transform maths).

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import carla
import numpy as np

from src.car_follow import screen, sim


class GeometryTests(unittest.TestCase):
    def test_box_corners_match_carla(self):
        for rot in ((0.0, 0.0, 0.0), (3.0, 47.0, -2.0), (-8.0, -135.0, 5.0)):
            t = carla.Transform(carla.Location(10.0, -4.0, 1.5), carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))
            center, extent = (0.1, 0.0, 0.7), (2.3, 1.0, 0.75)
            ours = screen.box_vertices((10.0, -4.0, 1.5), rot, center, extent)
            for v in ours:
                pass
            signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            theirs = [t.transform(carla.Location(center[0] + sx * extent[0], center[1] + sy * extent[1],
                                                 center[2] + sz * extent[2])) for sx, sy, sz in signs]
            np.testing.assert_allclose(ours, [[p.x, p.y, p.z] for p in theirs], atol=1e-4)

    def test_projection_matches_the_carla_camera_maths(self):
        eye, yaw, pitch = (5.0, 3.0, 20.0), 30.0, -35.0
        cam = screen.Camera(eye, yaw, pitch, 960, 540, 90.0)
        t = carla.Transform(carla.Location(*eye), carla.Rotation(pitch=pitch, yaw=yaw))
        pts = np.array([[20.0, 12.0, 1.0], [25.0, 5.0, 0.0], [18.0, 20.0, 2.0]])
        box = sim.image_box(np.array(t.get_inverse_matrix()), pts, 960, 540, 90.0)
        uv, depth = cam.project(pts)
        self.assertTrue((depth > 0).all())
        np.testing.assert_allclose([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], box, atol=1e-3)

    def test_chase_camera_sees_the_drone_and_the_car(self):
        drone, car = np.array([0.0, 0.0, 15.0]), np.array([20.0, 0.0, 0.8])
        eye, (yaw, pitch) = screen.chase_pose(drone, 0.0, car)
        cam = screen.Camera(eye, yaw, pitch, *screen.SIDE, 90.0)
        uv, depth = cam.project([drone, car])
        self.assertTrue((depth > 0).all())
        self.assertTrue(((uv >= 0) & (uv <= screen.SIDE)).all())
        self.assertGreater(uv[0, 1], uv[1, 1])  # the drone sits below the car in the image


class DrawingTests(unittest.TestCase):
    def test_drone_is_drawn_in_front_and_not_behind(self):
        cam = screen.Camera((0.0, 0.0, 0.0), 0.0, 0.0, 480, 270, 90.0)
        img = np.zeros((270, 480, 3), np.uint8)
        self.assertTrue(screen.draw_drone(img, cam, (6.0, 0.0, 0.0), 30.0))
        self.assertGreater(img[120:150, 220:260].sum(), 0)
        img = np.zeros((270, 480, 3), np.uint8)
        self.assertFalse(screen.draw_drone(img, cam, (-6.0, 0.0, 0.0), 0.0))
        self.assertEqual(img.sum(), 0)

    def test_map_window_puts_the_target_at_the_centre(self):
        raster = screen.MapRaster(np.full((400, 400, 3), 90, np.uint8), origin=(-50.0, -50.0), res=0.25)
        np.testing.assert_allclose(raster.to_world(raster.to_pixel([12.0, -7.0])), [12.0, -7.0])
        text = screen.Text()
        cars = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.3, 1.0, 0.8), "target")]
        img = screen.top_map(raster, text, (0.0, 0.0), cars, (-20.0, 0.0, 15.0), 0.0)
        self.assertEqual(img.shape, (screen.SIDE[1], screen.SIDE[0], 3))
        np.testing.assert_array_equal(img[135, 240], screen.GREEN)

    def test_frame_layout(self):
        text = screen.Text()
        first = screen.first_person(np.zeros((540, 960, 3), np.uint8), text, (400, 200, 520, 290), "GT #12", 25.0,
                                    "follow the white SUV in the left lane", "왼쪽 차선의 흰색 SUV를 따라가",
                                    "t 1.0 s")
        frame = screen.compose(first, np.zeros((270, 480, 3), np.uint8), np.zeros((270, 480, 3), np.uint8))
        self.assertEqual(frame.shape, (540, 1440, 3))
        self.assertGreater(frame[:, :960].sum(), 0)


if __name__ == "__main__":
    unittest.main()
