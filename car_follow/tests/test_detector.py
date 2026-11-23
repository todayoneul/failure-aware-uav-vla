"""Demo 1 step 5: the detector interface (src/car_follow/detector.py) without any model. No simulator, no GPU.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import detector
from src.car_follow.detector import Box


class BoxTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(detector.iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)
        self.assertEqual(detector.iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_dedupe_keeps_the_best_of_overlapping_boxes_across_labels(self):
        boxes = [Box(0, 0, 10, 10, 0.19, "truck"), Box(0, 0, 10, 10.5, 0.84, "car"), Box(30, 0, 40, 10, 0.5, "car")]
        kept = detector.dedupe(boxes)
        self.assertEqual([(b.label, b.score) for b in kept], [("car", 0.84), ("car", 0.5)])

    def test_match_truth_is_one_to_one(self):
        boxes = [Box(0, 0, 10, 10, 0.9, "car"), Box(1, 0, 11, 10, 0.8, "car"), Box(50, 50, 60, 60, 0.7, "car")]
        truth = [(0, 0, 10, 10)]
        got, unmatched = detector.match_truth(boxes, truth)
        self.assertEqual(got, {0: (0, 1.0)})
        self.assertEqual(unmatched, [1, 2])


class GroundTruthTests(unittest.TestCase):
    def test_visible_boxes_of_recorded_cars(self):
        ids = np.zeros((40, 60), np.int32)
        ids[10:20, 10:30] = 101          # 200 px
        ids[30:32, 40:45] = 202          # 10 px: too small
        ids[0:5, :] = 999                # not a recorded car
        boxes = detector.make("gt").detect(detector.Frame(None, ids, (101, 202)))
        self.assertEqual([b.xyxy for b in boxes], [(10.0, 10.0, 30.0, 20.0)])
        self.assertEqual(boxes[0].score, 1.0)

    def test_unknown_detector(self):
        with self.assertRaises(ValueError):
            detector.make("nope")


if __name__ == "__main__":
    unittest.main()
