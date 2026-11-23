"""Demo 1 step 6: the VLM pick around the model (src/car_follow/vlm.py), with a fake model. No GPU.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import vlm


class FakePicker:
    """Answers by box position or by number: `by` = "box" always favours the box at `box` (seen through the number
    drawn on it), "number" always favours the number 1, whatever box carries it."""

    def __init__(self, by, box=0, numbers_seen=None):
        self.by, self.box, self.calls = by, box, []

    def label_logits(self, rgb, text, labels):
        self.calls.append(text)
        logits = np.zeros(len(labels))
        if self.by == "number":
            logits[0] = 5.0
        else:
            logits[self.current_numbers[self.box] - 1] = 5.0
        return logits


class PickTests(unittest.TestCase):
    def setUp(self):
        self.bgr = np.zeros((540, 960, 3), np.uint8)
        self.boxes = [(100, 100, 150, 150), (300, 100, 350, 150), (500, 100, 550, 150)]

    def run_pick(self, picker):
        original = vlm.draw_marks

        def spy(bgr, boxes, labels, **kw):        # remember which number each box got in this shuffle
            picker.current_numbers = list(labels)
            return original(bgr, boxes, labels, **kw)
        vlm.draw_marks = spy
        try:
            return vlm.pick(picker, self.bgr, self.boxes, "follow the white SUV", permutations=3, seed=1)
        finally:
            vlm.draw_marks = original

    def test_a_model_that_sees_the_box_keeps_its_answer_under_shuffles(self):
        r = self.run_pick(FakePicker("box", box=2))
        self.assertEqual(r.top, 2)
        self.assertEqual(r.agreement, 1.0)
        self.assertGreater(r.margin, 0.9)
        self.assertAlmostEqual(float(r.probs.sum()), 1.0)

    def test_a_number_bias_is_spread_out_by_the_shuffles(self):
        r = self.run_pick(FakePicker("number"))
        self.assertLess(r.agreement, 1.0)          # the favoured box moves with the number 1
        self.assertLess(r.margin, 0.9)
        self.assertEqual(len(r.shuffles), 3)
        self.assertEqual(r.shuffles[0]["numbers"], [1, 2, 3])   # the first shuffle keeps the plain order

    def test_limits(self):
        self.assertIsNone(vlm.pick(FakePicker("box"), self.bgr, [], "x"))
        with self.assertRaises(ValueError):
            vlm.pick(FakePicker("box"), self.bgr, [(0, 0, 1, 1)] * 10, "x")
        self.assertEqual(vlm.candidates([None] * 12, list(range(12)))[:3], [11, 10, 9])
        self.assertEqual(len(vlm.candidates([None] * 12, list(range(12)))), vlm.MAX_CANDIDATES)

    def test_marks_are_drawn_without_touching_the_input(self):
        img = vlm.draw_marks(self.bgr, self.boxes, [1, 2, 3])
        self.assertEqual(int(self.bgr.sum()), 0)
        self.assertGreater(int(img.sum()), 0)


if __name__ == "__main__":
    unittest.main()
