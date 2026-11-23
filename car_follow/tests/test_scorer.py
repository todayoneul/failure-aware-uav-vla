"""Demo 1 step 4: the scorer's definitions (which car the box points at, switches, hold, the summary). No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import scorer

CARS = [101, 202, 303]          # actor IDs of the recorded cars: target, a look-alike, a background car
KIND = ["target", "lookalike", "background"]


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((40, 60), np.int32)
        self.img[10:20, 10:30] = 101      # target, 200 px
        self.img[10:20, 30:36] = 202      # look-alike, 60 px
        self.img[0:5, 0:60] = 999         # something that is not a recorded car

    def test_the_car_with_most_pixels_owns_the_box(self):
        owner, share = scorer.box_owner(self.img, (0, 0, 60, 40), CARS)
        self.assertEqual(owner, 0)
        self.assertAlmostEqual(share, 200 / 260)
        self.assertEqual(scorer.box_owner(self.img, (28, 8, 40, 22), CARS)[0], 1)

    def test_no_box_or_no_car(self):
        self.assertEqual(scorer.box_owner(self.img, None, CARS), (-1, 0.0))
        self.assertEqual(scorer.box_owner(self.img, (0, 0, 60, 5), CARS), (-1, 0.0))
        self.assertEqual(scorer.box_owner(self.img, (-30, -30, -5, -5), CARS), (-1, 0.0))

    def test_mask_box(self):
        self.assertEqual(scorer.mask_box(self.img, 101), (10.0, 10.0, 30.0, 20.0))
        self.assertIsNone(scorer.mask_box(self.img, 404))


class SwitchTests(unittest.TestCase):
    def test_short_flicker_is_not_a_switch(self):
        self.assertEqual(scorer.switches([0] * 10 + [1] * 4 + [0] * 10), [])
        self.assertEqual(scorer.switches([0, 1] * 20), [])

    def test_five_ticks_on_another_car_is_a_switch(self):
        self.assertEqual(scorer.switches([0] * 10 + [1] * 5 + [0] * 3), [(1, 10, 5)])

    def test_reattaching_to_another_car_after_a_loss_is_a_switch(self):
        self.assertEqual(scorer.switches([0] * 5 + [-1] * 3 + [2] * 6), [(2, 8, 6)])

    def test_a_wrong_start_before_the_first_lock_is_not_a_switch(self):
        self.assertEqual(scorer.switches([1] * 10 + [0] * 10), [])
        self.assertEqual(scorer.switches([-1] * 20), [])


class ScoreTests(unittest.TestCase):
    def test_trial_and_summary(self):
        n = 100
        owner = np.zeros(n, int)
        owner[40:50] = 1                  # a 10-tick switch to the look-alike
        target_px = np.full(n, 500)
        target_px[:10] = 0                # not visible at first
        pos = np.zeros((n, 3, 3))
        pos[:, 1, 0] = 8.0                # the look-alike drives 8 m beside the target the whole time
        pos[:, 2, 0] = 50.0
        row = scorer.score_trial(owner, target_px, KIND, pos, 1.5)
        self.assertFalse(row["switch_free"])
        self.assertEqual(row["switches"][0]["kind"], "lookalike")
        self.assertEqual(row["visible_ticks"], 90)
        self.assertEqual(row["held_ticks"], 80)
        self.assertAlmostEqual(row["near_lookalike_s"], 5.0)
        clean = scorer.score_trial(np.zeros(n, int), target_px, KIND, pos, 2.0, first_pick=True)
        s = scorer.summarize([row, clean])
        self.assertEqual(s["switch_free"], 0.5)
        self.assertAlmostEqual(s["hold"], 170 / 180, places=4)
        self.assertEqual(s["first_pick"], 1.0)
        self.assertEqual(s["switch_partners"], {"lookalike": 1, "background": 0})
        self.assertFalse(s["pass"]["switch_free"])
        self.assertTrue(s["pass"]["distance_error_m"])


if __name__ == "__main__":
    unittest.main()
