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
        n = 200
        owner = np.zeros(n, int)
        owner[140:150] = 1                # a 10-tick switch to the look-alike
        target_px = np.full(n, 500)
        target_px[:10] = 0                # not visible at first (camera warm-up)
        target_px[60:90] = 0              # hidden 1.5 s: normal tracking, stays in the main window
        owner[60:90] = -1
        pos = np.zeros((n, 3, 3))
        pos[:, 1, 0] = 8.0                # the look-alike drives 8 m beside the target the whole time
        pos[:, 2, 0] = 50.0
        dist = np.full(n, 1.5)
        dist[:100] = 9.0                  # the first 5 s are left out
        row = scorer.score_trial(owner, target_px, KIND, pos, dist, np.full(n, 1.0))
        self.assertFalse(row["switch_free"])
        self.assertEqual(row["switches"][0]["kind"], "lookalike")
        self.assertEqual(row["visible_ticks"], 160)
        self.assertEqual(row["held_ticks"], 150)
        self.assertEqual(row["distance_error_m"], 1.5)
        self.assertEqual(row["horizontal_error_m"], 1.0)
        self.assertEqual(row["long_occlusions"], [])
        self.assertIsNone(row["after"])
        self.assertAlmostEqual(row["near_lookalike_s"], 10.0)
        clean = scorer.score_trial(np.zeros(n, int), target_px, KIND, pos, np.full(n, 2.0), first_pick=True)
        s = scorer.summarize([row, clean])
        self.assertEqual(s["switch_free"], 0.5)
        self.assertAlmostEqual(s["hold"], 310 / 320, places=4)
        self.assertEqual(s["first_pick"], 1.0)
        self.assertEqual(s["switch_partners"], {"lookalike": 1, "background": 0})
        self.assertFalse(s["pass"]["switch_free"])
        self.assertTrue(s["pass"]["distance_error_m"])
        self.assertEqual(s["after_long_occlusion"]["trials"], 0)

    def test_a_long_occlusion_splits_the_trial(self):
        n = 400
        target_px = np.full(n, 500)
        target_px[150:210] = 0            # hidden 3 s from 7.5 s
        owner = np.zeros(n, int)
        owner[150:210] = -1
        owner[210:230] = 2                # back on a background car first ...
        pos = np.zeros((n, 3, 3))
        pos[:, 1, 0], pos[:, 2, 0] = 30.0, 50.0
        dist = np.r_[np.full(150, 1.0), np.full(n - 150, 6.0)]
        row = scorer.score_trial(owner, target_px, KIND, pos, dist)
        self.assertEqual(row["long_occlusions"], [{"start_s": 7.5, "length_s": 3.0}])
        self.assertEqual(row["main_s"], 7.5)
        self.assertTrue(row["switch_free"])           # the main window is clean
        self.assertEqual(row["hold"], 1.0)
        self.assertEqual(row["distance_error_m"], 1.0)
        after = row["after"]
        self.assertFalse(after["switch_free"])        # the box came back on another car
        self.assertTrue(after["reacquired"])          # ... and then on the target
        self.assertEqual(after["reacquired_after_s"], 1.0)
        self.assertEqual(after["distance_error_m"], 6.0)
        s = scorer.summarize([row])
        self.assertEqual(s["after_long_occlusion"]["trials"], 1)
        self.assertEqual(s["after_long_occlusion"]["reacquired"], 1)
        self.assertTrue(s["pass"]["switch_free"])

    def test_avoidance_record(self):
        dt = 0.05
        offset = np.zeros(200)
        offset[50:100] = 3.0              # one 2.5 s climb
        need = np.where(offset > 0, 3.0, 0.0)
        rise = offset * 0.9
        nearest = np.full(200, 12.0)
        nearest[0] = 0.5                  # camera warm-up, left out
        nearest[60] = 2.0
        a = scorer.avoidance(offset, need, np.zeros(200), ["clear"] * 200, rise, nearest, dt)
        self.assertEqual((a["time_s"], a["events"], a["offset_max_m"]), (2.5, 1, 3.0))
        self.assertEqual(a["rise_max_m"], 2.7)
        self.assertEqual((a["path_need_s"], a["los_need_s"]), (2.5, 0.0))
        self.assertEqual((a["nearest_min_m"], a["nearest_below_clearance_s"]), (2.0, 0.05))


if __name__ == "__main__":
    unittest.main()
