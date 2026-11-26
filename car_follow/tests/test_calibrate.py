"""Stage 3 calibration (docs/failure_plan.md 6) against values worked out by hand and the ranks of table 7.
No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import math
import unittest

import numpy as np

from src.car_follow import calibrate


class AlarmRuleTests(unittest.TestCase):
    def test_k_ticks_in_a_row(self):
        s = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, np.nan])
        np.testing.assert_array_equal(calibrate.alarm_from_score(s, 0.5, k=3),
                                      [False, False, False, False, False, False, True, False])

    def test_flight_score_is_the_largest_firing_threshold(self):
        rng = np.random.default_rng(0)
        for k in (1, 2, 4):
            s = rng.normal(size=200)
            mask = np.zeros(200, bool)
            mask[30:150] = True
            f = calibrate.flight_score(s, mask, k)
            self.assertTrue((calibrate.alarm_from_score(s, f, k) & mask).any())
            self.assertFalse((calibrate.alarm_from_score(s, math.nextafter(f, math.inf), k) & mask).any())
        self.assertEqual(calibrate.flight_score(np.ones(5), np.zeros(5, bool)), -math.inf)


class ThresholdTests(unittest.TestCase):
    def test_ranks_of_table_7(self):
        for n, alpha, rank in ((40, 0.10, 37), (60, 0.10, 55), (100, 0.10, 91),
                               (40, 0.05, 39), (60, 0.05, 58), (100, 0.05, 96)):
            s = list(range(1, n + 1))
            self.assertEqual(calibrate.rank_threshold(s, alpha), math.nextafter(rank, math.inf))
        for n, rank in ((40, 39), (60, 58), (100, 95)):
            self.assertEqual(calibrate.pac_rank(n, 0.10, 0.10), rank)

    def test_crc_equals_the_rank_with_one_flight_per_trial(self):
        rng = np.random.default_rng(1)
        s = rng.normal(size=60).tolist()
        out = calibrate.crc_threshold([[v] for v in s], 0.10)
        self.assertEqual(out["tau"], calibrate.rank_threshold(s, 0.10))
        self.assertEqual(sum(v >= out["tau"] for v in s), 5)          # 60 - 55
        self.assertLessEqual(out["bound"], 0.10)

    def test_crc_with_two_flights_per_trial(self):
        # n 9, alpha 0.2: (9 R + 1) / 10 <= 0.2 needs R <= 1/9, that is at most two flights of the 18 alarming.
        trials = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0],
                  [15.0, 16.0], [17.0, 18.0]]
        out = calibrate.crc_threshold(trials, 0.20)
        self.assertEqual(out["tau"], math.nextafter(16.0, math.inf))
        self.assertAlmostEqual(out["risk"], 1 / 9, places=4)

    def test_too_few_trials(self):
        out = calibrate.crc_threshold([[1.0]] * 5, 0.10)              # 1 / 6 > 0.1 even with no alarm
        self.assertFalse(out["feasible"])
        self.assertEqual(out["tau"], math.inf)

    def test_strict_rate_at_the_threshold(self):
        main = [[float(i)] for i in range(20)]
        strict = [[float(i) + (5.0 if i % 4 == 0 else 0.0)] for i in range(20)]
        out = calibrate.calibrate(main, strict, 0.10)
        tau = out["threshold"]["tau"]
        self.assertEqual(tau, math.nextafter(18.0, math.inf))          # rank ceil(21 * 0.9) = 19
        self.assertAlmostEqual(out["strict_risk_at_threshold"], sum(s[0] >= tau for s in strict) / 20)
        self.assertGreater(out["strict_threshold"]["tau"], tau)


if __name__ == "__main__":
    unittest.main()
