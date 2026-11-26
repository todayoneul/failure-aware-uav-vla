"""Stage 3 metrics (docs/failure_plan.md 3.1, 6) on synthetic labels and alarms, against values worked out by hand.
No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import failure_metrics as fm

N, LOCK = 400, 25


def labels(t_fail=None, failure=None, near=(), recovered=None):
    return {"ticks": N, "lock_tick": LOCK, "t_fail_tick": t_fail, "failure": failure, "recovered_tick": recovered,
            "near_miss": [{"start_tick": s, "ticks": n, "why": why} for s, n, why in near],
            "near_miss_flight": any("onset" not in why for _, _, why in near)}


def alarm_at(*ticks):
    a = np.zeros(N, bool)
    for t in ticks:
        a[t] = True
    return a


class MaskTests(unittest.TestCase):
    def test_normal_stretch(self):
        m = fm.normal_mask(labels(t_fail=200, failure="drift"))
        self.assertEqual((np.flatnonzero(m)[0], np.flatnonzero(m)[-1], m.sum()), (25, 139, 115))   # to 200 - 60
        self.assertEqual(fm.normal_mask(labels()).sum(), N - LOCK)

    def test_near_miss_windows_and_strict(self):
        lab = labels(near=[(150, 10, ["estimate"])])
        m = fm.normal_mask(lab)
        self.assertFalse(m[90:160].any())          # the run and the 3 s before it
        self.assertTrue(m[89] and m[160])
        self.assertTrue(fm.normal_mask(lab, strict=True)[90:160].all())


class FlightTests(unittest.TestCase):
    def test_false_alarm(self):
        r = fm.flight_metrics(labels(), alarm_at(300))
        self.assertTrue(r["false_alarm"] and r["false_alarm_strict"])
        self.assertEqual(r["first_false_alarm_s"], 15.0)
        self.assertIsNone(r["detected"])
        self.assertFalse(fm.flight_metrics(labels(), alarm_at(10))["false_alarm"])   # before the lock

    def test_alarm_in_a_near_miss_is_strict_only(self):
        r = fm.flight_metrics(labels(near=[(150, 10, ["box"])]), alarm_at(120))
        self.assertFalse(r["false_alarm"])
        self.assertTrue(r["false_alarm_strict"])
        self.assertTrue(r["near_miss_alarm"] and r["near_miss_flight"])

    def test_detection_delay_and_wrong_follow_time(self):
        lab = labels(t_fail=200, failure="switch_box")
        early = fm.flight_metrics(lab, alarm_at(170))
        self.assertEqual((early["detected"], early["delay_s"], early["wrong_follow_s"]), (True, -1.5, 0.0))
        self.assertFalse(early["false_alarm"])
        late = fm.flight_metrics(lab, alarm_at(220))
        self.assertEqual((late["delay_s"], late["wrong_follow_s"]), (1.0, 1.0))
        too_early = fm.flight_metrics(lab, alarm_at(100))
        self.assertTrue(too_early["false_alarm"])
        self.assertFalse(too_early["detected"])
        self.assertEqual(too_early["wrong_follow_s"], 10.0)     # to the end: (400 - 200) ticks

    def test_wrong_follow_time_ends_at_recovery(self):
        lab = labels(t_fail=200, failure="drift", recovered=260)
        r = fm.flight_metrics(lab, alarm_at(300))                # after the recovery: not a detection
        self.assertFalse(r["detected"])
        self.assertEqual(r["wrong_follow_s"], 3.0)

    def test_lead_before(self):
        self.assertEqual(fm.lead_before(alarm_at(150), 200), 2.5)
        self.assertIsNone(fm.lead_before(alarm_at(50), 200))


class BoundTests(unittest.TestCase):
    def test_clopper_pearson_values_of_the_plan(self):
        lo, hi = fm.clopper_pearson(9, 90)
        self.assertAlmostEqual(hi, 0.1814, places=4)      # docs/failure_plan.md 6.3
        self.assertAlmostEqual(fm.clopper_pearson(30, 30)[0], 0.8843, places=4)   # 4.4
        self.assertAlmostEqual(fm.clopper_pearson(27, 30)[0], 0.7347, places=4)
        self.assertEqual(fm.clopper_pearson(0, 10)[0], 0.0)
        self.assertEqual(fm.clopper_pearson(10, 10)[1], 1.0)

    def test_summary(self):
        rows = [fm.flight_metrics(labels(), alarm_at(300)), fm.flight_metrics(labels(), alarm_at()),
                fm.flight_metrics(labels(t_fail=200, failure="drift"), alarm_at(220)),
                fm.flight_metrics(labels(t_fail=200, failure="loss"), alarm_at())]
        s = fm.summarize(rows, trial_of=["a", "a", "b", "b"])
        self.assertEqual((s["false_alarm"]["k"], s["false_alarm"]["n"]), (1, 4))
        self.assertEqual((s["trial_false_alarm"]["k"], s["trial_false_alarm"]["n"]), (1, 2))
        self.assertEqual((s["failures"], s["detected"]["k"]), (2, 1))
        self.assertEqual(s["wrong_follow_s"]["total"], 11.0)    # 1.0 + 10.0
        self.assertEqual(s["by_failure"], {"drift": {"flights": 1, "detected": 1},
                                           "loss": {"flights": 1, "detected": 0}})


if __name__ == "__main__":
    unittest.main()
