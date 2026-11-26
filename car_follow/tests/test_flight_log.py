"""Stage 3 flight record (docs/failure_plan.md 5.2): the logged prediction, gate and association are the tracker's
own, the pixel counts are exact, and logging leaves the tracker as it would be without it. No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import copy
import unittest

import numpy as np

from src.car_follow import flight_log, tracker

DT = 0.05


def det(x, y, score=0.9, box=None):
    return tracker.Detection(box or (x, y, x + 10.0, y + 10.0), score, np.array([x, y, 0.0]))


def scene(k):
    """The target moves +x at 10 m/s; a second car runs alongside 2.5 m away; one box has no depth."""
    x = 10.0 * k * DT
    dets = [det(x, 0.0), det(x + 0.5, 2.5, score=0.3), tracker.Detection((0.0, 0.0, 5.0, 5.0), 0.8, None)]
    if k in (6, 7):
        dets = dets[1:]                     # the target is missed for two ticks
    return dets


class FlightLogTests(unittest.TestCase):
    def fly(self, ticks=12, log=True):
        mot = tracker.Tracker()
        mot.update([det(0.0, 0.0), det(0.5, 2.5, score=0.7)], DT)
        target = mot.tracks[0].id
        flog = flight_log.FlightLog(ticks, [11, 22])
        expected = {}
        for k in range(1, ticks):
            dets = scene(k)
            if log:
                t = mot.get(target)
                f = copy.deepcopy(t.filter)
                f.predict(DT)
                expected[k] = (f.position[:2].copy(), 3.0 + 4.0 * f.since_update)
                flog.before_update(mot.tracks, DT)
            mot.update(dets, DT)
            if log:
                flog.after(k, dets, mot.tracks, target)
        return mot, flog, expected, target

    def test_prediction_gate_and_innovation_are_the_trackers(self):
        mot, flog, expected, _ = self.fly()
        a = flog.arrays()
        for k, (pred, gate) in expected.items():
            np.testing.assert_allclose(a["tgt_pred"][k], pred, atol=1e-5)
            self.assertAlmostEqual(float(a["tgt_gate"][k]), gate, places=5)
            x = 10.0 * k * DT
            if k in (6, 7):
                self.assertTrue(np.isnan(a["tgt_innov"][k]))
                self.assertEqual(a["tgt_round"][k], 0)
            else:
                self.assertAlmostEqual(float(a["tgt_innov"][k]), float(np.linalg.norm(pred - [x, 0.0])), places=4)
                self.assertEqual(a["tgt_round"][k], 1)
        self.assertGreater(a["tgt_gate"][7], a["tgt_gate"][5])        # the gate grows while unseen
        self.assertEqual(a["tgt_tracks_in_gate"][3], 1)                # the car alongside is within 3 m
        self.assertEqual(a["tgt_in_gate"][3], 2)
        self.assertLess(a["tgt_near1"][3], a["tgt_near2"][3])

    def test_detections_and_tracks_are_kept_flat(self):
        mot, flog, _, target = self.fly()
        a = flog.arrays()
        self.assertEqual(len(a["det_tick"]), sum(len(scene(k)) for k in range(1, 12)))
        no_depth = np.isnan(a["det_pos"]).all(axis=1)
        self.assertEqual(int(no_depth.sum()), 11)
        last = a["trk_tick"] == 11
        self.assertEqual(sorted(a["trk_id"][last].tolist()), sorted(t.id for t in mot.tracks))
        row = np.flatnonzero(last & (a["trk_id"] == target))[0]
        np.testing.assert_allclose(a["trk_pos"][row], mot.get(target).position, atol=1e-5)
        unmatched = a["trk_tick"] == 6
        self.assertFalse(a["trk_matched"][unmatched & (a["trk_id"] == target)][0])
        self.assertTrue(np.isnan(a["trk_box"][unmatched & (a["trk_id"] == target)]).all())

    def test_logging_leaves_the_tracker_unchanged(self):
        with_log, _, _, _ = self.fly(log=True)
        without, _, _, _ = self.fly(log=False)
        self.assertEqual([t.id for t in with_log.tracks], [t.id for t in without.tracks])
        for a, b in zip(with_log.tracks, without.tracks):
            np.testing.assert_array_equal(a.filter.x, b.filter.x)
            np.testing.assert_array_equal(a.filter.P, b.filter.P)
            self.assertEqual((a.box, a.score, a.hits, a.matched), (b.box, b.score, b.hits, b.matched))

    def test_pixels(self):
        img = np.zeros((20, 30), np.int64)
        img[2:5, 2:12] = 505            # 30 px
        img[10:12, 0:4] = 17            # 8 px
        img[15:20, 20:30] = 999         # not a recorded car
        flog = flight_log.FlightLog(2, [505, 17, 42])
        flog.pixels(1, img)
        np.testing.assert_array_equal(flog.car_px[1], [30, 8, 0])
        np.testing.assert_array_equal(flog.car_px[0], [0, 0, 0])

    def test_empty_record(self):
        a = flight_log.FlightLog(3, [1]).arrays()
        self.assertEqual(a["det_box"].shape, (0, 4))
        self.assertEqual(a["trk_pos"].shape, (0, 3))
        self.assertTrue(np.isnan(a["tgt_pred"]).all())


if __name__ == "__main__":
    unittest.main()
