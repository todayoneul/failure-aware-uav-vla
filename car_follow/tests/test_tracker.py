"""Demo 1 step 5: the world-coordinate ByteTrack-style tracker (src/car_follow/tracker.py). No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import tracker

DT = 0.05


def det(x, y, score=1.0, tag=None):
    return tracker.Detection(box=(tag, x, y), score=score, position=np.array([x, y, 0.0]))


def ids_of(tracks):
    return sorted(t.id for t in tracks)


class TrackerTests(unittest.TestCase):
    def test_two_cars_side_by_side_keep_their_tracks(self):
        rng = np.random.default_rng(1)
        mot = tracker.Tracker()
        owner = {}
        for k in range(300):
            a = det(10.0 * k * DT + rng.normal(0, 0.3), 0.0 + rng.normal(0, 0.3), tag="a")
            b = det(10.0 * k * DT + rng.normal(0, 0.3), 3.5 + rng.normal(0, 0.3), tag="b")
            for t in mot.update([a, b], DT):
                if t.matched:
                    owner.setdefault(t.id, set()).add(t.box[0])
        self.assertEqual(len(mot.tracks), 2)
        self.assertTrue(all(len(v) == 1 for v in owner.values()))   # no track ever took the other car

    def test_overtaking_in_the_next_lane_keeps_identities(self):
        mot = tracker.Tracker()
        owner = {}
        for k in range(400):                       # b overtakes a at 4 m/s more, one lane over
            a = det(10.0 * k * DT, 0.0, tag="a")
            b = det(-10.0 + 14.0 * k * DT, 3.5, tag="b")
            for t in mot.update([a, b], DT):
                if t.matched:
                    owner.setdefault(t.id, set()).add(t.box[0])
        self.assertEqual(len(owner), 2)
        self.assertTrue(all(len(v) == 1 for v in owner.values()))

    def test_a_short_gap_is_bridged_and_a_long_one_is_not(self):
        for gap_s, same in ((1.0, True), (2.0, False)):
            mot = tracker.Tracker()
            first = None
            for k in range(200):
                hidden = 60 <= k < 60 + int(gap_s / DT)
                mot.update([] if hidden else [det(8.0 * k * DT, 0.0)], DT)
                if k == 10:
                    first = ids_of(mot.tracks)
            self.assertEqual(ids_of(mot.tracks) == first, same, gap_s)

    def test_low_scores_extend_tracks_but_do_not_start_them(self):
        mot = tracker.Tracker()
        mot.update([det(0.0, 0.0, 0.3)], DT)
        self.assertEqual(mot.tracks, [])
        mot.update([det(0.0, 0.0, 0.9)], DT)
        tid = mot.tracks[0].id
        for k in range(40):                         # partly hidden: low scores keep it alive
            mot.update([det(0.05 * k, 0.0, 0.3)], DT)
        self.assertEqual(ids_of(mot.tracks), [tid])
        self.assertTrue(mot.tracks[0].matched)

    def test_a_detection_outside_the_gate_starts_a_new_track(self):
        mot = tracker.Tracker()
        mot.update([det(0.0, 0.0)], DT)
        mot.update([det(10.0, 0.0)], DT)
        self.assertEqual(len(mot.tracks), 2)

    def test_detections_without_depth_are_ignored(self):
        mot = tracker.Tracker()
        mot.update([tracker.Detection((0, 0, 1, 1), 1.0, None)], DT)
        self.assertEqual(mot.tracks, [])


if __name__ == "__main__":
    unittest.main()
