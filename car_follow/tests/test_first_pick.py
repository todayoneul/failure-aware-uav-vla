"""Demo 1 config 3: the first pick in flight (src/car_follow/first_pick.py). No GPU, no simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import first_pick, tracker, vlm

DT = 0.05


def det(x, score, position=True):
    return tracker.Detection(box=(float(x), 10.0, x + 20.0, 30.0), score=score,
                             position=np.array([float(x), 0.0, 0.0]) if position else None)


class Prefers:
    """A fake VLM that always prefers one number."""

    def __init__(self, number, gap=20.0):
        self.number, self.gap = number, gap

    def label_logits(self, rgb, text, labels):
        return np.array([self.gap if int(label) == self.number else 0.0 for label in labels])


def request(model, threshold=8.0):
    return first_pick.Request(model=model, sentence="follow the car", seed=1, threshold=threshold, permutations=1)


class CandidateTests(unittest.TestCase):
    def test_weak_boxes_are_left_out_but_a_box_without_a_position_stays(self):
        dets = [det(0, 0.2), det(30, 0.9), det(60, 0.4), det(90, 0.8, position=False)]
        self.assertEqual(first_pick.candidate_indices(dets), [1, 3, 2])

    def test_at_most_nine_candidates_with_the_highest_scores(self):
        dets = [det(30 * i, 0.3 + i * 0.05) for i in range(12)]
        keep = first_pick.candidate_indices(dets)
        self.assertEqual(len(keep), vlm.MAX_CANDIDATES)
        self.assertEqual(keep[0], 11)
        self.assertNotIn(0, keep)


class SignalTests(unittest.TestCase):
    def test_one_candidate_cannot_be_verified(self):
        self.assertEqual(first_pick.signal(1, float("inf"), 8.0), first_pick.CANNOT_VERIFY)

    def test_the_threshold_separates_clear_from_ambiguous(self):
        self.assertEqual(first_pick.signal(3, 8.0, 8.0), first_pick.CLEAR)
        self.assertEqual(first_pick.signal(3, 7.9, 8.0), first_pick.AMBIGUOUS)


class ChooseTests(unittest.TestCase):
    bgr = np.zeros((100, 200, 3), np.uint8)

    def test_returns_the_detection_the_model_named(self):
        dets = [det(0, 0.4), det(50, 0.9)]
        r = request(Prefers(2))             # candidates are ordered by score, so number 2 is the 0.4 box
        self.assertIs(first_pick.choose(r, self.bgr, dets), dets[0])
        self.assertEqual(r.record["signal"], first_pick.CLEAR)

    def test_a_low_ratio_is_recorded_as_ambiguous(self):
        r = request(Prefers(1, gap=1.0))
        first_pick.choose(r, self.bgr, [det(0, 0.9), det(50, 0.8)])
        self.assertEqual(r.record["signal"], first_pick.AMBIGUOUS)

    def test_a_single_candidate_records_no_margin(self):
        r = request(Prefers(1))
        first_pick.choose(r, self.bgr, [det(0, 0.9)])
        self.assertEqual(r.record["signal"], first_pick.CANNOT_VERIFY)
        self.assertIsNone(r.record["margin"])
        self.assertIsNone(r.record["log_ratio"])

    def test_no_candidate_means_ask_again(self):
        r = request(Prefers(1))
        self.assertIsNone(first_pick.choose(r, self.bgr, [det(0, 0.1)]))
        self.assertIsNone(r.record)


class HandoffTests(unittest.TestCase):
    def test_the_box_is_carried_until_it_has_a_position(self):
        first = det(0, 0.9, position=False)
        h = first_pick.Handoff(first.box, DT)
        self.assertIsNone(h.step([det(2, 0.8, position=False), det(100, 0.9)]))
        later = det(4, 0.8)
        self.assertIs(h.step([det(100, 0.9), later]), later)

    def test_it_does_not_jump_to_a_car_that_does_not_overlap(self):
        h = first_pick.Handoff(det(0, 0.9).box, DT)
        self.assertIsNone(h.step([det(100, 0.9)]))
        self.assertEqual(h.missed, 1)

    def test_it_gives_up_after_the_tracker_buffer(self):
        h = first_pick.Handoff(det(0, 0.9).box, DT, patience_s=0.1)
        for _ in range(3):
            h.step([])
        self.assertTrue(h.lost)


class AdoptTests(unittest.TestCase):
    def test_a_detection_with_a_track_keeps_it(self):
        mot = tracker.Tracker()
        d = det(0, 0.9)
        mot.update([d], DT)
        self.assertIs(mot.adopt(d, DT), mot.tracks[0])
        self.assertEqual(len(mot.tracks), 1)

    def test_a_weak_detection_the_vlm_named_gets_a_track(self):
        mot = tracker.Tracker()
        d = det(0, 0.4)                     # below the score that starts a track
        mot.update([d], DT)
        self.assertEqual(mot.tracks, [])
        t = mot.adopt(d, DT)
        self.assertIs(mot.get(t.id), t)
        self.assertIs(t.box, d.box)

    def test_a_detection_without_a_position_gets_none(self):
        self.assertIsNone(tracker.Tracker().adopt(det(0, 0.9, position=False), DT))


class SummaryTests(unittest.TestCase):
    @staticmethod
    def pick(right, sig, seconds=1.0, candidate=True):
        return {"right": right, "signal": sig, "seconds": seconds, "margin": 0.9, "log_ratio": 9.0,
                "picked_kind": "target" if right else "lookalike", "target_is_candidate": candidate}

    def test_counts_by_signal_and_budget(self):
        rows = {"a": {"pick": self.pick(True, first_pick.CLEAR), "held_after_lock": 90, "visible_after_lock": 100},
                "b": {"pick": self.pick(False, first_pick.AMBIGUOUS, seconds=1.5, candidate=False),
                      "held_after_lock": 0, "visible_after_lock": 100},
                "c": {"pick": None}}
        s = first_pick.summarize(rows)
        self.assertEqual((s["picks"], s["right"], s["no_pick"]), (2, 1, ["c"]))
        self.assertEqual(s["by_signal"][first_pick.AMBIGUOUS], {"n": 1, "right": 0})
        self.assertEqual(s["over_delay_budget"], 1)
        self.assertEqual(s["target_not_a_candidate"], ["b"])
        self.assertEqual(s["hold_after_lock"], 0.45)


if __name__ == "__main__":
    unittest.main()
