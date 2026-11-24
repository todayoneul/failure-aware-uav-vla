"""Demo 1 config 3: the video overlay of the first pick (src/car_follow/pick_view.py). Display only; no simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import pick_view, tracker

DT = 0.05


def record(n=3, top=2, log_ratio=11.0, signal="clear", lock_tick=None):
    probs = [0.0] * n
    probs[top] = 1.0
    return {"candidates": n, "boxes": [[100.0 * i, 200.0, 100.0 * i + 60, 240.0] for i in range(n)],
            "scores": [0.9] * n, "top": top, "probs": probs, "log_ratio": log_ratio if n > 1 else None,
            "signal": signal, "seconds": 0.8, "tick": 0, "lock_tick": lock_tick}


class PhaseTests(unittest.TestCase):
    def test_the_steps_follow_the_pick_record(self):
        self.assertEqual(pick_view.phase(0, None, DT), pick_view.INPUT)
        r = record()
        self.assertEqual(pick_view.phase(10, r, DT), pick_view.CHOOSING)     # 0.5 s < 0.8 s of computing
        self.assertEqual(pick_view.phase(20, r, DT), pick_view.ANSWERED)
        r["lock_tick"] = 25
        self.assertEqual(pick_view.phase(30, r, DT), pick_view.LOCK)
        self.assertEqual(pick_view.phase(60, r, DT), pick_view.TRACK)

    def test_a_lost_track_and_a_climb_show_on_the_strip(self):
        self.assertEqual(pick_view.step_states(pick_view.TRACK, False, 0.0)[3], pick_view.WARN)
        self.assertEqual(pick_view.step_states(pick_view.TRACK, True, 1.5)[4], pick_view.ON)
        self.assertEqual(pick_view.climb(0.05, 0.1), 0.0)


class CarryTests(unittest.TestCase):
    def test_boxes_follow_the_overlapping_detection_or_stay(self):
        v = pick_view.PickView(8.0, 1.25, DT)
        r = record(n=2, top=1)
        v.carry(r, [])
        moved = tracker.Detection(box=(5.0, 200.0, 65.0, 240.0), score=0.9, position=None)
        v.carry(r, [moved])
        self.assertEqual(v.boxes[0], moved.box)
        self.assertEqual(v.boxes[1], tuple(r["boxes"][1]))


class DrawTests(unittest.TestCase):
    def test_every_phase_draws(self):
        v = pick_view.PickView(8.0, 1.25, DT)
        r = record()
        for k, lock in ((0, None), (20, None), (26, 25), (80, 25)):
            r["lock_tick"] = lock
            img = np.zeros((540, 960, 3), np.uint8)
            v.carry(r, [])
            v.draw(img, k, r, tracking=k < 80, climb_m=1.2)
            self.assertGreater(img.sum(), 0)

    def test_one_candidate_draws_without_a_ratio(self):
        r = record(n=1, top=0, signal="cannot_verify")
        self.assertIn("확인 불가", pick_view.answer_lines(r, 8.0)[-1])
        still = pick_view.still(np.zeros((540, 960, 3), np.uint8), r, 8.0, "follow the car", "en")
        self.assertEqual(still.shape, (540, 960, 3))


if __name__ == "__main__":
    unittest.main()
