"""Stage 3 failure trials (docs/failure_plan.md 4): the distractor's words per level and the checks that decide from
a recording alone whether an event happened. No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import random
import unittest

import numpy as np

from src.car_follow import failure_trials as ft

DT = 0.05
N = 1200
TYPES = {"a.sedan1": {"type": "sedan", "target": True}, "a.sedan2": {"type": "sedan", "target": True},
         "a.suv": {"type": "SUV", "target": True}, "a.police": {"type": "sedan", "target": False}}
RED, DARK_RED, BLUE = "200,20,20", "150,10,10", "20,40,200"
COLORS = {"a.sedan1": [RED, DARK_RED, BLUE], "a.sedan2": [RED], "a.suv": [RED], "a.police": [RED]}


def recording(cars=3):
    """Every car drives along +x at 8 m/s in the lane y = 0 unless moved; car 1 is the distractor."""
    t = np.arange(N) * DT
    pos = np.zeros((N, cars, 3))
    pos[:, :, 0] = 8.0 * t[:, None]
    pos[:, 2, 1] = -40.0                     # a background car far to the side
    vel = np.zeros_like(pos)
    vel[:, :, 0] = 8.0
    return {"pos": pos, "rot": np.zeros_like(pos), "vel": vel,
            "kind": np.array(["target", "distractor", "background"][:cars])}


class DistractorTests(unittest.TestCase):
    def test_levels(self):
        rng = random.Random(0)
        self.assertEqual(ft.distractor_car(rng, "twin", "a.sedan1", RED, TYPES, COLORS), ("a.sedan1", RED, None))
        for _ in range(20):
            b, c, way = ft.distractor_car(rng, "near_twin", "a.sedan1", RED, TYPES, COLORS)
            self.assertEqual(TYPES[b]["type"], "sedan")
            self.assertTrue((b, c) in (("a.sedan2", RED), ("a.sedan1", DARK_RED)) or c != RED)
            self.assertNotEqual(b, "a.police")              # body colour must show from above
            self.assertEqual(way, "blueprint" if b != "a.sedan1" else "rgb")
            b, c, way = ft.distractor_car(rng, "other", "a.sedan1", RED, TYPES, COLORS)
            self.assertEqual((b, c, way), ("a.sedan1", BLUE, None))


class CheckTests(unittest.TestCase):
    def test_cut_in(self):
        rec = recording()
        rec["pos"][:, 1, 1] = 3.5                            # the next lane, beside the target
        k0 = int(20.0 / DT)
        rec["pos"][k0:, 1, 0] += 10.0                         # ahead by 10 m ...
        rec["pos"][k0:, 1, 1] = np.maximum(0.0, 3.5 - 0.1 * np.arange(N - k0))   # ... then across into the lane
        out = ft.check_cut_in(rec, 1)
        self.assertTrue(out["happened"])
        self.assertAlmostEqual(out["t_s"], 20.0 + 18 * DT, places=2)
        rec2 = recording()
        rec2["pos"][:, 1, 1] = 3.5
        self.assertFalse(ft.check_cut_in(rec2, 1)["happened"])          # stays in its lane
        rec3 = recording()
        rec3["pos"][:, 1, 0] += 10.0                                    # ahead in the lane from the start
        self.assertFalse(ft.check_cut_in(rec3, 1)["happened"])          # never came from the next lane

    def test_queue(self):
        rec = recording()
        rec["pos"][:, 1, 0] += 8.0                           # 8 m ahead in the same lane
        a, b = int(20.0 / DT), int(25.0 / DT)
        rec["vel"][a:b, :2, 0] = 0.0                         # both stand 5 s
        out = ft.check_queue(rec, 1)
        self.assertTrue(out["happened"])
        self.assertEqual((out["t_s"], out["stood_s"], out["gap_m"]), (20.0, 5.0, 8.0))
        rec["pos"][:, 2] = rec["pos"][:, 0] + np.array([4.0, 0.0, 0.0])   # a car between them
        self.assertFalse(ft.check_queue(rec, 1)["happened"])
        rec = recording()
        rec["pos"][:, 1, 0] += 8.0
        rec["vel"][a:, :2, 0] = 0.0                          # they never drive off
        self.assertFalse(ft.check_queue(rec, 1)["happened"])
        rec = recording()
        rec["pos"][:, 1, 0] += 8.0
        rec["vel"][int(5.0 / DT):int(10.0 / DT), :2, 0] = 0.0   # before the window
        self.assertFalse(ft.check_queue(rec, 1)["happened"])

    def test_occlusion(self):
        occ = np.zeros(N, bool)
        occ[int(30.0 / DT):int(31.5 / DT)] = True
        self.assertFalse(ft.check_occlusion(occ)["happened"])          # 1.5 s
        occ[int(40.0 / DT):int(43.0 / DT)] = True
        out = ft.check_occlusion(occ)
        self.assertEqual((out["happened"], out["t_s"], out["hidden_s"]), (True, 40.0, 3.0))

    def test_junction_free(self):
        class W:
            def __init__(self, j):
                self.is_junction = j
        self.assertEqual(ft.junction_free([W(False)] * 5 + [W(True)] + [W(False)] * 3), 20.0)
        self.assertEqual(ft.junction_free([W(False)] * 4), 16.0)

    def test_relation_signs(self):
        rec = recording()
        rec["rot"][:, :, 1] = 90.0                           # heading +y
        rec["pos"][:, 1] = rec["pos"][:, 0] + np.array([-3.0, 5.0, 0.0])
        along, lateral, dyaw, _, _ = ft.pair_relation(rec, 0, 1)
        self.assertAlmostEqual(along[0], 5.0)
        self.assertAlmostEqual(lateral[0], 3.0)               # -x is to the right when heading +y (CARLA, y right)
        self.assertEqual(dyaw[0], 0.0)


if __name__ == "__main__":
    unittest.main()
