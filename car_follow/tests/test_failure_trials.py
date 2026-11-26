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


def lane_recording(changes, partner=None, width=3.5, move_s=3.0):
    """Cars along +x at 8 m/s; car 0 (the target) changes lanes at the given crossing times (s, side) with a smooth
    move over `move_s`; with `partner`, car 1 is a distractor starting in the left lane 10 m ahead. The lane record
    keys lanes 10 + lane number (left: -1), as the map's neighbours would."""
    rec = recording()
    cars = [0] + ([1] if partner is not None else [])
    rec["kind"] = np.array(["target", "distractor" if partner is not None else "lookalike", "background"])
    t = np.arange(N) * DT
    if partner is not None:
        rec["pos"][:, 1, 0] += 10.0
    shape = {"key": np.zeros((N, len(cars)), np.int64), "left": np.zeros((N, len(cars)), np.int64),
             "right": np.zeros((N, len(cars)), np.int64), "off": np.zeros((N, len(cars)), np.float32)}
    for j, (car, plan) in enumerate(zip(cars, [changes] + ([partner] if partner is not None else []))):
        start_lane = 0 if car == 0 else -1
        y = np.full(N, start_lane * width)          # lateral position (+ right, as CARLA's y)
        for when, side in plan:
            step = -width if side == "left" else width
            ramp = np.clip((t - (when - move_s / 2)) / move_s, 0.0, 1.0)
            y = y + step * (0.5 - 0.5 * np.cos(np.pi * ramp))
        lane = np.round(y / width)
        rec["pos"][:, car, 1] = y
        shape["key"][:, j] = 10 + lane
        shape["left"][:, j] = 10 + lane - 1
        shape["right"][:, j] = 10 + lane + 1
        shape["off"][:, j] = y - lane * width
    rec.update(lane_cars=np.array(cars), lane_key=shape["key"], lane_left=shape["left"], lane_right=shape["right"],
               lane_off=shape["off"], lane_width=np.full((N, len(cars)), width, np.float32))
    return rec


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
        rec4 = recording()
        rec4["pos"][:, 1, 1] = 3.5
        rec4["pos"][:, 1, 0] -= 5.0                                     # 5 m behind in the next lane ...
        rec4["pos"][k0:, 1, 1] = np.maximum(0.0, 3.5 - 0.1 * np.arange(N - k0))   # ... then across behind the target
        self.assertFalse(ft.check_cut_in(rec4, 1)["happened"])          # not ahead
        out = ft.check_cut_in(rec4, 1, behind=True)
        self.assertTrue(out["happened"])
        self.assertAlmostEqual(out["ahead_m"], -5.0, places=3)

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

    def test_lane_changes(self):
        rec = lane_recording(changes=[(20.0, "left"), (26.0, "right")])
        ch = ft.lane_changes(rec, 0)
        self.assertEqual([c["side"] for c in ch], ["left", "right"])
        for c, k in zip(ch, (400, 520)):
            self.assertLessEqual(abs(c["tick"] - k), 1)                 # the crossing, mid-move
        self.assertTrue(ch[0]["start_tick"] < 400 <= ch[0]["end_tick"])
        # a 3 s cosine move of 3.5 m is over 0.4 m from both lane centres for 0.56 of it (1.68 s)
        self.assertAlmostEqual(ch[0]["end_s"] - ch[0]["start_s"], 1.68, delta=0.1)
        np.testing.assert_allclose(ft.lateral_track(rec, 0), rec["pos"][:, 0, 1], atol=1e-5)   # continuous
        rec = lane_recording(changes=[(20.0, "left"), (20.5, "right")])           # back within 1 s: a wobble
        self.assertEqual(ft.lane_changes(rec, 0), [])
        rec = lane_recording(changes=[])
        rec["lane_key"][600:, 0] = 999                                              # a new road segment
        self.assertEqual(ft.lane_changes(rec, 0), [])

    def test_check_lane(self):
        single = {"event": {"kind": "lane_single", "window_s": [15.0, 45.0]}}
        weave = {"event": {"kind": "lane_weave", "window_s": [15.0, 45.0]}}
        self.assertTrue(ft.check_lane(single, lane_recording(changes=[(20.0, "left")]))["happened"])
        self.assertFalse(ft.check_lane(single, lane_recording(changes=[(10.0, "left")]))["happened"])   # too early
        self.assertFalse(ft.check_lane(single, lane_recording(changes=[(20.0, "left"), (30.0, "right")]))["happened"])
        out = ft.check_lane(weave, lane_recording(changes=[(20.0, "left"), (27.0, "right"), (34.0, "left")]))
        self.assertEqual((out["happened"], out["count"]), (True, 3))
        self.assertFalse(ft.check_lane(weave, lane_recording(changes=[(20.0, "left")]))["happened"])
        swap = {"event": {"kind": "lane_swap", "window_s": [15.0, 45.0]}}
        rec = lane_recording(changes=[(20.0, "left")], partner=[(20.2, "right")])
        out = ft.check_lane(swap, rec)
        self.assertTrue(out["happened"], out)
        self.assertLess(out["lateral_before_m"], 0)                     # the partner was on the left ...
        self.assertGreater(out["lateral_after_m"], 0)                   # ... and ends on the right
        rec = lane_recording(changes=[(20.0, "left")], partner=[(25.0, "right")])
        self.assertFalse(ft.check_lane(swap, rec)["happened"])          # not at the same time

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
