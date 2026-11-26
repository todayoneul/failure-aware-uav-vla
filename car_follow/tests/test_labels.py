"""Stage 3 labels (docs/failure_plan.md 3) on synthetic flights: outcome events and their chain, recovery, near-misses
and exposures. No simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import follow, labels

N, LOCK = 400, 25
EXTENT = [2.3, 1.0, 0.75]


def flight(others=((0.0, 8.0),), speed=10.0):
    """A target driving along +x at `speed` m/s; `others` are (along, lateral) offsets of cars driving with it.
    Evidence of a clean flight: box on the target and the estimate on it from the lock."""
    t = np.arange(N) * labels.DT
    cars = 1 + len(others)
    pos = np.zeros((N, cars, 3), np.float32)
    pos[:, 0, 0] = speed * t
    for i, (along, lateral) in enumerate(others, start=1):
        pos[:, i, 0] = speed * t + along
        pos[:, i, 1] = lateral
    vel = np.zeros_like(pos)
    vel[:, :, 0] = speed
    rec = {"pos": pos, "rot": np.zeros_like(pos), "vel": vel, "box_center": np.zeros((cars, 3)),
           "box_extent": np.tile(EXTENT, (cars, 1)), "kind": np.array(["target"] + ["lookalike"] * len(others))}
    est = pos[:, 0].astype(float).copy()
    est[:LOCK] = np.nan
    owner = np.zeros(N, int)
    owner[:LOCK] = -1
    track = np.full(N, 7)
    track[:LOCK] = -1
    drone = np.array([follow.goal_point(p, 0.0) for p in pos[:, 0]])
    ev = {"owner": owner, "target_px": np.full(N, 500), "track_id": track, "est_pos": est, "log_position": drone,
          "hidden_by": np.full(N, -1)}
    return ev, rec


def label(ev, rec, match=(True, True), pick=(0, True)):
    return labels.label_flight(ev, rec, list(match), LOCK, pick)


class OutcomeTests(unittest.TestCase):
    def test_clean_flight_has_no_label(self):
        ev, rec = flight()
        out = label(ev, rec)
        self.assertIsNone(out["failure"])
        self.assertEqual(out["chain"], [])
        self.assertEqual(out["near_miss"], [])
        self.assertFalse(out["near_miss_flight"])
        self.assertEqual(out["exposure"]["crossing"], [])

    def test_box_switch(self):
        ev, rec = flight()
        ev["owner"][100:110] = 1
        out = label(ev, rec)
        self.assertEqual(out["failure"], "switch_box")
        self.assertEqual(out["t_fail_tick"], 100)
        self.assertEqual(out["t_fail_s"], 5.0)

    def test_short_box_flicker_is_a_near_miss_only_on_a_word_match_car(self):
        ev, rec = flight(others=((0.0, 8.0), (0.0, -8.0)))
        ev["owner"][60:63] = 1          # word-match car, 3 ticks
        ev["owner"][200:203] = 2        # another car, 3 ticks
        out = label(ev, rec, match=(True, True, False))
        self.assertIsNone(out["failure"])
        self.assertEqual([(m["start_tick"], m["ticks"], m["why"]) for m in out["near_miss"]], [(60, 3, ["box"])])
        self.assertTrue(out["near_miss_flight"])

    def test_drift_needs_half_a_second(self):
        ev, rec = flight()
        ev["est_pos"][80:88, 1] += 4.5      # 8 ticks over 4 m: a near-miss
        out = label(ev, rec)
        self.assertIsNone(out["failure"])
        self.assertEqual(out["near_miss"][0]["why"], ["estimate"])
        ev["est_pos"][80:90, 1] += 0.0
        ev["est_pos"][88:90, 1] += 4.5      # now 10 ticks
        out = label(ev, rec)
        self.assertEqual(out["failure"], "drift")
        self.assertEqual(out["t_fail_tick"], 80)
        self.assertEqual(out["chain"][0]["ticks"], 10)

    def test_chain_drift_then_estimate_then_box(self):
        # main-024's pattern: the estimate is pulled ahead onto the car in front, then the box follows.
        ev, rec = flight(others=((8.0, 0.0),))
        ev["est_pos"][80:100, 0] += 5.0     # 5 m ahead: drift, 3 m from the car in front (not nearer than 3 m)
        ev["est_pos"][100:, 0] += 7.5       # on the car in front
        ev["owner"][110:] = 1
        out = label(ev, rec)
        self.assertEqual([e["kind"] for e in out["chain"]], ["drift", "switch_estimate", "switch_box"])
        self.assertEqual([e["tick"] for e in out["chain"]], [80, 100, 110])
        self.assertEqual(out["failure"], "drift")
        self.assertIsNone(out["recovered_tick"])

    def test_recovery(self):
        ev, rec = flight()
        ev["est_pos"][80:96, 1] += 4.5
        out = label(ev, rec)
        self.assertEqual(out["failure"], "drift")
        self.assertEqual(out["recovered_tick"], 96)

    def test_onset_is_kept_apart_from_near_misses(self):
        ev, rec = flight()
        ev["est_pos"][70:80, 1] += 3.5      # building up ...
        ev["est_pos"][80:96, 1] += 4.5      # ... into a drift at 80
        out = label(ev, rec)
        self.assertEqual(out["t_fail_tick"], 80)
        self.assertEqual(len(out["near_miss"]), 1)
        self.assertIn("onset", out["near_miss"][0]["why"])
        self.assertFalse(out["near_miss_flight"])
        self.assertEqual(out["exposure_s"]["near_miss"], 0.0)

    def test_loss_by_hiding_and_by_track_drop_counts_once(self):
        ev, rec = flight()
        ev["target_px"][120:170] = 0        # 2.5 s hidden
        ev["track_id"][150:] = -1           # the tracker drops the track
        out = label(ev, rec)
        losses = [e for e in out["chain"] if e["kind"] == "loss"]
        self.assertEqual(len(losses), 1)
        self.assertEqual((losses[0]["tick"], losses[0]["cause"]), (120, "hidden"))
        ev, rec = flight()
        ev["track_id"][150:] = -1
        out = label(ev, rec)
        self.assertEqual((out["failure"], out["t_fail_tick"], out["chain"][0]["cause"]),
                         ("loss", 150, "track_dropped"))

    def test_short_hiding_is_not_a_loss(self):
        ev, rec = flight()
        ev["target_px"][120:160] = 0        # 2.0 s: normal tracking
        out = label(ev, rec)
        self.assertIsNone(out["failure"])
        self.assertEqual(out["exposure"]["occlusion"], [{"start_s": 6.0, "length_s": 2.0}])

    def test_wrong_first_pick(self):
        ev, rec = flight()
        out = label(ev, rec, pick=(3, False))
        self.assertEqual((out["failure"], out["t_fail_tick"]), ("first_pick", 3))


class ExposureTests(unittest.TestCase):
    def test_crossing_by_distance_and_by_image(self):
        ev, rec = flight(others=((0.0, 4.0),))           # alongside, 4 m: within 5 m
        self.assertEqual(label(ev, rec)["exposure_s"]["crossing"], N * labels.DT)
        ev, rec = flight(others=((6.0, 0.0),))           # 6 m ahead in the same lane: overlaps in the image
        self.assertEqual(label(ev, rec)["exposure_s"]["crossing"], N * labels.DT)
        ev, rec = flight(others=((14.0, 0.0),))          # 14 m ahead: above the target in the image, apart
        self.assertEqual(label(ev, rec)["exposure"]["crossing"], [])
        ev, rec = flight(others=((0.0, 4.0),))
        self.assertEqual(label(ev, rec, match=(True, False))["exposure"]["crossing"], [])

    def test_box_cover(self):
        _, rec = flight(others=((-5.0, 0.0), (5.0, 0.0), (10.0, 0.0), (0.0, 3.5)))
        cover = [float(labels.box_cover(rec, i)[100]) for i in (1, 2, 3, 4)]
        self.assertGreater(cover[0], cover[1])       # behind covers more than ahead (the camera is behind)
        self.assertGreater(cover[1], 0.1)
        self.assertEqual(cover[2:], [0.0, 0.0])      # 10 m ahead, or the next lane: apart

    def test_queue(self):
        ev, rec = flight(others=((8.0, 0.0),), speed=0.0)
        self.assertEqual(label(ev, rec)["exposure_s"]["queue"], N * labels.DT)
        ev, rec = flight(others=((8.0, 0.0),), speed=5.0)
        self.assertEqual(label(ev, rec)["exposure"]["queue"], [])
        ev, rec = flight(others=((8.0, 3.5),), speed=0.0)   # the next lane
        self.assertEqual(label(ev, rec)["exposure"]["queue"], [])

    def test_out_of_depth_fov(self):
        self.assertAlmostEqual(labels.DEPTH_LIMIT_DEG, 56.87, places=2)
        ev, rec = flight()
        self.assertEqual(label(ev, rec)["exposure"]["out_of_depth_fov"], [])   # 20 m behind, 15 up: 36.9 deg
        ev["log_position"][200:220] = rec["pos"][200:220, 0] + np.array([-5.0, 0.0, 15.0])   # 71.6 deg
        self.assertEqual(label(ev, rec)["exposure"]["out_of_depth_fov"], [{"start_s": 10.0, "length_s": 1.0}])


class ConditionTests(unittest.TestCase):
    def test_demo1_trials_are_the_normal_condition_with_twins(self):
        c = labels.conditions({"id": "dev-001"})
        self.assertEqual((c["condition"], c["distractor"], c["event"]), ("demo1", "twin", None))


if __name__ == "__main__":
    unittest.main()
