"""The lane-change comparison (src/car_follow/lane_phases.py): phases around the target's lane changes and the
measures taken in each. Synthetic flights, no simulator.

Run from the car_follow folder:
    uv run python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.car_follow import follow, lane_phases as lp

DT = 0.05
N = 1200


def change(start_s, end_s, side="left"):
    return {"tick": int((start_s + end_s) / 2 / DT), "side": side, "start_tick": int(start_s / DT),
            "end_tick": int(end_s / DT)}


def flight():
    """The target along +x at 8 m/s, the drone exactly on the goal; box always on the target."""
    t = np.arange(N) * DT
    pos = np.zeros((N, 2, 3))
    pos[:, :, 0] = 8.0 * t[:, None]
    pos[:, 1, 1] = 3.5
    rec = {"pos": pos, "rot": np.zeros_like(pos), "vel": np.tile([8.0, 0.0, 0.0], (N, 2, 1)),
           "kind": np.array(["target", "lookalike"])}
    goal = np.array([follow.goal_point(p, 0.0) for p in pos[:, 0]])
    ev = {"owner": np.zeros(N, int), "target_px": np.full(N, 500), "track_id": np.full(N, 3),
          "est_pos": pos[:, 0].copy(), "offset": np.zeros(N), "log_position": goal,
          "log_acceleration": np.zeros((N, 3)), "log_yaw": np.zeros(N)}
    return ev, rec


class PhaseTests(unittest.TestCase):
    def test_labels(self):
        ph = lp.phase_labels(N, [change(20.0, 22.0)], start_tick=100)
        self.assertEqual(ph[int(14.0 / DT)], "before")          # 20 - 1 (pad) - 5 = 14 s
        self.assertEqual(ph[int(13.9 / DT)], "")
        self.assertEqual(ph[int(19.0 / DT)], "during")
        self.assertEqual(ph[int(23.0 / DT)], "during")          # 22 + 1 pad
        self.assertEqual(ph[int(23.1 / DT)], "after")
        self.assertEqual(ph[int(28.0 / DT)], "after")
        self.assertEqual(ph[int(28.2 / DT)], "")
        ph = lp.phase_labels(N, [change(20.0, 22.0), change(28.0, 30.0, "right")])
        self.assertEqual(ph[int(25.0 / DT)], "between")
        self.assertEqual(lp.phase_labels(N, []).tolist(), [""] * N)

    def test_measures(self):
        ev, rec = flight()
        lock = 25
        a, b = int(20.5 / DT), int(21.0 / DT)
        ev["owner"][a:b] = 1                                    # 0.5 s on the look-alike during the change
        ev["log_position"][int(16.0 / DT):int(17.0 / DT), 0] += 2.0   # 2 m off for 1 s before it
        events = [{"kind": "switch_box", "tick": a}]
        out, phase = lp.flight_phases(ev, rec, lock, events, [change(20.0, 22.0)])
        self.assertEqual(out["phases"]["during"]["wrong_box_s"], 0.5)
        self.assertEqual(out["phases"]["during"]["events"]["switch_box"], 1)
        self.assertEqual(out["phases"]["before"]["events"]["switch_box"], 0)
        self.assertAlmostEqual(out["phases"]["before"]["distance_error"]["mean"], 2.0 * 1.0 / 5.0, places=3)
        self.assertEqual(out["phases"]["after"]["distance_error"]["mean"], 0.0)
        self.assertFalse(out["kept_same_car"])
        self.assertEqual(out["first_failure"]["phase"], "during")
        pooled = lp.pool([out["phases"], out["phases"]])
        self.assertEqual(pooled["during"]["wrong_box_s"], 1.0)
        self.assertEqual(pooled["during"]["flights"], 2)
        ev, rec = flight()
        out, _ = lp.flight_phases(ev, rec, lock, [], [change(20.0, 22.0)])
        self.assertTrue(out["kept_same_car"])
        self.assertEqual(out["phases"]["during"]["hold"], 1.0)
        out, _ = lp.flight_phases(ev, rec, lock, [{"kind": "first_pick", "tick": 20}], [change(20.0, 22.0)])
        self.assertFalse(out["kept_same_car"])                 # a wrong first pick never keeps the named car
        self.assertEqual(out["first_failure"]["kind"], "first_pick")

    def test_along_across(self):
        ev, rec = flight()
        rec["rot"][:, 0, 1] = 90.0                              # heading +y: across (right) is -x
        across, along = lp.split(np.tile([-2.0, 3.0, 0.0], (N, 1)), rec)
        self.assertAlmostEqual(float(across[0]), 2.0)
        self.assertAlmostEqual(float(along[0]), 3.0)
        rec = flight()[1]
        rec["vel"][int(20.0 / DT):int(23.0 / DT), 0, 0] = 12.0   # a 4 m/s jump right after a change at 20-22 s
        self.assertAlmostEqual(lp.speed_jump(rec, change(20.0, 22.0)), 4.0)

    def test_jitter(self):
        ev, _ = flight()
        ev["log_acceleration"][:, 1] = np.where(np.arange(N) % 2, 1.0, -1.0)
        ev["log_yaw"] = np.linspace(170.0, 190.0, N) % 360.0 - (np.linspace(170.0, 190.0, N) % 360.0 > 180) * 360.0
        acc, jerk, rate = lp.jitter(ev)
        self.assertAlmostEqual(float(acc.mean()), 1.0)
        self.assertAlmostEqual(float(np.abs(jerk[1:]).mean()), 2.0 / DT)
        self.assertLess(float(np.abs(rate[1:]).max()), 1.0)    # the wrap at 180 deg is not a spin


if __name__ == "__main__":
    unittest.main()
