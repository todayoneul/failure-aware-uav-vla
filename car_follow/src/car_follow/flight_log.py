"""Stage 3 flight record (docs/failure_plan.md 5.2): what perception already computes each tick, kept so that the
monitor's signals can be worked out later without flying again. Read-only: it never changes the detector's boxes,
the tracker, its filters or the flight. Pure numpy.

Per tick, added to the evidence of scripts/demo1_eval.py:
    car_px              visible instance pixels of every recorded car (T x cars; column 0 is the target)
    tgt_pred, tgt_gate  the target track's constant-velocity prediction (x, y) and gate this tick, as
                        Tracker.update uses them (NaN when there is no target track yet)
    tgt_innov           distance from that prediction to the detection the track was matched to (NaN: unmatched)
    tgt_round           matching round of that detection: 1 (score >= high), 2 (low <= score < high), 0 none
    tgt_near1, _near2   distances from the prediction to the nearest and second-nearest usable detections
    tgt_in_gate         usable detections within the gate
    tgt_tracks_in_gate  other tracks whose predictions lie within the gate
"Usable" is what the tracker can match: a world position and score >= low.
Variable-length per tick, flat with a tick index:
    det_tick, det_box (x0, y0, x1, y1), det_score, det_pos (x, y, z; NaN without depth)   every detection, in
                                                                                          the tracker's input order
    trk_tick, trk_id, trk_pos, trk_vel (vx, vy), trk_since (s since its last detection), trk_matched, trk_box
    (NaN when unmatched), trk_score (its last matched score), trk_hits                     every live track
"""
import numpy as np

from .tracker import TrackerParams

NAN4 = (np.nan,) * 4
NAN3 = (np.nan,) * 3


class FlightLog:
    def __init__(self, ticks, car_ids, params=TrackerParams()):
        self.p = params
        ids = np.asarray(car_ids)
        self.order = np.argsort(ids)
        self.sorted = ids[self.order]
        self.car_px = np.zeros((ticks, len(ids)), np.int32)
        nan = np.full(ticks, np.nan, np.float32)
        self.tgt = {"tgt_pred": np.full((ticks, 2), np.nan, np.float32), "tgt_gate": nan.copy(),
                    "tgt_innov": nan.copy(), "tgt_near1": nan.copy(), "tgt_near2": nan.copy(),
                    "tgt_in_gate": np.zeros(ticks, np.int16), "tgt_tracks_in_gate": np.zeros(ticks, np.int16),
                    "tgt_round": np.zeros(ticks, np.int8)}
        self.det = {k: [] for k in ("det_tick", "det_box", "det_score", "det_pos")}
        self.trk = {k: [] for k in ("trk_tick", "trk_id", "trk_pos", "trk_vel", "trk_since", "trk_matched",
                                    "trk_box", "trk_score", "trk_hits")}
        self._pred = {}

    def pixels(self, k, ids_img):
        """Visible pixels of every recorded car in the instance-ID image."""
        flat = np.asarray(ids_img).ravel()
        idx = np.clip(np.searchsorted(self.sorted, flat), 0, len(self.sorted) - 1)
        hit = self.sorted[idx] == flat
        self.car_px[k, self.order] = np.bincount(idx[hit], minlength=len(self.sorted))

    def before_update(self, tracks, dt):
        """Each track's prediction and gate for the coming Tracker.update (call just before it)."""
        self._pred = {}
        for t in tracks:
            x = t.filter.x
            if x is not None:
                self._pred[t.id] = (x[:2] + x[2:] * dt, self.p.gate_m + self.p.gate_growth * (t.filter.since_update + dt))

    def after(self, k, dets, tracks, target_id):
        """This tick's detections, the live tracks and the target track's association (call after perception)."""
        for d in dets:
            self.det["det_tick"].append(k)
            self.det["det_box"].append(tuple(d.box))
            self.det["det_score"].append(d.score)
            self.det["det_pos"].append(NAN3 if d.position is None else tuple(d.position))
        for t in tracks:
            self.trk["trk_tick"].append(k)
            self.trk["trk_id"].append(t.id)
            pos = t.filter.position
            vel = t.filter.velocity
            self.trk["trk_pos"].append(NAN3 if pos is None else tuple(pos))
            self.trk["trk_vel"].append((np.nan, np.nan) if vel is None else tuple(vel[:2]))
            self.trk["trk_since"].append(t.filter.since_update)
            self.trk["trk_matched"].append(bool(t.matched))
            self.trk["trk_box"].append(NAN4 if t.box is None else tuple(t.box))
            self.trk["trk_score"].append(t.score)
            self.trk["trk_hits"].append(t.hits)
        if target_id is None or target_id not in self._pred:
            return
        pred, gate = self._pred[target_id]
        g = self.tgt
        g["tgt_pred"][k], g["tgt_gate"][k] = pred, gate
        usable = [d for d in dets if d.position is not None and d.score >= self.p.low]
        dist = sorted(float(np.linalg.norm(np.asarray(d.position)[:2] - pred)) for d in usable)
        if dist:
            g["tgt_near1"][k] = dist[0]
        if len(dist) > 1:
            g["tgt_near2"][k] = dist[1]
        g["tgt_in_gate"][k] = sum(x <= gate for x in dist)
        g["tgt_tracks_in_gate"][k] = sum(float(np.linalg.norm(p - pred)) <= gate
                                         for i, (p, _) in self._pred.items() if i != target_id)
        live = next((t for t in tracks if t.id == target_id), None)
        if live is not None and live.matched:
            d = next((d for d in usable if d.box is live.box), None)
            if d is not None:
                g["tgt_innov"][k] = float(np.linalg.norm(np.asarray(d.position)[:2] - pred))
                g["tgt_round"][k] = 1 if d.score >= self.p.high else 2

    def arrays(self):
        """The record as arrays for np.savez."""
        out = {"car_px": self.car_px, **self.tgt}
        shapes = {"det_box": 4, "det_pos": 3, "trk_pos": 3, "trk_vel": 2, "trk_box": 4}
        # Detection positions and scores keep full precision, so the tracker can be run again on them exactly.
        types = {"det_tick": np.int32, "trk_tick": np.int32, "trk_id": np.int32, "trk_hits": np.int32,
                 "trk_matched": bool, "det_pos": np.float64, "det_score": np.float64}
        for part in (self.det, self.trk):
            for key, values in part.items():
                dtype = types.get(key, np.float32)
                if key in shapes:
                    out[key] = np.asarray(values, dtype).reshape(-1, shapes[key])
                else:
                    out[key] = np.asarray(values, dtype)
        return out
