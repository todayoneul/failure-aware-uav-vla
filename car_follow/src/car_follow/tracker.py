"""Multi-object tracker for demo 1 config 2 (docs/demo1_plan.md 2.2, docs/demo1.md 5): ByteTrack-style association
in world coordinates.

Every detection box is turned into a world position first (src/car_follow/estimate.py: the box-centre ray and the
depth map). Tracks are constant-velocity Kalman filters on those positions, so the drone's own motion drops out:
the plan's ego-motion compensation comes from the drone's known pose, and the Kalman prediction is what each
detection is matched against.

Each tick, as in ByteTrack:
    1. predict every track
    2. match detections with score >= `high` to tracks, nearest first, within a gate around the prediction
    3. match the remaining tracks to detections with `low` <= score < `high` (partly hidden cars score low)
    4. start a track from each unmatched detection with score >= `new`
    5. drop tracks that have gone `buffer_s` without a detection (decision 4.5: hidden up to 1-2 s is bridged)
Matching is greedy on the distance (nearest pairs first); with a handful of cars in view this rarely differs from
the optimal assignment. Pure numpy.
"""
from dataclasses import dataclass

import numpy as np

from .estimate import FilterParams, TargetFilter


@dataclass(frozen=True)
class TrackerParams:
    high: float = 0.5          # score for the first matching round
    low: float = 0.1           # lowest score used at all (second round)
    new: float = 0.6           # score to start a track
    gate_m: float = 3.0        # horizontal distance from a track's prediction ...
    gate_growth: float = 4.0   # ... plus this many metres per second without a detection
    buffer_s: float = 1.5      # a track without detections is kept this long


@dataclass
class Detection:
    box: tuple                 # (x0, y0, x1, y1) pixels
    score: float
    position: np.ndarray       # world position of the car's origin (None: no depth)


class Track:
    def __init__(self, track_id, det, dt, params):
        self.id = track_id
        self.filter = TargetFilter(params)
        self.filter.update(det.position, dt)
        self.box, self.score, self.hits = det.box, det.score, 1
        self.matched = True

    @property
    def position(self):
        return self.filter.position

    @property
    def since_update(self):
        return self.filter.since_update


def greedy_match(tracks, dets, gates):
    """[(track index, detection index)] by increasing horizontal distance, each within its track's gate."""
    if not tracks or not dets:
        return []
    tp = np.array([t.position[:2] for t in tracks])
    dp = np.array([d.position[:2] for d in dets])
    cost = np.linalg.norm(tp[:, None, :] - dp[None, :, :], axis=2)
    pairs, used_t, used_d = [], set(), set()
    for flat in np.argsort(cost, axis=None):
        i, j = np.unravel_index(flat, cost.shape)
        if cost[i, j] > gates[i]:
            break
        if i in used_t or j in used_d:
            continue
        pairs.append((int(i), int(j)))
        used_t.add(i)
        used_d.add(j)
    return pairs


class Tracker:
    def __init__(self, params=TrackerParams(), filter_params=FilterParams()):
        self.p, self.fp = params, filter_params
        self.tracks = []
        self.next_id = 1

    def get(self, track_id):
        return next((t for t in self.tracks if t.id == track_id), None)

    def adopt(self, det, dt):
        """The track holding `det` after this tick's update, started here if the detection was too weak to start one
        (the VLM's first pick may name a box under the `new` score). None if it has no world position."""
        found = next((t for t in self.tracks if t.box is det.box), None)
        if found is None and det.position is not None:
            found = Track(self.next_id, det, dt, self.fp)
            self.tracks.append(found)
            self.next_id += 1
        return found

    def update(self, detections, dt):
        """One tick. `detections`: [Detection]; those without a position are ignored. Returns the live tracks."""
        p = self.p
        dets = [d for d in detections if d.position is not None and d.score >= p.low]
        for t in self.tracks:
            t.filter.predict(dt)
            t.matched = False
            t.box = None
        high = [d for d in dets if d.score >= p.high]
        low = [d for d in dets if d.score < p.high]
        gates = [p.gate_m + p.gate_growth * t.since_update for t in self.tracks]
        free_tracks = list(range(len(self.tracks)))
        free_high = list(range(len(high)))
        for group, free in ((high, free_high), (low, None)):
            cand = list(range(len(group))) if free is None else free
            ts = [self.tracks[i] for i in free_tracks]
            pairs = greedy_match(ts, [group[j] for j in cand], [gates[i] for i in free_tracks])
            for a, b in pairs:
                t, d = ts[a], group[cand[b]]
                t.filter.update(d.position, dt)
                t.box, t.score, t.hits, t.matched = d.box, d.score, t.hits + 1, True
            matched_t = {free_tracks[a] for a, _ in pairs}
            if free is not None:
                taken = {cand[b] for _, b in pairs}
                free_high = [j for j in free_high if j not in taken]
            free_tracks = [i for i in free_tracks if i not in matched_t]
        for j in free_high:
            if high[j].score >= p.new:
                self.tracks.append(Track(self.next_id, high[j], dt, self.fp))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.since_update <= p.buffer_s]
        return self.tracks
