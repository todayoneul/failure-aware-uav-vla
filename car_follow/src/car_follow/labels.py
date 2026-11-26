"""Stage 3 labels (docs/failure_plan.md 3): when and how a flight failed, worked out after the flight from the
simulator's ground truth. The monitor never sees these. Pure numpy.

Inputs of one flight:
    ev      the evidence of scripts/demo1_eval.py (its .npz): owner, target_px, track_id, est_pos, box,
            log_position, ...
    rec     the traffic recording it flew over: pos (T x N x 3), rot, vel, box_center, box_extent, kind
    match   bool per recorded car: the sentence's car-type and colour words fit it (the target, index 0, included)
    lock    the tick the target's track was locked (config 3: the pick's lock tick), or None if it never was
    pick    (tick, right) of the first pick, or None for a configuration without one

Outcome labels (table 2), each an event with the tick it starts:
    first_pick       the first pick was not the target                          at the pick's tick
    switch_box       the box's owner on one other car for SWITCH_TICKS or more  at the run's start (as scorer.py)
    switch_estimate  the car nearest the estimate (within EST_NEAR_M) is one
                     other car for SWITCH_TICKS or more                         at the run's start
    drift            track alive, estimate over DRIFT_M from the target
                     horizontally for DRIFT_TICKS or more                       at the run's first tick over DRIFT_M
    loss             hidden longer than LOSS_HIDDEN_S, or the track dropped     whichever is earlier
The first event of a flight is its failure and its tick is t_fail; the later ones stay in order (the chain).

Exposure labels (table 3), as runs of ticks [(start, length)]:
    crossing          a word-match car within CROSS_M of the target, or the two cars' boxes overlap as seen from
                      the nominal follow pose (traffic only)
    queue             a word-match car in the target's lane within QUEUE_M, both slower than QUEUE_SPEED (traffic only)
    out_of_depth_fov  the true target lies deeper below the drone than the depth camera sees
    occlusion         the target's visible pixels below VISIBLE_PX
    near_miss         before t_fail (or anywhere, with no failure): the estimate over NEAR_MISS_M from the target,
                      or the box on a word-match car for 1 to SWITCH_TICKS - 1 ticks. A run that lasts into
                      t_fail is tagged 'onset': the failure building up, left out of the normal stretch like a
                      near-miss but not counted as one
"""
import math

import numpy as np

from . import clearance, estimate, follow, scorer, screen

DT = 0.05
SWITCH_TICKS = scorer.SWITCH_TICKS   # 5 ticks, 0.25 s
EST_NEAR_M = 3.0
DRIFT_M, DRIFT_S = 4.0, 0.5
DRIFT_TICKS = int(round(DRIFT_S / DT))
NEAR_MISS_M = 3.0
LOSS_HIDDEN_S = scorer.ALLOW_HIDDEN_S
VISIBLE_PX = scorer.VISIBLE_PX
CROSS_M = 5.0
QUEUE_M, QUEUE_SPEED, LANE_HALF_M, QUEUE_YAW_DEG = 10.0, 1.0, 1.75, 20.0
RECOVER_S = 1.0
CAMERA = {"width": 960, "height": 540, "fov": 90.0}   # trials.CAMERA (trials.py needs the simulator's module)
OUTCOMES = ("first_pick", "switch_box", "switch_estimate", "drift", "loss")
EXPOSURES = ("crossing", "queue", "out_of_depth_fov", "occlusion", "near_miss")


def depth_limit_deg(dcam=clearance.DepthCamera()):
    """How far below the horizon the depth camera sees: its mount pitch plus half its vertical field of view."""
    half_v = math.degrees(math.atan(math.tan(math.radians(dcam.fov) / 2.0) * dcam.height / dcam.width))
    return -dcam.mount_pitch + half_v


DEPTH_LIMIT_DEG = depth_limit_deg()   # 56.87


def true_runs(mask):
    """[(start, length)] of the runs of True."""
    return [(s, n) for v, s, n in scorer.runs(np.asarray(mask, dtype=int)) if v == 1]


def merge(spans):
    """Overlapping or touching [(start, length)] merged, in order."""
    out = []
    for s, n in sorted(spans):
        if out and s <= out[-1][0] + out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][0] + out[-1][1], s + n) - out[-1][0])
        else:
            out.append((s, n))
    return out


def flight_arrays(ev, rec, lock):
    """Per tick: `alive` (the target's track exists, from the lock on), `drift` (horizontal estimate error, NaN
    when not alive), `nearest` (recording index of the car nearest the estimate within EST_NEAR_M, -1 if none or not
    alive), `below` (degrees the true target lies below the drone's horizon)."""
    n = len(ev["owner"])
    tgt = rec["pos"][:n, 0].astype(float)
    est = np.asarray(ev["est_pos"], dtype=float)
    alive = (np.asarray(ev["track_id"]) >= 0) & np.isfinite(est).all(axis=1)
    alive[: (lock if lock is not None else n)] = False
    drift = np.full(n, np.nan)
    drift[alive] = np.linalg.norm(est[alive, :2] - tgt[alive, :2], axis=1)
    nearest = np.full(n, -1)
    if alive.any():
        d = np.linalg.norm(rec["pos"][:n, :, :2][alive] - est[alive, None, :2], axis=2)
        best = d.argmin(axis=1)
        nearest[alive] = np.where(d[np.arange(len(best)), best] < EST_NEAR_M, best, -1)
    drone = np.asarray(ev["log_position"], dtype=float)
    rel = drone - tgt
    below = np.degrees(np.arctan2(rel[:, 2], np.linalg.norm(rel[:, :2], axis=1)))
    return {"alive": alive, "drift": drift, "nearest": nearest, "below": below}


def outcome_events(ev, rec, lock, pick=None, arrays=None, dt=DT):
    """The outcome events of one flight, sorted by tick: [{'kind', 'tick', ...}]."""
    a = arrays or flight_arrays(ev, rec, lock)
    owner = np.asarray(ev["owner"])
    n = len(owner)
    events = []
    if pick is not None and not pick[1]:
        events.append({"kind": "first_pick", "tick": int(pick[0])})
    for c, s, m in scorer.switches(owner):
        events.append({"kind": "switch_box", "tick": int(s), "car": int(c), "ticks": int(m)})
    for c, s, m in scorer.switches(a["nearest"]):
        events.append({"kind": "switch_estimate", "tick": int(s), "car": int(c), "ticks": int(m)})
    over = a["alive"] & (np.nan_to_num(a["drift"]) > DRIFT_M)
    for s, m in true_runs(over):
        if m >= DRIFT_TICKS:
            events.append({"kind": "drift", "tick": int(s), "ticks": int(m),
                           "max_m": round(float(np.nanmax(a["drift"][s:s + m])), 2)})
    if lock is not None:
        hidden = [(s, m) for s, m in scorer.hidden_runs(ev["target_px"]) if m * dt > LOSS_HIDDEN_S and s + m > lock]
        if hidden:
            s, m = hidden[0]
            tag = np.asarray(ev["hidden_by"])[s:s + m] if "hidden_by" in ev else np.array([-1])
            tag = tag[tag >= 0]
            cause = scorer.TAGS.get(int(np.bincount(tag).argmax()), "unknown") if len(tag) else "unknown"
            events.append({"kind": "loss", "tick": int(max(s, lock)), "cause": "hidden", "hidden_by": cause,
                           "ticks": int(m)})
        track = np.asarray(ev["track_id"])
        after = np.flatnonzero(track[lock:] < 0)
        if len(after):
            events.append({"kind": "loss", "tick": int(lock + after[0]), "cause": "track_dropped"})
    events.sort(key=lambda e: (e["tick"], OUTCOMES.index(e["kind"])))
    # One loss per flight: the earlier of its two causes.
    out, lost = [], False
    for e in events:
        if e["kind"] == "loss":
            if lost:
                continue
            lost = True
        out.append(e)
    return out


def recovery_tick(ev, arrays, t_fail, dt=DT, hold_s=RECOVER_S):
    """The first tick after `t_fail` that starts `hold_s` of following the target properly again: through the whole
    window the track alive, its estimate within DRIFT_M of the target and the box on no other car, and the box on
    the target in at least half of it (a track coasting with no box, as under an overpass, is not a recovery).
    None if never."""
    owner = np.asarray(ev["owner"])
    n = len(owner)
    need = int(round(hold_s / dt))
    ok = arrays["alive"] & (np.nan_to_num(arrays["drift"], nan=np.inf) <= DRIFT_M) & np.isin(owner, (0, -1))
    on = (owner == 0).astype(int)
    for s, m in true_runs(ok[t_fail + 1:]):
        s += t_fail + 1
        if m < need:
            continue
        hits = np.convolve(on[s:s + m], np.ones(need, int), "valid")   # box-on-target ticks per window
        good = np.flatnonzero(hits * 2 >= need)
        if len(good) and s + good[0] + need <= n:
            return int(s + good[0])
    return None


def crossing_runs(rec, match, ticks=None, cross_m=CROSS_M, camera=CAMERA):
    """Runs where a word-match car (not the target) is within `cross_m` of the target horizontally, or its box
    overlaps the target's in the image of a camera at the nominal follow pose (20 m behind, 15 m up, aimed at the
    target). Traffic only: the same for every flight over this recording."""
    n = len(rec["pos"]) if ticks is None else ticks
    others = [i for i, m in enumerate(match) if m and i != 0]
    hit = np.zeros(n, bool)
    if not others:
        return []
    pos, rot = rec["pos"][:n].astype(float), rec["rot"][:n].astype(float)
    d = np.linalg.norm(pos[:, others, :2] - pos[:, :1, :2], axis=2)
    hit |= (d < cross_m).any(axis=1)
    near = np.flatnonzero(~hit & (d < 40.0).any(axis=1))   # the image check only where it can matter
    for k in near:
        yaw = float(rot[k, 0, 1])
        eye = follow.goal_point(pos[k, 0], yaw)
        aim = pos[k, 0] + np.array([0.0, 0.0, estimate.AIM_ABOVE_M])
        cyaw, cpitch = screen.look_at(eye, aim)
        cam = screen.Camera(eye, cyaw, cpitch, camera["width"], camera["height"], camera["fov"])
        tb = _image_box(cam, pos[k, 0], rot[k, 0], rec["box_center"][0], rec["box_extent"][0])
        if tb is None:
            continue
        for i in others:
            if d[k, others.index(i)] >= 40.0:
                continue
            ob = _image_box(cam, pos[k, i], rot[k, i], rec["box_center"][i], rec["box_extent"][i])
            if ob is not None and ob[0] < tb[2] and tb[0] < ob[2] and ob[1] < tb[3] and tb[1] < ob[3]:
                hit[k] = True
                break
    return true_runs(hit)


def _image_box(cam, position, rot, center, extent):
    uv, depth = cam.project(screen.box_vertices(position, rot, center, extent))
    if (depth <= 0.1).any():
        return None
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    if x1 < 0 or y1 < 0 or x0 > cam.width or y0 > cam.height:
        return None
    return x0, y0, x1, y1


def queue_runs(rec, match, ticks=None):
    """Runs where a word-match car (not the target) waits in the target's lane within QUEUE_M of it: heading within
    QUEUE_YAW_DEG, lateral offset under LANE_HALF_M in the target's frame, both slower than QUEUE_SPEED."""
    n = len(rec["pos"]) if ticks is None else ticks
    others = [i for i, m in enumerate(match) if m and i != 0]
    if not others:
        return []
    pos, rot, vel = (rec[k][:n].astype(float) for k in ("pos", "rot", "vel"))
    yaw = np.radians(rot[:, 0, 1])
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    side = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
    rel = pos[:, others, :2] - pos[:, :1, :2]
    along = (rel * fwd[:, None]).sum(axis=2)
    lateral = (rel * side[:, None]).sum(axis=2)
    dyaw = np.abs((rot[:, others, 1] - rot[:, :1, 1] + 180.0) % 360.0 - 180.0)
    slow = np.linalg.norm(vel[:, :, :2], axis=2) < QUEUE_SPEED
    hit = ((np.abs(along) <= QUEUE_M) & (np.abs(lateral) < LANE_HALF_M) & (dyaw < QUEUE_YAW_DEG)
           & slow[:, others] & slow[:, :1]).any(axis=1)
    return true_runs(hit)


def near_miss_runs(ev, arrays, match, t_fail=None):
    """Runs before `t_fail` (the whole flight if None) with the estimate over NEAR_MISS_M from the target, or the box
    on a word-match car other than the target for fewer than SWITCH_TICKS ticks. Each run is tagged with why."""
    owner = np.asarray(ev["owner"])
    n = len(owner)
    end = n if t_fail is None else t_fail
    far = arrays["alive"] & (np.nan_to_num(arrays["drift"]) > NEAR_MISS_M)
    spans = [(s, m, "estimate") for s, m in true_runs(far)]
    look = np.array([bool(match[c]) and c != 0 if c >= 0 else False for c in owner])
    for c, s, m in scorer.runs(owner):
        if c > 0 and look[s] and m < SWITCH_TICKS:
            spans.append((s, m, "box"))
    out = []
    for s, m in merge([(s, m) for s, m, _ in spans]):
        if s >= end:
            continue
        why = sorted({w for a, b, w in spans if a < s + m and s < a + b})
        if s + m >= end:   # lasts into the failure: its onset (kept out of the normal stretch, not counted as a
            why.append("onset")   # near-miss)
        out.append((s, min(m, end - s), why))
    return out


def label_flight(ev, rec, match, lock, pick=None, dt=DT, with_arrays=False):
    """All labels of one flight (JSON-ready). `with_arrays` also returns the per-tick arrays (not JSON)."""
    n = len(ev["owner"])
    a = flight_arrays(ev, rec, lock)
    events = outcome_events(ev, rec, lock, pick, a, dt)
    t_fail = events[0]["tick"] if events else None
    rec_tick = recovery_tick(ev, a, t_fail, dt) if t_fail is not None else None

    def spans(runs):
        return [{"start_s": round(s * dt, 2), "length_s": round(m * dt, 2)} for s, m in runs]

    after_lock = np.zeros(n, bool)
    if lock is not None:
        after_lock[lock:] = True
    occl = after_lock & (np.asarray(ev["target_px"]) < VISIBLE_PX)
    deep = after_lock & (a["below"] > DEPTH_LIMIT_DEG)
    near = near_miss_runs(ev, a, match, t_fail)
    exposure = {
        "crossing": crossing_runs(rec, match, n),
        "queue": queue_runs(rec, match, n),
        "out_of_depth_fov": true_runs(deep),
        "occlusion": true_runs(occl),
    }
    out = {
        "ticks": n,
        "lock_tick": lock,
        "failure": events[0]["kind"] if events else None,
        "t_fail_tick": t_fail,
        "t_fail_s": None if t_fail is None else round(t_fail * dt, 2),
        "chain": [dict(e, t_s=round(e["tick"] * dt, 2)) for e in events],
        "recovered_tick": rec_tick,
        "recovered_s": None if rec_tick is None else round(rec_tick * dt, 2),
        "drift_max_m": round(float(np.nanmax(a["drift"])), 2) if np.isfinite(a["drift"]).any() else None,
        "below_max_deg": round(float(a["below"][after_lock].max()), 1) if after_lock.any() else None,
        "near_miss": [{"start_tick": int(s), "ticks": int(m), "start_s": round(s * dt, 2),
                       "length_s": round(m * dt, 2), "why": why} for s, m, why in near],
        "exposure": {k: spans(v) for k, v in exposure.items()},
        "exposure_s": {k: round(sum(m for _, m in v) * dt, 2) for k, v in exposure.items()},
    }
    out["exposure_s"]["near_miss"] = round(sum(m for _, m, why in near if "onset" not in why) * dt, 2)
    out["near_miss_flight"] = any("onset" not in why for _, _, why in near)
    if with_arrays:
        return out, a
    return out


def conditions(trial):
    """The condition labels set when the trial was made (docs/failure_plan.md 3.3). Demo 1 trials carry none of the
    fields: they are the normal condition with twins."""
    return {
        "condition": trial.get("condition", "demo1"),
        "distractor": trial.get("distractor", "twin"),
        "event": trial.get("event"),
        "start_ambiguity": trial.get("start_ambiguity"),
        "degradation": trial.get("degradation"),
    }
