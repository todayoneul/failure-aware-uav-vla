"""Demo 1 scorer (docs/demo1_plan.md 3.2, docs/demo1.md 4): the plan's metrics from per-tick evidence of a flight.

Per tick a run keeps:
    owner        recording index of the car the tracking box points at: the car with the most instance pixels inside
                 the box (-1: no box, or no car pixels in it)
    target_px    the target's visible pixels in the drone camera (instance segmentation)
A switch is the box staying on one car other than the target for `SWITCH_TICKS` ticks or more after it has pointed
at the target at least once (a wrong first pick is the first-pick metric's business, not a switch).

Occlusions (docs/demo1.md 4.5, decided 2026-11-23): re-acquiring a target after a long occlusion is outside demo 1.
A hidden stretch of up to `ALLOW_HIDDEN_S` is part of normal tracking, as a tracker's track buffer bridges it.
Each trial is split at the start of its first longer occlusion:
    main     ticks before it (the whole trial if there is none); the pass lines apply here
    after    ticks from it to the end; reported separately (re-acquired or not, hold, switches, distance error)
The altitude guard's climbs (src/car_follow/clearance.py) are recorded separately as well. The distance error is
measured to the goal the drone was asked to hold: the true goal raised by the guard's offset. Pure numpy.
"""
import numpy as np

SWITCH_TICKS = 5            # 0.25 s at 20 Hz (docs/demo1_plan.md 3.2)
VISIBLE_PX = 30             # the target counts as on screen from this many pixels (as for the trial checks)
NEAR_M = 10.0               # "near a look-alike" (docs/demo1_plan.md 3.2, recorded value)
WARMUP_S = 5.0              # distance error leaves out the first 5 s (docs/demo1_plan.md 3.2)
ALLOW_HIDDEN_S = 2.0        # hidden this long or less is normal tracking (docs/demo1.md 4.5)
ACTIVE_M = 0.1              # the altitude guard counts as climbing above this offset
PASS = {"first_pick": 0.90, "switch_free": 0.90, "hold": 0.95, "distance_error_m": 3.0}


# CARLA 0.9.16 semantic tags (red channel of the instance segmentation image).
TAGS = {0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building", 4: "wall", 5: "fence", 6: "pole",
        7: "traffic light", 8: "traffic sign", 9: "vegetation", 10: "terrain", 11: "sky", 12: "pedestrian",
        13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train", 18: "motorcycle", 19: "bicycle", 20: "static",
        21: "dynamic", 22: "other", 23: "water", 24: "road line", 25: "ground", 26: "bridge", 27: "rail track",
        28: "guard rail"}


def occluder(tag_img, box):
    """The most common semantic tag inside `box` (the target's projected box while it is hidden), or -1."""
    if box is None:
        return -1
    h, w = tag_img.shape
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return -1
    tags, counts = np.unique(tag_img[y0:y1, x0:x1], return_counts=True)
    return int(tags[counts.argmax()])


def box_owner(ids_img, box, car_ids):
    """(recording index of the car with the most pixels inside `box`, that car's share of the car pixels in the box).
    `ids_img` is the instance-ID image, `box` (x0, y0, x1, y1) in pixels or None, `car_ids` the actor ID of every
    recorded car in recording order. (-1, 0.0) when there is no box or no car pixel in it."""
    if box is None:
        return -1, 0.0
    h, w = ids_img.shape
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return -1, 0.0
    ids, counts = np.unique(ids_img[y0:y1, x0:x1], return_counts=True)
    index = {int(a): i for i, a in enumerate(car_ids)}
    best, best_n, total = -1, 0, 0
    for a, n in zip(ids.tolist(), counts.tolist()):
        if a in index:
            total += n
            if n > best_n:
                best, best_n = index[a], n
    return (best, best_n / total) if total else (-1, 0.0)


def mask_box(ids_img, actor_id):
    """The pixel box (x0, y0, x1, y1) of one actor's visible pixels, or None."""
    ys, xs = np.nonzero(ids_img == actor_id)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def runs(values):
    """[(value, start, length)] of the runs of equal values."""
    values = np.asarray(values)
    if len(values) == 0:
        return []
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.r_[0, edges]
    lengths = np.diff(np.r_[starts, len(values)])
    return [(int(values[s]), int(s), int(n)) for s, n in zip(starts, lengths)]


def switches(owner, target=0, min_ticks=SWITCH_TICKS, locked=False):
    """Switch events: runs of at least `min_ticks` ticks on one car other than the target, after the box first
    pointed at the target (or from the start if `locked`). Returns [(car index, start tick, length)]."""
    owner = np.asarray(owner)
    if locked:
        first = -1
    else:
        hits = np.flatnonzero(owner == target)
        if len(hits) == 0:
            return []
        first = hits[0]
    return [(v, s, n) for v, s, n in runs(owner) if s > first and v not in (target, -1) and n >= min_ticks]


def near_lookalike(rec_pos, kind, target=0, near=NEAR_M):
    """Per tick: whether a look-alike is within `near` metres (horizontally) of the target."""
    look = [i for i, k in enumerate(kind) if str(k) == "lookalike"]
    if not look:
        return np.zeros(len(rec_pos), bool)
    d = np.linalg.norm(rec_pos[:, look, :2] - rec_pos[:, target:target + 1, :2], axis=2)
    return (d < near).any(axis=1)


def hidden_runs(target_px):
    """[(start tick, length)] of the stretches with the target hidden, after it was first visible."""
    visible = np.asarray(target_px) >= VISIBLE_PX
    seen = np.flatnonzero(visible)
    if len(seen) == 0:
        return []
    return [(s, n) for v, s, n in runs(visible.astype(int)) if v == 0 and s > seen[0]]


def split_tick(target_px, dt=0.05, allow_s=ALLOW_HIDDEN_S):
    """(tick where the first occlusion longer than `allow_s` starts, or None; [(start, length)] of all such)."""
    long = [(s, n) for s, n in hidden_runs(target_px) if n * dt > allow_s]
    return (long[0][0] if long else None), long


def _mean_after(values, lo, hi, warmup):
    if values is None:
        return None
    v = np.asarray(values, dtype=float)[max(lo, warmup):hi]
    return round(float(v.mean()), 3) if len(v) else None


def _ticks_after(lo, hi, warmup):
    return max(0, hi - max(lo, warmup))


def _window(owner, visible, kind, dt, lo, hi, locked):
    o, vis = owner[lo:hi], visible[lo:hi]
    events = switches(o, locked=locked)
    return {
        "switches": [{"car": int(c), "kind": str(kind[c]), "start_s": round((lo + s) * dt, 2), "ticks": n}
                     for c, s, n in events],
        "switch_free": not events,
        "visible_ticks": int(vis.sum()),
        "held_ticks": int((vis & (o == 0)).sum()),
        "hold": round(float((vis & (o == 0)).sum() / vis.sum()), 4) if vis.any() else None,
    }


def score_trial(owner, target_px, kind, rec_pos, distance_error, horizontal_error=None, first_pick=None, dt=0.05):
    """The plan's per-trial values. `distance_error` and `horizontal_error` are per tick (m): to the goal the drone
    was asked to hold, and to the true goal horizontally. `first_pick` is True/False for configs that pick the
    target, None otherwise. The top-level values are the main window's; `after` holds the window from the first
    long occlusion on (None if there is none)."""
    owner = np.asarray(owner)
    n = len(owner)
    visible = np.asarray(target_px) >= VISIBLE_PX
    cut, long = split_tick(target_px, dt)
    end = n if cut is None else cut
    warm = int(round(WARMUP_S / dt))
    near = near_lookalike(rec_pos, kind)
    row = {"first_pick": first_pick}
    row.update(_window(owner, visible, kind, dt, 0, end, locked=False))
    row.update(
        distance_error_m=_mean_after(distance_error, 0, end, warm),
        horizontal_error_m=_mean_after(horizontal_error, 0, end, warm),
        error_ticks=_ticks_after(0, end, warm),
        main_s=round(end * dt, 2),
        long_occlusions=[{"start_s": round(s * dt, 2), "length_s": round(m * dt, 2)} for s, m in long],
        hidden_s=round(sum(m for _, m in hidden_runs(target_px)) * dt, 2),
        near_lookalike_s=round(float(near.sum() * dt), 2),
        box_on_other_car_ticks=int(((owner != 0) & (owner != -1)).sum()),
        no_box_while_visible_ticks=int((visible & (owner == -1)).sum()),
        after=None,
    )
    if cut is not None:
        after = _window(owner, visible, kind, dt, cut, n, locked=True)
        back = np.flatnonzero(owner[cut + long[0][1]:] == 0)
        after.update(
            start_s=round(cut * dt, 2),
            reacquired=bool(len(back)),
            reacquired_after_s=round(float(back[0] * dt), 2) if len(back) else None,
            distance_error_m=_mean_after(distance_error, cut, n, warm),
            horizontal_error_m=_mean_after(horizontal_error, cut, n, warm),
            error_ticks=_ticks_after(cut, n, warm),
        )
        row["after"] = after
    return row


def avoidance(offset, path, los, los_state, rise, nearest, dt=0.05, clearance_m=3.0, max_offset_m=10.0, skip=2):
    """The altitude guard's record for one trial. Per tick: `offset` the guard's offset (m), `path` and `los` its two
    needs (m), `los_state` its line-of-sight state (str), `rise` the drone's height over the true nominal goal (m),
    `nearest` the nearest depth-map point (m). The first `skip` ticks (camera warm-up) are left out of `nearest`."""
    offset, rise, nearest = (np.asarray(a, dtype=float) for a in (offset, rise, nearest))
    active = offset > ACTIVE_M
    state = np.asarray(los_state).astype(str)
    near = nearest[skip:]
    return {
        "time_s": round(float(active.sum() * dt), 2),
        "events": sum(1 for v, _, _ in runs(active.astype(int)) if v == 1),
        "offset_max_m": round(float(offset.max()), 2),
        "offset_mean_while_active_m": round(float(offset[active].mean()), 2) if active.any() else None,
        "rise_max_m": round(float(rise.max()), 2),
        "rise_mean_while_active_m": round(float(rise[active].mean()), 2) if active.any() else None,
        "path_need_s": round(float((np.asarray(path) > 0).sum() * dt), 2),
        "path_beyond_max_s": round(float((np.asarray(path) > max_offset_m).sum() * dt), 2),
        "los_need_s": round(float((np.asarray(los) > 0).sum() * dt), 2),
        "los_unreachable_s": round(float((state == "unreachable").sum() * dt), 2),
        "nearest_min_m": round(float(near.min()), 2) if len(near) else None,
        "nearest_below_clearance_s": round(float((near < clearance_m).sum() * dt), 2),
    }


def _mean(rows, key):
    """Tick-weighted mean of the per-window means `key` (windows differ in length once trials are split)."""
    pairs = [(r[key], r.get("error_ticks", 1)) for r in rows if r.get(key) is not None and r.get("error_ticks", 1)]
    if not pairs:
        return None
    return round(float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs)), 3)


def summarize(rows):
    """Rates over trials and the pass lines (docs/demo1_plan.md 3.2) on the main windows; the after windows and the
    altitude guard separately. Hold is summed over all trials; the distance errors are averaged over all ticks
    (with full-length windows this equals the mean of the trial means)."""
    picks = [r["first_pick"] for r in rows if r["first_pick"] is not None]
    vis = sum(r["visible_ticks"] for r in rows)
    out = {
        "trials": len(rows),
        "first_pick": round(sum(picks) / len(picks), 4) if picks else None,
        "switch_free": round(sum(r["switch_free"] for r in rows) / len(rows), 4),
        "hold": round(sum(r["held_ticks"] for r in rows) / vis, 4) if vis else None,
        "distance_error_m": _mean(rows, "distance_error_m"),
        "horizontal_error_m": _mean(rows, "horizontal_error_m"),
        "switch_partners": {k: sum(s["kind"] == k for r in rows for s in r["switches"])
                            for k in ("lookalike", "background")},
        "near_lookalike_s": round(sum(r["near_lookalike_s"] for r in rows), 1),
    }
    out["pass"] = {
        "first_pick": None if out["first_pick"] is None else out["first_pick"] >= PASS["first_pick"],
        "switch_free": out["switch_free"] >= PASS["switch_free"],
        "hold": out["hold"] is not None and out["hold"] >= PASS["hold"],
        "distance_error_m": out["distance_error_m"] is not None and out["distance_error_m"] <= PASS["distance_error_m"],
    }
    later = [r["after"] for r in rows if r.get("after")]
    avis = sum(a["visible_ticks"] for a in later)
    out["after_long_occlusion"] = {
        "trials": len(later),
        "reacquired": sum(a["reacquired"] for a in later),
        "switch_free": sum(a["switch_free"] for a in later),
        "hold": round(sum(a["held_ticks"] for a in later) / avis, 4) if avis else None,
        "distance_error_m": _mean(later, "distance_error_m"),
        "horizontal_error_m": _mean(later, "horizontal_error_m"),
    }
    guard = [r["avoidance"] for r in rows if r.get("avoidance")]
    if guard:
        nearest = [g["nearest_min_m"] for g in guard if g["nearest_min_m"] is not None]
        out["avoidance"] = {
            "trials_climbing": sum(g["events"] > 0 for g in guard),
            "time_s": round(sum(g["time_s"] for g in guard), 1),
            "events": sum(g["events"] for g in guard),
            "offset_max_m": max(g["offset_max_m"] for g in guard),
            "rise_max_m": max(g["rise_max_m"] for g in guard),
            "path_need_s": round(sum(g["path_need_s"] for g in guard), 1),
            "path_beyond_max_s": round(sum(g.get("path_beyond_max_s", 0.0) for g in guard), 2),
            "los_need_s": round(sum(g["los_need_s"] for g in guard), 1),
            "los_unreachable_s": round(sum(g["los_unreachable_s"] for g in guard), 1),
            "nearest_min_m": min(nearest) if nearest else None,
            "nearest_below_clearance_s": round(sum(g["nearest_below_clearance_s"] for g in guard), 1),
        }
    return out
