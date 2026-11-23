"""Demo 1 scorer (docs/demo1_plan.md 3.2, docs/demo1.md 4): the plan's metrics from per-tick evidence of a flight.

Per tick a run keeps:
    owner        recording index of the car the tracking box points at: the car with the most instance pixels inside
                 the box (-1: no box, or no car pixels in it)
    target_px    the target's visible pixels in the drone camera (instance segmentation)
A switch is the box staying on one car other than the target for `SWITCH_TICKS` ticks or more after it has pointed
at the target at least once (a wrong first pick is the first-pick metric's business, not a switch). Pure numpy.
"""
import numpy as np

SWITCH_TICKS = 5            # 0.25 s at 20 Hz (docs/demo1_plan.md 3.2)
VISIBLE_PX = 30             # the target counts as on screen from this many pixels (as for the trial checks)
NEAR_M = 10.0               # "near a look-alike" (docs/demo1_plan.md 3.2, recorded value)
WARMUP_S = 5.0              # distance error leaves out the first 5 s (docs/demo1_plan.md 3.2)
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


def switches(owner, target=0, min_ticks=SWITCH_TICKS):
    """Switch events: runs of at least `min_ticks` ticks on one car other than the target, after the box first
    pointed at the target. Returns [(car index, start tick, length)]."""
    owner = np.asarray(owner)
    locked = np.flatnonzero(owner == target)
    if len(locked) == 0:
        return []
    return [(v, s, n) for v, s, n in runs(owner) if s > locked[0] and v not in (target, -1) and n >= min_ticks]


def near_lookalike(rec_pos, kind, target=0, near=NEAR_M):
    """Per tick: whether a look-alike is within `near` metres (horizontally) of the target."""
    look = [i for i, k in enumerate(kind) if str(k) == "lookalike"]
    if not look:
        return np.zeros(len(rec_pos), bool)
    d = np.linalg.norm(rec_pos[:, look, :2] - rec_pos[:, target:target + 1, :2], axis=2)
    return (d < near).any(axis=1)


def score_trial(owner, target_px, kind, rec_pos, distance_error_mean, first_pick=None, dt=0.05):
    """The plan's per-trial values. `first_pick` is True/False for configs that pick the target, None otherwise."""
    owner = np.asarray(owner)
    visible = np.asarray(target_px) >= VISIBLE_PX
    events = switches(owner)
    near = near_lookalike(rec_pos, kind)
    return {
        "first_pick": first_pick,
        "switches": [{"car": int(c), "kind": str(kind[c]), "start_s": round(s * dt, 2), "ticks": n}
                     for c, s, n in events],
        "switch_free": not events,
        "visible_ticks": int(visible.sum()),
        "held_ticks": int((visible & (owner == 0)).sum()),
        "hold": round(float((visible & (owner == 0)).sum() / visible.sum()), 4) if visible.any() else None,
        "distance_error_m": distance_error_mean,
        "near_lookalike_s": round(float(near.sum() * dt), 2),
        "box_on_other_car_ticks": int(((owner != 0) & (owner != -1)).sum()),
        "no_box_while_visible_ticks": int((visible & (owner == -1)).sum()),
    }


def summarize(rows):
    """Rates over trials and the pass lines (docs/demo1_plan.md 3.2). Hold is summed over all trials."""
    picks = [r["first_pick"] for r in rows if r["first_pick"] is not None]
    vis = sum(r["visible_ticks"] for r in rows)
    out = {
        "trials": len(rows),
        "first_pick": round(sum(picks) / len(picks), 4) if picks else None,
        "switch_free": round(sum(r["switch_free"] for r in rows) / len(rows), 4),
        "hold": round(sum(r["held_ticks"] for r in rows) / vis, 4) if vis else None,
        "distance_error_m": round(float(np.mean([r["distance_error_m"] for r in rows])), 3),
        "switch_partners": {k: sum(s["kind"] == k for r in rows for s in r["switches"])
                            for k in ("lookalike", "background")},
        "near_lookalike_s": round(sum(r["near_lookalike_s"] for r in rows), 1),
    }
    out["pass"] = {
        "first_pick": None if out["first_pick"] is None else out["first_pick"] >= PASS["first_pick"],
        "switch_free": out["switch_free"] >= PASS["switch_free"],
        "hold": out["hold"] is not None and out["hold"] >= PASS["hold"],
        "distance_error_m": out["distance_error_m"] <= PASS["distance_error_m"],
    }
    return out
