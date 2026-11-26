"""The lane-change comparison (condition A4, added on the user's request on 2026-11-26): every flight over a trial
whose target changes lanes is split, around those changes, into phases, and each phase gets the same measures.
Pure numpy: the flight's evidence (scripts/demo1_eval.py), its traffic recording with the lane record
(failure_trials.LaneProbe) and its outcome events (labels.py).

Phases (per tick, from the target's lane changes, failure_trials.lane_changes):
    before   BEFORE_S before the first change starts
    during   from each change's start to its end, both widened by PAD_S (the settle band of lane_changes leaves out
             the slow first and last part of the move)
    between  weave: after one change and before the next
    after    AFTER_S after the last change ends
Only ticks after the lock and after scorer.WARMUP_S count.

Measures per phase:
    wrong_box_s      the box on another car (its owner is not the target)
    no_box_s         no box while the target is on screen (VISIBLE_PX)
    hold             share of the target's visible ticks with the box on it
    lost_s           the target's track gone (after the lock)
    events           outcome events (labels.py) that start in the phase: switch_box, switch_estimate, drift, loss
    distance_error   to the goal the drone was asked to hold (the true goal raised by the guard's offset, as
                     demo1_eval.errors): mean and 95th percentile, m
    across_error,    its horizontal part across and along the target's heading: mean of the size, m (a lane change
    along_error      shows across, a change of the target's speed along)
    estimate_error   horizontal, the target track's estimate against the truth: mean and 95th percentile, m
    accel_rms        the drone's horizontal acceleration, root mean square, m/s^2
    accel_across_rms its part across the target's heading (the lane change) and along it (the target's speed
    accel_along_rms  changes), root mean square, m/s^2
    jerk_rms         its rate of change, root mean square, m/s^3
    yaw_rate_rms     the drone's yaw rate, root mean square, deg/s
    target_lat_speed the target's own lateral speed in its lane frame, root mean square, m/s (how hard the phase is)
    target_yaw_rate  the target's yaw rate, root mean square, deg/s (a curve in the phase, not a lane change)
    target_accel_rms the target's change of speed, root mean square, m/s^2 (after a forced change about half the
                     targets sped up from 8 to 13-14 m/s for a few seconds: a Traffic Manager effect, docs/failure.md 6)
Per change, speed_jump_mps: the target's highest speed from the change's start to 6 s after its end, less its mean
speed over the 3 s before the start.
A flight kept the car first named when its first pick was right and it has no switch (box or estimate) and no loss
from the lock to the end of the after phase.
"""
import numpy as np

from . import follow, scorer

DT = 0.05
BEFORE_S, AFTER_S = 5.0, 5.0
PAD_S = 1.0
PHASES = ("before", "during", "between", "after")
EVENT_KINDS = ("switch_box", "switch_estimate", "drift", "loss")


def phase_labels(n, changes, start_tick=0, dt=DT):
    """Per tick: the phase name, or '' outside every phase (and before `start_tick`)."""
    out = np.full(n, "", dtype="<U7")
    if not changes:
        return out
    pad, before, after = (int(round(x / dt)) for x in (PAD_S, BEFORE_S, AFTER_S))
    spans = [(max(0, c["start_tick"] - pad), min(n, c["end_tick"] + pad + 1)) for c in changes]
    first, last = spans[0][0], spans[-1][1]
    out[max(0, first - before):first] = "before"
    out[first:last] = "between"
    for a, b in spans:
        out[a:b] = "during"
    out[last:min(n, last + after)] = "after"
    out[:start_tick] = ""
    return out


def heading_frame(rec, n=None):
    """Per tick: the target's forward and right unit vectors (x, y), CARLA's frame (y to the right of x)."""
    yaw = np.radians(rec["rot"][:n, 0, 1].astype(float))
    fwd = np.c_[np.cos(yaw), np.sin(yaw)]
    return fwd, np.c_[-np.sin(yaw), np.cos(yaw)]


def goal_offset(ev, rec):
    """Per tick: the drone's position less the goal it was asked to hold (demo1_eval.errors with the guard on)."""
    pos, yaw = rec["pos"][:, 0].astype(float), rec["rot"][:, 0, 1].astype(float)
    goal = np.array([follow.goal_point(p, y) for p, y in zip(pos, yaw)])
    d = np.asarray(ev["log_position"], dtype=float) - goal
    d[:, 2] -= np.asarray(ev["offset"], dtype=float)
    return d


def distance_error(ev, rec):
    """Per tick: distance to the goal the drone was asked to hold (demo1_eval.errors with the guard on)."""
    return np.linalg.norm(goal_offset(ev, rec), axis=1)


def split(vectors, rec):
    """Per tick: (across, along) parts of horizontal vectors in the target's heading frame."""
    fwd, right = heading_frame(rec, len(vectors))
    v = np.asarray(vectors, dtype=float)[:, :2]
    return (v * right).sum(axis=1), (v * fwd).sum(axis=1)


def estimate_error(ev, rec, lock):
    n = len(ev["owner"])
    est = np.asarray(ev["est_pos"], dtype=float)
    alive = (np.asarray(ev["track_id"]) >= 0) & np.isfinite(est).all(axis=1)
    if lock is not None:
        alive[:lock] = False
    out = np.full(n, np.nan)
    out[alive] = np.linalg.norm(est[alive, :2] - rec["pos"][:n, 0, :2].astype(float)[alive], axis=1)
    return out


def jitter(ev, dt=DT):
    """Per tick: (horizontal acceleration m/s^2, jerk m/s^3, yaw rate deg/s) of the drone."""
    acc = np.asarray(ev["log_acceleration"], dtype=float)[:, :2]
    jerk = np.r_[np.zeros((1, 2)), np.diff(acc, axis=0) / dt]
    yaw = np.unwrap(np.radians(np.asarray(ev["log_yaw"], dtype=float)))
    rate = np.degrees(np.r_[0.0, np.diff(yaw) / dt])
    return np.linalg.norm(acc, axis=1), np.linalg.norm(jerk, axis=1), rate


def target_lateral_speed(rec):
    """Per tick: the target's speed across its own heading (m/s)."""
    vel, yaw = rec["vel"][:, 0, :2].astype(float), np.radians(rec["rot"][:, 0, 1].astype(float))
    return -vel[:, 0] * np.sin(yaw) + vel[:, 1] * np.cos(yaw)


def target_yaw_rate(rec, dt=DT):
    yaw = np.unwrap(np.radians(rec["rot"][:, 0, 1].astype(float)))
    return np.degrees(np.r_[0.0, np.diff(yaw) / dt])


def target_speed(rec):
    return np.linalg.norm(rec["vel"][:, 0, :2].astype(float), axis=1)


def speed_jump(rec, change, dt=DT):
    """The target's highest speed from the change's start to 6 s after its end, less its mean speed over the 3 s
    before the start (m/s)."""
    v = target_speed(rec)
    a, b = change["start_tick"], change["end_tick"]
    before = v[max(0, a - int(round(3.0 / dt))):a]
    after = v[a:min(len(v), b + int(round(6.0 / dt)) + 1)]
    if not len(before) or not len(after):
        return None
    return round(float(after.max() - before.mean()), 2)


def _stats(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean": None, "p95": None}
    return {"mean": round(float(v.mean()), 3), "p95": round(float(np.percentile(v, 95)), 3)}


def _rms(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return round(float(np.sqrt(np.mean(v ** 2))), 3) if len(v) else None


def per_tick(ev, rec, lock, dt=DT):
    """The per-tick series the measures are taken from (also for plotting)."""
    n = len(ev["owner"])
    acc, jerk, rate = jitter(ev, dt)
    off = goal_offset(ev, rec)
    across, along = split(off, rec)
    acc_across, acc_along = split(ev["log_acceleration"], rec)
    v = target_speed(rec)[:n]
    return {"owner": np.asarray(ev["owner"]), "visible": np.asarray(ev["target_px"]) >= scorer.VISIBLE_PX,
            "lost": np.asarray(ev["track_id"]) < 0, "distance": np.linalg.norm(off, axis=1),
            "across": np.abs(across), "along": np.abs(along),
            "estimate": estimate_error(ev, rec, lock), "accel": acc, "accel_across": acc_across,
            "accel_along": acc_along, "jerk": jerk, "yaw_rate": rate,
            "target_lat": target_lateral_speed(rec)[:n], "target_yaw": target_yaw_rate(rec, dt)[:n],
            "target_accel": np.r_[0.0, np.diff(v) / dt]}


def measures(series, mask, events, dt=DT):
    """The measures of the ticks in `mask`."""
    s = series
    owner, visible = s["owner"][mask], s["visible"][mask]
    return {
        "seconds": round(float(mask.sum() * dt), 2),
        "wrong_box_s": round(float(((owner >= 0) & (owner != 0)).sum() * dt), 2),
        "no_box_s": round(float(((owner < 0) & visible).sum() * dt), 2),
        "visible_ticks": int(visible.sum()),
        "held_ticks": int((visible & (owner == 0)).sum()),
        "hold": round(float((visible & (owner == 0)).sum() / visible.sum()), 4) if visible.any() else None,
        "lost_s": round(float(s["lost"][mask].sum() * dt), 2),
        "events": {k: sum(1 for e in events if e["kind"] == k and mask[e["tick"]]) for k in EVENT_KINDS},
        "distance_error": _stats(s["distance"][mask]),
        "across_error": _stats(s["across"][mask]),
        "along_error": _stats(s["along"][mask]),
        "estimate_error": _stats(s["estimate"][mask]),
        "accel_rms": _rms(s["accel"][mask]),
        "accel_across_rms": _rms(s["accel_across"][mask]),
        "accel_along_rms": _rms(s["accel_along"][mask]),
        "jerk_rms": _rms(s["jerk"][mask]),
        "yaw_rate_rms": _rms(s["yaw_rate"][mask]),
        "target_lat_speed": _rms(s["target_lat"][mask]),
        "target_yaw_rate": _rms(s["target_yaw"][mask]),
        "target_accel_rms": _rms(s["target_accel"][mask]),
    }


def flight_phases(ev, rec, lock, events, changes, dt=DT):
    """Per phase the measures of one flight, its phase per tick, and whether it kept the car first named."""
    n = len(ev["owner"])
    start = max(lock if lock is not None else n, int(round(scorer.WARMUP_S / dt)))
    phase = phase_labels(n, changes, start, dt)
    series = per_tick(ev, rec, lock, dt)
    out = {p: measures(series, phase == p, events, dt) for p in PHASES if (phase == p).any()}
    end = int(np.flatnonzero(phase == "after")[-1]) if (phase == "after").any() else n - 1
    bad = [e for e in events if e["kind"] == "first_pick" or (
        e["kind"] in ("switch_box", "switch_estimate", "loss") and lock is not None and lock <= e["tick"] <= end)]
    first_bad = bad[0] if bad else None
    kept = lock is not None and first_bad is None
    return {"phases": out, "kept_same_car": bool(kept),
            "first_failure": None if first_bad is None else {"kind": first_bad["kind"],
                                                              "t_s": round(first_bad["tick"] * dt, 2),
                                                              "phase": str(phase[first_bad["tick"]]) or "outside"},
            "changes": [dict(c, speed_jump_mps=speed_jump(rec, c, dt)) for c in changes]}, phase


def pool(flights):
    """Pooled measures of several flights' phases (each a flight_phases()['phases'] dict): sums of seconds and
    events, tick-weighted means, worst 95th percentile and root mean square of the per-flight RMS values."""
    out = {}
    for p in PHASES:
        rows = [f[p] for f in flights if p in f]
        if not rows:
            continue
        w = np.array([r["seconds"] for r in rows])

        def wmean(key, sub=None):
            vals = [(r[key][sub] if sub else r[key]) for r in rows]
            ok = [(v, x) for v, x in zip(vals, w) if v is not None]
            if not ok:
                return None
            v, x = np.array(ok).T
            return round(float((v * x).sum() / x.sum()), 3)

        def wrms(key):
            ok = [(r[key], x) for r, x in zip(rows, w) if r[key] is not None]
            if not ok:
                return None
            v, x = np.array(ok).T
            return round(float(np.sqrt((v ** 2 * x).sum() / x.sum())), 3)

        visible = sum(r["visible_ticks"] for r in rows)
        out[p] = {
            "flights": len(rows),
            "seconds": round(float(w.sum()), 2),
            "wrong_box_s": round(sum(r["wrong_box_s"] for r in rows), 2),
            "no_box_s": round(sum(r["no_box_s"] for r in rows), 2),
            "hold": round(sum(r["held_ticks"] for r in rows) / visible, 4) if visible else None,
            "lost_s": round(sum(r["lost_s"] for r in rows), 2),
            "events": {k: sum(r["events"][k] for r in rows) for k in EVENT_KINDS},
            "distance_error_mean": wmean("distance_error", "mean"),
            "across_error_mean": wmean("across_error", "mean"),
            "along_error_mean": wmean("along_error", "mean"),
            "distance_error_p95_worst": max((r["distance_error"]["p95"] for r in rows
                                             if r["distance_error"]["p95"] is not None), default=None),
            "estimate_error_mean": wmean("estimate_error", "mean"),
            "estimate_error_p95_worst": max((r["estimate_error"]["p95"] for r in rows
                                             if r["estimate_error"]["p95"] is not None), default=None),
            "accel_rms": wrms("accel_rms"),
            "accel_across_rms": wrms("accel_across_rms"),
            "accel_along_rms": wrms("accel_along_rms"),
            "jerk_rms": wrms("jerk_rms"),
            "yaw_rate_rms": wrms("yaw_rate_rms"),
            "target_lat_speed": wrms("target_lat_speed"),
            "target_yaw_rate": wrms("target_yaw_rate"),
            "target_accel_rms": wrms("target_accel_rms"),
        }
    return out
