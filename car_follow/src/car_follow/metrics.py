"""Follow quality of one flight (docs/demo1_plan.md 3.2, docs/demo1.md 2): how far the drone is from the true goal
point, how well the camera points at the target, and how smoothly it flies. Pure numpy.

The switch and hold metrics of the plan need the tracker and come with the scorer (implementation step 4).
"""
import math

import numpy as np

from . import follow


def camera_angles(yaw, pitch, eye, point):
    """(horizontal, vertical) angle in degrees of `point` off the camera's optical axis (positive: right, up)."""
    y, p = np.radians(yaw), np.radians(pitch)
    forward = np.stack([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], axis=-1)
    right = np.stack([-np.sin(y), np.cos(y), np.zeros_like(y)], axis=-1)
    up = np.cross(forward, right)  # CARLA is left-handed: forward x right points up
    d = np.asarray(point) - np.asarray(eye)
    f, r, u = (d * forward).sum(-1), (d * right).sum(-1), (d * up).sum(-1)
    return np.degrees(np.arctan2(r, f)), np.degrees(np.arctan2(u, np.hypot(f, r)))


def _stats(x):
    x = np.asarray(x, dtype=float)
    return {"mean": round(float(x.mean()), 3), "p95": round(float(np.percentile(x, 95)), 3),
            "max": round(float(x.max()), 3)}


def follow_quality(log, target, limits, dt=0.05, warmup_s=5.0, camera=None):
    """Metrics of a flight `log` (follow.simulate's layout) against the true `target` track
    (position T x 3, yaw T, aim T x 3). Ticks before `warmup_s` are left out (docs/demo1_plan.md 3.2)."""
    camera = camera or {"width": 960, "height": 540, "fov": 90.0}
    keep = log["t"] >= warmup_s
    goal = np.array([follow.goal_point(p, y) for p, y in zip(target["position"], target["yaw"])])
    err = log["position"] - goal
    # The error split into range (horizontal distance to the target), height and bearing (angle off "behind").
    rel = log["position"] - np.asarray(target["position"])
    rng = np.linalg.norm(rel[:, :2], axis=1)
    back = -np.array([follow.heading(y)[:2] for y in target["yaw"]])
    cosb = (rel[:, :2] * back).sum(1) / np.maximum(rng, 1e-6)
    bearing = np.degrees(np.arccos(np.clip(cosb, -1.0, 1.0)))
    h, v = camera_angles(log["yaw"], log["pitch"], log["position"], target["aim"])
    half_h = camera["fov"] / 2.0
    half_v = math.degrees(math.atan(math.tan(math.radians(half_h)) * camera["height"] / camera["width"]))
    in_view = (np.abs(h) < half_h) & (np.abs(v) < half_v)
    acc_h = np.linalg.norm(log["acceleration"][:, :2], axis=1)
    jerk = np.linalg.norm(np.diff(log["acceleration"], axis=0), axis=1) / dt
    yaw_rate = np.abs((np.diff(log["yaw"]) + 180.0) % 360.0 - 180.0) / dt
    speed_t = np.linalg.norm(np.diff(target["position"], axis=0), axis=1) / dt
    k1 = keep[1:]
    return {
        "distance_error_m": _stats(np.linalg.norm(err, axis=1)[keep]),
        "horizontal_error_m": _stats(np.linalg.norm(err[:, :2], axis=1)[keep]),
        "vertical_error_m": _stats(np.abs(err[:, 2])[keep]),
        "range_error_m": _stats(np.abs(rng - follow.BEHIND_M)[keep]),
        "bearing_error_deg": _stats(bearing[keep]),
        "aim_error_deg": _stats(np.hypot(h, v)[keep]),
        "target_in_view": round(float(in_view[keep].mean()), 4),
        "drone_speed_max": round(float(np.linalg.norm(log["velocity"][:, :2], axis=1).max()), 2),
        "accel_h_max": round(float(acc_h.max()), 2),
        "accel_limited": round(float((acc_h[keep] >= 0.99 * limits.accel_h).mean()), 4),
        "jerk_rms": round(float(np.sqrt((jerk[k1] ** 2).mean())), 3),
        "jerk_max": round(float(jerk[k1].max()), 2),
        "yaw_rate_max": round(float(yaw_rate.max()), 1),
        "yaw_limited": round(float((yaw_rate[k1] >= 0.99 * limits.yaw_rate).mean()), 4),
        "target_speed_mean": round(float(speed_t.mean()), 2),
        "target_speed_max": round(float(speed_t.max()), 2),
    }
