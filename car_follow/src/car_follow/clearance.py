"""Vision-based active altitude (docs/demo1.md 4.6): a depth map in, a height offset over the follow goal out.

A low-resolution depth camera fixed to the drone body looks forward and a little down. On a real drone a stereo
pair or a RealSense does this job. Each tick the guard reads the depth map and decides how far above the nominal
follow height (15 m over the target) the drone should fly:

    path            Points of the depth map inside a corridor along the drone's motion, up to `lookahead` ahead
                    and within `half_width_m` to either side, that reach higher than `clearance_m` below the
                    nominal height. The needed offset puts the drone `clearance_m` over the highest of them.
    line of sight   At the pixel where the target should be, the depth map is much shorter than the target's own
                    depth, so something stands in between. The needed offset is the one at which the line from the
                    drone to the target passes `los_margin_m` over every blocking point. Climbing cannot help when
                    the target is under a structure (depth points right above it, or a blocker right at it) or
                    the offset would exceed `max_offset_m`. Then the guard reports the view as unreachable and does
                    not climb for it.
    hold, return    The offset climbs as soon as it is needed, limited in rate and acceleration. After the need has
                    gone it holds for `hold_s`, then returns slowly and smoothly to the nominal height.

Inputs are only the depth map (metres along the optical axis), the camera's pose, the drone's velocity and an
estimate of the target point (ground truth in config 1, the tracker's estimate in configs 2-3). The guard never
reads the map or casts simulator rays. A monocular depth model can therefore replace the simulator's depth camera:
it only has to produce the same depth map. Pure numpy.
"""
import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class DepthCamera:
    width: int = 160
    height: int = 120
    fov: float = 90.0           # horizontal field of view, degrees (vertical: 73.7 at 160 x 120)
    mount_pitch: float = -20.0  # fixed to the body: yaw turns with the drone, pitch stays here
    max_range_m: float = 60.0   # farther pixels (and the sky) are ignored

    @property
    def focal(self):
        return self.width / (2.0 * math.tan(math.radians(self.fov) / 2.0))


@dataclass(frozen=True)
class Params:
    lookahead_s: float = 3.0       # corridor length: drone speed x this ...
    lookahead_min_m: float = 12.0  # ... and at least this (a hovering drone still looks ahead)
    half_width_m: float = 3.0      # corridor half width
    clearance_m: float = 3.0       # vertical clearance over obstacles in the corridor
    los_gap_m: float = 3.0         # a pixel blocks the view if it is this much nearer than the target
    los_window_px: int = 2         # half size of the pixel window around the target's pixel (5 x 5)
    los_share: float = 0.5         # share of blocking pixels in the window that makes the view blocked
    los_margin_m: float = 1.0      # the line of sight should pass this much over a blocking point
    los_min_r_m: float = 4.0       # a blocker nearer the target than this (along the line) means it is under it
    cover_radius_m: float = 3.0    # points within this (horizontally) of the target ...
    cover_height_m: float = 4.0    # ... and this much above it mean the target is under a structure
    max_offset_m: float = 10.0
    climb_rate: float = 3.0        # m/s (the drone's vertical limit is 4 m/s)
    climb_accel: float = 2.0       # m/s^2
    hold_s: float = 1.5            # stay up this long after the need has gone
    return_rate: float = 1.0       # m/s
    return_accel: float = 0.5      # m/s^2


LOS_STATES = ("clear", "blocked", "unreachable", "unseen")


@dataclass(frozen=True)
class Decision:
    offset: float        # height to add to the follow goal this tick, m
    need: float          # what the depth map asks for this tick, m (0 to max_offset_m)
    path: float          # need from the corridor, m
    los: float           # need from the line of sight, m
    los_state: str       # one of LOS_STATES
    nearest_m: float     # distance to the nearest point of the depth map (inf if none)


def axes(yaw, pitch):
    """(forward, right, up) unit vectors of a camera looking along (yaw, pitch), no roll, in CARLA world axes."""
    y, p = math.radians(yaw), math.radians(pitch)
    forward = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.array([-math.sin(y), math.cos(y), 0.0])
    return forward, right, np.cross(forward, right)  # CARLA is left-handed: forward x right points up


def pixel_rays(cam):
    """Per pixel (H x W x 2): the (right, up) offsets of its ray at unit depth along the optical axis."""
    xs = (np.arange(cam.width) + 0.5 - cam.width / 2.0) / cam.focal
    ys = (np.arange(cam.height) + 0.5 - cam.height / 2.0) / cam.focal
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, -gy], axis=-1)


def back_project(depth, cam, eye, yaw, rays=None, mask=None):
    """World points (N x 3) of the depth map's pixels nearer than `cam.max_range_m` (and inside `mask`)."""
    rays = pixel_rays(cam) if rays is None else rays
    keep = depth < cam.max_range_m
    if mask is not None:
        keep &= mask
    d = depth[keep][:, None]
    f, r, u = axes(yaw, cam.mount_pitch)
    ray = rays[keep]
    return np.asarray(eye, dtype=float) + d * (f + ray[:, :1] * r + ray[:, 1:] * u)


def project(point, cam, eye, yaw):
    """(x, y) pixel and depth along the optical axis of a world point, as the depth camera sees it."""
    f, r, u = axes(yaw, cam.mount_pitch)
    d = np.asarray(point, dtype=float) - np.asarray(eye, dtype=float)
    z = float(d @ f)
    if z <= 1e-6:
        return None, z
    return (cam.focal * float(d @ r) / z + cam.width / 2.0, -cam.focal * float(d @ u) / z + cam.height / 2.0), z


def corridor_direction(velocity, yaw, min_speed=1.0):
    """Horizontal unit vector of the drone's motion, or of its heading when it barely moves."""
    v = np.asarray(velocity, dtype=float)[:2]
    s = float(np.linalg.norm(v))
    if s > min_speed:
        return v / s, s
    y = math.radians(yaw)
    return np.array([math.cos(y), math.sin(y)]), s


def path_need(points, eye, velocity, yaw, base_z, p=Params()):
    """Offset over `base_z` (the nominal follow height) that keeps `p.clearance_m` over the corridor's obstacles."""
    if len(points) == 0:
        return 0.0
    d, speed = corridor_direction(velocity, yaw)
    length = max(p.lookahead_min_m, speed * p.lookahead_s)
    rel = points[:, :2] - np.asarray(eye, dtype=float)[:2]
    along = rel @ d
    lateral = np.abs(rel @ np.array([-d[1], d[0]]))
    z = points[(along > 0.0) & (along < length) & (lateral < p.half_width_m), 2]
    z = z[z > base_z - p.clearance_m]
    return float(z.max() + p.clearance_m - base_z) if len(z) else 0.0


def covered(points, target, p=Params()):
    """Whether depth points stand right above the target (it is under a deck or a roof)."""
    if len(points) == 0:
        return False
    target = np.asarray(target, dtype=float)
    near = np.linalg.norm(points[:, :2] - target[:2], axis=1) < p.cover_radius_m
    return bool((points[near, 2] > target[2] + p.cover_height_m).any())


def los_need(depth, cam, eye, yaw, target, base_z, p=Params(), rays=None, points=None):
    """(offset over `base_z` that clears the line of sight to `target`, state). The state is "clear", "blocked"
    (the offset clears it), "unreachable" (no offset up to `p.max_offset_m` would) or "unseen" (target outside the
    depth image). `points` (the depth map's world points) lets it tell a target under a structure."""
    px, z = project(target, cam, eye, yaw)
    if px is None or not (0.0 <= px[0] < cam.width and 0.0 <= px[1] < cam.height):
        return 0.0, "unseen"
    cx, cy, k = int(px[0]), int(px[1]), p.los_window_px
    x0, x1 = max(0, cx - k), min(cam.width, cx + k + 1)
    y0, y1 = max(0, cy - k), min(cam.height, cy + k + 1)
    window = depth[y0:y1, x0:x1]
    blocking = window < z - p.los_gap_m
    if blocking.mean() < p.los_share:
        return 0.0, "clear"
    if points is not None and covered(points, target, p):
        return 0.0, "unreachable"
    mask = np.zeros(depth.shape, bool)
    mask[y0:y1, x0:x1] = blocking
    pts = back_project(depth, replace(cam, max_range_m=np.inf), eye, yaw, rays, mask)
    target = np.asarray(target, dtype=float)
    line = np.asarray(eye, dtype=float)[:2] - target[:2]
    reach = float(np.linalg.norm(line))           # horizontal distance drone - target
    if reach < 1e-3:
        return 0.0, "unreachable"
    s = (pts[:, :2] - target[:2]) @ (line / reach)  # each blocker's distance from the target along the line
    if (s < p.los_min_r_m).any():
        return 0.0, "unreachable"
    h = pts[:, 2] - target[2]
    need = float(target[2] + ((h + p.los_margin_m) * reach / s).max() - base_z)
    if need > p.max_offset_m:
        return 0.0, "unreachable"
    return max(need, 0.0), "blocked"


class AltitudeGuard:
    """Turns each depth map into the height offset the follow controller adds to its goal."""

    def __init__(self, cam=DepthCamera(), params=Params()):
        self.cam, self.p = cam, params
        self.rays = pixel_rays(cam)
        self.offset, self.rate = 0.0, 0.0
        self.since_need = math.inf

    def update(self, depth, eye, yaw, velocity, base_z, target, dt):
        """One tick. `depth` (H x W, metres), `eye` and `yaw` the camera's position and heading, `velocity` the
        drone's, `base_z` the nominal follow height and `target` the estimated target point (or None)."""
        p = self.p
        pts = back_project(depth, self.cam, eye, yaw, self.rays)
        path = path_need(pts, eye, velocity, yaw, base_z, p)
        los, state = (0.0, "unseen") if target is None else los_need(depth, self.cam, eye, yaw, target, base_z, p,
                                                                     self.rays, pts)
        need = float(np.clip(max(path, los), 0.0, p.max_offset_m))
        self.since_need = 0.0 if need > 0.0 and need >= self.offset - 0.25 else self.since_need + dt
        if need > self.offset:
            goal, vmax, accel = need, p.climb_rate, p.climb_accel
        elif self.since_need >= p.hold_s:
            goal, vmax, accel = need, p.return_rate, p.return_accel
        else:
            goal, vmax, accel = self.offset, p.return_rate, p.climb_accel
        e = goal - self.offset
        want = math.copysign(min(vmax, math.sqrt(2.0 * accel * abs(e)), 2.0 * abs(e)), e)
        a = p.climb_accel if want > self.rate else max(accel, p.return_accel)
        self.rate += float(np.clip(want - self.rate, -a * dt, a * dt))
        self.offset = float(np.clip(self.offset + self.rate * dt, 0.0, p.max_offset_m))
        if self.offset in (0.0, p.max_offset_m):
            self.rate = 0.0
        nearest = float(np.linalg.norm(pts - np.asarray(eye, dtype=float), axis=1).min()) if len(pts) else math.inf
        return Decision(self.offset, need, path, los, state, nearest)
