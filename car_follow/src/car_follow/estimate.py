"""Target position from a tracking box (docs/demo1_plan.md 2.3, docs/demo1.md 5): where the car is, from the drone's
camera pose, the box and the depth map, and a constant-velocity Kalman filter over those measurements.

Measurements of one box (the drone knows its own pose, as a real drone does from GPS and IMU):
    flat        the ray through the box's bottom centre meets the flat ground z = `ground_z` (the plan's method;
                wrong where the road is not at that height, e.g. on the elevated ring road)
    depth       the ray through the box's centre meets the first surface in the depth map (the depth camera of
                the altitude guard, src/car_follow/clearance.py); no flat-ground assumption
Both land on the car's near surface, not its centre, so each is pushed on along the ray (horizontally) by a fixed
distance fitted on dev trials. The filter tracks the car's origin (its ground point) in x, y with constant velocity
and smooths its height. Pure numpy.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import clearance


# Fitted on dev 20 with ground-truth boxes (docs/demo1.md 5.2): the depth point lands on the car's near surface.
DEPTH_PUSH_M = 1.30   # on along the box-centre ray, horizontally, to the car's origin
DEPTH_LIFT_M = -1.38  # and down to its ground point
AIM_ABOVE_M = 0.75    # the camera aims this far above the estimated ground point (about the box centre)


def pixel_ray(eye, yaw, pitch, width, height, fov, pixel):
    """Unit world direction of the ray through `pixel` (x, y) of a camera at `eye` looking along (yaw, pitch)."""
    f, r, u = clearance.axes(yaw, pitch)
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    d = f + (pixel[0] - width / 2.0) / focal * r - (pixel[1] - height / 2.0) / focal * u
    return d / np.linalg.norm(d)


def on_plane(eye, ray, z):
    """Where the ray from `eye` meets the plane at height `z` (None if it does not go down to it)."""
    if ray[2] >= -1e-6 or eye[2] <= z:
        return None
    return np.asarray(eye, dtype=float) + (z - eye[2]) / ray[2] * ray


def along_depth(eye, ray, depth, dcam, yaw):
    """Distance along the ray from `eye` to the first surface in the depth map of the depth camera at the same
    `eye` (yaw `yaw`, the camera's fixed mount pitch), or None if the ray leaves its image or sees nothing."""
    px, z = clearance.project(np.asarray(eye, dtype=float) + ray, dcam, eye, yaw)
    if px is None or not (0.0 <= px[0] < dcam.width and 0.0 <= px[1] < dcam.height):
        return None
    d = float(depth[int(px[1]), int(px[0])])
    return d / z if d < dcam.max_range_m else None  # planar depth -> distance along the unit ray


def push(point, ray, distance):
    """`point` moved on horizontally along the ray by `distance`."""
    h = np.array([ray[0], ray[1], 0.0])
    n = float(np.linalg.norm(h))
    return point if n < 1e-6 else point + distance * h / n


def measure(box, eye, yaw, pitch, cam, depth=None, dcam=None, ground_z=0.0):
    """Raw measurements of one box (x0, y0, x1, y1): {"flat": point or None, "depth": point or None,
    "ray_bottom": ..., "ray_centre": ...}. `cam` has width, height and fov."""
    w, h, fov = cam["width"], cam["height"], cam["fov"]
    bottom = pixel_ray(eye, yaw, pitch, w, h, fov, ((box[0] + box[2]) / 2.0, box[3]))
    centre = pixel_ray(eye, yaw, pitch, w, h, fov, ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
    out = {"ray_bottom": bottom, "ray_centre": centre, "flat": on_plane(eye, bottom, ground_z), "depth": None}
    if depth is not None:
        t = along_depth(eye, centre, depth, dcam, yaw)
        if t is not None:
            out["depth"] = np.asarray(eye, dtype=float) + t * centre
    return out


def locate(box, eye, yaw, pitch, cam, depth, dcam):
    """The car's origin (ground point) from a box by the depth map, with the dev-fitted correction; None if the
    box-centre ray finds no depth."""
    m = measure(box, eye, yaw, pitch, cam, depth, dcam)
    if m["depth"] is None:
        return None
    return push(m["depth"], m["ray_centre"], DEPTH_PUSH_M) + np.array([0.0, 0.0, DEPTH_LIFT_M])


@dataclass(frozen=True)
class FilterParams:
    accel_sigma: float = 3.0     # m/s^2, white-noise acceleration of the constant-velocity model
    meas_sigma: float = 0.7      # m, measurement noise (x, y)
    height_s: float = 0.5        # s, time constant of the height smoothing
    heading_min_speed: float = 1.0  # m/s; below this the heading holds its last value


class TargetFilter:
    """Constant-velocity Kalman filter on the target's ground point: state (x, y, vx, vy); height smoothed apart."""

    def __init__(self, params=FilterParams()):
        self.p = params
        self.x = None
        self.P = None
        self.z = None
        self.yaw = None
        self.since_update = 0.0

    def predict(self, dt):
        if self.x is None:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.p.accel_sigma ** 2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q * G @ G.T
        self.since_update += dt

    def update(self, point, dt):
        point = np.asarray(point, dtype=float)
        if self.x is None:
            self.x = np.array([point[0], point[1], 0.0, 0.0])
            self.P = np.diag([self.p.meas_sigma ** 2] * 2 + [25.0, 25.0])
            self.z = point[2]
        else:
            H = np.zeros((2, 4))
            H[0, 0] = H[1, 1] = 1.0
            R = np.eye(2) * self.p.meas_sigma ** 2
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (point[:2] - H @ self.x)
            self.P = (np.eye(4) - K @ H) @ self.P
            self.z += dt / (self.p.height_s + dt) * (point[2] - self.z)
        self.since_update = 0.0
        v = self.x[2:]
        if float(np.linalg.norm(v)) > self.p.heading_min_speed:
            self.yaw = math.degrees(math.atan2(v[1], v[0]))

    @property
    def position(self):
        return None if self.x is None else np.array([self.x[0], self.x[1], self.z])

    @property
    def velocity(self):
        return None if self.x is None else np.array([self.x[2], self.x[3], 0.0])
