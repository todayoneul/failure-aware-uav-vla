"""Demo 1 follow control (docs/demo1_plan.md 2.3, docs/demo1.md 2): hold the point `behind` metres behind the target
along its heading and `above` metres over it, and keep the camera on the target.

The controller sends velocity commands, like an offboard velocity setpoint on a real autopilot:
    command = goal velocity (+ lag x goal acceleration, to make up for the drone's response lag)
              + correction towards the goal point
The correction is a square-root controller (as in ArduPilot's position controller): proportional (kp) near the
goal, and far from it no faster than the drone can stop at `correction_accel`, so a large error does not make it
overshoot. The goal velocity and acceleration come from the goal point's own motion (filtered differences), so the
same code works on ground truth (config 1) and later on the Kalman estimate (configs 2-3).

The heading that places the goal point is filtered (`heading_s`, like the responsiveness filter of PX4 Follow Me):
when the target turns, the goal swings round it gradually instead of jumping to the new "behind".
Pure numpy, no simulator: `simulate` runs the loop over a recorded target track, and the live loop calls
`Follower.command` and `drone.step` in the same order.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import drone

BEHIND_M = 20.0
ABOVE_M = 15.0


def heading(yaw_deg):
    y = math.radians(yaw_deg)
    return np.array([math.cos(y), math.sin(y), 0.0])


def goal_from(target_position, direction, behind=BEHIND_M, above=ABOVE_M):
    """The goal point `behind` metres back along the horizontal unit `direction` and `above` metres up."""
    return np.asarray(target_position, dtype=float) - behind * direction + np.array([0.0, 0.0, above])


def goal_point(target_position, target_yaw, behind=BEHIND_M, above=ABOVE_M):
    """The true goal point of docs/demo1_plan.md 3.2: behind the target along its heading."""
    return goal_from(target_position, heading(target_yaw), behind, above)


def aim_angles(eye, point):
    """(yaw, pitch) in degrees that point a camera at `eye` towards `point`."""
    d = np.asarray(point, dtype=float) - np.asarray(eye, dtype=float)
    return math.degrees(math.atan2(d[1], d[0])), math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))


def default_kp(lag_s):
    """Position gain that makes the loop critically damped for a first-order velocity lag (kp = 1 / (4 lag)),
    kept between 0.25 and 1.0 per second."""
    return 1.0 if lag_s <= 0.0 else float(np.clip(1.0 / (4.0 * lag_s), 0.25, 1.0))


def sqrt_correction(error, kp, accel):
    """Velocity towards the goal for a position `error` vector: kp x error near it, and no more than the speed from
    which `accel` stops the drone in the remaining distance (continuous where the two meet)."""
    d = float(np.linalg.norm(error))
    if d == 0.0:
        return np.zeros_like(error)
    linear = accel / (kp * kp)
    speed = kp * d if d <= linear else math.sqrt(2.0 * accel * (d - linear / 2.0))
    return error / d * speed


@dataclass(frozen=True)
class Gains:
    kp: float = None             # 1/s; None: default_kp(lag)
    lead: bool = True            # add lag x goal acceleration
    smooth_s: float = 0.25       # time constant of the goal velocity / acceleration filters, s
    heading_s: float = 1.0       # time constant of the heading filter that places the goal, s (0: none; docs/demo1.md 2.7)
    correction_accel: float = 2.0  # deceleration the correction plans with, m/s^2 (half the horizontal limit)


class Follower:
    def __init__(self, limits, gains=Gains(), behind=BEHIND_M, above=ABOVE_M):
        self.limits, self.gains, self.behind, self.above = limits, gains, behind, above
        self.kp = default_kp(limits.lag_s) if gains.kp is None else gains.kp
        self.direction = None
        self.prev_goal = None
        self.goal_velocity = np.zeros(3)
        self.goal_acceleration = np.zeros(3)

    def _direction(self, target_yaw, dt):
        h = heading(target_yaw)
        if self.direction is None or self.gains.heading_s <= 0.0:
            self.direction = h
        else:
            d = self.direction + dt / (self.gains.heading_s + dt) * (h - self.direction)
            n = float(np.linalg.norm(d))
            self.direction = d / n if n > 1e-6 else h
        return self.direction

    def command(self, state, target_position, target_yaw, aim_point, dt):
        """The velocity, yaw and pitch command for this tick, and the goal point it aims for."""
        goal = goal_from(target_position, self._direction(target_yaw, dt), self.behind, self.above)
        if self.prev_goal is not None:
            alpha = dt / (self.gains.smooth_s + dt)
            raw = (goal - self.prev_goal) / dt
            new_velocity = self.goal_velocity + alpha * (raw - self.goal_velocity)
            raw_acc = (new_velocity - self.goal_velocity) / dt
            self.goal_acceleration = self.goal_acceleration + alpha * (raw_acc - self.goal_acceleration)
            self.goal_velocity = new_velocity
        self.prev_goal = goal
        v = self.goal_velocity + sqrt_correction(goal - state.position, self.kp, self.gains.correction_accel)
        if self.gains.lead and self.limits.lag_s > 0.0:
            v = v + self.limits.lag_s * self.goal_acceleration
        yaw, pitch = aim_angles(state.position, aim_point)
        return drone.Command(v, yaw, pitch), goal


def start_state(drone_start):
    """The drone state at a trial's `drone_start` transform [x, y, z, pitch, yaw, roll], at rest."""
    x, y, z, pitch, yaw, _ = drone_start
    return drone.State(np.array([x, y, z], dtype=float), yaw=yaw, pitch=pitch)


def simulate(track, start, limits, gains=Gains(), dt=0.05):
    """Fly the follow loop over a recorded target track and return the per-tick log (numpy arrays).

    `track` has per-tick arrays: position (T x 3, the target's origin), yaw (T), aim (T x 3, its box centre).
    Log row k holds the drone state at tick k and the goal the follower used at tick k."""
    follower = Follower(limits, gains)
    state = start.copy()
    n = len(track["yaw"])
    log = {k: np.zeros((n, 3)) for k in ("position", "velocity", "acceleration", "goal", "command")}
    log["yaw"], log["pitch"] = np.zeros(n), np.zeros(n)
    for k in range(n):
        cmd, goal = follower.command(state, track["position"][k], track["yaw"][k], track["aim"][k], dt)
        log["position"][k], log["velocity"][k], log["acceleration"][k] = state.position, state.velocity, \
            state.acceleration
        log["yaw"][k], log["pitch"][k], log["goal"][k], log["command"][k] = state.yaw, state.pitch, goal, \
            cmd.velocity
        state = drone.step(state, cmd, limits, dt)
    log["t"] = np.arange(n) * dt
    return log
