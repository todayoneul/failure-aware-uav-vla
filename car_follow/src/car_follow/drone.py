"""The drone of Phase 2 (docs/demo1.md 2): a point mass that follows velocity commands with a first-order lag,
within limits on speed, acceleration and turn rate. No flight model: CARLA has none that runs in step with its
synchronous ticks (docs/carla_air_check.md), so the drone is integrated here, once per tick, and stays reproducible.

The camera is on a gimbal: its yaw is the body yaw (turned by the body at `yaw_rate`), its pitch turns on its own
at `pitch_rate`. Positions are CARLA world metres (x forward, y right, z up); angles are degrees.
Pure numpy, no simulator.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Limits:
    speed_h: float = 25.0      # horizontal speed, m/s (docs/demo1.md 2.7; the plan began at 20)
    accel_h: float = 6.0       # horizontal acceleration, m/s^2 (docs/demo1.md 2.7; the plan began at 4)
    speed_v: float = 4.0       # vertical speed, m/s
    accel_v: float = 2.0       # vertical acceleration, m/s^2
    yaw_rate: float = 60.0     # body turn rate, deg/s (docs/demo1_plan.md 2.3)
    pitch_rate: float = 90.0   # gimbal pitch rate, deg/s
    pitch_min: float = -90.0   # gimbal looks straight down at most
    pitch_max: float = 0.0     # and level at least
    lag_s: float = 0.5         # time constant of the velocity response, s (0: none; docs/demo1.md 2.7)


@dataclass
class State:
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    pitch: float = 0.0
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self):
        return State(self.position.copy(), self.velocity.copy(), self.yaw, self.pitch, self.acceleration.copy())


@dataclass(frozen=True)
class Command:
    velocity: np.ndarray       # wanted velocity, m/s
    yaw: float                 # wanted body (camera) yaw, deg
    pitch: float               # wanted gimbal pitch, deg


def wrap(deg):
    """An angle in [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def clip_norm(v, limit):
    n = float(np.linalg.norm(v))
    return v if n <= limit or n == 0.0 else v * (limit / n)


def limit_velocity(v, limits):
    out = np.array(v, dtype=float)
    out[:2] = clip_norm(out[:2], limits.speed_h)
    out[2] = float(np.clip(out[2], -limits.speed_v, limits.speed_v))
    return out


def turn(angle, wanted, rate, dt):
    return angle + float(np.clip(wrap(wanted - angle), -rate * dt, rate * dt))


def step(state, command, limits, dt):
    """The state one tick (`dt` seconds) later under `command`."""
    v_cmd = limit_velocity(command.velocity, limits)
    # Exact first-order response over the tick (stable for any dt), then the acceleration limits.
    gain = 1.0 if limits.lag_s <= 0.0 else 1.0 - math.exp(-dt / limits.lag_s)
    accel = (v_cmd - state.velocity) * gain / dt
    accel[:2] = clip_norm(accel[:2], limits.accel_h)
    accel[2] = float(np.clip(accel[2], -limits.accel_v, limits.accel_v))
    velocity = state.velocity + accel * dt
    position = state.position + 0.5 * (state.velocity + velocity) * dt
    yaw = wrap(turn(state.yaw, command.yaw, limits.yaw_rate, dt))
    pitch = float(np.clip(turn(state.pitch, command.pitch, limits.pitch_rate, dt), limits.pitch_min,
                          limits.pitch_max))
    return State(position, velocity, yaw, pitch, accel)
