"""CARLA-Air probe 3 (check 4): step responses of the AirSim SimpleFlight drone in real time (async mode).

Velocity steps 0 -> 10 -> 0 m/s and 0 -> 20 m/s along NED x, and a yaw-rate step, sampled by polling
getMultirotorState as fast as the RPC allows. The results set the lag and limits of our own dynamics model.
"""
import json
from pathlib import Path
import math
import time

import airsim
import carla

ac = airsim.MultirotorClient(port=41451, timeout_value=15)
ac.confirmConnection()
ac.enableApiControl(True)
ac.armDisarm(True)
world = carla.Client("localhost", 2000).get_world()


def sample(seconds):
    rows = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        s = ac.getMultirotorState()
        k = s.kinematics_estimated
        v = k.linear_velocity
        q = k.orientation
        yaw = math.degrees(airsim.to_eularian_angles(q)[2])
        rows.append((s.timestamp / 1e9, v.x_val, v.y_val, v.z_val, yaw, k.position.z_val))
    return rows


def step_stats(rows, axis, v0, v1):
    t0 = rows[0][0]
    span = v1 - v0
    out = {"samples": len(rows), "poll_hz": round(len(rows) / (rows[-1][0] - t0), 1)}
    for frac in (0.1, 0.63, 0.9):
        hit = next((r[0] - t0 for r in rows if (r[axis] - v0) / span >= frac), None)
        out[f"t{int(frac * 100)}_s"] = None if hit is None else round(hit, 3)
    vals = [r[axis] for r in rows]
    peak = max(vals) if span > 0 else min(vals)
    out["peak"] = round(peak, 2)
    out["overshoot_pct"] = round(100.0 * (peak - v1) / span, 1) if span else None
    out["final"] = round(sum(vals[-20:]) / 20, 2)
    acc = [abs((b[axis] - a[axis]) / (b[0] - a[0])) for a, b in zip(rows, rows[1:]) if b[0] - a[0] > 1e-4]
    out["max_accel"] = round(max(acc), 2) if acc else None
    out["max_vz"] = round(max(abs(r[3]) for r in rows), 2)
    return out


res = {}
z_hold = 27.45 - 60.0  # NED z for about 60 m above the start ground
t = time.monotonic()
ac.moveToZAsync(z_hold, 8, timeout_sec=40).join()
ac.hoverAsync().join()
time.sleep(2.0)
res["climb_s"] = round(time.monotonic() - t, 1)
res["collision_before"] = ac.simGetCollisionInfo().has_collided

ac.moveByVelocityZAsync(10.0, 0.0, z_hold, duration=30.0)
res["vx_0_to_10"] = step_stats(sample(5.0), 1, 0.0, 10.0)
ac.moveByVelocityZAsync(0.0, 0.0, z_hold, duration=30.0)
res["vx_10_to_0"] = step_stats(sample(5.0), 1, 10.0, 0.0)
ac.moveByVelocityZAsync(-20.0, 0.0, z_hold, duration=30.0)
res["vx_0_to_minus20"] = step_stats(sample(7.0), 1, 0.0, -20.0)
ac.moveByVelocityZAsync(0.0, 0.0, z_hold, duration=30.0)
res["vx_minus20_to_0"] = step_stats(sample(6.0), 1, -20.0, 0.0)

ac.hoverAsync().join()
ac.rotateByYawRateAsync(60.0, 3.0)
rows = sample(3.5)
yaw = [r[4] for r in rows]
rates = []
for a, b in zip(rows, rows[1:]):
    d = (b[4] - a[4] + 180.0) % 360.0 - 180.0
    if b[0] - a[0] > 1e-4:
        rates.append(d / (b[0] - a[0]))
mid = rates[len(rates) // 3: 2 * len(rates) // 3]
res["yaw_rate_60"] = {"mean_rate_mid_dps": round(sum(mid) / len(mid), 1),
                      "t_to_30dps_s": round(next((rows[i + 1][0] - rows[0][0] for i, r in enumerate(rates) if r >= 30.0), -1), 3)}
res["collision_after"] = ac.simGetCollisionInfo().has_collided
ac.hoverAsync().join()

# Async frame rate of the world while the drone flies.
f0 = world.get_snapshot().frame
time.sleep(3.0)
res["async_world_fps"] = round((world.get_snapshot().frame - f0) / 3.0, 1)

print(json.dumps(res, indent=1))
OUT = Path(__file__).resolve().parents[2] / "outputs" / "carla_air_check"
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / (Path(__file__).stem + ".json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
