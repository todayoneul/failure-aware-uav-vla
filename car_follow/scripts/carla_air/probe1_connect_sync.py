"""CARLA-Air probe 1: both clients connect (check 1) and whether the drone follows world.tick() (check 2)."""
import json
from pathlib import Path
import math
import time

import airsim
import carla

out = {}


def drone_actor(world):
    hits = [a for a in world.get_actors() if "drone" in a.type_id.lower()]
    return hits[0] if hits else None


def pos_carla(world):
    d = drone_actor(world)
    l = d.get_location()
    return (l.x, l.y, l.z)


def pos_air(ac):
    s = ac.getMultirotorState()
    p = s.kinematics_estimated.position
    return (p.x_val, p.y_val, p.z_val), s.timestamp


def dist(a, b):
    return math.dist(a, b)


client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
out["server_version"] = client.get_server_version()
out["client_version"] = client.get_client_version()
world = client.get_world()
out["map"] = world.get_map().name
types = {}
for a in world.get_actors():
    key = a.type_id.split(".")[0]
    types[key] = types.get(key, 0) + 1
out["actor_kinds"] = types
d = drone_actor(world)
out["drone_actor"] = None if d is None else {"id": d.id, "type_id": d.type_id, "attributes": dict(d.attributes),
                                             "location": list(pos_carla(world))}
s0 = world.get_settings()
out["settings_before"] = {"sync": s0.synchronous_mode, "dt": s0.fixed_delta_seconds}
print(json.dumps(out, indent=1))

ac = airsim.MultirotorClient(port=41451, timeout_value=15)
ac.confirmConnection()
out["airsim_ping"] = ac.ping()
pa, ts = pos_air(ac)
out["airsim_start"] = {"pos_ned": pa, "timestamp_ns": ts}
if d is not None:
    pc = pos_carla(world)
    out["offset_air_minus_carla"] = [pa[0] - pc[0], pa[1] - pc[1], pa[2] + pc[2]]
print(json.dumps({k: out[k] for k in ("airsim_ping", "airsim_start", "offset_air_minus_carla") if k in out}, indent=1))

# Take off in async mode and climb to 20 m above the start.
ac.enableApiControl(True)
ac.armDisarm(True)
t = time.monotonic()
ac.takeoffAsync(timeout_sec=20).join()
ac.moveToZAsync(pa[2] - 20.0, 5, timeout_sec=30).join()
out["takeoff_climb_wall_s"] = round(time.monotonic() - t, 1)
ac.hoverAsync().join()
time.sleep(2.0)

try:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.tick()
    res = {}
    a0, t0 = pos_air(ac)
    c0 = pos_carla(world)
    # Fly forward at 5 m/s (NED x) and do not tick for 3 s of wall time.
    ac.moveByVelocityAsync(5.0, 0.0, 0.0, duration=60.0)
    w = time.monotonic()
    time.sleep(3.0)
    a1, t1 = pos_air(ac)
    c1 = pos_carla(world)
    res["no_tick_3s"] = {"air_moved_m": round(dist(a0, a1), 2), "carla_moved_m": round(dist(c0, c1), 2),
                         "air_clock_s": round((t1 - t0) / 1e9, 3)}
    # Now tick 60 times (3 s of simulation at 20 Hz) as fast as possible.
    per_tick = []
    w = time.monotonic()
    prev = pos_carla(world)
    for i in range(60):
        world.tick()
        cur = pos_carla(world)
        per_tick.append(round(dist(prev, cur), 3))
        prev = cur
    wall = time.monotonic() - w
    a2, t2 = pos_air(ac)
    c2 = pos_carla(world)
    res["ticks_60"] = {"wall_s": round(wall, 2), "air_moved_m": round(dist(a1, a2), 2),
                       "carla_moved_m": round(dist(c1, c2), 2), "air_clock_s": round((t2 - t1) / 1e9, 3),
                       "carla_step_m_first20": per_tick[:20]}
    out["sync_test"] = res
    print(json.dumps(res, indent=1))
finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    try:
        ac.hoverAsync().join()
    except Exception as e:  # noqa: BLE001
        print("hover failed:", e)

OUT = Path(__file__).resolve().parents[2] / "outputs" / "carla_air_check"
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / (Path(__file__).stem + ".json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=str)
print("saved")
