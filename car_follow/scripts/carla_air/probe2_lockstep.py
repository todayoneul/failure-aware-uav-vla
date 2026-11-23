"""CARLA-Air probe 2: can AirSim pause / continue-for-time give lockstep with CARLA world.tick()?

The drone is airborne from probe 1. A CARLA car is spawned and driven at a set velocity so we can see
whether CARLA moves while AirSim is paused.
"""
import json
from pathlib import Path
import math
import threading
import time

import airsim
import carla

DT = 0.05


def drone_actor(world):
    return [a for a in world.get_actors() if a.type_id == "airsim.drone"][0]


def loc(actor):
    l = actor.get_location()
    return (l.x, l.y, l.z)


def air(ac):
    s = ac.getMultirotorState()
    p = s.kinematics_estimated.position
    return (p.x_val, p.y_val, p.z_val), s.timestamp / 1e9


client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
ac = airsim.MultirotorClient(port=41451, timeout_value=15)
ac.confirmConnection()
ac.enableApiControl(True)
res = {}

# A car 20 m from the drone on the nearest road, driven by set_target_velocity (no Traffic Manager needed).
drone = drone_actor(world)
wp = world.get_map().get_waypoint(drone.get_location())
bp = world.get_blueprint_library().find("vehicle.tesla.model3")
car = None
for w in [wp] + wp.next(20.0) + wp.next(40.0):
    t = w.transform
    t.location.z += 0.5
    car = world.try_spawn_actor(bp, t)
    if car:
        break
res["car_spawned"] = car is not None

try:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)
    for _ in range(10):
        world.tick()
    if car:
        car.set_target_velocity(carla.Vector3D(*[5.0 * c for c in (car.get_transform().get_forward_vector().x,
                                                                    car.get_transform().get_forward_vector().y, 0.0)]))

    # A: pause AirSim. Does the drone stop, and does CARLA still tick (car moves)?
    ac.moveByVelocityAsync(5.0, 0.0, 0.0, duration=600.0)
    time.sleep(1.0)
    ac.simPause(True)
    a0, t0 = air(ac)
    time.sleep(2.0)
    a1, t1 = air(ac)
    c0 = loc(car) if car else None
    w = time.monotonic()
    tick_ok = True

    def tick_n(n):
        global tick_ok
        try:
            for _ in range(n):
                world.tick(10.0)
        except Exception as e:  # noqa: BLE001
            tick_ok = repr(e)

    th = threading.Thread(target=tick_n, args=(20,))
    th.start()
    th.join(30.0)
    a2, t2 = air(ac)
    res["A_paused"] = {
        "is_paused": ac.simIsPause(),
        "air_moved_while_paused_2s_m": round(math.dist(a0, a1), 3),
        "air_clock_while_paused_s": round(t1 - t0, 3),
        "carla_20_ticks_ok": tick_ok if not th.is_alive() else "tick hung",
        "carla_20_ticks_wall_s": round(time.monotonic() - w, 2),
        "car_moved_in_20_ticks_m": round(math.dist(c0, loc(car)), 2) if car else None,
        "air_moved_during_ticks_m": round(math.dist(a1, a2), 3),
    }
    print(json.dumps(res, indent=1), flush=True)

    # B: per step simContinueForTime(DT) then world.tick(). Measure AirSim clock and motion per step.
    steps = []
    pa, pt = air(ac)
    pc = loc(drone_actor(world))
    w = time.monotonic()
    for i in range(40):
        ac.simContinueForTime(DT)
        # wait until AirSim is paused again
        deadline = time.monotonic() + 2.0
        while not ac.simIsPause() and time.monotonic() < deadline:
            time.sleep(0.001)
        world.tick(10.0)
        a, t = air(ac)
        c = loc(drone_actor(world))
        steps.append({"air_dt": round(t - pt, 4), "air_step_m": round(math.dist(pa, a), 4),
                      "carla_step_m": round(math.dist(pc, c), 4)})
        pa, pt, pc = a, t, c
    wall = time.monotonic() - w
    adt = [s["air_dt"] for s in steps[5:]]
    am = [s["air_step_m"] for s in steps[5:]]
    res["B_continue_for_time"] = {"wall_s_40_steps": round(wall, 2),
                                  "air_dt_mean": round(sum(adt) / len(adt), 4), "air_dt_min": min(adt),
                                  "air_dt_max": max(adt), "air_step_m_mean": round(sum(am) / len(am), 4),
                                  "first8": steps[:8]}
    print(json.dumps(res["B_continue_for_time"], indent=1), flush=True)

    # C: per step simContinueForFrames(1) then world.tick().
    steps = []
    pa, pt = air(ac)
    w = time.monotonic()
    for i in range(40):
        ac.simContinueForFrames(1)
        world.tick(10.0)
        deadline = time.monotonic() + 2.0
        while not ac.simIsPause() and time.monotonic() < deadline:
            time.sleep(0.001)
        a, t = air(ac)
        steps.append({"air_dt": round(t - pt, 4), "air_step_m": round(math.dist(pa, a), 4)})
        pa, pt = a, t
    wall = time.monotonic() - w
    adt = [s["air_dt"] for s in steps[5:]]
    res["C_continue_for_frames"] = {"wall_s_40_steps": round(wall, 2),
                                    "air_dt_mean": round(sum(adt) / len(adt), 4), "air_dt_min": min(adt),
                                    "air_dt_max": max(adt), "first8": steps[:8]}
    print(json.dumps(res["C_continue_for_frames"], indent=1), flush=True)
finally:
    try:
        ac.simPause(False)
        ac.hoverAsync()
    except Exception as e:  # noqa: BLE001
        print("airsim restore failed:", e)
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    if car:
        car.destroy()

OUT = Path(__file__).resolve().parents[2] / "outputs" / "carla_air_check"
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / (Path(__file__).stem + ".json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, default=str)
print("saved")
