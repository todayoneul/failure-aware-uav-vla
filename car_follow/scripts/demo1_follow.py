"""Demo 1 step 2 (docs/demo1.md 2): the drone follows the target on ground truth (config 1).

In config 1 the drone uses the target's true position and does not change anything in the world, so the follow loop
can be flown offline over recorded traffic, as many times as needed:

    record   run trials in CARLA with every car on autopilot and record each car's pose and velocity every tick
    offline  fly the follow loop over the recordings with several settings and score each flight
    live     fly the loop inside CARLA, recording the traffic again (compared with an earlier recording to see
             whether the trial replays the same way); --video writes the demo screen (docs/demo1.md 3): first
             person with the lock-on box, chase view with the drone drawn in, top-down map

Every trial starts from the same state: traffic lights reset, the trial's Traffic Manager seed, cars spawned in a
fixed order. Outputs go to outputs/demo1_follow/ (local only).

Run from the car_follow folder:
    uv run python scripts/demo1_follow.py record --split dev --run rec1
    uv run python scripts/demo1_follow.py offline --run rec1 --lag 0 0.5 1 2
    uv run python scripts/demo1_follow.py live --split dev --ids dev-001 --lag 0.5 --compare rec1 --video
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.car_follow import drone, follow, metrics, screen, sim, trials  # noqa: E402

OUT = ROOT / "outputs" / "demo1_follow"
XODR = ROOT / "assets" / "carla-0.9.16" / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{trials.MAP}.xodr"
HZ = 20.0
DT = 1.0 / HZ


def roles(actors):
    out = [("target", actors["target"])]
    out += [("lookalike", a) for a in actors["lookalikes"]]
    out += [("background", a) for a in actors["background"]]
    return [(r, a) for r, a in out if a is not None]


def start_trial(client, world, tm, trial):
    """Clear the world and spawn `trial` from a fixed state. Returns the spawned actors."""
    sim.clear(client, world)
    world.tick()
    world.reset_all_traffic_lights()
    tm.set_random_device_seed(int(trial["seed"]))
    return trials.spawn(world, trial, tm)


class Recorder:
    """Pose, velocity and box of every spawned car of a trial, every tick."""

    def __init__(self, actors, ticks):
        self.roles = roles(actors)
        n = len(self.roles)
        self.ids = np.array([a.id for _, a in self.roles])
        self.kind = np.array([r for r, _ in self.roles])
        self.box_center = np.array([[a.bounding_box.location.x, a.bounding_box.location.y,
                                     a.bounding_box.location.z] for _, a in self.roles])
        self.box_extent = np.array([[a.bounding_box.extent.x, a.bounding_box.extent.y,
                                     a.bounding_box.extent.z] for _, a in self.roles])
        self.pos = np.zeros((ticks, n, 3), np.float32)
        self.rot = np.zeros((ticks, n, 3), np.float32)
        self.vel = np.zeros((ticks, n, 3), np.float32)
        self.aim = np.zeros((ticks, 3))
        self.target = actors["target"]

    def take(self, world, k):
        snap = world.get_snapshot()
        for i, (_, a) in enumerate(self.roles):
            s = snap.find(a.id)
            if s is None:
                continue
            t, v = s.get_transform(), s.get_velocity()
            self.pos[k, i] = (t.location.x, t.location.y, t.location.z)
            self.rot[k, i] = (t.rotation.pitch, t.rotation.yaw, t.rotation.roll)
            self.vel[k, i] = (v.x, v.y, v.z)
            if a.id == self.target.id:
                c = t.transform(self.target.bounding_box.location)
                self.aim[k] = (c.x, c.y, c.z)
        return snap.frame

    def save(self, path, trial):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, trial_id=trial["id"], ids=self.ids, kind=self.kind, box_center=self.box_center,
                            box_extent=self.box_extent, pos=self.pos, rot=self.rot, vel=self.vel, aim=self.aim)


def target_track(rec):
    """The target's track (follow.simulate layout) from a saved recording."""
    i = int(np.flatnonzero(rec["kind"] == "target")[0])
    return {"position": rec["pos"][:, i].astype(float), "yaw": rec["rot"][:, i, 1].astype(float),
            "aim": rec["aim"].astype(float)}


def select(split, ids):
    found = trials.load_trials(split)
    return [t for t in found if not ids or t["id"] in ids]


def cmd_record(args):
    ticks = int(trials.DURATION_S * HZ)
    out = OUT / "traffic" / args.run
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, tm):
        for trial in select(args.split, args.ids):
            w = time.monotonic()
            actors = start_trial(client, world, tm, trial)
            rec = Recorder(actors, ticks)
            for k in range(ticks):
                world.tick()
                rec.take(world, k)
            rec.save(out / f"{trial['id']}.npz", trial)
            track = target_track(np.load(out / f"{trial['id']}.npz"))
            speed = np.linalg.norm(np.diff(track["position"], axis=0), axis=1) / DT
            print(f"{trial['id']}: {len(rec.roles)} cars, {time.monotonic() - w:.1f} s, target speed mean "
                  f"{speed.mean():.1f} max {speed.max():.1f} m/s, moved {np.linalg.norm(track['position'][-1] - track['position'][0]):.0f} m",
                  flush=True)


def settings_from(args):
    for speed in args.speed:
        for accel in args.accel:
            for lag in args.lag:
                for hs in args.heading:
                    yield drone.Limits(lag_s=lag, speed_h=speed, accel_h=accel), follow.Gains(heading_s=hs)


def summary(rows):
    keys = ("distance_error_m", "range_error_m", "vertical_error_m", "bearing_error_deg", "aim_error_deg")
    out = {f"{k}_mean": round(float(np.mean([r[k]["mean"] for r in rows])), 3) for k in keys}
    out.update({f"{k}_p95_worst": round(float(np.max([r[k]["p95"] for r in rows])), 3) for k in keys})
    out["distance_error_m_max"] = round(float(np.max([r["distance_error_m"]["max"] for r in rows])), 2)
    out["trials_over_3m"] = int(sum(r["distance_error_m"]["mean"] > 3.0 for r in rows))
    out["target_in_view_min"] = round(float(np.min([r["target_in_view"] for r in rows])), 4)
    for k in ("jerk_rms", "accel_limited", "yaw_limited"):
        out[f"{k}_mean"] = round(float(np.mean([r[k] for r in rows])), 4)
    out["jerk_max"] = round(float(np.max([r["jerk_max"] for r in rows])), 1)
    out["drone_speed_max"] = round(float(np.max([r["drone_speed_max"] for r in rows])), 1)
    out["target_speed_max"] = round(float(np.max([r["target_speed_max"] for r in rows])), 1)
    return out


def cmd_offline(args):
    paths = sorted((OUT / "traffic" / args.run).glob("*.npz"))
    starts = {t["id"]: t for s in ("dev", "pilot", "smoke") for t in trials.load_trials(s)}
    result = {"run": args.run, "trials": [p.stem for p in paths], "settings": []}
    for limits, gains in settings_from(args):
        rows = {}
        for p in paths:
            track = target_track(np.load(p))
            log = follow.simulate(track, follow.start_state(starts[p.stem]["drone_start"]), limits, gains, DT)
            rows[p.stem] = metrics.follow_quality(log, track, limits, DT)
        s = summary(list(rows.values()))
        result["settings"].append({"speed_h": limits.speed_h, "accel_h": limits.accel_h, "lag_s": limits.lag_s, "heading_s": gains.heading_s,
                                   "kp": follow.default_kp(limits.lag_s), "summary": s, "per_trial": rows})
        print(f"speed {limits.speed_h:.0f} accel {limits.accel_h:.0f} lag {limits.lag_s:.2f} heading {gains.heading_s:.1f}: dist mean {s['distance_error_m_mean']:.2f} "
              f"p95 worst {s['distance_error_m_p95_worst']:.1f} max {s['distance_error_m_max']:.1f}, over 3 m "
              f"{s['trials_over_3m']}, range mean {s['range_error_m_mean']:.2f}, bearing mean "
              f"{s['bearing_error_deg_mean']:.1f}, aim p95 worst {s['aim_error_deg_p95_worst']:.1f}, in view min "
              f"{s['target_in_view_min']:.3f}, jerk rms {s['jerk_rms_mean']:.2f} max {s['jerk_max']:.0f}, "
              f"accel limited {s['accel_limited_mean']:.3f}, speed max {s['drone_speed_max']:.1f} "
              f"(target {s['target_speed_max']:.1f})", flush=True)
    out = OUT / f"offline_{args.run}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


def cmd_live(args):
    import carla
    import cv2
    ticks = int(trials.DURATION_S * HZ)
    limits = drone.Limits(lag_s=args.lag[0], speed_h=args.speed[0], accel_h=args.accel[0])
    gains = follow.Gains(heading_s=args.heading[0])
    run = args.run or f"live_v{args.speed[0]:g}_a{args.accel[0]:g}_lag{args.lag[0]:g}_h{args.heading[0]:g}"
    report = []
    raster = screen.MapRaster.cached(XODR, OUT / "map", trials.MAP) if args.video else None
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, tm):
        for trial in select(args.split, args.ids):
            actors = start_trial(client, world, tm, trial)
            rec = Recorder(actors, ticks)
            target = actors["target"]
            state = follow.start_state(trial["drone_start"])
            follower = follow.Follower(limits, gains)
            cams, writer, scr = None, None, None
            if args.video:
                pose = carla.Transform(carla.Location(*state.position), carla.Rotation(pitch=state.pitch, yaw=state.yaw))
                cams = {"first": sim.camera(world, "sensor.camera.rgb", *screen.FIRST, trials.CAMERA["fov"], pose),
                        "chase": sim.camera(world, "sensor.camera.rgb", *screen.SIDE, trials.CAMERA["fov"], pose)}
                (OUT / "video").mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(OUT / "video" / f"{trial['id']}_{run}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), HZ, screen.SIZE)
                scr = screen.Screen(raster, trial["sentence"]["en"], trial["sentence"]["ko"], hz=HZ,
                                    fov=trials.CAMERA["fov"])
            aim = np.array(trial["target"]["transform"][:3]) + [0.0, 0.0, 0.8]
            log = {k: np.zeros((ticks, 3)) for k in ("position", "velocity", "acceleration", "goal", "command")}
            log["yaw"], log["pitch"] = np.zeros(ticks), np.zeros(ticks)
            w = time.monotonic()
            try:
                for k in range(ticks):
                    if cams:
                        cams["first"][0].set_transform(carla.Transform(
                            carla.Location(*state.position), carla.Rotation(pitch=state.pitch, yaw=state.yaw)))
                        eye, (cyaw, cpitch) = screen.chase_pose(state.position, state.yaw, aim)
                        cams["chase"][0].set_transform(carla.Transform(carla.Location(*eye),
                                                                       carla.Rotation(pitch=cpitch, yaw=cyaw)))
                    world.tick()
                    frame = rec.take(world, k)
                    tpos, tyaw, aim = rec.pos[k, 0].astype(float), float(rec.rot[k, 0, 1]), rec.aim[k].copy()
                    cmd, goal = follower.command(state, tpos, tyaw, aim, DT)
                    for key, val in (("position", state.position), ("velocity", state.velocity),
                                     ("acceleration", state.acceleration), ("goal", goal),
                                     ("command", cmd.velocity)):
                        log[key][k] = val
                    log["yaw"][k], log["pitch"][k] = state.yaw, state.pitch
                    if writer is not None:
                        err = float(np.linalg.norm(state.position - follow.goal_point(tpos, tyaw)))
                        status = (f"t {k * DT:4.1f} s    config 1 (ground truth)    drone {np.linalg.norm(state.velocity[:2]):4.1f} m/s"
                                  f"    goal error {err:4.1f} m")
                        cars = [(rec.pos[k, i], rec.rot[k, i], rec.box_center[i], rec.box_extent[i], str(rec.kind[i]))
                                for i in range(len(rec.ids))]
                        image = scr.frame(sim.bgr(sim.frame_at(cams["first"][1], frame)),
                                          sim.bgr(sim.frame_at(cams["chase"][1], frame)),
                                          screen.Camera(eye, cyaw, cpitch, *screen.SIDE, trials.CAMERA["fov"]),
                                          state.position, state.yaw, state.pitch, cars, 0, f"GT #{target.id}",
                                          status, goal)
                        writer.write(image)
                    state = drone.step(state, cmd, limits, DT)
            finally:
                for cam, _ in (cams or {}).values():
                    cam.stop()
                    cam.destroy()
                if writer is not None:
                    writer.release()
            wall = time.monotonic() - w
            path = OUT / "traffic" / run / f"{trial['id']}.npz"
            rec.save(path, trial)
            log["t"] = np.arange(ticks) * DT
            track = target_track(np.load(path))
            q = metrics.follow_quality(log, track, limits, DT)
            row = {"trial": trial["id"], "ticks_per_s": round(ticks / wall, 1), "gpu_mib": sim.gpu_memory_mib(),
                   "quality": q}
            if args.compare:
                old = np.load(OUT / "traffic" / args.compare / f"{trial['id']}.npz")
                same_ids = old["pos"].shape == rec.pos.shape
                row["replay_target_max_diff_m"] = round(float(np.abs(old["pos"][:, 0] - rec.pos[:, 0]).max()), 4)
                row["replay_all_max_diff_m"] = round(float(np.abs(old["pos"] - rec.pos).max()), 4) if same_ids else None
                off = follow.simulate(target_track(old), follow.start_state(trial["drone_start"]), limits, gains, DT)
                row["offline_vs_live_drone_max_diff_m"] = round(float(np.abs(off["position"] - log["position"]).max()), 4)
            report.append(row)
            print(json.dumps(row), flush=True)
    out = OUT / f"live_{run}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("record", "offline", "live"))
    p.add_argument("--split", default="dev", choices=("dev", "pilot", "smoke"))
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--run", default=None, help="name of the traffic recording (record, offline) or of the live run")
    p.add_argument("--compare", default=None, help="live: a recording to compare the replay with")
    p.add_argument("--lag", type=float, nargs="+", default=[0.5])
    p.add_argument("--heading", type=float, nargs="+", default=[follow.Gains().heading_s])
    p.add_argument("--speed", type=float, nargs="+", default=[drone.Limits().speed_h],
                   help="offline: horizontal speed limits of the drone to try, m/s")
    p.add_argument("--accel", type=float, nargs="+", default=[drone.Limits().accel_h],
                   help="offline: horizontal acceleration limits of the drone to try, m/s^2")
    p.add_argument("--video", action="store_true")
    p.add_argument("--no-start", action="store_true", help="use a simulator that is already running")
    args = p.parse_args()
    if args.mode in ("record", "offline") and not args.run:
        p.error("--run is needed")
    {"record": cmd_record, "offline": cmd_offline, "live": cmd_live}[args.mode](args)


if __name__ == "__main__":
    main()
