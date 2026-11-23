"""Demo 1 evaluation on replayed traffic (docs/demo1.md 4): fly a configuration over the recorded traffic of each
trial (src/car_follow/replay.py), keep per-tick evidence, and score it (src/car_follow/scorer.py).

    config 1  the tracking box is the target's ground-truth 2D box (around its visible instance pixels) and the drone
              follows the target's true position (docs/demo1_plan.md 3.4). Configs 2 and 3 come with steps 5 and 6.

The altitude guard (src/car_follow/clearance.py, docs/demo1.md 4.6) reads a 160 x 120 depth camera on the drone:
    --guard on       it raises the follow goal (default)
    --guard observe  it runs and is recorded, but the drone ignores it (for comparison)
    --guard off      no depth camera

Per trial the evidence (owner of the box, target pixels, guard, drone log) goes to outputs/demo1_eval/<name>/<trial>.npz
and the scores to outputs/demo1_eval/<name>.json (local only).

Run from the car_follow folder:
    uv run python scripts/demo1_eval.py --config 1 --split dev --run rec1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_follow import DT, HZ, OUT as FOLLOW_OUT, XODR, select, target_track  # noqa: E402
from src.car_follow import clearance, drone, estimate, follow, replay, scorer, screen, sim, trials  # noqa: E402

OUT = ROOT / "outputs" / "demo1_eval"


def fly_trial(client, world, trial, rec, config, guard_mode="on", raster=None, video=None, measure=False):
    """One trial on replayed traffic. Returns the per-tick evidence, the drone log and the wall time."""
    ticks = len(rec["pos"])
    cam = trials.CAMERA
    rp = replay.Replay(client, world, rec, trial)
    target_id = rp.ids[0]
    track = target_track(rec)
    limits, gains = drone.Limits(), follow.Gains()
    state = follow.start_state(trial["drone_start"])
    follower = follow.Follower(limits, gains)
    dcam = clearance.DepthCamera()
    guard = clearance.AltitudeGuard(dcam) if guard_mode != "off" else None

    def pose():
        return carla.Transform(carla.Location(*state.position), carla.Rotation(pitch=state.pitch, yaw=state.yaw))

    def depth_pose():
        return carla.Transform(carla.Location(*state.position), carla.Rotation(pitch=dcam.mount_pitch, yaw=state.yaw))

    cams = {"seg": sim.camera(world, "sensor.camera.instance_segmentation", cam["width"], cam["height"], cam["fov"],
                              pose())}
    if guard is not None:
        cams["depth"] = sim.camera(world, "sensor.camera.depth", dcam.width, dcam.height, dcam.fov, depth_pose())
    writer = scr = None
    if video:
        cams["rgb"] = sim.camera(world, "sensor.camera.rgb", cam["width"], cam["height"], cam["fov"], pose())
        cams["chase"] = sim.camera(world, "sensor.camera.rgb", *screen.SIDE, cam["fov"], pose())
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), HZ, screen.SIZE)
        scr = screen.Screen(raster, trial["sentence"]["en"], trial["sentence"]["ko"], hz=HZ, fov=cam["fov"])
    ev = {"owner": np.full(ticks, -1), "share": np.zeros(ticks), "target_px": np.zeros(ticks, int),
          "box": np.full((ticks, 4), np.nan), "perceive_s": np.zeros(ticks), "hidden_by": np.full(ticks, -1),
          "offset": np.zeros(ticks), "need_path": np.zeros(ticks), "need_los": np.zeros(ticks),
          "los_state": np.full(ticks, "unseen", dtype="<U11"), "nearest": np.full(ticks, np.inf),
          "guard_s": np.zeros(ticks), "depth_at_target": np.full(ticks, np.nan),
          "target_depth": np.full(ticks, np.nan)}
    if measure:  # raw position measurements of the box (src/car_follow/estimate.py), for scripts/demo1_estimate.py
        for key in ("m_flat", "m_depth", "ray_b", "ray_c"):
            ev[key] = np.full((ticks, 3), np.nan)
    log = {k: np.zeros((ticks, 3)) for k in ("position", "velocity", "acceleration", "goal", "command")}
    log["yaw"], log["pitch"] = np.zeros(ticks), np.zeros(ticks)
    aim = track["aim"][0]
    w0 = time.monotonic()
    try:
        for k in range(ticks):
            rp.apply(k, wait=True)
            for name in ("seg", "rgb"):
                if name in cams:
                    cams[name][0].set_transform(pose())
            if "depth" in cams:
                cams["depth"][0].set_transform(depth_pose())
            if "chase" in cams:
                eye, (cyaw, cpitch) = screen.chase_pose(state.position, state.yaw, aim)
                cams["chase"][0].set_transform(carla.Transform(carla.Location(*eye),
                                                               carla.Rotation(pitch=cpitch, yaw=cyaw)))
            frame = world.tick()
            seg_img = sim.frame_at(cams["seg"][1], frame)
            ids_img = sim.instance_ids(seg_img)
            p0 = time.monotonic()
            if config == 1:
                box = scorer.mask_box(ids_img, target_id)
                tpos, tyaw, aim = track["position"][k], track["yaw"][k], track["aim"][k]
            else:
                raise NotImplementedError("configs 2 and 3 come with implementation steps 5 and 6")
            ev["perceive_s"][k] = time.monotonic() - p0
            offset = 0.0
            if guard is not None:
                depth = sim.depth_m(sim.frame_at(cams["depth"][1], frame))
                g0 = time.monotonic()
                d = guard.update(depth, state.position, state.yaw, state.velocity, tpos[2] + follow.ABOVE_M, aim, DT)
                ev["guard_s"][k] = time.monotonic() - g0
                ev["offset"][k], ev["need_path"][k], ev["need_los"][k] = d.offset, d.path, d.los
                ev["los_state"][k], ev["nearest"][k] = d.los_state, d.nearest_m
                px, z = clearance.project(aim, dcam, state.position, state.yaw)
                if px is not None and 0 <= px[0] < dcam.width and 0 <= px[1] < dcam.height:
                    ev["depth_at_target"][k], ev["target_depth"][k] = depth[int(px[1]), int(px[0])], z
                if guard_mode == "on":
                    offset = d.offset
                if measure and box is not None:
                    m = estimate.measure(box, state.position, state.yaw, state.pitch, cam, depth, dcam)
                    ev["ray_b"][k], ev["ray_c"][k] = m["ray_bottom"], m["ray_centre"]
                    for key in ("flat", "depth"):
                        if m[key] is not None:
                            ev[f"m_{key}"][k] = m[key]
            ev["owner"][k], ev["share"][k] = scorer.box_owner(ids_img, box, rp.ids)
            ev["target_px"][k] = int((ids_img == target_id).sum())
            if ev["target_px"][k] < scorer.VISIBLE_PX:  # what covers the target's projected box while it is hidden
                cam_now = screen.Camera(state.position, state.yaw, state.pitch, cam["width"], cam["height"], cam["fov"])
                uv, depth_v = cam_now.project(screen.box_vertices(rec["pos"][k, 0], rec["rot"][k, 0],
                                                                  rec["box_center"][0], rec["box_extent"][0]))
                if (depth_v > 0.1).all():
                    ev["hidden_by"][k] = scorer.occluder(sim.semantic_tags(seg_img), (*uv.min(0), *uv.max(0)))
            if box is not None:
                ev["box"][k] = box
            cmd, goal = follower.command(state, tpos, tyaw, aim, DT, offset=offset)
            for key, val in (("position", state.position), ("velocity", state.velocity),
                             ("acceleration", state.acceleration), ("goal", goal), ("command", cmd.velocity)):
                log[key][k] = val
            log["yaw"][k], log["pitch"][k] = state.yaw, state.pitch
            if writer is not None:
                cars = [(rec["pos"][k, i], rec["rot"][k, i], rec["box_center"][i], rec["box_extent"][i],
                         str(rec["kind"][i])) for i in range(len(rp.ids))]
                err = float(np.linalg.norm(state.position - follow.goal_point(tpos, tyaw) - [0.0, 0.0, offset]))
                owner = ev["owner"][k]
                climb = ""
                if guard is not None and ev["offset"][k] > scorer.ACTIVE_M:
                    need = max(ev["need_los"][k], ev["need_path"][k])
                    why = "hold" if need < ev["offset"][k] - 0.25 else (
                        "LOS" if ev["need_los"][k] >= ev["need_path"][k] else "path")
                    climb = f"    climb +{ev['offset'][k]:.1f} m ({why})"
                status = (f"t {k * DT:4.1f} s    config {config}    box on "
                          f"{'target' if owner == 0 else ('none' if owner < 0 else rec['kind'][owner])}"
                          f"    goal error {err:4.1f} m{climb}")
                writer.write(scr.frame(sim.bgr(sim.frame_at(cams["rgb"][1], frame)),
                                       sim.bgr(sim.frame_at(cams["chase"][1], frame)),
                                       screen.Camera(eye, cyaw, cpitch, *screen.SIDE, cam["fov"]), state.position,
                                       state.yaw, state.pitch, cars, 0, f"GT #{target_id}", status, goal))
            state = drone.step(state, cmd, limits, DT)
        wall = time.monotonic() - w0
    finally:
        for s, _ in cams.values():
            s.stop()
            s.destroy()
        if writer is not None:
            writer.release()
        rp.destroy()
        world.tick()
    log["t"] = np.arange(ticks) * DT
    return ev, log, wall


def errors(log, track, offset):
    """Per tick: distance to the goal the drone was asked to hold (true goal raised by the guard's offset), horizontal
    distance to the true goal, and the height over the true nominal goal."""
    goal = np.array([follow.goal_point(p, y) for p, y in zip(track["position"], track["yaw"])])
    d = log["position"] - goal
    raised = d - np.c_[np.zeros((len(d), 2)), offset]
    return np.linalg.norm(raised, axis=1), np.linalg.norm(d[:, :2], axis=1), d[:, 2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=int, default=1, choices=(1,))
    p.add_argument("--split", default="dev", choices=("dev", "pilot", "smoke"))
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--run", default="rec1", help="the traffic recording to replay (outputs/demo1_follow/traffic/<run>)")
    p.add_argument("--guard", default="on", choices=("on", "observe", "off"))
    p.add_argument("--name", default=None, help="output name (default: config<N>_<split>_<run>_guard-<mode>)")
    p.add_argument("--video", nargs="*", default=None, help="trial ids to write the screen video for")
    p.add_argument("--measure", action="store_true", help="also keep the raw position measurements of the box")
    p.add_argument("--no-start", action="store_true")
    args = p.parse_args()
    name = args.name or f"config{args.config}_{args.split}_{args.run}_guard-{args.guard}" + (
        "_measure" if args.measure else "")
    if args.measure and args.guard == "off":
        p.error("--measure needs the depth camera (--guard on or observe)")
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    raster = screen.MapRaster.cached(XODR, FOLLOW_OUT / "map", trials.MAP) if args.video else None
    rows = {}
    t0 = time.monotonic()
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, _tm):
        sim.clear(client, world)
        world.tick()
        for trial in select(args.split, args.ids):
            rec = dict(np.load(FOLLOW_OUT / "traffic" / args.run / f"{trial['id']}.npz"))
            world.reset_all_traffic_lights()
            video = out / f"{trial['id']}.mp4" if args.video is not None and trial["id"] in args.video else None
            ev, log, wall = fly_trial(client, world, trial, rec, args.config, args.guard, raster, video,
                                      args.measure)
            track = target_track(rec)
            applied = ev["offset"] if args.guard == "on" else np.zeros(len(ev["offset"]))  # observe: not flown
            dist, horiz, rise = errors(log, track, applied)
            row = scorer.score_trial(ev["owner"], ev["target_px"], rec["kind"], rec["pos"], dist, horiz,
                                     first_pick=None, dt=DT)
            if args.guard != "off":
                row["avoidance"] = scorer.avoidance(ev["offset"], ev["need_path"], ev["need_los"], ev["los_state"], rise,
                                                    ev["nearest"], DT)
                row["guard_ms_mean"] = round(1000 * float(ev["guard_s"].mean()), 2)
            hidden = ev["hidden_by"][ev["hidden_by"] >= 0]
            tags, counts = np.unique(hidden, return_counts=True)
            row.update(hidden_by={scorer.TAGS.get(int(t), str(t)): int(c) for t, c in zip(tags, counts)},
                       ticks_per_s=round(len(ev["owner"]) / wall, 1), gpu_mib=sim.gpu_memory_mib(),
                       perceive_ms_mean=round(1000 * float(ev["perceive_s"].mean()), 2))
            np.savez_compressed(out / f"{trial['id']}.npz", **ev, **{f"log_{k}": v for k, v in log.items()})
            rows[trial["id"]] = row
            a = row.get("avoidance") or {}
            after = row["after"]
            print(f"{trial['id']}: switches {len(row['switches'])}, hold {row['hold']}, distance error "
                  f"{row['distance_error_m']}, horizontal {row['horizontal_error_m']}, main {row['main_s']} s, "
                  f"long occlusions {[o['length_s'] for o in row['long_occlusions']]}"
                  + (f", after: reacquired {after['reacquired']}" if after else "")
                  + (f", climb {a.get('time_s')} s / {a.get('events')} events / max +{a.get('offset_max_m')} m "
                     f"(path {a.get('path_need_s')} s, LOS {a.get('los_need_s')} s, unreachable "
                     f"{a.get('los_unreachable_s')} s), nearest {a.get('nearest_min_m')} m" if a else "")
                  + f", {row['ticks_per_s']} ticks/s, {row['gpu_mib']} MiB", flush=True)
    summary = scorer.summarize(list(rows.values()))
    result = {"config": args.config, "split": args.split, "recording": args.run, "guard": args.guard,
              "summary": summary, "trials": rows}
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {(OUT / f'{name}.json').relative_to(ROOT)}, elapsed {time.monotonic() - t0:.0f} s")


if __name__ == "__main__":
    main()
