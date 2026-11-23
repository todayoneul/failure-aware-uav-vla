"""Demo 1 step 4, before the scorer (docs/demo1.md 4.1): does replaying recorded traffic (physics off, every car
put at its recorded pose each tick in one batch) give normal RGB images, instance segmentation actor IDs and 3D
boxes, and how fast is it? One trial, in one simulator session:

  A  speed without cameras: 1,200 replayed ticks, batch applied with and without waiting for the reply
  B  poses without cameras: after every tick, every car's pose against the recording (both batch modes)
  C  with the demo cameras (first-person RGB and instance segmentation, chase RGB) and the drone flying the follow
     loop over the recorded target: target pixels, its mask against the projected 3D box (ours and CARLA's), the
     screen video and a few still images
  D  for the speed comparison: the same trial driven live by the Traffic Manager, same cameras and processing

Run from the car_follow folder:
    uv run python scripts/demo1_replay_check.py --trial dev-013 --run rec1
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
from scripts.demo1_follow import DT, HZ, OUT as FOLLOW_OUT, XODR, Recorder, start_trial, target_track  # noqa: E402
from src.car_follow import drone, follow, replay, screen, sim, trials  # noqa: E402

OUT = ROOT / "outputs" / "demo1_replay_check"
TICKS = int(trials.DURATION_S * HZ)
STILLS = (100, 500, 840)


def pose_error(world, ids, rec, k):
    snap = world.get_snapshot()
    worst_pos, worst_rot = 0.0, 0.0
    for i, aid in enumerate(ids):
        t = snap.find(aid).get_transform()
        p = np.array([t.location.x, t.location.y, t.location.z])
        r = np.array([t.rotation.pitch, t.rotation.yaw, t.rotation.roll])
        worst_pos = max(worst_pos, float(np.abs(p - rec["pos"][k, i]).max()))
        worst_rot = max(worst_rot, float(np.abs((r - rec["rot"][k, i] + 180.0) % 360.0 - 180.0).max()))
    return worst_pos, worst_rot


def pass_speed(world, rp, wait):
    w = time.monotonic()
    for k in range(TICKS):
        rp.apply(k, wait=wait)
        world.tick()
    return round(TICKS / (time.monotonic() - w), 1)


def pass_poses(world, rp, rec, wait):
    worst = [0.0, 0.0]
    for k in range(TICKS):
        rp.apply(k, wait=wait)
        world.tick()
        p, r = pose_error(world, rp.ids, rec, k)
        worst = [max(worst[0], p), max(worst[1], r)]
    return {"position_m": worst[0], "rotation_deg": worst[1]}


def box_of(cam, rec, k, i):
    uv, depth = cam.project(screen.box_vertices(rec["pos"][k, i], rec["rot"][k, i], rec["box_center"][i],
                                                rec["box_extent"][i]))
    if (depth <= 0.1).any():
        return None
    return (*uv.min(0), *uv.max(0))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pass_cameras(client, world, trial, rec, rp, raster, tag):
    """Fly the follow loop with the demo cameras. `rp` is a Replay, or None when the Traffic Manager drives."""
    cam = trials.CAMERA
    track = target_track(rec)
    state = follow.start_state(trial["drone_start"])
    follower = follow.Follower(drone.Limits(), follow.Gains())
    if rp is None:  # start the live trial first: start_trial clears every sensor in the world
        actors = start_trial(client, world, client.get_trafficmanager(8000), trial)
        live = Recorder(actors, TICKS)
        ids, target = list(live.ids), actors["target"]
    else:
        ids, target = rp.ids, rp.actors[0]
    pose = carla.Transform(carla.Location(*state.position), carla.Rotation(pitch=state.pitch, yaw=state.yaw))
    cams = {"rgb": sim.camera(world, "sensor.camera.rgb", cam["width"], cam["height"], cam["fov"], pose),
            "seg": sim.camera(world, "sensor.camera.instance_segmentation", cam["width"], cam["height"], cam["fov"],
                              pose),
            "chase": sim.camera(world, "sensor.camera.rgb", *screen.SIDE, cam["fov"], pose)}
    look_idx = [i for i, k in enumerate(rec["kind"]) if k == "lookalike"]
    scr = screen.Screen(raster, trial["sentence"]["en"], trial["sentence"]["ko"], hz=HZ, fov=cam["fov"])
    writer = cv2.VideoWriter(str(OUT / f"{trial['id']}_{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), HZ, screen.SIZE)
    t_apply = t_tick = t_img = t_proc = 0.0
    rows, aim = [], track["aim"][0]
    try:
        w0 = time.monotonic()
        for k in range(TICKS):
            a = time.monotonic()
            if rp is not None:
                rp.apply(k, wait=True)
            b = time.monotonic()
            cams["rgb"][0].set_transform(carla.Transform(carla.Location(*state.position),
                                                         carla.Rotation(pitch=state.pitch, yaw=state.yaw)))
            cams["seg"][0].set_transform(carla.Transform(carla.Location(*state.position),
                                                         carla.Rotation(pitch=state.pitch, yaw=state.yaw)))
            eye, (cyaw, cpitch) = screen.chase_pose(state.position, state.yaw, aim)
            cams["chase"][0].set_transform(carla.Transform(carla.Location(*eye), carla.Rotation(pitch=cpitch, yaw=cyaw)))
            frame = world.tick()
            c = time.monotonic()
            rgb = sim.bgr(sim.frame_at(cams["rgb"][1], frame))
            seg = sim.frame_at(cams["seg"][1], frame)
            chase = sim.bgr(sim.frame_at(cams["chase"][1], frame))
            d = time.monotonic()
            if rp is None:
                live.take(world, k)
                src = {"pos": live.pos, "rot": live.rot, "box_center": live.box_center, "box_extent": live.box_extent}
            else:
                src = rec
            ids_img = sim.instance_ids(seg)
            first_cam = screen.Camera(state.position, state.yaw, state.pitch, cam["width"], cam["height"], cam["fov"])
            ours = box_of(first_cam, src, k, 0)
            w2c = np.array(carla.Transform(carla.Location(*state.position),
                                           carla.Rotation(pitch=state.pitch, yaw=state.yaw)).get_inverse_matrix())
            verts = [[v.x, v.y, v.z] for v in target.bounding_box.get_world_vertices(target.get_transform())]
            theirs = sim.image_box(w2c, verts, cam["width"], cam["height"], cam["fov"])
            mask = ids_img == target.id
            n = int(mask.sum())
            row = {"k": k, "target_px": n,
                   "lookalike_px": [int((ids_img == ids[i]).sum()) for i in look_idx],
                   "box_diff_px": (float(np.abs(np.array(ours) - np.array(theirs)).max())
                                   if ours is not None and theirs is not None else None)}
            if n and ours is not None:
                ys, xs = np.nonzero(mask)
                mbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                inside = (xs >= ours[0] - 1) & (xs <= ours[2] + 1) & (ys >= ours[1] - 1) & (ys <= ours[3] + 1)
                row.update(mask_inside_box=float(inside.mean()), mask_box_iou=iou(mbox, ours))
            rows.append(row)
            tpos, tyaw = src["pos"][k, 0].astype(float), float(src["rot"][k, 0, 1])
            aim = screen.box_vertices(tpos, src["rot"][k, 0], src["box_center"][0], (0, 0, 0))[0]
            cmd, goal = follower.command(state, tpos, tyaw, aim, DT)
            cars = [(src["pos"][k, i], src["rot"][k, i], src["box_center"][i], src["box_extent"][i], str(rec["kind"][i]))
                    for i in range(len(ids))]
            frame_img = scr.frame(rgb.copy(), chase, screen.Camera(eye, cyaw, cpitch, *screen.SIDE, cam["fov"]),
                                  state.position, state.yaw, state.pitch, cars, 0, f"GT #{target.id}",
                                  f"t {k * DT:4.1f} s    replay check ({tag})", goal)
            writer.write(frame_img)
            if k in STILLS:
                vis = rgb.copy()
                if ours is not None:
                    cv2.rectangle(vis, tuple(int(v) for v in ours[:2]), tuple(int(v) for v in ours[2:]), (0, 255, 0), 1)
                if theirs is not None:
                    cv2.rectangle(vis, tuple(int(v) for v in theirs[:2]), tuple(int(v) for v in theirs[2:]),
                                  (255, 0, 255), 1)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
                ids_vis = ids_img.astype(np.uint32)
                colors = np.stack([(ids_vis * 37) % 255, (ids_vis * 91) % 255, (ids_vis * 151) % 255], -1)
                colors[ids_vis == 0] = 0
                cv2.imwrite(str(OUT / f"{trial['id']}_{tag}_k{k}_rgb.png"), vis)
                cv2.imwrite(str(OUT / f"{trial['id']}_{tag}_k{k}_seg.png"), colors.astype(np.uint8))
            state = drone.step(state, cmd, drone.Limits(), DT)
            e = time.monotonic()
            t_apply += b - a
            t_tick += c - b
            t_img += d - c
            t_proc += e - d
        wall = time.monotonic() - w0
    finally:
        for s, _ in cams.values():
            s.stop()
            s.destroy()
        writer.release()
    vis = [r for r in rows if r["target_px"] >= trials.VISIBLE_MIN_PIXELS]
    diffs = [r["box_diff_px"] for r in rows if r["box_diff_px"] is not None]
    out = {"ticks_per_s": round(TICKS / wall, 1),
           "seconds_per_tick": {"apply": round(t_apply / TICKS, 4), "tick": round(t_tick / TICKS, 4),
                                "images": round(t_img / TICKS, 4), "processing": round(t_proc / TICKS, 4)},
           "gpu_mib": sim.gpu_memory_mib(),
           "target_visible_ticks": len(vis), "target_px_median": int(np.median([r["target_px"] for r in vis])) if vis else 0,
           "lookalike_visible_ticks": [sum(r["lookalike_px"][j] >= trials.VISIBLE_MIN_PIXELS for r in rows)
                                       for j in range(len(look_idx))],
           "box_ours_vs_carla_px_max": round(max(diffs), 3) if diffs else None,
           "mask_inside_box_min": round(min(r["mask_inside_box"] for r in vis if "mask_inside_box" in r), 4),
           "mask_inside_box_mean": round(float(np.mean([r["mask_inside_box"] for r in vis if "mask_inside_box" in r])), 4),
           "mask_box_iou_median": round(float(np.median([r["mask_box_iou"] for r in vis if "mask_box_iou" in r])), 3)}
    if rp is None:
        sim.clear(client, world)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial", default="dev-013")
    p.add_argument("--run", default="rec1", help="the recording to replay (outputs/demo1_follow/traffic/<run>)")
    p.add_argument("--no-start", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    trial = next(t for s in ("dev", "pilot", "smoke") for t in trials.load_trials(s) if t["id"] == args.trial)
    rec = dict(np.load(FOLLOW_OUT / "traffic" / args.run / f"{args.trial}.npz"))
    raster = screen.MapRaster.cached(XODR, FOLLOW_OUT / "map", trials.MAP)
    result = {"trial": args.trial, "recording": args.run, "cars": int(len(rec["ids"]))}
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, tm):
        sim.clear(client, world)
        world.tick()
        world.reset_all_traffic_lights()
        rp = replay.Replay(client, world, rec, trial)
        world.tick()
        result["A_ticks_per_s_no_cameras"] = {"batch_wait": pass_speed(world, rp, True),
                                              "batch_no_wait": pass_speed(world, rp, False)}
        print(json.dumps(result), flush=True)
        result["B_pose_error_max"] = {"batch_wait": pass_poses(world, rp, rec, True),
                                      "batch_no_wait": pass_poses(world, rp, rec, False)}
        print(json.dumps(result["B_pose_error_max"]), flush=True)
        result["C_replay_with_cameras"] = pass_cameras(client, world, trial, rec, rp, raster, "replay")
        print(json.dumps(result["C_replay_with_cameras"]), flush=True)
        rp.destroy()
        world.tick()
        world.reset_all_traffic_lights()
        result["D_live_traffic_manager_with_cameras"] = pass_cameras(client, world, trial, rec, None, raster, "live")
        print(json.dumps(result["D_live_traffic_manager_with_cameras"]), flush=True)
    (OUT / f"{args.trial}_{args.run}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {(OUT / f'{args.trial}_{args.run}.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
