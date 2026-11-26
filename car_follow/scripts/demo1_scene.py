"""Demo 1: draw a scored flight again from its evidence (docs/demo1.md 9.3): the drone's logged pose and the tracker's
box over the same replayed traffic, on the three-view screen with the config 3 overlay (src/car_follow/pick_view.py).

Nothing is flown or decided again, so the picture is the scored flight itself. A re-flight would not repeat it exactly
(the detector and the VLM vary slightly between runs on the GPU). As a check, the car under the logged box is measured
again on the new instance image and compared with the scored owner of every tick.

Writes outputs/demo1_eval/<result>_scenes/<trial>_<lang>.mp4 (H.264) and, for --stills, one PNG per time and a JPG
sheet of them.

A result that flew a failure trial run (demo1_eval.py --failure-run) takes its trials and recordings from
outputs/failure_trials/<name> (no --split or --run).

Run from the car_follow folder:
    uv run python scripts/demo1_scene.py --result config3_main_main_a --split main --open-sealed --run main_a \\
        --scenes main-024:ko --stills main-024:ko:46,49,51.5,54
    uv run python scripts/demo1_scene.py --result config3_a4_dev_guard-on_rfdetr-small_qlora1 --scenes dev-A4c-01:en
"""
import argparse
import json
import sys
import time
from pathlib import Path

import av
import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_follow import DT, HZ, OUT as FOLLOW_OUT, XODR, select, target_track  # noqa: E402
from scripts.failure_labels import sources  # noqa: E402
from src.car_follow import estimate, follow, pick_view, replay, scorer, screen, sim, tracker, trials  # noqa: E402

EVAL = ROOT / "outputs" / "demo1_eval"
WARMUP_TICKS = 2
TAG = "본평가 비행 기록을 다시 그림"


def climb_text(z, k):
    if z["offset"][k] <= scorer.ACTIVE_M:
        return ""
    need_los, need_path, off = z["need_los"][k], z["need_path"][k], z["offset"][k]
    why = "hold" if max(need_los, need_path) < off - 0.25 else ("LOS" if need_los >= need_path else "path")
    return f"    climb +{off:.1f} m ({why})"


def render(client, world, raster, trial, lang, rec, z, row, result, video_path, stills):
    """One scored flight drawn again. Returns (ticks whose re-measured owner matches the scored one, ticks with a box),
    and the still images {time: image}."""
    cam = trials.CAMERA
    rp = replay.Replay(client, world, rec, trial)
    track = target_track(rec)
    pick = row.get("pick")
    view = pick_view.PickView(result["ambiguity_threshold"], result["pick_delay_s"], DT) if pick else None
    scr = screen.Screen(raster, trial["sentence"]["en"], trial["sentence"]["ko"], hz=HZ, fov=cam["fov"], given=lang)
    pos, yaw, pitch = z["log_position"], z["log_yaw"], z["log_pitch"]

    def pose(k):
        return carla.Transform(carla.Location(*pos[k]), carla.Rotation(pitch=float(pitch[k]), yaw=float(yaw[k])))

    cams = {"rgb": sim.camera(world, "sensor.camera.rgb", cam["width"], cam["height"], cam["fov"], pose(0)),
            "seg": sim.camera(world, "sensor.camera.instance_segmentation", cam["width"], cam["height"], cam["fov"],
                              pose(0)),
            "chase": sim.camera(world, "sensor.camera.rgb", *screen.SIDE, cam["fov"], pose(0))}
    out = av.open(str(video_path), "w")
    stream = out.add_stream("libx264", rate=int(HZ))
    stream.width, stream.height, stream.pix_fmt = screen.SIZE[0], screen.SIZE[1], "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}
    want = {int(round(t / DT)): t for t in stills}
    shots, same, boxed = {}, 0, 0
    aim = track["aim"][0]
    try:
        for _ in range(WARMUP_TICKS):
            rp.apply(0, wait=True)
            world.tick()
        for k in range(len(pos)):
            rp.apply(k, wait=True)
            cams["rgb"][0].set_transform(pose(k))
            cams["seg"][0].set_transform(pose(k))
            eye, (cyaw, cpitch) = screen.chase_pose(pos[k], yaw[k], aim)   # as in the flight: last tick's aim
            cams["chase"][0].set_transform(carla.Transform(carla.Location(*eye), carla.Rotation(pitch=cpitch, yaw=cyaw)))
            frame = world.tick()
            ids_img = sim.instance_ids(sim.frame_at(cams["seg"][1], frame))
            bgr = sim.bgr(sim.frame_at(cams["rgb"][1], frame))
            chase = sim.bgr(sim.frame_at(cams["chase"][1], frame))
            if np.isfinite(z["est_pos"][k]).all():
                aim = z["est_pos"][k] + np.array([0.0, 0.0, estimate.AIM_ABOVE_M])
            box = tuple(z["box"][k]) if np.isfinite(z["box"][k]).all() else None
            if box is not None:
                boxed += 1
                same += int(scorer.box_owner(ids_img, box, rp.ids)[0] == z["owner"][k])
            tid = int(z["track_id"][k])
            owner = int(z["owner"][k])
            err = float(np.linalg.norm(pos[k] - follow.goal_point(track["position"][k], track["yaw"][k])
                                       - [0.0, 0.0, z["offset"][k]]))
            status = (f"t {k * DT:4.1f} s    config 3    box on "
                      f"{'target' if owner == 0 else ('none' if owner < 0 else rec['kind'][owner])}"
                      f"    goal error {err:4.1f} m{climb_text(z, k)}")
            cars = [(rec["pos"][k, i], rec["rot"][k, i], rec["box_center"][i], rec["box_extent"][i],
                     str(rec["kind"][i])) for i in range(len(rp.ids))]
            img = scr.frame(bgr, chase, screen.Camera(eye, cyaw, cpitch, *screen.SIDE, cam["fov"]), pos[k], yaw[k],
                            pitch[k], cars, 0, f"track #{tid}" if tid >= 0 else "lost", status, z["log_goal"][k],
                            track_box=box, distance=float(np.linalg.norm(aim - pos[k])))
            first = img[:, :screen.FIRST[0]]
            if view is not None:   # the candidates carried on ground-truth boxes, for display only
                dets = [tracker.Detection(b, 1.0, None) for b in (scorer.mask_box(ids_img, a) for a in rp.ids)
                        if b is not None]
                # the saved record holds the final lock tick; before it, the flight did not have one yet
                now = pick if pick["lock_tick"] is None or k >= pick["lock_tick"] else dict(pick, lock_tick=None)
                view.carry(now, dets)
                view.draw(first, k, now, tid >= 0, pick_view.climb(float(z["offset"][k]), scorer.ACTIVE_M))
            tw = int(view.text.font(13).getlength(TAG)) if view is not None else 200
            (view.text if view is not None else screen.Text()).draw(
                first, [(TAG, (screen.FIRST[0] - tw - 10, 6), 13, (190, 230, 255), False)])
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")):
                out.mux(packet)
            if k in want:
                shots[want[k]] = img.copy()
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
        for s, _ in cams.values():
            s.stop()
            s.destroy()
        rp.destroy()
        world.tick()
    return same, boxed, shots


def sheet(images, width=960):
    rows = [cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1]))), interpolation=cv2.INTER_AREA)
            for im in images]
    return np.concatenate(rows, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", required=True, help="the evaluation's name (outputs/demo1_eval/<result>.json)")
    p.add_argument("--split", default="dev", choices=("dev", "pilot", "smoke", "main"))
    p.add_argument("--open-sealed", action="store_true", help="needed with --split main")
    p.add_argument("--run", default=None, help="the traffic recording the evaluation replayed (demo 1 splits)")
    p.add_argument("--scenes", nargs="+", required=True, help="trial:lang, e.g. main-024:ko")
    p.add_argument("--stills", nargs="*", default=[], help="trial:lang:t1,t2,... seconds to save as pictures")
    p.add_argument("--no-start", action="store_true")
    args = p.parse_args()
    global TAG
    result = json.loads((EVAL / f"{args.result}.json").read_text(encoding="utf-8"))
    if "failure_run" in result:
        if set(result["split"].split("+")) & {"main", "calibration", "pilot", "smoke"} and not args.open_sealed:
            p.error(f"{args.result} flew sealed trials: --open-sealed is needed")
        by_id, traffic = sources(result, args.open_sealed)
        TAG = f"{result['split']} 비행 기록을 다시 그림"
    else:
        if args.split == "main" and not args.open_sealed:
            p.error("--split main needs --open-sealed")
        if args.run is None:
            p.error("--run is needed for a demo 1 split")
        by_id, traffic = None, FOLLOW_OUT / "traffic" / args.run
        if args.split != "main":
            TAG = f"{args.split} 비행 기록을 다시 그림"
    stills = {}
    for s in args.stills:
        tid, lang, times = s.split(":")
        stills[(tid, lang)] = [float(t) for t in times.split(",")]
    out = EVAL / f"{args.result}_scenes"
    out.mkdir(parents=True, exist_ok=True)
    raster = screen.MapRaster.cached(XODR, FOLLOW_OUT / "map", trials.MAP)
    scenes = [tuple(s.split(":")) for s in args.scenes]
    found = by_id if by_id is not None else {
        t["id"]: t for t in select(args.split, sorted({t for t, _ in scenes}), args.open_sealed)}
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, _tm):
        sim.clear(client, world)
        world.tick()
        for tid, lang in scenes:
            w = time.monotonic()
            rec = dict(np.load(traffic / f"{tid}.npz"))
            z = np.load(EVAL / args.result / f"{tid}_{lang}.npz")
            row = result["languages"][lang]["trials"][tid]
            world.reset_all_traffic_lights()
            same, boxed, shots = render(client, world, raster, found[tid], lang, rec, z, row, result,
                                        out / f"{tid}_{lang}.mp4", stills.get((tid, lang), []))
            for t, im in sorted(shots.items()):
                cv2.imwrite(str(out / f"{tid}_{lang}_{t:05.1f}s.png"), im)
            if shots:
                cv2.imwrite(str(out / f"{tid}_{lang}_sheet.jpg"), sheet([im for _, im in sorted(shots.items())]),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"{tid}_{lang}: owner re-measured on {boxed} boxed ticks, same as scored on {same} "
                  f"({same / max(boxed, 1):.2%}), {len(shots)} stills, {time.monotonic() - w:.0f} s", flush=True)


if __name__ == "__main__":
    main()
