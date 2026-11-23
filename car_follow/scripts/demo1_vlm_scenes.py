"""Demo 1 step 6 (docs/demo1.md 6.2): start frames for the VLM pick.

    --new N   draw N new start scenes in the dev map cells with new seeds (same generator and checks as the trial
              lists, scripts/make_demo1_trials.py; starts of the open trials are not reused). They are for choosing
              the VLM and tuning the prompt; the definitions go to configs/demo1_vlm_dev.json.
    --dev     the start frame of each dev trial as config 3 will see it: the replayed traffic at tick 0 after the
              camera warm-up, from the drone start. These are kept for the 90% check only.

Each scene is saved to outputs/demo1_vlm/scenes/<id>.png (drone RGB), <id>.npz (instance IDs) and <id>.json (the
sentence, the target's and the look-alikes' actor IDs, every car's actor ID). Pilot, smoke and main are not touched.

Run from the car_follow folder (starts and stops the simulator):
    uv run python scripts/demo1_vlm_scenes.py --new 100
    uv run python scripts/demo1_vlm_scenes.py --dev
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_follow import HZ, OUT as FOLLOW_OUT, select  # noqa: E402
from src.car_follow import follow, replay, sim, trials  # noqa: E402

OUT = ROOT / "outputs" / "demo1_vlm" / "scenes"
VLM_DEV_PATH = ROOT / "configs" / "demo1_vlm_dev.json"
SEED_BASE = 500000
CAM = trials.CAMERA


def capture(world, pose, ticks=5):
    """(BGR image, instance-ID image) from `pose` after `ticks` ticks (new cameras need a couple to settle)."""
    rgb = sim.camera(world, "sensor.camera.rgb", CAM["width"], CAM["height"], CAM["fov"], pose)
    seg = sim.camera(world, "sensor.camera.instance_segmentation", CAM["width"], CAM["height"], CAM["fov"], pose)
    try:
        for _ in range(ticks):
            frame = world.tick()
        return sim.bgr(sim.frame_at(rgb[1], frame)), sim.instance_ids(sim.frame_at(seg[1], frame))
    finally:
        for s, _ in (rgb, seg):
            s.stop()
            s.destroy()


def save(scene_id, bgr, ids, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{scene_id}.png"), bgr)
    np.savez_compressed(OUT / f"{scene_id}.npz", ids=ids.astype(np.int32))
    (OUT / f"{scene_id}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def new_scenes(client, world, count, max_attempts):
    types = trials.load_types()
    library = world.get_blueprint_library()
    colors = {b: (list(library.find(b).get_attribute("color").recommended_values)
                  if library.find(b).has_attribute("color") else []) for b in types}
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    pools = collections.defaultdict(lambda: collections.defaultdict(list))
    for wp in trials.candidate_waypoints(world_map):
        cell = trials.cell_of(wp.transform.location.x, wp.transform.location.y)
        pools[trials.cell_split(cell)][cell].append(wp)
    used = {tuple(round(v) for v in t["target"]["transform"][:2])
            for split in ("dev", "pilot", "smoke") for t in trials.load_trials(split)}
    kept, rejected = [], collections.Counter()
    for attempt in range(max_attempts):
        if len(kept) == count:
            break
        seed = SEED_BASE + attempt
        rng = random.Random(seed)
        intent = trials.INTENTS[len(kept) % len(trials.INTENTS)]
        trial = trials.plan(rng, intent, pools["dev"], types, colors, spawn_points)
        start = tuple(round(v) for v in trial["target"]["transform"][:2])
        if start in used:
            rejected["start already used"] += 1
            continue
        actors = trials.spawn(world, trial)
        try:
            if actors["target"] is None or None in actors["lookalikes"]:
                rejected["target or look-alike did not spawn"] += 1
                continue
            trial["background"] = [b for b, a in zip(trial["background"], actors["background"]) if a is not None]
            bgr, ids = capture(world, trials.as_transform(trial["drone_start"]))
            found, counts = np.unique(ids, return_counts=True)
            pixels = dict(zip(found.tolist(), counts.tolist()))
            options, check = trials.describe(world_map, trial, actors, pixels)
            if check["target_pixels"] < trials.TARGET_MIN_PIXELS:
                rejected["target hidden at the start"] += 1
                continue
            if check["visible_lookalikes"] == 0:
                rejected["no look-alike in view"] += 1
                continue
            if intent not in options:
                rejected[f"'{intent}' does not name the target alone"] += 1
                continue
            t = trial["target"]
            scene_id = f"vlmdev-{len(kept) + 1:03d}"
            sentence = trials.sentence(intent, types[t["blueprint"]], t["color"], t["index_from_left"], t["lanes"])
            trial = {"id": scene_id, "split": "vlm_dev", "seed": seed, **trial, "sentence": sentence,
                     "check": {**check, "expressions_possible": options}}
            cars = [actors["target"]] + actors["lookalikes"] + [a for a in actors["background"] if a is not None]
            save(scene_id, bgr, ids, {"id": scene_id, "source": "vlm_dev", "sentence": sentence,
                                      "target_id": actors["target"].id,
                                      "lookalike_ids": [a.id for a in actors["lookalikes"]],
                                      "car_ids": [a.id for a in cars]})
            used.add(start)
            kept.append(trial)
            print(f"{scene_id}: seed {seed}, {sentence['expression']}, {sentence['en']}", flush=True)
        finally:
            sim.clear(client, world)
    VLM_DEV_PATH.write_text(json.dumps({"_comment": "VLM dev scenes (docs/demo1.md 6.2): start scenes in the dev "
                                        "cells with new seeds, for choosing the VLM and tuning the prompt. Not "
                                        "flown; only their start frames are used.", "rejected": dict(rejected),
                                        "scenes": kept}, indent=1, ensure_ascii=False), encoding="utf-8")
    return kept, rejected


def dev_scenes(client, world, run, ids):
    out = []
    for trial in select("dev", ids):
        rec = dict(np.load(FOLLOW_OUT / "traffic" / run / f"{trial['id']}.npz"))
        world.reset_all_traffic_lights()
        rp = replay.Replay(client, world, rec, trial)
        try:
            for _ in range(2):            # camera warm-up as in scripts/demo1_eval.py
                rp.apply(0, wait=True)
                world.tick()
            rp.apply(0, wait=True)
            s = follow.start_state(trial["drone_start"])
            pose = carla.Transform(carla.Location(*s.position), carla.Rotation(pitch=s.pitch, yaw=s.yaw))
            bgr, ids_img = capture(world, pose, ticks=3)
        finally:
            rp.destroy()
            world.tick()
        kinds = [str(k) for k in rec["kind"]]
        save(trial["id"], bgr, ids_img, {"id": trial["id"], "source": "dev", "sentence": trial["sentence"],
                                         "target_id": int(rp.ids[0]),
                                         "lookalike_ids": [int(a) for a, k in zip(rp.ids, kinds) if k == "lookalike"],
                                         "car_ids": [int(a) for a in rp.ids]})
        out.append(trial["id"])
        print(f"{trial['id']}: {trial['sentence']['en']}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", type=int, default=0)
    p.add_argument("--dev", action="store_true")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--run", default="rec1")
    p.add_argument("--max-attempts", type=int, default=600)
    p.add_argument("--no-start", action="store_true")
    args = p.parse_args()
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, _tm):
        sim.clear(client, world)
        world.tick()
        if args.new:
            kept, rejected = new_scenes(client, world, args.new, args.max_attempts)
            print(f"kept {len(kept)}; rejected {dict(rejected)}; wrote {VLM_DEV_PATH.relative_to(ROOT)}")
        if args.dev:
            print(f"dev start frames: {len(dev_scenes(client, world, args.run, args.ids))}")


if __name__ == "__main__":
    main()
