"""Demo 1 step 6 (docs/demo1.md 7): start frames for the QLoRA training of the VLM first pick.

Same generator and checks as the VLM dev scenes (scripts/demo1_vlm_scenes.py), in the dev map cells with new seeds,
and with the drone's start pose jittered a little. About a quarter of the dev cells are kept out of training
(vlm_data.unseen_cell), and starts within HOLDOUT_M of a start of the dev or VLM dev scenes are not used; a training
start may repeat up to MAX_PER_START times with different cars. Pilot, smoke and main are not touched.

Each scene goes to outputs/demo1_vlm/train/scenes/<id>.png (drone RGB), <id>.npz (instance IDs) and <id>.json (the
look-alike group with lanes and positions, every car's actor ID).

Run from the car_follow folder (starts and stops the simulator; resumes where it stopped):
    uv run python scripts/demo1_vlm_train_scenes.py --count 800
"""
import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_follow import HZ  # noqa: E402
from scripts.demo1_vlm_scenes import VLM_DEV_PATH, capture  # noqa: E402
from src.car_follow import sim, trials, vlm_data  # noqa: E402

OUT = ROOT / "outputs" / "demo1_vlm" / "train" / "scenes"
SEED_BASE = 600000
HOLDOUT_M = 3.0
MAX_PER_START = 4
JITTER = {"xy": 1.5, "z": 1.0, "pitch": 2.0, "yaw": 4.0}


def held_out_starts():
    starts = [t["target"]["transform"][:2] for split in ("dev", "pilot", "smoke") for t in trials.load_trials(split)]
    starts += [s["target"]["transform"][:2] for s in json.loads(VLM_DEV_PATH.read_text(encoding="utf-8"))["scenes"]]
    return np.asarray(starts, dtype=float)


def jittered(rng, drone_start):
    x, y, z, pitch, yaw, roll = drone_start
    return carla.Transform(carla.Location(x + rng.uniform(-JITTER["xy"], JITTER["xy"]),
                                          y + rng.uniform(-JITTER["xy"], JITTER["xy"]),
                                          z + rng.uniform(-JITTER["z"], JITTER["z"])),
                           carla.Rotation(pitch=pitch + rng.uniform(-JITTER["pitch"], JITTER["pitch"]),
                                          yaw=yaw + rng.uniform(-JITTER["yaw"], JITTER["yaw"]), roll=roll))


def group_record(world_map, trial, actors, pixels):
    """The look-alike group (target first) with lane and position along the target's road, as trials.describe sees it."""
    target = actors["target"]
    t = target.get_transform()
    forward = t.get_forward_vector()
    road, lane = trial["target"]["road"], trial["target"]["lane"]
    cars = []
    for role, actor in [("target", target)] + [("lookalike", a) for a in actors["lookalikes"]]:
        wp = world_map.get_waypoint(actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        index, lanes = trials.lane_layout(wp)
        d = actor.get_location() - t.location
        same = wp.road_id == road and wp.lane_id * lane > 0
        cars.append({"id": actor.id, "role": role, "pixels": int(pixels.get(actor.id, 0)), "same_road": same,
                     "index_from_left": index if same else -1, "lanes": lanes if same else -1,
                     "along_m": round(d.x * forward.x + d.y * forward.y, 3)})
    return cars


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=800)
    p.add_argument("--max-attempts", type=int, default=6000)
    p.add_argument("--no-start", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    old = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("*.json"))]
    per_start = collections.Counter(tuple(m["start"]) for m in old)
    first = max((m["seed"] - SEED_BASE + 1 for m in old), default=0)
    kept = len(old)
    held = held_out_starts()
    rejected = collections.Counter()
    t0 = time.monotonic()
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, _tm):
        sim.clear(client, world)
        world.tick()
        types = trials.load_types()
        library = world.get_blueprint_library()
        colors = {b: (list(library.find(b).get_attribute("color").recommended_values)
                      if library.find(b).has_attribute("color") else []) for b in types}
        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()
        pool = collections.defaultdict(list)
        total, unseen, removed = 0, 0, 0
        for wp in trials.candidate_waypoints(world_map):
            cell = trials.cell_of(wp.transform.location.x, wp.transform.location.y)
            if trials.cell_split(cell) != "dev":
                continue
            total += 1
            if vlm_data.unseen_cell(cell):
                unseen += 1
                continue
            at = np.array([wp.transform.location.x, wp.transform.location.y])
            if np.min(np.hypot(*(held - at).T)) < HOLDOUT_M:
                removed += 1
                continue
            pool[cell].append(wp)
        print(f"dev-cell starts: {total}, in cells kept out: {unseen}, near open scenes: {removed}, usable: "
              f"{total - unseen - removed} in {len(pool)} cells", flush=True)
        for attempt in range(first, args.max_attempts):
            if kept >= args.count:
                break
            seed = SEED_BASE + attempt
            rng = random.Random(seed)
            intent = trials.INTENTS[attempt % len(trials.INTENTS)]
            trial = trials.plan(rng, intent, pool, types, colors, spawn_points)
            start = tuple(round(v, 1) for v in trial["target"]["transform"][:2])
            if per_start[start] >= MAX_PER_START:
                rejected["start used enough"] += 1
                continue
            actors = trials.spawn(world, trial)
            try:
                if actors["target"] is None or None in actors["lookalikes"]:
                    rejected["target or look-alike did not spawn"] += 1
                    continue
                bgr, ids = capture(world, jittered(rng, trial["drone_start"]))
                found, counts = np.unique(ids, return_counts=True)
                pixels = dict(zip(found.tolist(), counts.tolist()))
                cars = group_record(world_map, trial, actors, pixels)
                names = vlm_data.nameable(cars)
                if not names:
                    rejected["no car can be named alone"] += 1
                    continue
                kept += 1
                scene_id = f"vlmtrain-{kept:04d}"
                t = trial["target"]
                everyone = [actors["target"]] + actors["lookalikes"] + [a for a in actors["background"]
                                                                        if a is not None]
                meta = {"id": scene_id, "source": "vlm_train", "seed": seed, "start": list(start),
                        "blueprint": t["blueprint"], "family": trials.color_family(t["color"]), "cars": cars,
                        "car_ids": [a.id for a in everyone]}
                cv2.imwrite(str(OUT / f"{scene_id}.png"), bgr)
                np.savez_compressed(OUT / f"{scene_id}.npz", ids=ids.astype(np.int32))
                (OUT / f"{scene_id}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
                per_start[start] += 1
                if kept % 25 == 0:
                    rate = (kept - len(old)) / max(time.monotonic() - t0, 1e-3)
                    print(f"{kept}/{args.count} scenes, {rate * 60:.1f}/min, seed {seed}, rejected {dict(rejected)}",
                          flush=True)
            finally:
                sim.clear(client, world)
    print(f"kept {kept}; distinct starts {len(per_start)}; rejected {dict(rejected)}; "
          f"{(time.monotonic() - t0) / 60:.1f} min")
    if kept < args.count:
        print(f"only {kept} of {args.count} scenes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
