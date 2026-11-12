"""Make the demo 1 trial lists (docs/phase2_demo1.md 1): dev, pilot and smoke (open) and main (sealed).

Each trial is drawn, spawned in Town05 and looked at from the drone start before it is kept: the target and every
look-alike must spawn, the target must show at least 300 pixels, at least one look-alike must be in view, and the
sentence built for the trial must name the target among the matches in view. The main trials start in map cells of
their own (trials.cell_split). Pictures of the open trials' starts go to outputs/phase2/demo1_trials/; none are made of
the main trials. --catalog only draws every target blueprint in every clear colour, to check the colour names by eye.

Run from the repository root (starts and stops the simulator):
    uv run python scripts/make_demo1_trials.py
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.car_follow import sim, trials  # noqa: E402

OUT = ROOT / "outputs" / "phase2" / "demo1_trials"
SEED_BASE = {"dev": 100000, "pilot": 200000, "smoke": 300000, "main": 900000}
POOL = {"dev": "dev", "pilot": "gate", "smoke": "gate", "main": "main"}


def picture(image, trial, actors):
    world_to_camera = np.array(trials.as_transform(trial["drone_start"]).get_inverse_matrix())
    cam = trials.CAMERA
    marks = [(actors["target"], (60, 220, 60), "target")]
    marks += [(a, (0, 140, 255), f"look-alike {i + 1}") for i, a in enumerate(actors["lookalikes"])]
    for actor, colour, label in marks:
        vertices = [(v.x, v.y, v.z) for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
        box = sim.image_box(world_to_camera, vertices, cam["width"], cam["height"], cam["fov"])
        if box is None:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(image, label, (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    s = trial["sentence"]
    band = image[:56].copy()
    cv2.rectangle(band, (0, 0), (image.shape[1], 56), (0, 0, 0), -1)
    image[:56] = cv2.addWeighted(band, 0.6, image[:56], 0.4, 0)
    for row, text in enumerate([f"{trial['id']}  [{s['expression']}]", s["en"]]):
        cv2.putText(image, text, (10, 22 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def catalog(client, world, wp, types, colors, path, tile=(200, 150), columns=8):
    """One picture of every target blueprint in every clear colour as the drone sees it at the start, each labelled
    with the words a sentence would use, to check the colour families by eye."""
    cam = trials.CAMERA
    pose = trials.drone_start(wp)
    world_to_camera = np.array(pose.get_inverse_matrix())
    rgb, frames = sim.camera(world, "sensor.camera.rgb", cam["width"], cam["height"], cam["fov"], pose)
    tiles = []
    try:
        for blueprint in sorted(b for b, e in types.items() if e["target"]):
            for color in (c for c in colors[blueprint] if trials.clear_color(c)):
                bp = world.get_blueprint_library().find(blueprint)
                bp.set_attribute("color", color)
                car = world.spawn_actor(bp, trials.spawn_transform(wp))
                for _ in range(4):
                    frame = world.tick()
                image = sim.bgr(sim.frame_at(frames, frame))
                vertices = [(v.x, v.y, v.z) for v in car.bounding_box.get_world_vertices(car.get_transform())]
                x0, y0, x1, y1 = sim.image_box(world_to_camera, vertices, cam["width"], cam["height"], cam["fov"])
                cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.9
                crop = image[max(0, int(cy - half * 0.75)):int(cy + half * 0.75), max(0, int(cx - half)):int(cx + half)]
                crop = cv2.resize(crop, tile)
                label = f"{trials.color_family(color)} {types[blueprint]['type']}"
                for text, row in ((label, 14), (f"{blueprint.split('.', 1)[1]} {color}", tile[1] - 6)):
                    cv2.putText(crop, text, (4, row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(crop, text, (4, row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(crop)
                car.destroy()
    finally:
        rgb.stop()
        rgb.destroy()
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len([t for t in tiles if t.any()])


def make_split(client, world, split, count, pools, types, colors, spawn_points, used, max_attempts):
    kept, rejected, tried = [], collections.Counter(), 0
    world_map = world.get_map()
    for attempt in range(max_attempts):
        if len(kept) == count:
            break
        tried += 1
        seed = SEED_BASE[split] + attempt
        rng = random.Random(seed)
        intent = trials.INTENTS[len(kept) % len(trials.INTENTS)]
        trial = trials.plan(rng, intent, pools[POOL[split]], types, colors, spawn_points)
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
            pixels, image = trials.start_view(world, trial, actors, rgb=split != "main")
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
            trial = {"id": f"{split}-{len(kept) + 1:03d}", "split": split, "seed": seed, **trial,
                     "sentence": trials.sentence(intent, types[t["blueprint"]], t["color"], t["index_from_left"],
                                                 t["lanes"]),
                     "check": {**check, "expressions_possible": options}}
            used.add(start)
            kept.append(trial)
            if image is not None:
                (OUT / split).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(OUT / split / f"{trial['id']}.jpg"), picture(image, trial, actors),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        finally:
            sim.clear(client, world)
    if len(kept) < count:
        raise RuntimeError(f"{split}: only {len(kept)} of {count} trials after {max_attempts} attempts: {dict(rejected)}")
    return kept, {"attempts": tried, "rejected": dict(rejected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=int, default=20)
    parser.add_argument("--pilot", type=int, default=3)
    parser.add_argument("--smoke", type=int, default=10)
    parser.add_argument("--main", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=400, help="per split")
    parser.add_argument("--no-start", action="store_true", help="use a simulator that is already running")
    parser.add_argument("--catalog", action="store_true",
                        help="only draw every target blueprint in every clear colour (outputs/phase2/demo1_trials/)")
    args = parser.parse_args()

    types = trials.load_types()
    with sim.session(trials.MAP, start=not args.no_start) as (client, world, _tm):
        library = world.get_blueprint_library()
        colors = {b: (list(library.find(b).get_attribute("color").recommended_values)
                      if library.find(b).has_attribute("color") else []) for b in types}
        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()
        pools = collections.defaultdict(lambda: collections.defaultdict(list))
        for wp in trials.candidate_waypoints(world_map):
            cell = trials.cell_of(wp.transform.location.x, wp.transform.location.y)
            pools[trials.cell_split(cell)][cell].append(wp)
        notes = {"candidates": {k: sum(map(len, v.values())) for k, v in pools.items()},
                 "candidate_cells": {k: len(v) for k, v in pools.items()},
                 "candidate_roads": {k: len({wp.road_id for c in v.values() for wp in c}) for k, v in pools.items()}}
        print("candidate start waypoints:", notes["candidates"], "cells:", notes["candidate_cells"],
              "roads:", notes["candidate_roads"])
        if args.catalog:
            first = pools["dev"][sorted(pools["dev"])[0]][0]
            n = catalog(client, world, first, types, colors, OUT / "catalog.jpg")
            print(f"catalog: {n} cars -> {(OUT / 'catalog.jpg').relative_to(ROOT)}")
            return

        used, splits, stats = set(), {}, {}
        for split in ("main", "pilot", "smoke", "dev"):
            kept, stats[split] = make_split(client, world, split, getattr(args, split), pools, types, colors,
                                            spawn_points, used, args.max_attempts)
            splits[split] = kept
            print(split, len(kept), stats[split])
        notes["generation"] = stats
        main_trials = splits.pop("main")
        trials.write_trials(splits, main_trials, notes=notes)

    print(f"wrote {trials.TRIALS_PATH.relative_to(ROOT)} and {trials.MAIN_PATH.relative_to(ROOT)} (sealed)")
    print(json.dumps({s: collections.Counter(t["sentence"]["expression"] for t in v) for s, v in splits.items()}))


if __name__ == "__main__":
    main()
