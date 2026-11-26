"""Make the stage 3 failure trials (docs/failure_plan.md 4, src/car_follow/failure_trials.py). For now the
generator pilot of docs/failure_plan.md 7, step 3: in dev cells, draw trials of each condition, check the start
as demo 1 does, record the traffic with the event script running, and check from the recording alone whether the
event happened. A condition passes when the event happened in 70% or more of its recorded trials.

Run from the car_follow folder (starts and stops the simulator):
    uv run python scripts/make_failure_trials.py pilot --conditions A1 A2 A3 --count 10
    uv run python scripts/make_failure_trials.py pilot --conditions A4a A4b A4c --list dev --name a4_dev
Outputs go to outputs/failure_trials/<name>/ (local only): pilot.json, trials/<id>.json and traffic/<id>.npz.
--list dev names the trials dev-<condition>-NN with split dev and draws them from their own seeds (A4 only for now:
the lane-change condition added on 2026-11-26 is checked on its own dev trials, apart from batch A's dev list).
--tight gives A1 and A2 trials the tight event values (failure_trials.TIGHT: a closer cut-in, a closer queue). Every
row with a distractor also reports how much of the target's image box (nominal follow pose) the distractor covered.
"""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_follow import DT, HZ, Recorder  # noqa: E402
from src.car_follow import failure_trials, labels, sim, trials  # noqa: E402

OUT = ROOT / "outputs" / "failure_trials"
PILOT_SEED = 1_050_000       # the pilot's own seeds (the dev list starts at 1,100,000, docs/failure_plan.md 4.3)
# A4 (the target changes lanes) seeds below the pilot's: generator 1,010,000-1,039,999, its dev trials 1,040,000+
A4_SEED = {"generator": {"A4a": 1_010_000, "A4b": 1_020_000, "A4c": 1_030_000},
           "dev": {"A4a": 1_040_000, "A4b": 1_043_000, "A4c": 1_046_000}}
PASS_YIELD = 0.70


def seed_for(condition, listing, offset, attempt):
    if condition in A4_SEED[listing]:
        return A4_SEED[listing][condition] + offset + attempt
    if listing != "generator":
        raise ValueError(f"no {listing} seeds for {condition}: batch A's lists are made in step 5")
    return PILOT_SEED + 10_000 * failure_trials.CONDITIONS.index(condition) + offset + attempt


def level_of(condition, kept):
    if condition in ("A1", "A2"):
        return failure_trials.LEVELS[kept % 3]
    return "near_twin" if condition == "A4c" else "twin"


def draw(client, world, world_map, rng, condition, level, cells, types, colors, spawn_points):
    """(trial, None) for a trial whose start passes demo 1's checks, or (None, reason)."""
    trial = failure_trials.plan(rng, condition, level, world_map, cells, types, colors, spawn_points)
    if trial is None:
        return None, "no distractor or no slot for it"
    t = trial["target"]
    wp = world_map.get_waypoint(trials.as_transform(t["transform"]).location)
    if condition == "A1" or condition.startswith("A4"):
        free_m = failure_trials.A1_FREE_M if condition == "A1" else failure_trials.A4_FREE_M[condition]
        free = failure_trials.junction_free(failure_trials.straight_route(wp, free_m + 20.0))
        if free < free_m:
            return None, "a junction too close on the straight-on route"
        trial["event"]["junction_free_m"] = free
    if condition == "A2":
        light, dist = failure_trials.light_ahead(world, wp, failure_trials.A2_LIGHT_M[1])
        if light is None or dist < failure_trials.A2_LIGHT_M[0]:
            return None, "no signalled junction at the right distance"
        trial["event"]["light_m"] = round(dist, 1)
    if condition == "A3":
        under = failure_trials.underpass_ahead(world, wp)
        if under is None or not (failure_trials.UNDERPASS_FROM_M <= under[0] <= failure_trials.UNDERPASS_TO_M) \
                or under[1] < failure_trials.UNDERPASS_MIN_M:
            return None, "no underpass at the right distance on the straight-on route"
        trial["event"]["underpass_m"] = list(under[:2])
        trial["event"]["underpass_xy"] = list(under[2:])
    actors = failure_trials.spawn(world, trial)
    try:
        if actors["target"] is None or None in actors["lookalikes"]:
            return None, "target or look-alike did not spawn"
        if trial.get("distractor_car") and actors["distractor"] is None:
            return None, "distractor did not spawn"
        trial["background"] = [b for b, a in zip(trial["background"], actors["background"]) if a is not None]
        pixels, _ = trials.start_view(world, trial, actors)
        matches = failure_trials.word_match_actors(trial, actors, types)
        options, check = trials.describe(world_map, trial, {**actors, "lookalikes": matches}, pixels)
        if check["target_pixels"] < trials.TARGET_MIN_PIXELS:
            return None, "target hidden at the start"
        if check["visible_lookalikes"] == 0:
            return None, "no look-alike in view"
        if trial["intent"] not in options:
            return None, f"'{trial['intent']}' does not name the target alone"
        trial["sentence"] = trials.sentence(trial["intent"], types[t["blueprint"]], t["color"], t["index_from_left"],
                                            t["lanes"])
        trial["check"] = {**check, "expressions_possible": options}
        return trial, None
    finally:
        sim.clear(client, world)



def record(client, world, tm, trial, path):
    """Record the trial's traffic with its event script; returns (event log, check result, occluded per tick)."""
    ticks = int(trials.DURATION_S * HZ)
    sim.clear(client, world)
    world.tick()
    world.reset_all_traffic_lights()
    tm.set_random_device_seed(int(trial["seed"]))
    actors = failure_trials.spawn(world, trial, tm)
    try:
        if actors["target"] is None or None in actors["lookalikes"] or (
                trial.get("distractor_car") and actors["distractor"] is None):
            return {"state": "not recorded: a car did not spawn"}, {"happened": False, "reason": "spawn"}, None
        d = [actors["distractor"]] if actors["distractor"] is not None else []
        rec = Recorder({"target": actors["target"], "lookalikes": actors["lookalikes"],
                        "background": d + actors["background"]}, ticks)
        if d:   # the recorder knows target, look-alikes and background; the distractor comes first in the latter
            rec.kind[1 + len(actors["lookalikes"])] = "distractor"
        event = failure_trials.script(trial, DT)
        event.setup(world, tm, actors)
        probe = failure_trials.OcclusionProbe(ticks)
        # The map lanes of the word-match cars (target, look-alikes, distractor), read from the client's map copy
        lanes = failure_trials.LaneProbe(ticks, list(enumerate([actors["target"]] + actors["lookalikes"] + d)))
        world_map = world.get_map()
        for k in range(ticks):
            event.step(k, world, tm, actors)
            world.tick()
            rec.take(world, k)
            probe.step(k, world, actors["target"])
            lanes.step(k, world_map)
        rec.save(path, trial)
        saved = dict(np.load(path))
        np.savez_compressed(path, **saved, **lanes.arrays())
        occluded = probe.per_tick()
        result = failure_trials.check(trial, dict(np.load(path)), occluded, DT)
        return event.log, result, occluded
    finally:
        world.freeze_all_traffic_lights(False)
        sim.clear(client, world)


def cover(path, window=failure_trials.WINDOW_S):
    """How much of the target's image box (nominal follow pose) the distractor covered from the window's start on:
    the largest share and the seconds over 10% (None without a distractor)."""
    if not path.exists():
        return None
    rec = dict(np.load(path))
    found = np.flatnonzero(rec["kind"] == "distractor")
    if not len(found):
        return None
    share = labels.box_cover(rec, int(found[0]))
    share[:int(window[0] / DT)] = 0.0
    return {"max": round(float(share.max()), 3), "over_10pct_s": round(float((share > 0.1).sum() * DT), 2)}


def runs_s(mask):
    """[(start s, length s)] of the True runs, or None."""
    if mask is None:
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(round(a * DT, 2), round((b - a) * DT, 2)) for a, b in zip(edges[::2], edges[1::2])]


def travelled(path):
    """How far the target had driven by 15 s and by 45 s (m)."""
    if not path.exists():
        return None
    p = np.load(path)["pos"][:, 0, :2].astype(float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return [round(float(d[min(len(d) - 1, int(s / DT))]), 1) for s in (15.0, 45.0)]


def cmd_pilot(args):
    types = trials.load_types()
    out = OUT / args.name
    (out / "traffic").mkdir(parents=True, exist_ok=True)
    rows, rejected = [], collections.defaultdict(collections.Counter)
    t0 = time.monotonic()
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, tm):
        library = world.get_blueprint_library()
        colors = {b: (list(library.find(b).get_attribute("color").recommended_values)
                      if library.find(b).has_attribute("color") else []) for b in types}
        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()
        cells = collections.defaultdict(list)
        for wp in trials.candidate_waypoints(world_map):
            cell = trials.cell_of(wp.transform.location.x, wp.transform.location.y)
            if trials.cell_split(cell) == "dev":
                cells[cell].append(wp)
        for condition in args.conditions:
            kept = 0
            for attempt in range(args.max_draws):
                if kept == args.count:
                    break
                seed = seed_for(condition, args.list, args.seed_offset, attempt)
                level = level_of(condition, kept)
                trial, why = draw(client, world, world_map, random.Random(seed), condition, level, cells, types, colors,
                                  spawn_points)
                if trial is None:
                    rejected[condition][why] += 1
                    continue
                kept += 1
                if args.tight and condition in failure_trials.TIGHT:
                    trial["event"].update(failure_trials.TIGHT[condition])
                if args.cut_behind and condition == "A1":
                    trial["event"].update(failure_trials.CUT_BEHIND)
                # 'gen-': the generator's own trial run, not the gate list called pilot (docs/failure_plan.md 4.4)
                prefix = "gen" if args.list == "generator" else args.list
                trial = {"id": f"{prefix}-{condition}-{kept:02d}", "split": args.list, "seed": seed, **trial}
                w = time.monotonic()
                log, result, occluded = record(client, world, tm, trial, out / "traffic" / f"{trial['id']}.npz")
                row = {"id": trial["id"], "condition": condition, "level": level, "seed": seed,
                       "intent": trial["intent"], "target": trial["target"]["blueprint"],
                       "distractor_car": trial.get("distractor_car"), "event": trial["event"], "script": log,
                       "check": result, "occluded_s": None if occluded is None else round(float(occluded.sum() * DT), 2),
                       "occluded_runs": runs_s(occluded), "travelled_m": travelled(out / "traffic" / f"{trial['id']}.npz"),
                       "seconds": round(time.monotonic() - w, 1)}
                row["cover"] = cover(out / "traffic" / f"{trial['id']}.npz")
                rows.append(row)
                (out / "trials").mkdir(exist_ok=True)
                (out / "trials" / f"{trial['id']}.json").write_text(json.dumps(trial, indent=1, ensure_ascii=False),
                                                                     encoding="utf-8")
                print(f"{trial['id']} [{level}] script {log.get('state')}, blocked {log.get('blocked_ticks')}, event {result}, "
                      f"{row['seconds']} s", flush=True)
    summary = {}
    for condition in args.conditions:
        rs = [r for r in rows if r["condition"] == condition]
        ok = sum(bool(r["check"].get("happened")) for r in rs)
        by_level = {lv: f"{sum(bool(r['check'].get('happened')) for r in rs if r['level'] == lv)}/"
                        f"{sum(r['level'] == lv for r in rs)}" for lv in sorted({r['level'] for r in rs})}
        summary[condition] = {"recorded": len(rs), "happened": ok, "yield": round(ok / len(rs), 3) if rs else None,
                              "passes": bool(rs) and ok / len(rs) >= PASS_YIELD, "by_level": by_level,
                              "draws_rejected": dict(rejected[condition])}
    result = {"list": args.list, "tight": args.tight, "cut_behind": args.cut_behind, "seed_base": {c: seed_for(c, args.list, 0, 0) for c in args.conditions},
              "seed_offset": args.seed_offset, "pass_yield": PASS_YIELD, "window_s": list(failure_trials.WINDOW_S),
              "summary": summary, "trials": rows, "elapsed_s": round(time.monotonic() - t0, 1)}
    (out / "pilot.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {(out / 'pilot.json').relative_to(ROOT)}, {result['elapsed_s']} s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pilot", help="the generator pilot in dev cells")
    s.add_argument("--conditions", nargs="+", default=["A1", "A2", "A3"], choices=failure_trials.CONDITIONS)
    s.add_argument("--count", type=int, default=10, help="recorded trials per condition")
    s.add_argument("--max-draws", type=int, default=3000, help="per condition (draws that fail a screen cost no recording)")
    s.add_argument("--seed-offset", type=int, default=0,
                   help="draw from other seeds (the tuning used offset 0; the confirming run fresh ones)")
    s.add_argument("--name", default="pilot", help="output folder under outputs/failure_trials")
    s.add_argument("--tight", action="store_true",
                   help="A1 and A2 with the tight event values (failure_trials.TIGHT, proposed after step 4)")
    s.add_argument("--cut-behind", action="store_true",
                   help="A1 cuts in behind the target, between it and the drone (failure_trials.CUT_BEHIND)")
    s.add_argument("--list", default="generator", choices=("generator", "dev"),
                   help="dev: the trials are dev trials (ids dev-..., split dev, own seeds; A4 only)")
    s.add_argument("--no-start", action="store_true", help="use a simulator that is already running")
    args = p.parse_args()
    {"pilot": cmd_pilot}[args.cmd](args)


if __name__ == "__main__":
    main()
