"""Make the stage 3 failure trials (docs/failure_plan.md 4, src/car_follow/failure_trials.py). For now the
generator pilot of docs/failure_plan.md 7, step 3: in dev cells, draw trials of each condition, check the start
as demo 1 does, record the traffic with the event script running, and check from the recording alone whether the
event happened. A condition passes when the event happened in 70% or more of its recorded trials.

Step 5 (docs/failure_plan.md 4.3-4.4): `lists` makes the lists of failure_trials.LISTS. dev and aux come from the
dev cells (seeds from 1,100,000), calibration, pilot, smoke and main from the gate and main cells pooled (seeds from
1,900,000), each trial's list fixed by its seed's sha256 before it is drawn. A trial is kept only when its event
happened. Sealed lists print counts and reasons only; their trial files and traffic manifests get a sha256 in
configs/failure_trials.json.

Run from the car_follow folder (starts and stops the simulator):
    uv run python scripts/make_failure_trials.py lists --lists dev aux
    uv run python scripts/make_failure_trials.py lists --lists pooled
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
import hashlib
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


LIST_SEED = {"open": 1_100_000, "pooled": 1_900_000}   # docs/failure_plan.md 4.3
SEED_STRIDE = 20_000                                       # per condition: CONDITIONS.index(condition) * SEED_STRIDE
PREFIX = {"dev": "fdev", "aux": "faux", "calibration": "fcal", "pilot": "fpilot", "smoke": "fsmoke", "main": "fmain"}
INDEX = ROOT / "configs" / "failure_trials.json"
LISTS_OUT = OUT / "lists"


def pooled_list(seed, condition):
    """The pooled list a seed belongs to for `condition`: its sha256 against the lists' shares of the condition."""
    need = [(name, failure_trials.LISTS[name].get(condition, 0)) for name in failure_trials.SEALED_LISTS]
    total = sum(n for _, n in need)
    u = int(hashlib.sha256(f"failure-list:{seed}".encode()).hexdigest(), 16) / float(16 ** 64)
    acc = 0.0
    for name, n in need:
        acc += n / total
        if u < acc:
            return name
    return need[-1][0]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_list(name, trials_kept, rows, stats):
    """configs/failure_trials_<name>.json, the run folder's pilot.json and traffic manifest, and the index entry."""
    folder = LISTS_OUT / name
    trials_kept = sorted(trials_kept, key=lambda t: t["id"])
    body = (json.dumps({"list": name, "trials": trials_kept}, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    path = ROOT / "configs" / f"failure_trials_{name}.json"
    path.write_bytes(body)
    manifest = "".join(f"{t['id']} {sha256_bytes((folder / 'traffic' / (t['id'] + '.npz')).read_bytes())}\n"
                       for t in trials_kept)
    (folder / "traffic_manifest.txt").write_text(manifest, encoding="utf-8", newline="\n")
    (folder / "pilot.json").write_text(json.dumps({"list": name, "trials": rows, "stats": stats}, indent=1,
                                                  ensure_ascii=False), encoding="utf-8")
    index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {
        "map": trials.MAP, "notes": {"plan": "docs/failure_plan.md 4.3-4.4 (option 1, 2026-11-26)",
                                     "sizes": failure_trials.LISTS, "seeds": LIST_SEED, "seed_stride": SEED_STRIDE},
        "splits": {}}
    entry = {"file": path.name, "count": len(trials_kept), "sha256": sha256_bytes(body),
             "traffic_manifest_sha256": sha256_bytes(manifest.encode("utf-8")),
             "traffic_folder": str(folder.relative_to(ROOT)).replace("\\", "/"), "generation": stats}
    if name in failure_trials.SEALED_LISTS:
        index[name] = dict(entry, sealed=True)
    else:
        index["splits"][name] = trials_kept
        index[f"{name}_info"] = entry
    INDEX.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def cmd_lists(args):
    types = trials.load_types()
    groups = [["dev"], ["aux"]] if args.lists != ["pooled"] else [list(failure_trials.SEALED_LISTS)]
    groups = [g for g in groups if g == list(failure_trials.SEALED_LISTS) or g[0] in args.lists]
    t0 = time.monotonic()
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, tm):
        library = world.get_blueprint_library()
        colors = {b: (list(library.find(b).get_attribute("color").recommended_values)
                      if library.find(b).has_attribute("color") else []) for b in types}
        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()
        for group in groups:
            pooled = group == list(failure_trials.SEALED_LISTS)
            cells = collections.defaultdict(list)
            for wp in trials.candidate_waypoints(world_map):
                cell = trials.cell_of(wp.transform.location.x, wp.transform.location.y)
                if (trials.cell_split(cell) in ("gate", "main")) == pooled:
                    cells[cell].append(wp)
            kept = {name: [] for name in group}
            rows = {name: [] for name in group}
            stats = {name: {} for name in group}
            conditions = [c for c in failure_trials.CONDITIONS if any(failure_trials.LISTS[n].get(c) for n in group)]
            for condition in conditions:
                quota = {n: failure_trials.LISTS[n].get(condition, 0) for n in group}
                count = {n: 0 for n in group}
                st = {n: {"quota": quota[n], "attempts": 0, "recorded": 0, "kept": 0,
                          "rejected": collections.Counter()} for n in group if quota[n]}
                base = LIST_SEED["pooled" if pooled else "open"] + SEED_STRIDE * failure_trials.CONDITIONS.index(condition)
                for attempt in range(SEED_STRIDE):
                    if all(count[n] >= quota[n] for n in group):
                        break
                    seed = base + attempt
                    name = pooled_list(seed, condition) if pooled else group[0]
                    if count[name] >= quota[name]:
                        continue
                    st[name]["attempts"] += 1
                    level = level_of(condition, count[name])
                    trial, why = draw(client, world, world_map, random.Random(seed), condition, level, cells, types,
                                      colors, spawn_points)
                    if trial is None:
                        st[name]["rejected"][why] += 1
                        continue
                    tid = f"{PREFIX[name]}-{condition}-{count[name] + 1:03d}"
                    trial = {"id": tid, "split": name, "seed": seed, **trial}
                    traffic = LISTS_OUT / name / "traffic" / f"{tid}.npz"
                    log, result, occluded = record(client, world, tm, trial, traffic)
                    st[name]["recorded"] += 1
                    if result.get("happened") is False:
                        st[name]["rejected"]["event did not happen"] += 1
                        traffic.unlink(missing_ok=True)
                        continue
                    count[name] += 1
                    st[name]["kept"] += 1
                    kept[name].append(trial)
                    (LISTS_OUT / name / "trials").mkdir(parents=True, exist_ok=True)
                    (LISTS_OUT / name / "trials" / f"{tid}.json").write_text(
                        json.dumps(trial, indent=1, ensure_ascii=False), encoding="utf-8")
                    rows[name].append({"id": tid, "split": name, "condition": condition, "level": level, "seed": seed,
                                       "intent": trial["intent"], "event": trial["event"], "script": log,
                                       "check": result, "cover": cover(traffic)})
                    if name in failure_trials.OPEN_LISTS:
                        print(f"{tid} [{level}, {trial['intent']}] event {result.get('happened')}", flush=True)
                for n in st:
                    st[n]["rejected"] = dict(st[n]["rejected"])
                    stats[n][condition] = st[n]
                    # sealed lists: counts and reasons only
                    print(f"{n} {condition}: kept {st[n]['kept']}/{quota[n]}, recorded {st[n]['recorded']}, "
                          f"draws {st[n]['attempts']}, rejected {st[n]['rejected']}", flush=True)
            for name in group:
                write_list(name, kept[name], rows[name], stats[name])
                print(f"wrote {name}: {len(kept[name])} trials ({'sealed' if name in failure_trials.SEALED_LISTS else 'open'})",
                      flush=True)
    print(f"elapsed {time.monotonic() - t0:.0f} s")


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
    s = sub.add_parser("lists", help="step 5: the lists of failure_trials.LISTS (sealed ones print counts only)")
    s.add_argument("--lists", nargs="+", default=["dev", "aux"], choices=("dev", "aux", "pooled"),
                   help="pooled: calibration, pilot, smoke and main together (their seeds are shared out by sha256)")
    s.add_argument("--no-start", action="store_true", help="use a simulator that is already running")
    args = p.parse_args()
    {"pilot": cmd_pilot, "lists": cmd_lists}[args.cmd](args)


if __name__ == "__main__":
    main()
