"""Stage 3 labels of the flights of one evaluation run (docs/failure_plan.md 3, src/car_follow/labels.py), from its
evidence, the traffic it flew over and its trials. No simulator: nothing is flown again.

Reads outputs/demo1_eval/<run>.json and <run>/<trial>[_<lang>].npz, writes outputs/failure_labels/<run>.json (local
only) and prints one line per flight. A run on a sealed split needs --open-sealed.

Run from the car_follow folder:
    uv run python scripts/failure_labels.py --run config3_dev_rec1_qlora1_v2
    uv run python scripts/failure_labels.py --run config3_main_main_a --open-sealed --ids main-024
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.car_follow import labels, replay, trials  # noqa: E402

EVAL = ROOT / "outputs" / "demo1_eval"
TRAFFIC = ROOT / "outputs" / "demo1_follow" / "traffic"
OUT = ROOT / "outputs" / "failure_labels"
SEALED = ("main", "calibration")


def word_match(rec, trial, types):
    """Per recorded car: whether the sentence's car-type and colour words fit it (docs/failure_plan.md 3.2)."""
    t = trial["target"]
    kind, family = types[t["blueprint"]]["type"], trials.color_family(t["color"])
    out = []
    for bp, color in replay.blueprints(rec, trial):   # older recordings take them from the trial
        bp, color = str(bp), str(color)
        out.append(bool(color) and bp in types and types[bp]["type"] == kind and trials.color_family(color) == family)
    return out


def flights(result):
    """[(trial id, language or None, score row)] of a run's result."""
    if "languages" in result:
        return [(tid, lang, row) for lang, part in result["languages"].items() for tid, row in part["trials"].items()]
    return [(tid, None, row) for tid, row in result["trials"].items()]


def lock_and_pick(row, ev):
    pick = row.get("pick")
    if "pick" in row:   # config 3: the VLM's first pick
        if not pick:
            return None, (0, False)
        return pick.get("lock_tick"), (pick["tick"], bool(row["first_pick"]))
    seen = np.flatnonzero(np.asarray(ev["track_id"]) >= 0)   # config 2: ground truth picks the track
    return (int(seen[0]) if len(seen) else None), None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="an evaluation run under outputs/demo1_eval")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--open-sealed", action="store_true", help="needed for a run on a sealed split")
    p.add_argument("--name", default=None, help="output name (default: the run's, with the ids)")
    args = p.parse_args()
    result = json.loads((EVAL / f"{args.run}.json").read_text(encoding="utf-8"))
    split = result["split"]
    if split in SEALED and not args.open_sealed:
        p.error(f"{args.run} flew the sealed {split} split: --open-sealed is needed")
    by_id = {t["id"]: t for t in trials.load_trials(split, open_sealed=split in SEALED)}
    types = trials.load_types()
    t0 = time.monotonic()
    rows = []
    for tid, lang, row in flights(result):
        if args.ids and tid not in args.ids:
            continue
        stem = tid if lang is None else f"{tid}_{lang}"
        ev = dict(np.load(EVAL / args.run / f"{stem}.npz"))
        rec = dict(np.load(TRAFFIC / result["recording"] / f"{tid}.npz"))
        match = word_match(rec, by_id[tid], types)
        lock, pick = lock_and_pick(row, ev)
        lab = labels.label_flight(ev, rec, match, lock, pick)
        lab.update(trial=tid, language=lang, word_match_cars=int(sum(match)) - 1,
                   conditions=labels.conditions(by_id[tid]))
        rows.append(lab)
        chain = " -> ".join(f"{e['kind']} {e['t_s']:.2f}" for e in lab["chain"]) or "none"
        near = ", ".join(f"{m['start_s']:.2f}+{m['length_s']:.2f} {'/'.join(m['why'])}" for m in lab["near_miss"])
        print(f"{stem}: {chain}; recovered {lab['recovered_s']}; drift max {lab['drift_max_m']} m; near-miss "
              f"[{near}]; exposure {lab['exposure_s']}", flush=True)
    counts = {"flights": len(rows),
              "by_failure": {k: sum(r["failure"] == k for r in rows) for k in labels.OUTCOMES},
              "no_failure": sum(r["failure"] is None for r in rows),
              "near_miss_flights": sum(r["near_miss_flight"] for r in rows),
              "exposure_s": {k: round(sum(r["exposure_s"][k] for r in rows), 2) for k in labels.EXPOSURES}}
    OUT.mkdir(parents=True, exist_ok=True)
    name = args.name or (args.run + ("_" + "_".join(args.ids) if args.ids else ""))
    out = {"run": args.run, "split": split, "recording": result["recording"],
           "rules": {"switch_ticks": labels.SWITCH_TICKS, "estimate_near_m": labels.EST_NEAR_M,
                     "drift_m": labels.DRIFT_M, "drift_s": labels.DRIFT_S, "near_miss_m": labels.NEAR_MISS_M,
                     "loss_hidden_s": labels.LOSS_HIDDEN_S, "cross_m": labels.CROSS_M, "queue_m": labels.QUEUE_M,
                     "depth_limit_deg": round(labels.DEPTH_LIMIT_DEG, 2), "recover_s": labels.RECOVER_S},
           "counts": counts, "flights": rows}
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts, indent=1))
    print(f"wrote {(OUT / f'{name}.json').relative_to(ROOT)}, {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
