"""The lane-change comparison of one evaluation run over A4 trials (src/car_follow/lane_phases.py): per flight and
per condition (A4a single, A4b weave, A4c swap), the switches, losses, distance and estimate errors and the drone's
jitter before, during, between and after the target's lane changes, and whether the car first named was kept.
No simulator: nothing is flown again.

Reads outputs/demo1_eval/<run>.json and <run>/<trial>_<lang>.npz and the failure trial run it flew (demo1_eval.py
--failure-run), writes outputs/failure_labels/<run>_lanes.json (local only) and prints the tables.

Run from the car_follow folder:
    uv run python scripts/lane_change_report.py --run config3_a4_dev_guard-on_rfdetr-small_qlora1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.failure_labels import (EVAL, FAILURE_SEALED, OUT, SEALED, flights, lock_and_pick,  # noqa: E402
                                    sources, word_match)
from src.car_follow import failure_trials, labels, lane_phases, trials  # noqa: E402

CONDITION_NAMES = {"A4a": "single", "A4b": "weave", "A4c": "swap"}


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def print_table(by_condition):
    """Two tables: what the tracker did, and how the drone moved (with how hard the target made it)."""
    print(f"{'cond':5} {'phase':8} {'n':>3} {'s':>6} {'wrong s':>7} {'nobox s':>7} {'hold':>6} {'lost s':>6} "
          f"{'sw box':>6} {'sw est':>6} {'drift':>5} {'loss':>4} {'est m':>5} {'p95':>5}")
    for cond, part in by_condition.items():
        for ph, r in part["pooled"].items():
            e = r["events"]
            print(f"{cond:5} {ph:8} {r['flights']:>3} {r['seconds']:>6.1f} {r['wrong_box_s']:>7.2f} "
                  f"{r['no_box_s']:>7.2f} {fmt(r['hold'], 3):>6} {r['lost_s']:>6.2f} {e['switch_box']:>6} "
                  f"{e['switch_estimate']:>6} {e['drift']:>5} {e['loss']:>4} {fmt(r['estimate_error_mean']):>5} "
                  f"{fmt(r['estimate_error_p95_worst'], 1):>5}")
    print(f"{'cond':5} {'phase':8} {'dist m':>6} {'p95':>5} {'across':>6} {'along':>6} {'acc':>5} {'acc x':>5} "
          f"{'acc y':>5} {'jerk':>6} {'yaw':>5} {'tgt yaw':>7} {'tgt acc':>7} {'tgt lat':>7}")
    for cond, part in by_condition.items():
        for ph, r in part["pooled"].items():
            print(f"{cond:5} {ph:8} {fmt(r['distance_error_mean']):>6} {fmt(r['distance_error_p95_worst'], 1):>5} "
                  f"{fmt(r['across_error_mean']):>6} {fmt(r['along_error_mean']):>6} {fmt(r['accel_rms']):>5} "
                  f"{fmt(r['accel_across_rms']):>5} {fmt(r['accel_along_rms']):>5} {fmt(r['jerk_rms'], 1):>6} "
                  f"{fmt(r['yaw_rate_rms'], 1):>5} {fmt(r['target_yaw_rate'], 1):>7} {fmt(r['target_accel_rms']):>7} "
                  f"{fmt(r['target_lat_speed']):>7}")
    print("acc x: across the target's heading, acc y: along it (m/s^2, RMS); yaw in deg/s RMS")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="an evaluation run under outputs/demo1_eval that flew A4 trials")
    p.add_argument("--open-sealed", action="store_true", help="needed for a run on sealed trials")
    args = p.parse_args()
    result = json.loads((EVAL / f"{args.run}.json").read_text(encoding="utf-8"))
    if "failure_run" not in result:
        p.error(f"{args.run} did not fly a failure trial run")
    if set(result["split"].split("+")) & (set(SEALED) | set(FAILURE_SEALED)) and not args.open_sealed:
        p.error(f"{args.run} flew sealed trials: --open-sealed is needed")
    by_id, traffic = sources(result, args.open_sealed)
    types = trials.load_types()
    t0 = time.monotonic()
    rows = []
    for tid, lang, row in flights(result):
        trial = by_id[tid]
        if (trial.get("event") or {}).get("kind") not in failure_trials.LANE_KINDS:
            continue
        stem = tid if lang is None else f"{tid}_{lang}"
        ev = dict(np.load(EVAL / args.run / f"{stem}.npz"))
        rec = dict(np.load(traffic / f"{tid}.npz"))
        match = word_match(rec, trial, types)
        lock, pick = lock_and_pick(row, ev)
        lab = labels.label_flight(ev, rec, match, lock, pick)
        changes = failure_trials.lane_changes(rec, 0)
        out, _ = lane_phases.flight_phases(ev, rec, lock, lab["chain"], changes)
        out.update(trial=tid, language=lang, condition=trial["condition"], intent=trial["intent"],
                   first_pick=row.get("first_pick"), failure=lab["failure"], t_fail_s=lab["t_fail_s"],
                   chain=[{k: e[k] for k in ("kind", "t_s")} for e in lab["chain"]],
                   switches=len(row["switches"]), hold=row["hold"], distance_error_m=row["distance_error_m"])
        rows.append(out)
        ch = ", ".join(f"{c['side']} {c['start_s']:.1f}-{c['end_s']:.1f} (speed {c['speed_jump_mps']:+.1f})"
                       for c in out["changes"])
        first = out["first_failure"]
        print(f"{stem} [{trial['condition']}, {trial['intent']}]: changes {ch}; pick "
              f"{'right' if row.get('first_pick') else 'WRONG'}; kept {out['kept_same_car']}"
              + (f" (first {first['kind']} {first['t_s']} s, {first['phase']})" if first else "")
              + f"; chain {' -> '.join(e['kind'] + ' ' + str(e['t_s']) for e in out['chain']) or 'none'}", flush=True)
    by_condition = {}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        by_condition[cond] = {
            "name": CONDITION_NAMES.get(cond, cond), "flights": len(rs),
            "first_pick_right": sum(bool(r["first_pick"]) for r in rs),
            "kept_same_car": sum(r["kept_same_car"] for r in rs),
            "changes": sum(len(r["changes"]) for r in rs),
            "speed_jumps_over_3": sum(1 for r in rs for c in r["changes"] if (c["speed_jump_mps"] or 0) > 3.0),
            "first_failures": [{"trial": r["trial"], "language": r["language"], **r["first_failure"]}
                               for r in rs if r["first_failure"]],
            "pooled": lane_phases.pool([r["phases"] for r in rs]),
        }
    print_table(by_condition)
    for cond, part in by_condition.items():
        print(f"{cond} ({part['name']}): kept the car first named {part['kept_same_car']}/{part['flights']}, "
              f"first pick right {part['first_pick_right']}/{part['flights']}, {part['changes']} lane changes, "
              f"{part['speed_jumps_over_3']} with the target speeding up over 3 m/s")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"run": args.run, "failure_run": result["failure_run"], "split": result["split"],
           "rules": {"before_s": lane_phases.BEFORE_S, "after_s": lane_phases.AFTER_S, "pad_s": lane_phases.PAD_S,
                     "settle_m": failure_trials.SETTLE_M, "settle_cap_s": failure_trials.SETTLE_CAP_S,
                     "min_stay_s": failure_trials.MIN_STAY_S},
           "by_condition": by_condition, "flights": rows}
    path = OUT / f"{args.run}_lanes.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}, {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
