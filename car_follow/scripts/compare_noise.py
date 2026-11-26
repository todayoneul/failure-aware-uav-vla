"""Is run B within the run-to-run noise of two runs A1 and A2 of the same flights (docs/failure.md 3)? The simulator
is not bit-reproducible (depth readings differ by millimetres from the first ticks), so a change that should not
touch the flight is checked against how much two flights of the same code already differ.

Pass, as fixed in outputs/demo1_eval/stage3_logging_rule.json before B flew:
    switch counts equal in every flight
    for distance_error_m, horizontal_error_m and hold: max |B - A1| and max |B - A2| over the flights at most twice
    max |A1 - A2| (hold: at least 0.02)
    max drone position difference B vs A1 at most twice that of A1 vs A2

Run from the car_follow folder:
    uv run python scripts/compare_noise.py check2_gt_a1 check2_gt_a2 check2_gt_b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "demo1_eval"
METRICS = ("distance_error_m", "horizontal_error_m", "hold")
FLOOR = {"hold": 0.02}


def rows(name):
    r = json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))
    if "languages" in r:
        return {f"{t}_{lang}": row for lang, part in r["languages"].items() for t, row in part["trials"].items()}
    return r["trials"]


def diff(a, b):
    return None if a is None or b is None else abs(a - b)


def pair(x, y, ra, rb):
    """Per-flight differences between runs x and y."""
    out = {}
    for stem in sorted(set(ra) & set(rb)):
        za, zb = np.load(EVAL / x / f"{stem}.npz"), np.load(EVAL / y / f"{stem}.npz")
        p = np.linalg.norm(za["log_position"] - zb["log_position"], axis=1)
        apart = np.flatnonzero(p > 0)
        out[stem] = {m: diff(ra[stem][m], rb[stem][m]) for m in METRICS}
        out[stem].update(switches_equal=len(ra[stem]["switches"]) == len(rb[stem]["switches"]),
                         position_max_m=float(p.max()), apart_tick=int(apart[0]) if len(apart) else None)
    return out


def worst(per, key):
    v = [d[key] for d in per.values() if d[key] is not None]
    return max(v) if v else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a1")
    p.add_argument("a2")
    p.add_argument("b")
    args = p.parse_args()
    r1, r2, rb = rows(args.a1), rows(args.a2), rows(args.b)
    noise, b1, b2 = pair(args.a1, args.a2, r1, r2), pair(args.b, args.a1, rb, r1), pair(args.b, args.a2, rb, r2)
    checks = {"switches": all(d["switches_equal"] for d in list(b1.values()) + list(b2.values()))}
    table = {}
    for m in METRICS:
        limit = max(2 * worst(noise, m), FLOOR.get(m, 0.0))
        table[m] = {"a1_a2": round(worst(noise, m), 4), "b_a1": round(worst(b1, m), 4), "b_a2": round(worst(b2, m), 4),
                    "limit": round(limit, 4)}
        checks[m] = worst(b1, m) <= limit and worst(b2, m) <= limit
    limit = 2 * worst(noise, "position_max_m")
    table["position_max_m"] = {"a1_a2": round(worst(noise, "position_max_m"), 3),
                               "b_a1": round(worst(b1, "position_max_m"), 3), "limit": round(limit, 3)}
    checks["position"] = worst(b1, "position_max_m") <= limit
    ticks = {"a1_a2": sorted(d["apart_tick"] for d in noise.values() if d["apart_tick"] is not None),
             "b_a1": sorted(d["apart_tick"] for d in b1.values() if d["apart_tick"] is not None)}
    out = {"flights": len(b1), "worst": table, "drone_apart_tick": ticks, "checks": checks,
           "passed": all(checks.values())}
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
