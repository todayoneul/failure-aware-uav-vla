"""Compare two evaluation runs flight by flight (docs/failure_plan.md 5.3): are the evidence arrays and the score rows
bit-identical? Wall-clock timings are left out. Keys only one run has (a newer, longer record) are listed, not
compared.

Run from the car_follow folder:
    uv run python scripts/compare_runs.py check2_gt_a1 check2_gt_a2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "demo1_eval"
TIMING_ARRAYS = {"perceive_s", "detect_s", "guard_s"}
TIMING_FIELDS = {"ticks_per_s", "gpu_mib", "perceive_ms_mean", "guard_ms_mean", "detect_ms_mean", "detect_ms_p95",
                 "seconds", "seconds_mean", "seconds_max"}


def strip(value):
    """A score row without its wall-clock fields, at any depth."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in TIMING_FIELDS}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def rows(result):
    if "languages" in result:
        return {f"{t}_{lang}": row for lang, part in result["languages"].items() for t, row in part["trials"].items()}
    return dict(result["trials"])


def same_array(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def compare(run_a, run_b):
    ra = json.loads((EVAL / f"{run_a}.json").read_text(encoding="utf-8"))
    rb = json.loads((EVAL / f"{run_b}.json").read_text(encoding="utf-8"))
    a_rows, b_rows = rows(ra), rows(rb)
    out = {"a": run_a, "b": run_b, "flights": 0, "identical_flights": 0, "differences": [], "only_in_a": set(),
           "only_in_b": set()}
    for stem in sorted(set(a_rows) & set(b_rows)):
        out["flights"] += 1
        diffs = []
        if strip(a_rows[stem]) != strip(b_rows[stem]):
            keys = sorted(k for k in set(a_rows[stem]) | set(b_rows[stem])
                          if strip(a_rows[stem].get(k)) != strip(b_rows[stem].get(k)))
            diffs.append(f"score row: {keys}")
        za, zb = np.load(EVAL / run_a / f"{stem}.npz"), np.load(EVAL / run_b / f"{stem}.npz")
        out["only_in_a"] |= set(za.files) - set(zb.files)
        out["only_in_b"] |= set(zb.files) - set(za.files)
        for key in sorted((set(za.files) & set(zb.files)) - TIMING_ARRAYS):
            if not same_array(za[key], zb[key]):
                a, b = za[key], zb[key]
                first = None
                if a.shape == b.shape and a.dtype.kind in "fiub":
                    bad = ~((a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else (a == b))
                    first = int(np.argwhere(bad)[0][0]) if bad.any() else None
                diffs.append(f"{key} (first differing tick {first})")
        if diffs:
            out["differences"].append({"flight": stem, "what": diffs})
        else:
            out["identical_flights"] += 1
    out["only_in_a"], out["only_in_b"] = sorted(out["only_in_a"]), sorted(out["only_in_b"])
    out["identical"] = out["identical_flights"] == out["flights"] and out["flights"] > 0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a")
    p.add_argument("run_b")
    args = p.parse_args()
    out = compare(args.run_a, args.run_b)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["identical"] else 1)


if __name__ == "__main__":
    main()
