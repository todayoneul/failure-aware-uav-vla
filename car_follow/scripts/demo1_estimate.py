"""Demo 1 step 5 (docs/demo1.md 5.2): how well the target's position comes out of a box, before any detector.

Reads the evidence of a `demo1_eval.py --measure` run (config 1: the drone follows the truth, the box is the
target's ground-truth box, and every tick keeps the raw measurements of src/car_follow/estimate.py). For each method
(flat ground, depth map) it fits the push along the ray and the height offset on these dev trials, then reports
the error of single measurements and of the Kalman filter run over them (open loop: the filter does not fly the
drone), on the ground and on the elevated road separately.

Run from the car_follow folder:
    uv run python scripts/demo1_estimate.py --name config1_dev_rec1_guard-on_measure
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_follow import DT, OUT as FOLLOW_OUT, select, target_track  # noqa: E402
from src.car_follow import estimate  # noqa: E402

EVAL = ROOT / "outputs" / "demo1_eval"
ELEVATED_M = 3.0   # the target counts as on the elevated road above this height


def horizontal_unit(ray):
    h = ray[:, :2]
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {"n": int(len(x)), "median": round(float(np.median(x)), 3), "mean": round(float(x.mean()), 3),
            "p95": round(float(np.percentile(x, 95)), 3), "over_2m": round(float((x > 2.0).mean()), 4)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--run", default="rec1")
    args = p.parse_args()
    data = []
    for trial in select(args.split, None):
        path = EVAL / args.name / f"{trial['id']}.npz"
        if not path.exists():
            continue
        ev = np.load(path)
        rec = np.load(FOLLOW_OUT / "traffic" / args.run / f"{trial['id']}.npz")
        track = target_track(dict(rec))
        data.append((trial["id"], ev, track))
    result = {"trials": len(data), "methods": {}}
    for method, ray_key in (("flat", "ray_b"), ("depth", "ray_c")):
        m = np.concatenate([ev[f"m_{method}"] for _, ev, _ in data])
        ray = np.concatenate([ev[ray_key] for _, ev, _ in data])
        truth = np.concatenate([t["position"] for _, _, t in data])
        ok = np.isfinite(m).all(axis=1)
        u = horizontal_unit(ray[ok])
        diff = truth[ok] - m[ok]
        push = float(np.median((diff[:, :2] * u).sum(1)))             # along the ray, horizontally
        lift = float(np.median(diff[:, 2]))                            # truth height minus measured height
        out = {"push_m": round(push, 3), "lift_m": round(lift, 3), "measured_share": round(float(ok.mean()), 4)}
        elevated = truth[:, 2] > ELEVATED_M
        for label, est in (("raw", m), ("pushed", None)):
            e = np.full(len(m), np.nan)
            z = np.full(len(m), np.nan)
            if est is None:
                est = m.copy()
                est[ok, :2] += push * u
                est[ok, 2] += lift
            e[ok] = np.linalg.norm(est[ok, :2] - truth[ok, :2], axis=1)
            z[ok] = np.abs(est[ok, 2] - truth[ok, 2])
            out[label] = {"horizontal_ground": stats(e[~elevated]), "horizontal_elevated": stats(e[elevated]),
                          "height": stats(z)}
        # Open-loop Kalman filter over the pushed measurements, trial by trial.
        fe, se, he = [], [], []
        for _, ev, track in data:
            mm, rr = ev[f"m_{method}"], ev[ray_key]
            f = estimate.TargetFilter()
            for k in range(len(mm)):
                f.predict(DT)
                if np.isfinite(mm[k]).all():
                    q = estimate.push(mm[k], rr[k], push) + np.array([0.0, 0.0, lift])
                    f.update(q, DT)
                if f.position is None or k * DT < 5.0:
                    continue
                fe.append(float(np.linalg.norm(f.position[:2] - track["position"][k, :2])))
                if k > 0:
                    v = (track["position"][k] - track["position"][k - 1]) / DT
                    sp = float(np.linalg.norm(v[:2]))
                    se.append(abs(float(np.linalg.norm(f.velocity[:2])) - sp))
                    if sp > 3.0 and f.yaw is not None:
                        he.append(abs((f.yaw - track["yaw"][k] + 180.0) % 360.0 - 180.0))
        out["filtered"] = {"horizontal": stats(fe), "speed": stats(se), "heading_deg_moving": stats(he)}
        result["methods"][method] = out
    print(json.dumps(result, indent=1))
    path = EVAL / f"{args.name}_estimate.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
