"""Check the stage 3 flight record of an evaluation run (docs/failure_plan.md 5.3, docs/failure.md 3): does the new
record agree with the evidence the flight already kept, and does it hold everything the tracker saw?

Per flight:
    pixels       car_px column 0 (the target) equals target_px on every tick
    innovation   tgt_innov is finite exactly where the target track was matched (target_score finite), from the
                 tick after the target track is first followed
    tracks       the target track's logged position equals the followed estimate (est_pos) whenever it is alive
    replay       the tracker run again offline on the logged detections gives the same target track and the same
                 estimate, bit for bit (configs 2 and 3; a pick carried by the image handoff is not replayed)

Run from the car_follow folder:
    uv run python scripts/check_flight_record.py --run check2_gt_b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.car_follow import detector, tracker  # noqa: E402

EVAL = ROOT / "outputs" / "demo1_eval"
DT = 1.0 / 20.0   # scripts/demo1_follow.py HZ


def detections(z, starts, k):
    out = []
    for i in range(starts[k], starts[k + 1]):
        pos = z["det_pos"][i]
        out.append(tracker.Detection(tuple(float(v) for v in z["det_box"][i]), float(z["det_score"][i]),
                                     None if np.isnan(pos).any() else pos.astype(float)))
    return out


def replay(z, row):
    """(track_id, est_pos) per tick from the logged detections, or None when the flight cannot be replayed."""
    ticks = len(z["owner"])
    starts = np.searchsorted(z["det_tick"], np.arange(ticks + 1))
    pick = row.get("pick")
    if "pick" in row and (not pick or pick.get("handoff_ticks")):
        return None
    mot = tracker.Tracker()
    target, follow, pending = None, None, None
    tid, est = np.full(ticks, -1), np.full((ticks, 3), np.nan)
    for k in range(ticks):
        dets = detections(z, starts, k)
        mot.update(dets, DT)
        if pick is None:          # config 2: ground truth names the track the tick it is first seen
            if target is None and z["track_id"][k] >= 0:
                target = int(z["track_id"][k])
                follow = mot.get(target).filter
        else:                     # config 3: the VLM's box is adopted at its tick, followed from the lock
            if k == pick["tick"]:
                box = pick["boxes"][pick["top"]]
                chosen = max(dets, key=lambda d: detector.iou(d.box, box))
                pending = mot.adopt(chosen, DT)
            if pending is not None and pick.get("lock_tick") is not None and k >= pick["lock_tick"]:
                target, follow, pending = pending.id, pending.filter, None
        live = mot.get(target) if target is not None else None
        tid[k] = target if live is not None else -1
        if follow is not None:
            est[k] = follow.position
    return tid, est


def check(z, row):
    out = {"pixels": bool(np.array_equal(z["car_px"][:, 0], z["target_px"]))}
    followed = np.flatnonzero(z["track_id"] >= 0)
    alive = z["track_id"] >= 0
    alive[: int(followed[0]) + 1 if len(followed) else len(alive)] = False
    out["innovation"] = bool(np.array_equal(np.isfinite(z["tgt_innov"][alive]), np.isfinite(z["target_score"][alive])))
    tid = z["track_id"]
    rows = [(int(k), i) for i, k in enumerate(z["trk_tick"]) if tid[k] >= 0 and z["trk_id"][i] == tid[k]]
    diff = max((float(np.abs(z["trk_pos"][i] - z["est_pos"][k]).max()) for k, i in rows), default=0.0)
    out["tracks"] = diff < 1e-3 and len(rows) == int((tid >= 0).sum())
    r = replay(z, row)
    if r is None:
        out["replay"] = None
    else:
        out["replay"] = bool(np.array_equal(r[0], tid) and np.array_equal(r[1], z["est_pos"], equal_nan=True))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    result = json.loads((EVAL / f"{args.run}.json").read_text(encoding="utf-8"))
    rows = ({f"{t}_{lang}": r for lang, part in result["languages"].items() for t, r in part["trials"].items()}
            if "languages" in result else result["trials"])
    bad = 0
    for stem, row in sorted(rows.items()):
        z = dict(np.load(EVAL / args.run / f"{stem}.npz"))
        if "car_px" not in z:
            raise SystemExit(f"{stem} has no stage 3 record (flown before it existed)")
        out = check(z, row)
        ok = all(v is not False for v in out.values())
        bad += not ok
        print(f"{stem}: {out}, detections {len(z['det_tick'])}, track rows {len(z['trk_tick'])}")
    print("all agree" if not bad else f"{bad} flights disagree")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
