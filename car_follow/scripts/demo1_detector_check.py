"""Demo 1 step 5 (docs/demo1.md 5.5): how well a detector finds the cars in the drone's view, before tracking.

Flies config 1 (the drone follows the true target, altitude guard on) over the replayed traffic of each trial, and
every `--every` ticks runs the detector on the drone's RGB frame. Each recorded car with at least 30 visible pixels
is a ground-truth box (around its visible pixels); a detection matches it at IoU >= 0.5 (also reported at 0.3, since
a partly hidden car's visible box is smaller than the car). Reports recall by score threshold, car kind, visible
size and distance, false detections per frame, time per frame and GPU memory. A few frames with misses are saved.

Run from the car_follow folder:
    uv run python scripts/demo1_detector_check.py --detector rfdetr-small --split dev --every 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_eval import OUT as EVAL_OUT, fly_trial  # noqa: E402
from scripts.demo1_follow import HZ, OUT as FOLLOW_OUT, select  # noqa: E402
from src.car_follow import detector, scorer, sim, trials  # noqa: E402

OUT = EVAL_OUT / "detector_check"
PX_BINS = (30, 100, 300, 1000, 3000, 10 ** 9)
DIST_BINS = (0, 30, 50, 80, 10 ** 9)
THRESHOLDS = (0.3, 0.5)


class Probe:
    def __init__(self, det, every, trial_id, shots):
        self.det, self.every, self.trial_id, self.shots = det, every, trial_id, shots
        self.rows, self.false, self.times, self.frames = [], [], [], 0

    def __call__(self, k, bgr, ids_img, rp, rec, state):
        if k % self.every:
            return
        boxes, dt = detector.timed(self.det, detector.Frame(bgr, ids_img, tuple(rp.ids)))
        self.times.append(dt)
        self.frames += 1
        ids, counts = np.unique(ids_img, return_counts=True)
        index = {int(a): i for i, a in enumerate(rp.ids)}
        truth, small = [], []
        for a, n in zip(ids.tolist(), counts.tolist()):
            if a in index:
                box = scorer.mask_box(ids_img, a)
                (truth if n >= scorer.VISIBLE_PX else small).append((index[a], n, box))
        for i, n, box in truth:
            # the best score among detections overlapping the car at IoU >= 0.5 and at IoU >= 0.3
            s50 = max((b.score for b in boxes if detector.iou(b.xyxy, box) >= 0.5), default=0.0)
            s30 = max((b.score for b in boxes if detector.iou(b.xyxy, box) >= 0.3), default=0.0)
            dist = float(np.linalg.norm(rec["pos"][k, i] - state.position))
            self.rows.append((k, i, str(rec["kind"][i]), n, dist, s50, s30))
        for th in THRESHOLDS:
            kept = [b for b in boxes if b.score >= th]
            got, unmatched = detector.match_truth(kept, [t[2] for t in truth + small], min_iou=0.3)
            self.false.append((th, len(unmatched)))
        missed = [t for t, r in zip(truth, self.rows[-len(truth):]) if r[5] < 0.5] if truth else []
        big_miss = [t for t in missed if t[1] >= 300]
        if big_miss and len(self.shots) < 12:
            img = bgr.copy()
            for b in boxes:
                if b.score >= 0.3:
                    cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), (0, 255, 255), 1)
                    cv2.putText(img, f"{b.score:.2f}", (int(b.x0), int(b.y0) - 2), 0, 0.4, (0, 255, 255), 1)
            for _, n, box in big_miss:
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            self.shots.append((f"{self.trial_id}_t{k * 0.05:05.1f}", img))


def recall(rows, th, min_iou=0.5, where=None):
    """Share of cars with a detection of score >= th overlapping them at IoU >= min_iou (0.5 or 0.3)."""
    sel = [r for r in rows if where is None or where(r)]
    if not sel:
        return None
    col = 5 if min_iou == 0.5 else 6
    hit = sum(1 for r in sel if r[col] >= th)
    return {"n": len(sel), "recall": round(hit / len(sel), 4)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", default="rfdetr-small", choices=sorted(detector.MODELS))
    p.add_argument("--split", default="dev", choices=("dev", "pilot", "smoke"))
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--run", default="rec1")
    p.add_argument("--every", type=int, default=10, help="run the detector every this many ticks")
    p.add_argument("--no-start", action="store_true")
    args = p.parse_args()
    out = OUT / args.detector
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    det = detector.make(args.detector)
    load_s = time.monotonic() - t0
    rows, false, times, frames, shots = [], [], [], 0, []
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, _tm):
        sim.clear(client, world)
        world.tick()
        for trial in select(args.split, args.ids):
            rec = dict(np.load(FOLLOW_OUT / "traffic" / args.run / f"{trial['id']}.npz"))
            world.reset_all_traffic_lights()
            probe = Probe(det, args.every, trial["id"], shots)
            fly_trial(client, world, trial, rec, 1, "on", probe=probe)
            rows += [(trial["id"],) + r for r in probe.rows]
            false += probe.false
            times += probe.times
            frames += probe.frames
            r = recall([x[1:] for x in rows if x[0] == trial["id"]], 0.5)
            print(f"{trial['id']}: frames {probe.frames}, cars {len(probe.rows)}, recall@0.5 {r and r['recall']}, "
                  f"{1000 * np.median(probe.times):.1f} ms", flush=True)
        gpu_total = sim.gpu_memory_mib()
    r = [x[1:] for x in rows]
    kinds = ("target", "lookalike", "background")
    result = {
        "detector": det.info(), "load_s": round(load_s, 1), "split": args.split, "every": args.every,
        "frames": frames, "cars": len(r),
        "match": "IoU with the car's visible-pixel box; a hit needs IoU >= min_iou and score >= threshold",
        "recall": {f"score>={th}": {
            "all_iou0.5": recall(r, th), "all_iou0.3": recall(r, th, 0.3),
            "by_kind": {k: recall(r, th, where=lambda x, k=k: x[2] == k) for k in kinds},
            "by_px": {f"{lo}-{hi}": recall(r, th, where=lambda x, lo=lo, hi=hi: lo <= x[3] < hi)
                      for lo, hi in zip(PX_BINS[:-1], PX_BINS[1:])},
            "by_distance_m": {f"{lo}-{hi}": recall(r, th, where=lambda x, lo=lo, hi=hi: lo <= x[4] < hi)
                              for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:])},
        } for th in THRESHOLDS},
        "false_per_frame": {f"score>={th}": round(float(np.mean([n for t, n in false if t == th])), 3)
                            for th in THRESHOLDS},
        "ms_per_frame": {"mean": round(1000 * float(np.mean(times)), 2),
                         "median": round(1000 * float(np.median(times)), 2),
                         "p95": round(1000 * float(np.percentile(times, 95)), 2),
                         "max": round(1000 * float(np.max(times)), 2)},
        "gpu_total_mib": gpu_total,
    }
    det_info = det.info()
    result["detector"]["vram_peak_mib"] = det_info.get("vram_peak_mib")
    path = EVAL_OUT / f"detector_check_{args.detector}_{args.split}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    np.save(out / "rows.npy", np.array(rows, dtype=object), allow_pickle=True)
    for name, img in shots:
        cv2.imwrite(str(out / f"miss_{name}.jpg"), img)
    print(json.dumps(result, indent=1))
    print(f"wrote {path.relative_to(ROOT)}, {len(shots)} miss frames in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
