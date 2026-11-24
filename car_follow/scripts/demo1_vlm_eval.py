"""Demo 1 step 6 (docs/demo1.md 6): zero-shot first pick by VLMs on saved start frames (no simulator).

For each scene (scripts/demo1_vlm_scenes.py) the detector's boxes are the candidates (RF-DETR-Small, score >= 0.3,
at most 9, cached once per detector). Each VLM is then asked, for the English and the Korean sentence, which
numbered box the sentence names (src/car_follow/vlm.py: 3 shuffles of the numbers, probabilities averaged per box).
A pick is right when its box overlaps the target's visible-pixel box at IoU >= 0.5.

Reports per model, scene set and language: accuracy (all scenes, and when the target is among the candidates),
by expression, what the wrong picks were, the margin and the shuffle agreement of right and wrong picks (and how
well the margin separates them), time per pick and GPU memory. Wrong picks are drawn to outputs/demo1_vlm/<set>/.

Run from the car_follow folder:
    uv run python scripts/demo1_vlm_eval.py --set vlm_dev --models qwen3-vl-2b internvl3.5-2b
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.car_follow import detector, scorer, trials, vlm, vlm_data, vlm_metrics  # noqa: E402

BASE = ROOT / "outputs" / "demo1_vlm"
SCENES = BASE / "scenes"
CANDIDATE_SCORE = 0.3


def load_scenes(which):
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SCENES.glob("*.json"))]
    return [m for m in metas if m["source"] == which]


def scene_plans():
    """{scene id: its trial entry} for the VLM dev and dev scenes (target car, lane, start)."""
    vlm_dev = json.loads((ROOT / "configs" / "demo1_vlm_dev.json").read_text(encoding="utf-8"))["scenes"]
    return {t["id"]: t for t in vlm_dev + trials.load_trials("dev")}


def held_out_sentence(plan, lang, types):
    """The scene's command in the wording never used in training (src/car_follow/vlm_data.py)."""
    t = plan["target"]
    return vlm_data.render(vlm_data.HELD_OUT[lang][plan["sentence"]["expression"]], lang,
                           trials.color_family(t["color"]), types[t["blueprint"]], t["index_from_left"], t["lanes"])


def candidate_table(scenes, det_name):
    """{scene id: {boxes, scores, target, lookalikes, kinds}} from the detector, cached per detector."""
    path = BASE / f"candidates_{det_name}.json"
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [s for s in scenes if s["id"] not in table]
    if todo:
        det = detector.make(det_name)
        for s in todo:
            bgr = cv2.imread(str(SCENES / f"{s['id']}.png"))
            ids = np.load(SCENES / f"{s['id']}.npz")["ids"]
            boxes = [b for b in det.detect(detector.Frame(bgr)) if b.score >= CANDIDATE_SCORE]
            keep = vlm.candidates([b.xyxy for b in boxes], [b.score for b in boxes])
            boxes = [boxes[i] for i in keep]
            truth = scorer.mask_box(ids, s["target_id"])
            kinds = []
            for b in boxes:   # which recorded car each candidate box is (most pixels inside), for the error kinds
                owner, _ = scorer.box_owner(ids, b.xyxy, s["car_ids"])
                a = s["car_ids"][owner] if owner >= 0 else None
                kinds.append("target" if a == s["target_id"] else "lookalike" if a in s["lookalike_ids"]
                             else "background" if a is not None else "none")
            ious = [detector.iou(b.xyxy, truth) if truth else 0.0 for b in boxes]
            target = int(np.argmax(ious)) if ious and max(ious) >= 0.5 else None
            table[s["id"]] = {"boxes": [list(b.xyxy) for b in boxes], "scores": [b.score for b in boxes],
                              "target": target, "kinds": kinds}
        if hasattr(det, "_torch"):
            del det
            import torch
            torch.cuda.empty_cache()
        path.write_text(json.dumps(table, indent=1), encoding="utf-8")
    return table


def auroc(pos, neg):
    """P(a right pick's margin > a wrong pick's margin)."""
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    return round(float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                       / (len(pos) * len(neg))), 3)


def summarize(rows):
    out = {}
    for lang in ("en", "ko"):
        r = [x for x in rows if x["lang"] == lang]
        if not r:
            continue
        right = [x for x in r if x["right"]]
        wrong = [x for x in r if not x["right"]]
        with_target = [x for x in r if x["target_in_candidates"]]
        out[lang] = {
            "n": len(r), "accuracy": round(len(right) / len(r), 4),
            "accuracy_when_target_is_a_candidate": round(sum(x["right"] for x in with_target) / len(with_target), 4)
            if with_target else None,
            "target_not_a_candidate": len(r) - len(with_target),
            "by_expression": {e: round(np.mean([x["right"] for x in r if x["expression"] == e]), 3)
                              for e in sorted({x["expression"] for x in r})},
            "wrong_picked": {k: sum(1 for x in wrong if x["picked_kind"] == k)
                             for k in ("lookalike", "background", "none")},
            "margin_median": {"right": round(float(np.median([x["margin"] for x in right])), 3) if right else None,
                              "wrong": round(float(np.median([x["margin"] for x in wrong])), 3) if wrong else None},
            "margin_auroc": auroc([x["margin"] for x in right], [x["margin"] for x in wrong]),
            "agreement_all_3": round(np.mean([x["agreement"] == 1.0 for x in r]), 3),
            "accuracy_when_agreement_all_3": round(np.mean([x["right"] for x in r if x["agreement"] == 1.0]), 3)
            if any(x["agreement"] == 1.0 for x in r) else None,
            "seconds_per_pick": round(float(np.mean([x["seconds"] for x in r])), 3),
            "reliability": vlm_metrics.reliability(r),
            "unseen_cells": vlm_metrics.reliability([x for x in r if x.get("unseen_cell")]),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="vlm_dev", choices=("vlm_dev", "dev"))
    p.add_argument("--models", nargs="+", default=["qwen3-vl-2b"], choices=sorted(vlm.MODELS))
    p.add_argument("--detector", default="rfdetr-small")
    p.add_argument("--prompt", default="v1", choices=sorted(vlm.PROMPTS))
    p.add_argument("--permutations", type=int, default=3)
    p.add_argument("--langs", nargs="+", default=["en", "ko"])
    p.add_argument("--limit", type=int, default=None, help="only the first N scenes (quick checks)")
    p.add_argument("--tag", default="", help="suffix of the output name")
    p.add_argument("--adapter", default=None, help="a trained LoRA folder (scripts/demo1_vlm_train.py)")
    p.add_argument("--wording", default="trial", choices=("trial", "held_out"),
                   help="held_out: the commands in a wording never used in training")
    args = p.parse_args()
    import torch
    scenes = load_scenes(args.set)[:args.limit]
    table = candidate_table(scenes, args.detector)
    plans = scene_plans()
    types = trials.load_types()
    unseen = {i for i, t in plans.items() if vlm_data.unseen_cell(trials.cell_of(*t["target"]["transform"][:2]))}
    results = {}
    for name in args.models:
        model = vlm.make(name, offline=True, adapter=args.adapter)
        first = scenes[0]
        bgr = cv2.imread(str(SCENES / f"{first['id']}.png"))
        c = table[first["id"]]
        free = model.generate(np.ascontiguousarray(vlm.draw_marks(bgr, c["boxes"], range(1, len(c["boxes"]) + 1))
                                                  [:, :, ::-1]),
                              vlm.PROMPTS[args.prompt].format(sentence=first["sentence"]["en"]))
        torch.cuda.reset_peak_memory_stats()
        rows = []
        bad = BASE / f"{args.set}_wrong" / f"{name}_{args.prompt}{args.tag}"
        for s in scenes:
            c = table[s["id"]]
            bgr = cv2.imread(str(SCENES / f"{s['id']}.png"))
            for lang in args.langs:
                said = (s["sentence"][lang] if args.wording == "trial"
                        else held_out_sentence(plans[s["id"]], lang, types))
                r = vlm.pick(model, bgr, c["boxes"], said, args.permutations, args.prompt,
                             seed=zlib.crc32(f"{s['id']}/{lang}".encode()))
                if r is None:
                    rows.append({"scene": s["id"], "lang": lang, "right": False, "target_in_candidates": False,
                                 "expression": s["sentence"]["expression"], "picked_kind": "none", "margin": 0.0,
                                 "agreement": 0.0, "seconds": 0.0})
                    continue
                right = c["target"] is not None and r.top == c["target"]
                rows.append({"scene": s["id"], "lang": lang, "right": bool(right), "said": said,
                             "unseen_cell": s["id"] in unseen, "target_in_candidates": c["target"] is not None,
                             "expression": s["sentence"]["expression"], "candidates": len(c["boxes"]),
                             "picked": r.top, "picked_kind": c["kinds"][r.top], "target": c["target"],
                             "probs": r.probs.round(4).tolist(), "margin": round(r.margin, 4),
                             "log_ratio": round(r.log_ratio, 3), "agreement": r.agreement,
                             "shuffles": r.shuffles, "seconds": round(r.seconds, 3)})
                if not right:
                    bad.mkdir(parents=True, exist_ok=True)
                    img = vlm.draw_marks(bgr, c["boxes"], range(1, len(c["boxes"]) + 1))
                    if c["target"] is not None:
                        x0, y0, x1, y1 = (int(v) for v in c["boxes"][c["target"]])
                        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 220, 0), 2)
                    x0, y0, x1, y1 = (int(v) for v in c["boxes"][r.top])
                    cv2.rectangle(img, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (0, 0, 255), 2)
                    cv2.putText(img, f"{lang}: {said[:60]}  p={r.probs.max():.2f} m={r.margin:.2f}",
                                (8, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(str(bad / f"{s['id']}_{lang}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        info = model.info()
        info["peak_mib_during_picks"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        info["free_answer_example"] = free
        results[name] = {"info": info, "summary": summarize(rows), "rows": rows}
        print(name, json.dumps({"info": info, "summary": results[name]["summary"]}, indent=1, ensure_ascii=False),
              flush=True)
        model.unload()
        del model
    path = BASE / f"eval_{args.set}_{args.prompt}{args.tag}.json"
    path.write_text(json.dumps({"set": args.set, "scenes": len(scenes), "detector": args.detector,
                                "candidate_score": CANDIDATE_SCORE, "permutations": args.permutations,
                                "adapter": args.adapter, "wording": args.wording,
                                "prompt": vlm.PROMPTS[args.prompt], "results": results}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
