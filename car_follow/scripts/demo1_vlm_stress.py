"""Demo 1 step 6 (docs/demo1.md 7): does the VLM read the command, and does its margin drop when it cannot answer?

With few wrong picks left, "the margin of wrong picks" cannot be measured on the normal commands. This script changes
the question on the same VLM dev scenes (saved frames, simulator off), three ways:
    other        the command names a different look-alike. A model that reads the sentence moves its pick there;
                 one that found the answer some other way stays on the original target.
    no_position  the command has no position ("follow the white hatchback") and at least two candidates fit it.
    removed      the true box is taken out of the candidates, as if the detector had missed it. Every pick is wrong.

For no_position and removed a good ambiguity signal is low. Each is compared with the normal picks of the same model
(--normal, a saved scripts/demo1_vlm_eval.py result) by AUROC of the margin and of log(p1 / p2), where 1.0 means
every normal pick scores above every changed one.

Run from the car_follow folder:
    uv run python scripts/demo1_vlm_stress.py --normal vlm_dev_v2 --tag zero_shot
    uv run python scripts/demo1_vlm_stress.py --normal vlm_dev_v2_qlora1 --tag qlora1 \
        --adapter outputs/demo1_vlm/lora/qlora1/best
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
from scripts.demo1_vlm_eval import BASE, SCENES, candidate_table, load_scenes, scene_plans  # noqa: E402
from src.car_follow import detector, scorer, trials, vlm, vlm_data, vlm_metrics  # noqa: E402

BARE = {"en": "follow the {car}", "ko": "{car} 따라가"}


def group_of(scene, plan, ids, boxes):
    """The target (first) and its look-alikes, with their lane, order along the road, pixels and candidate box."""
    t = plan["target"]
    if len(scene["lookalike_ids"]) != len(plan["lookalikes"]):
        raise ValueError(f"{scene['id']}: {len(scene['lookalike_ids'])} recorded look-alikes, "
                         f"{len(plan['lookalikes'])} planned")
    cars = [{"id": scene["target_id"], "index_from_left": t["index_from_left"], "lanes": t["lanes"], "along_m": 0.0}]
    for actor, entry in zip(scene["lookalike_ids"], plan["lookalikes"]):
        cars.append({"id": actor, "index_from_left": t["index_from_left"] + entry["lane_offset"], "lanes": t["lanes"],
                     "along_m": entry["along_m"]})
    for c in cars:
        c["pixels"] = int((ids == c["id"]).sum())
        truth = scorer.mask_box(ids, c["id"])
        ious = [detector.iou(b, truth) for b in boxes] if truth else []
        c["box"] = int(np.argmax(ious)) if ious and max(ious) >= 0.5 else None
    return cars


def row(scene, lang, condition, said, result, right, **extra):
    return {"scene": scene["id"], "lang": lang, "condition": condition, "said": said, "right": bool(right),
            "picked": result.top, "probs": result.probs.round(4).tolist(), "margin": round(result.margin, 4),
            "log_ratio": round(result.log_ratio, 3), "agreement": result.agreement, **extra}


def signal(rows, normal):
    """How far the changed commands sit below the normal picks of the same model."""
    margin = [r["margin"] for r in rows]
    log_ratio = [r["log_ratio"] for r in rows]
    return {"n": len(rows), "margin_median": round(float(np.median(margin)), 3),
            "margin_at_least": {str(t): round(float(np.mean(np.array(margin) >= t)), 3)
                                for t in vlm_metrics.THRESHOLDS},
            "all_3_agree": round(float(np.mean([r["agreement"] == 1.0 for r in rows])), 3),
            "auroc_margin": round(vlm_metrics.auroc([r["margin"] for r in normal], margin), 3),
            "auroc_log_ratio": round(vlm_metrics.auroc([r["log_ratio"] for r in normal], log_ratio), 3)}


def summarize(rows, normal):
    out = {"normal": {"n": len(normal), "accuracy": round(float(np.mean([r["right"] for r in normal])), 4),
                      "margin_median": round(float(np.median([r["margin"] for r in normal])), 3),
                      "margin_at_least": {str(t): round(float(np.mean(np.array([r["margin"] for r in normal]) >= t)),
                                                        3) for t in vlm_metrics.THRESHOLDS},
                      "all_3_agree": round(float(np.mean([r["agreement"] == 1.0 for r in normal])), 3)}}
    other = [r for r in rows if r["condition"] == "other"]
    if other:
        out["other"] = {"n": len(other), "picked_the_named_car": round(float(np.mean([r["right"] for r in other])), 4),
                        "stayed_on_the_original_target": round(float(np.mean([r["stayed"] for r in other])), 4)}
    for condition in ("no_position", "removed"):
        part = [r for r in rows if r["condition"] == condition]
        if part:
            out[condition] = {lang: signal([r for r in part if r["lang"] == lang],
                                           [r for r in normal if r["lang"] == lang]) for lang in ("en", "ko")}
            out[condition]["both"] = signal(part, normal)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--normal", required=True, help="saved evaluation of the same model, e.g. vlm_dev_v2_qlora1")
    p.add_argument("--tag", required=True)
    p.add_argument("--set", default="vlm_dev", choices=("vlm_dev", "dev"))
    p.add_argument("--model", default="qwen3-vl-4b-4bit", choices=sorted(vlm.MODELS))
    p.add_argument("--adapter", default=None)
    p.add_argument("--prompt", default="v2", choices=sorted(vlm.PROMPTS))
    p.add_argument("--detector", default="rfdetr-small")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    saved = json.loads((BASE / f"eval_{args.normal}.json").read_text(encoding="utf-8"))
    if saved.get("adapter") != args.adapter:
        raise SystemExit(f"{args.normal} was run with adapter {saved.get('adapter')}, not {args.adapter}")
    # A single candidate has margin 1 by definition: it says nothing about how sure the model is.
    normal = [r for r in saved["results"][args.model]["rows"]
              if r["target_in_candidates"] and "probs" in r and len(r["probs"]) >= 2]
    scenes = load_scenes(args.set)[:args.limit]
    table = candidate_table(scenes, args.detector)
    plans = scene_plans()
    types = trials.load_types()
    model = vlm.make(args.model, offline=True, adapter=args.adapter)
    rows = []
    skipped = {"other": 0, "no_position": 0, "removed": 0}
    for k, s in enumerate(scenes):
        c = table[s["id"]]
        boxes, plan = c["boxes"], plans[s["id"]]
        bgr = cv2.imread(str(SCENES / f"{s['id']}.png"))
        ids = np.load(SCENES / f"{s['id']}.npz")["ids"]
        cars = group_of(s, plan, ids, boxes)
        t = plan["target"]
        family, entry = trials.color_family(t["color"]), types[t["blueprint"]]
        named = vlm_data.other_target(cars, s["target_id"], s["sentence"]["expression"])
        fitting = {x["box"] for x in cars if x["box"] is not None and x["pixels"] >= trials.VISIBLE_MIN_PIXELS}
        rest = [i for i in range(len(boxes)) if i != c["target"]]
        skipped["other"] += named is None or c["target"] is None
        skipped["no_position"] += len(fitting) < 2
        skipped["removed"] += c["target"] is None or len(rest) < 2
        for lang in ("en", "ko"):
            if named is not None and c["target"] is not None:
                car, expression = named
                said = vlm_data.render(vlm_data.TEMPLATES[lang][expression][0], lang, family, entry,
                                       car["index_from_left"], car["lanes"])
                r = vlm.pick(model, bgr, boxes, said, 3, args.prompt, seed=zlib.crc32(f"{s['id']}/{lang}/o".encode()))
                rows.append(row(s, lang, "other", said, r, r.top == car["box"], stayed=r.top == c["target"],
                                expression=expression))
            if len(fitting) >= 2:
                said = BARE[lang].format(car=vlm_data.car_name(lang, family, entry))
                r = vlm.pick(model, bgr, boxes, said, 3, args.prompt, seed=zlib.crc32(f"{s['id']}/{lang}/n".encode()))
                rows.append(row(s, lang, "no_position", said, r, r.top in fitting, fitting=len(fitting)))
            if c["target"] is not None and len(rest) >= 2:
                said = s["sentence"][lang]
                r = vlm.pick(model, bgr, [boxes[i] for i in rest], said, 3, args.prompt,
                             seed=zlib.crc32(f"{s['id']}/{lang}/r".encode()))
                rows.append(row(s, lang, "removed", said, r, False, left=len(rest)))
        if (k + 1) % 20 == 0:
            print(f"{k + 1}/{len(scenes)} scenes", flush=True)
    model.unload()
    summary = summarize(rows, normal)
    print(json.dumps({"skipped_scenes": skipped, **summary}, indent=1, ensure_ascii=False), flush=True)
    path = BASE / f"stress_{args.tag}.json"
    path.write_text(json.dumps({"normal": args.normal, "adapter": args.adapter, "prompt": args.prompt,
                                "skipped_scenes": skipped, "summary": summary, "rows": rows}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
