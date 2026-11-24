"""Demo 1 step 6 (docs/demo1.md 7): QLoRA on the VLM first pick (simulator off).

The base model stays in 4-bit; LoRA adapters go on the language model's attention and MLP layers (the vision encoder is
not trained). One example is one forward pass over a saved training scene: the detector's boxes with freshly shuffled
numbers, a sentence (half the trial lists' wording, half other wordings, never the held-out ones), and a cross-entropy
loss over the candidate numbers at the answer position, the same logits the pick reads (src/car_follow/vlm.py).

Checkpoints are compared on the 100 VLM dev scenes (accuracy, then NLL). The dev 20 are not read here.

Run from the car_follow folder:
    uv run python scripts/demo1_vlm_train.py --name pilot --limit-scenes 8 --epochs 6 --eval-every 0
    uv run python scripts/demo1_vlm_train.py --name qlora1 --epochs 2
"""
import argparse
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_vlm_eval import BASE, CANDIDATE_SCORE, SCENES, candidate_table, load_scenes  # noqa: E402
from src.car_follow import detector, scorer, trials, vlm, vlm_data, vlm_metrics  # noqa: E402

TRAIN = BASE / "train"
TRAIN_SCENES = TRAIN / "scenes"
LORA = BASE / "lora"
TARGETS = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
           r"|mlp\.(gate_proj|up_proj|down_proj))")


def train_table(metas, det_name):
    """{scene: {boxes, members: {actor id: index of its box}}} from the detector, cached."""
    path = TRAIN / f"candidates_{det_name}.json"
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [m for m in metas if m["id"] not in table]
    if todo:
        det = detector.make(det_name)
        for k, m in enumerate(todo):
            bgr = cv2.imread(str(TRAIN_SCENES / f"{m['id']}.png"))
            ids = np.load(TRAIN_SCENES / f"{m['id']}.npz")["ids"]
            boxes = [b for b in det.detect(detector.Frame(bgr)) if b.score >= CANDIDATE_SCORE]
            boxes = [boxes[i] for i in vlm.candidates([b.xyxy for b in boxes], [b.score for b in boxes])]
            members = {}
            for c in m["cars"]:
                truth = scorer.mask_box(ids, c["id"])
                ious = [detector.iou(b.xyxy, truth) for b in boxes] if truth else []
                if ious and max(ious) >= 0.5:
                    members[str(c["id"])] = int(np.argmax(ious))
            table[m["id"]] = {"boxes": [[float(v) for v in b.xyxy] for b in boxes],
                              "scores": [float(b.score) for b in boxes], "members": members}
            if (k + 1) % 200 == 0:
                print(f"candidates: {k + 1}/{len(todo)}", flush=True)
        del det
        import torch
        torch.cuda.empty_cache()
        path.write_text(json.dumps(table), encoding="utf-8")
    return table


def build_examples(metas, table):
    out = []
    for m in metas:
        t = table[m["id"]]
        if len(t["boxes"]) < 2:
            continue
        for car, expression in vlm_data.nameable(m["cars"]):
            box = t["members"].get(str(car["id"]))
            if box is not None:
                out.append({"scene": m["id"], "box": box, "expression": expression, "family": m["family"],
                            "blueprint": m["blueprint"], "index": car["index_from_left"], "lanes": car["lanes"]})
    return out


def sentence_for(rng, ex, lang, types):
    options = vlm_data.TEMPLATES[lang][ex["expression"]]
    template = options[0] if rng.random() < 0.5 else rng.choice(options[1:])
    return vlm_data.render(template, lang, ex["family"], types[ex["blueprint"]], ex["index"], ex["lanes"])


def validate(picker, scenes, table, prompt):
    """Accuracy and NLL on the VLM dev scenes, also for the part in map cells kept out of training."""
    config = json.loads((ROOT / "configs" / "demo1_vlm_dev.json").read_text(encoding="utf-8"))["scenes"]
    unseen = {s["id"] for s in config if vlm_data.unseen_cell(trials.cell_of(*s["target"]["transform"][:2]))}
    rows = []
    for s in scenes:
        c = table[s["id"]]
        bgr = cv2.imread(str(SCENES / f"{s['id']}.png"))
        for lang in ("en", "ko"):
            r = vlm.pick(picker, bgr, c["boxes"], s["sentence"][lang], 3, prompt,
                         seed=zlib.crc32(f"{s['id']}/{lang}".encode()))
            if r is None:
                continue
            rows.append({"scene": s["id"], "lang": lang, "right": c["target"] is not None and r.top == c["target"],
                         "target": c["target"], "probs": r.probs.round(4).tolist(), "margin": r.margin,
                         "agreement": r.agreement})
    out = {"all": vlm_metrics.reliability(rows)}
    for lang in ("en", "ko"):
        out[lang] = round(float(np.mean([x["right"] for x in rows if x["lang"] == lang])), 4)
    out["unseen_cells"] = round(float(np.mean([x["right"] for x in rows if x["scene"] in unseen])), 4)
    out["unseen_n"] = sum(x["scene"] in unseen for x in rows)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--base", default="qwen3-vl-4b-4bit")
    p.add_argument("--detector", default="rfdetr-small")
    p.add_argument("--prompt", default="v2", choices=sorted(vlm.PROMPTS))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=1500, help="examples between validations (0: only at the end)")
    p.add_argument("--limit-scenes", type=int, default=None)
    p.add_argument("--max-minutes", type=float, default=150.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed-test", type=int, default=0, help="stop after this many examples, without validating")
    args = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    out = LORA / args.name
    out.mkdir(parents=True, exist_ok=True)

    metas = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(TRAIN_SCENES.glob("*.json"))]
    table = train_table(metas, args.detector)
    metas = metas[:args.limit_scenes]
    examples = build_examples(metas, table)
    types = trials.load_types()
    counts = {e: sum(x["expression"] == e for x in examples) for e in ("lane", "front", "back")}
    print(f"{len(metas)} scenes, {len(examples)} (car, expression) pairs {counts}, x2 languages", flush=True)
    val_scenes = load_scenes("vlm_dev")
    val_table = candidate_table(val_scenes, args.detector)

    picker = vlm.make(args.base, offline=True)
    model = get_peft_model(picker.model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                                    bias="none", target_modules=TARGETS))
    picker.model = model
    # Without checkpointing one 960x540 example kept 11.0 GiB of activations, more than the card has beside other apps.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [q for q in model.parameters() if q.requires_grad]
    print(f"trainable: {sum(q.numel() for q in trainable) / 1e6:.1f} M parameters", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    per_epoch = 2 * len(examples)
    total = int(per_epoch * args.epochs)
    steps = max(1, total // args.accumulate)
    warm = max(1, min(50, steps // 10))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warm, 1.0) * (
        0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))))

    def checkpoint(done, history, best):
        model.eval()
        v = validate(picker, val_scenes, val_table, args.prompt)
        model.train()
        score = (round((v["en"] + v["ko"]) / 2, 4), -v["all"]["nll"])
        row = {"examples": done, "en": v["en"], "ko": v["ko"], "unseen_cells": v["unseen_cells"],
               "unseen_n": v["unseen_n"], "nll": v["all"]["nll"], "ece": v["all"]["ece"],
               "margin_auroc": v["all"]["margin_auroc"], "wrong_with_margin": v["all"]["wrong_with_margin"]}
        history.append(row)
        print("VAL", json.dumps(row), flush=True)
        model.save_pretrained(out / "last")
        if best is None or score > best:
            best = score
            model.save_pretrained(out / "best")
            print(f"best so far at {done} examples", flush=True)
        (out / "log.json").write_text(json.dumps({"args": vars(args), "pairs": len(examples), "counts": counts,
                                                  "history": history}, indent=1), encoding="utf-8")
        return best

    history, best = [], None
    model.train()
    t0 = time.monotonic()
    done, seen_loss, seen_right, seen = 0, 0.0, 0, 0
    items = []
    stop = False
    optimizer.zero_grad(set_to_none=True)
    while done < total and not stop:
        if not items:
            items = [(e, lang) for e in examples for lang in ("en", "ko")]
            rng.shuffle(items)
        ex, lang = items.pop()
        c = table[ex["scene"]]
        n = len(c["boxes"])
        numbers = np_rng.permutation(np.arange(1, n + 1))
        bgr = cv2.imread(str(TRAIN_SCENES / f"{ex['scene']}.png"))
        rgb = np.ascontiguousarray(vlm.draw_marks(bgr, c["boxes"], numbers)[:, :, ::-1])
        text = vlm.PROMPTS[args.prompt].format(sentence=sentence_for(rng, ex, lang, types))
        _, inputs = picker.prompt_inputs(rgb, text)
        ids = [picker.token_id(str(k)) for k in range(1, n + 1)]
        label = torch.tensor([int(numbers[ex["box"]]) - 1], device=model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]
        logits = logits.float()[ids]
        loss = torch.nn.functional.cross_entropy(logits[None], label)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is {loss.item()} at example {done}")
        (loss / args.accumulate).backward()
        done += 1
        seen += 1
        seen_loss += loss.item()
        seen_right += int(logits.argmax().item() == label.item())
        if done % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
        if done % 100 == 0:
            spent = time.monotonic() - t0
            print(f"{done}/{total} loss {seen_loss / seen:.3f} acc {seen_right / seen:.3f} "
                  f"lr {schedule.get_last_lr()[0]:.1e} {done / spent:.2f} ex/s "
                  f"eta {(total - done) / max(done / spent, 1e-6) / 60:.0f} min "
                  f"gpu {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)
            seen, seen_loss, seen_right = 0, 0.0, 0
        if args.speed_test and done >= args.speed_test:
            spent = time.monotonic() - t0
            print(f"speed test: {done / spent:.2f} ex/s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB",
                  flush=True)
            return
        if args.eval_every and done % args.eval_every == 0 and done < total:
            best = checkpoint(done, history, best)
        if (time.monotonic() - t0) / 60 > args.max_minutes:
            print(f"stopping at the {args.max_minutes:.0f} minute limit", flush=True)
            stop = True
    best = checkpoint(done, history, best)
    print(f"done: {done} examples in {(time.monotonic() - t0) / 60:.1f} min; adapters in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
