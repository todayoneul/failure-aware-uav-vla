"""Demo 1 step 6 (docs/demo1.md 7): accuracy and the ambiguity signal, side by side, from saved evaluations.

Reads outputs/demo1_vlm/eval_<set>_<prompt><tag>.json (scripts/demo1_vlm_eval.py). Rows saved before the unseen-cell
flag existed are flagged here from the scene lists, so a zero-shot baseline and a trained model are split the same way.

Run from the car_follow folder:
    uv run python scripts/demo1_vlm_report.py vlm_dev_v2:qwen3-vl-4b-4bit vlm_dev_v2_qlora1:qwen3-vl-4b-4bit
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_vlm_eval import BASE, scene_plans  # noqa: E402
from src.car_follow import trials, vlm_data, vlm_metrics  # noqa: E402


def line(label, r):
    if r is None:
        return f"{label:<24} (none)"
    keep = r["keep"]["0.8"]
    wrong = r["wrong_with_margin"]
    return (f"{label:<24} n={r['n']:<4} acc {r['accuracy'] * 100:5.1f}%  ece {r['ece']:.3f}  nll {r['nll']:.3f}  "
            f"conf R/W {r['confidence']['right']}/{r['confidence']['wrong']}  auroc {r['margin_auroc']}  "
            f"wrong m>=.5/.8/.9 {wrong['0.5']}/{wrong['0.8']}/{wrong['0.9']}  "
            f"keep m>=.8 {keep['kept']}@{keep['accuracy']}  "
            f"3/3 {r['keep_agreement']['kept']}@{r['keep_agreement']['accuracy']}")


def main():
    plans = scene_plans()
    unseen = {i for i, t in plans.items() if vlm_data.unseen_cell(trials.cell_of(*t["target"]["transform"][:2]))}
    out = {}
    for spec in sys.argv[1:]:
        tag, model = spec.split(":")
        data = json.loads((BASE / f"eval_{tag}.json").read_text(encoding="utf-8"))
        rows = data["results"][model]["rows"]
        for x in rows:
            x["unseen_cell"] = x["scene"] in unseen
        print(f"\n{tag} / {model} (adapter: {data.get('adapter')}, wording: {data.get('wording', 'trial')})")
        out[spec] = {}
        for lang in ("en", "ko"):
            r = [x for x in rows if x["lang"] == lang]
            out[spec][lang] = vlm_metrics.reliability(r)
            out[spec][f"{lang}_unseen"] = vlm_metrics.reliability([x for x in r if x["unseen_cell"]])
            print(line(lang, out[spec][lang]))
            print(line(f"{lang}, unseen cells", out[spec][f"{lang}_unseen"]))
        out[spec]["both"] = vlm_metrics.reliability(rows)
        print(line("en+ko", out[spec]["both"]))
    return out


if __name__ == "__main__":
    main()
