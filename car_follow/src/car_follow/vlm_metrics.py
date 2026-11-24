"""Is the VLM's top-1 / top-2 margin still an honest ambiguity signal? (docs/demo1.md 7)

`reliability(rows)` takes one row per pick with `right`, `probs`, `margin`, `agreement` and `target` (the index of the
true box among the candidates, None if the detector missed it) and reports:
    ece                    expected calibration error of p(top-1), 10 equal-width bins
    nll                    mean -log p(true box), over the picks whose target is a candidate
    confidence             median p(top-1) of right and of wrong picks
    margin_auroc           P(a right pick has a larger margin than a wrong pick)
    wrong_with_margin      share of the wrong picks whose margin is at least 0.5 / 0.8 / 0.9 (confidently wrong)
    keep                   at each margin threshold: share of picks kept and accuracy of those
    keep_agreement         the same for "all shuffles agree"
"""
import numpy as np

THRESHOLDS = (0.5, 0.8, 0.9)


def auroc(pos, neg):
    """P(a value from `pos` > a value from `neg`), ties count one half."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def ece(confidence, right, bins=10):
    confidence, right = np.asarray(confidence, dtype=float), np.asarray(right, dtype=float)
    if len(confidence) == 0:
        return None
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        inside = index == b
        if inside.any():
            total += inside.mean() * abs(right[inside].mean() - confidence[inside].mean())
    return float(total)


def _keep(mask, right):
    return {"kept": round(float(mask.mean()), 3), "accuracy": round(float(right[mask].mean()), 3) if mask.any()
            else None}


def reliability(rows, thresholds=THRESHOLDS):
    rows = [r for r in rows if "probs" in r]
    if not rows:
        return None
    right = np.array([bool(r["right"]) for r in rows])
    conf = np.array([max(r["probs"]) for r in rows])
    margin = np.array([r["margin"] for r in rows])
    agree = np.array([r["agreement"] == 1.0 for r in rows])
    with_target = [r for r in rows if r.get("target") is not None]
    nll = float(np.mean([-np.log(max(r["probs"][r["target"]], 1e-4)) for r in with_target])) if with_target else None
    wrong = margin[~right]
    return {
        "n": len(rows), "accuracy": round(float(right.mean()), 4), "ece": round(ece(conf, right), 4),
        "nll": round(nll, 4) if nll is not None else None,
        "confidence": {"right": round(float(np.median(conf[right])), 3) if right.any() else None,
                       "wrong": round(float(np.median(conf[~right])), 3) if (~right).any() else None},
        "margin_auroc": None if right.all() or not right.any() else round(auroc(margin[right], wrong), 3),
        "wrong_with_margin": {str(t): round(float((wrong >= t).mean()), 3) if len(wrong) else None
                              for t in thresholds},
        "keep": {str(t): _keep(margin >= t, right) for t in thresholds},
        "keep_agreement": _keep(agree, right),
    }
