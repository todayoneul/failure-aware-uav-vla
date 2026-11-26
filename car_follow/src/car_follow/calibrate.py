"""Stage 3 trial-level calibration of a monitor's threshold (docs/failure_plan.md 6). Pure numpy.

A monitor gives a score per tick; its alarm fires at tick t when the score has been at or above the threshold tau
for the last `k` ticks. The unit of calibration is the trial: each trial's loss at tau is the share of its flights
(English and Korean) with an alarm in the normal stretch (failure_metrics.normal_mask), so 0, 0.5 or 1.

Conformal risk control (bounded loss, B = 1) picks
    tau_hat = min { tau : (n * R_hat(tau) + B) / (n + 1) <= alpha }
over the n calibration trials, which bounds a new trial's expected loss, the per-flight probability of an alarm in
the normal stretch, by alpha for exchangeable trials. With 0/1 losses (one flight per trial) this is the
ceil((n + 1)(1 - alpha))-th smallest per-trial score (`rank_threshold`).

The strict variant (docs/failure_plan.md 6.2) is the same with the strict stretch (near-miss windows kept): its
rate at tau_hat is reported, and the threshold strict calibration would have given is kept for reference.
"""
import math

import numpy as np

from .failure_metrics import binomial_cdf


def alarm_from_score(score, tau, k=1):
    """Bool per tick: the score at or above `tau` for the last `k` ticks (NaN counts as below)."""
    above = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf) >= tau
    if k <= 1:
        return above
    run = np.zeros(len(above), int)
    count = 0
    for i, a in enumerate(above):
        count = count + 1 if a else 0
        run[i] = count
    return run >= k


def flight_score(score, mask, k=1):
    """The flight's score over `mask`: the largest tau whose alarm (rule `k`) fires somewhere in `mask`, that is the
    largest k-tick running minimum of the score at a masked tick. -inf if nothing in `mask` can fire."""
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)
    if k > 1:
        lows = np.full(len(s), -np.inf)
        for i in range(k - 1, len(s)):
            lows[i] = s[i - k + 1:i + 1].min()
        s = lows
    m = np.asarray(mask, dtype=bool)
    return float(s[m].max()) if m.any() else -math.inf


def risk(trial_scores, tau):
    """Mean over trials of the share of their flights whose flight score reaches `tau` (alarm in the stretch)."""
    return float(np.mean([np.mean([s >= tau for s in flights]) for flights in trial_scores]))


def crc_threshold(trial_scores, alpha, bound=1.0):
    """Conformal risk control over trials. `trial_scores`: per trial, the flight scores of its flights. Returns
    {'tau', 'feasible', 'risk', 'bound', 'n', 'alpha'}: the smallest tau (just above a flight score) with
    (n R_hat + B) / (n + 1) <= alpha; tau = inf and feasible False if even no alarm at all fails (n too small)."""
    n = len(trial_scores)
    values = sorted({s for flights in trial_scores for s in flights if s > -math.inf})
    candidates = [-math.inf] + [math.nextafter(v, math.inf) for v in values]
    for tau in candidates:
        r = risk(trial_scores, tau)
        b = (n * r + bound) / (n + 1)
        if b <= alpha:
            return {"tau": tau, "feasible": True, "risk": round(r, 4), "bound": round(b, 4), "n": n, "alpha": alpha}
    return {"tau": math.inf, "feasible": False, "risk": 0.0, "bound": round(bound / (n + 1), 4), "n": n,
            "alpha": alpha}


def rank_threshold(scores, alpha):
    """The 0/1-loss form: just above the ceil((n + 1)(1 - alpha))-th smallest score; inf if that rank exceeds n."""
    s = sorted(scores)
    n = len(s)
    rank = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    return math.inf if rank > n else math.nextafter(s[rank - 1], math.inf)


def pac_rank(n, alpha, delta):
    """With 0/1 losses, the smallest rank whose threshold keeps the false-alarm rate at or under `alpha` with
    probability at least 1 - `delta` over the draw of the n calibration trials (None if no rank does). The rate of
    the rank-r threshold is 1 - U_(r), U_(r) ~ Beta(r, n + 1 - r), so P(rate <= alpha) = P(Bin(n, 1 - alpha) < r)."""
    for r in range(1, n + 1):
        if binomial_cdf(r - 1, n, 1.0 - alpha) >= 1.0 - delta:
            return r
    return None


def calibrate(trial_scores, strict_scores, alpha):
    """Threshold from the normal-stretch scores, with the strict rate at that threshold and the strict threshold
    for reference (docs/failure_plan.md 6.2). Both inputs: per trial, the flight scores."""
    main = crc_threshold(trial_scores, alpha)
    strict = crc_threshold(strict_scores, alpha)
    return {
        "threshold": main,
        "strict_risk_at_threshold": round(risk(strict_scores, main["tau"]), 4),
        "strict_threshold": strict,
    }
