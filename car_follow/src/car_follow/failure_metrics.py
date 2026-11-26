"""Stage 3 metrics of a monitor's alarms against the labels (docs/failure_plan.md 3.1 and 6, docs/research_plan.md
6). Pure numpy.

Per flight, `alarm` is a bool per tick (the monitor's alarm rule already applied) and `labels` is
labels.label_flight's output.

    normal stretch        from the lock to t_fail - L (the flight's end with no failure), without the near-miss
                          windows (each near-miss run and the L s before it). The strict stretch keeps them.
    false alarm           an alarm anywhere in the normal stretch (the quantity calibrated, docs/failure_plan.md 6.1)
    detection             the first alarm from t_fail - L until the flight recovers on its own (or ends)
    delay                 that alarm's tick minus t_fail (negative: it came early)
    wrong-follow time     t_fail to the alarm (0 if the alarm came early); without one, to the recovery, else to the
                          flight's end (the main metric)

Reporting rule (decided 2026-11-26, docs/failure_plan.md 6.5): the pooled detection rate is never reported alone. The
failure groups below are reported side by side with it, each with its own detection rate (95% interval), lead time
and wrong-follow time, because occlusion losses are frequent and easy to see (no box) and would inflate a pooled rate:
    silent      switch_box, switch_estimate, drift: the box or estimate on the wrong car while the system looks fine
    loss        the target hidden or its track dropped, split by cause (hidden, track_dropped)
    first_pick  the first pick was not the target
When the rows carry a monitor's declared failure type ('declared') or the response's outcome ('reacquired'), each group
also gets the type accuracy and the reacquisition rate.
"""
import math

import numpy as np

from .labels import DT

L_S = 3.0
FAILURE_GROUPS = {"silent": ("switch_box", "switch_estimate", "drift"), "loss": ("loss",),
                  "first_pick": ("first_pick",)}


def group_of(kind):
    for g, kinds in FAILURE_GROUPS.items():
        if kind in kinds:
            return g
    return None


def _ticks(seconds, dt=DT):
    return int(round(seconds / dt))


def normal_mask(labels, n=None, lead_s=L_S, strict=False, dt=DT):
    """Bool per tick: the flight's normal stretch (see the module's notes)."""
    n = labels["ticks"] if n is None else n
    mask = np.zeros(n, bool)
    lock = labels["lock_tick"]
    if lock is None:
        return mask
    lead = _ticks(lead_s, dt)
    t_fail = labels["t_fail_tick"]
    end = n if t_fail is None else max(lock, t_fail - lead)
    mask[lock:end] = True
    if not strict:
        for nm in labels["near_miss"]:
            s = nm["start_tick"]
            mask[max(0, s - lead): s + nm["ticks"]] = False
    return mask


def near_miss_mask(labels, n=None, lead_s=L_S, dt=DT):
    """Bool per tick: the near-miss windows (each run, not an onset, and the L s before it)."""
    n = labels["ticks"] if n is None else n
    mask = np.zeros(n, bool)
    lead = _ticks(lead_s, dt)
    for nm in labels["near_miss"]:
        if "onset" in nm["why"]:
            continue
        s = nm["start_tick"]
        mask[max(0, s - lead): s + nm["ticks"]] = True
    return mask


def first_true(mask, lo=0, hi=None):
    hits = np.flatnonzero(np.asarray(mask)[lo:hi])
    return int(lo + hits[0]) if len(hits) else None


def rising_edges(mask):
    m = np.asarray(mask, dtype=int)
    return int((np.diff(np.r_[0, m]) == 1).sum())


def flight_metrics(labels, alarm, lead_s=L_S, dt=DT):
    """The metrics of one flight's alarms (JSON-ready)."""
    alarm = np.asarray(alarm, dtype=bool)
    n = len(alarm)
    normal = normal_mask(labels, n, lead_s, strict=False, dt=dt)
    strict = normal_mask(labels, n, lead_s, strict=True, dt=dt)
    near = near_miss_mask(labels, n, lead_s, dt)
    fa = first_true(alarm & normal)
    minutes = normal.sum() * dt / 60.0
    chain = labels.get("chain") or []
    row = {
        "failure": labels["failure"],
        "group": group_of(labels["failure"]),
        "cause": chain[0].get("cause") if chain else None,
        "false_alarm": fa is not None,
        "false_alarm_strict": first_true(alarm & strict) is not None,
        "first_false_alarm_s": None if fa is None else round(fa * dt, 2),
        "normal_s": round(float(normal.sum() * dt), 2),
        "strict_normal_s": round(float(strict.sum() * dt), 2),
        "alarms_per_min": round(rising_edges(alarm & normal) / minutes, 3) if minutes > 0 else None,
        "near_miss_flight": bool(labels.get("near_miss_flight", False)),
        "near_miss_alarm": bool((alarm & near).any()),
        "detected": None, "delay_s": None, "wrong_follow_s": None,
    }
    t_fail = labels["t_fail_tick"]
    if t_fail is None:
        return row
    lead = _ticks(lead_s, dt)
    lo = max(0, t_fail - lead)
    if labels["lock_tick"] is not None and labels["failure"] != "first_pick":
        lo = max(lo, labels["lock_tick"])
    recovered = labels["recovered_tick"]
    hi = n if recovered is None else recovered
    a = first_true(alarm, lo, hi)
    row["detected"] = a is not None
    if a is not None:
        row["delay_s"] = round((a - t_fail) * dt, 2)
        row["wrong_follow_s"] = round(max(0, a - t_fail) * dt, 2)
    else:
        row["wrong_follow_s"] = round(((n if recovered is None else recovered) - t_fail) * dt, 2)
    return row


def lead_before(alarm, start, window_s=5.0, dt=DT):
    """Seconds by which the first alarm within `window_s` before tick `start` (up to `start`) came before it, or
    None. For alarms that predict an exposure, such as a crossing."""
    lo = max(0, start - _ticks(window_s, dt))
    a = first_true(alarm, lo, start + 1)
    return None if a is None else round((start - a) * dt, 2)


def binomial_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k + 1)))


def clopper_pearson(k, n, conf=0.95):
    """Two-sided exact (lower, upper) bounds of a binomial proportion k / n."""
    if n == 0:
        return None, None
    tail = (1.0 - conf) / 2.0

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - binomial_cdf(k - 1, n, p) >= tail)
    upper = 1.0 if k == n else solve(lambda p: binomial_cdf(k, n, p) <= tail)
    return lower, upper


def _rate(k, n):
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    lo, hi = clopper_pearson(k, n)
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def summarize(rows, trial_of=None):
    """Over flights `rows` (flight_metrics outputs). `trial_of` (one trial id per row) adds the trial-level false
    alarm rate: a trial counts when any of its flights had one."""
    out = {
        "flights": len(rows),
        "false_alarm": _rate(sum(r["false_alarm"] for r in rows), len(rows)),
        "false_alarm_strict": _rate(sum(r["false_alarm_strict"] for r in rows), len(rows)),
        "near_miss_flights": sum(r["near_miss_flight"] for r in rows),
        "near_miss_alarm_flights": sum(r["near_miss_alarm"] for r in rows),
    }
    per_min = [r["alarms_per_min"] for r in rows if r["alarms_per_min"] is not None]
    out["alarms_per_min_mean"] = round(float(np.mean(per_min)), 3) if per_min else None
    if trial_of is not None:
        trials = {}
        for r, t in zip(rows, trial_of):
            trials.setdefault(t, []).append(r)
        out["trial_false_alarm"] = _rate(sum(any(r["false_alarm"] for r in v) for v in trials.values()), len(trials))
        out["trial_false_alarm_strict"] = _rate(sum(any(r["false_alarm_strict"] for r in v) for v in trials.values()),
                                                len(trials))
    fails = [r for r in rows if r["failure"] is not None]
    out["failures"] = len(fails)
    out["detected"] = _rate(sum(bool(r["detected"]) for r in fails), len(fails))
    delays = [r["delay_s"] for r in fails if r["delay_s"] is not None]
    out["delay_s"] = ({"median": round(float(np.median(delays)), 2), "p90": round(float(np.percentile(delays, 90)), 2),
                       "early": sum(d <= 0 for d in delays)} if delays else None)
    wrong = [r["wrong_follow_s"] for r in fails]
    out["wrong_follow_s"] = ({"total": round(float(np.sum(wrong)), 2), "median": round(float(np.median(wrong)), 2),
                              "max": round(float(np.max(wrong)), 2)} if wrong else None)
    kinds = sorted({r["failure"] for r in fails})
    out["by_failure"] = {k: {"flights": sum(r["failure"] == k for r in fails),
                             "detected": sum(bool(r["detected"]) for r in fails if r["failure"] == k)} for k in kinds}
    out["by_group"] = {g: group_summary([r for r in fails if group_of(r["failure"]) == g])
                       for g in FAILURE_GROUPS}
    losses = [r for r in fails if r["failure"] == "loss"]
    out["by_group"]["loss"]["by_cause"] = {c: group_summary([r for r in losses if r.get("cause") == c])
                                           for c in sorted({r.get("cause") for r in losses if r.get("cause")})}
    return out


def group_summary(rows):
    """Detection rate (95% interval), lead time, wrong-follow time of one failure group's flights; type accuracy and
    reacquisition rate when the rows carry 'declared' (the monitor's failure type) and 'reacquired'."""
    out = {"flights": len(rows), "detected": _rate(sum(bool(r["detected"]) for r in rows), len(rows))}
    delays = [r["delay_s"] for r in rows if r["delay_s"] is not None]
    early = [-d for d in delays if d <= 0]
    out["lead"] = {"early": len(early), "early_share": round(len(early) / len(rows), 4) if rows else None,
                   "lead_s_median": round(float(np.median(early)), 2) if early else None,
                   "delay_s_median": round(float(np.median(delays)), 2) if delays else None}
    wrong = [r["wrong_follow_s"] for r in rows if r["wrong_follow_s"] is not None]
    out["wrong_follow_s"] = ({"total": round(float(np.sum(wrong)), 2), "median": round(float(np.median(wrong)), 2)}
                             if wrong else None)
    typed = [r for r in rows if r.get("declared") is not None]
    if typed:
        out["type_accuracy"] = _rate(sum(group_of(r["declared"]) == group_of(r["failure"]) or
                                         r["declared"] == r["failure"] for r in typed), len(typed))
    responded = [r for r in rows if r.get("reacquired") is not None]
    if responded:
        out["reacquired"] = _rate(sum(bool(r["reacquired"]) for r in responded), len(responded))
    return out
