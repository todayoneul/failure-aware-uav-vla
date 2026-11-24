"""Config 3's first pick in flight (docs/demo1_plan.md 2.1, docs/demo1.md 8): the VLM names one detected car.

The pick sees only what the drone has: the RGB frame, the detector's boxes and the sentence. Ground truth is used
afterwards, to score it. The simulator waits for the VLM, so the pick's time is charged in simulated time: the drone
hovers with no lock for `PICK_DELAY_S` (a fixed budget, so that flights repeat) while the tracker keeps running, and
then follows the picked track. A pick that took longer than the budget is counted in the summary.

The VLM chooses among all boxes of the saved-frame evaluation, including cars the depth camera cannot see (it looks
less steeply down than the RGB camera, so the cars nearest the drone fall outside it). The tracker needs a world
position, so a picked box without one is carried in the image (`Handoff`, the drone is hovering) until a detection of
that car has a position.

The ambiguity signal is recorded and changes nothing in the flight (the monitor is a later stage):
    clear           at least two candidates and log(p1 / p2) >= the threshold fixed on VLM dev (docs/demo1.md 7.5)
    ambiguous       at least two candidates and log(p1 / p2) below it
    cannot_verify   one candidate: its margin is 1 by definition and says nothing, so none is recorded
"""
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

from . import detector, vlm

CANDIDATE_SCORE = 0.3      # as in the saved-frame evaluation (scripts/demo1_vlm_eval.py)
PICK_DELAY_S = 1.25        # the pick took 1.19 s next to the simulator (docs/demo1.md 7.6)
HANDOFF_IOU = 0.3          # the picked box continues on the detection overlapping it most, at least this much
HANDOFF_PATIENCE_S = 1.5   # as long as the tracker's buffer: no matching detection for longer gives the pick up
CLEAR, AMBIGUOUS, CANNOT_VERIFY = "clear", "ambiguous", "cannot_verify"


@dataclass
class Request:
    """One flight's pick: the model, the sentence, and (once made) the record of the pick."""
    model: object
    sentence: str
    seed: int
    threshold: float
    prompt: str = "v2"
    permutations: int = 3
    delay_s: float = PICK_DELAY_S
    record: dict = None


def seed_for(trial_id, lang):
    """The same shuffle seed as the saved-frame evaluation."""
    return zlib.crc32(f"{trial_id}/{lang}".encode())


def candidate_indices(dets):
    """Indices of the detections the VLM may choose from, highest score first: score >= `CANDIDATE_SCORE`, at most
    `vlm.MAX_CANDIDATES`, with or without a world position (the same candidates as the saved-frame evaluation)."""
    usable = [i for i, d in enumerate(dets) if d.score >= CANDIDATE_SCORE]
    keep = vlm.candidates([dets[i].box for i in usable], [dets[i].score for i in usable])
    return [usable[j] for j in keep]


def signal(candidates, log_ratio, threshold):
    if candidates < 2:
        return CANNOT_VERIFY
    return CLEAR if log_ratio >= threshold else AMBIGUOUS


def choose(request, bgr, dets):
    """Ask the VLM which of `dets` the sentence names. Returns the chosen detection (None: no candidate on this tick,
    ask again on the next) and fills `request.record`."""
    keep = candidate_indices(dets)
    if not keep:
        return None
    r = vlm.pick(request.model, bgr, [dets[i].box for i in keep], request.sentence, request.permutations,
                 request.prompt, seed=request.seed)
    n = len(keep)
    request.record = {
        "candidates": n,
        "boxes": [[round(float(v), 2) for v in dets[i].box] for i in keep],
        "scores": [round(float(dets[i].score), 4) for i in keep],
        "top": r.top,
        "probs": r.probs.round(4).tolist(),
        "margin": round(r.margin, 4) if n >= 2 else None,
        "log_ratio": round(r.log_ratio, 3) if n >= 2 else None,
        "agreement": round(r.agreement, 3),
        "signal": signal(n, r.log_ratio, request.threshold),
        "seconds": round(r.seconds, 3),
    }
    return dets[keep[r.top]]


class Handoff:
    """Carries a picked box through the image until a detection of that car has a world position."""

    def __init__(self, box, dt, min_iou=HANDOFF_IOU, patience_s=HANDOFF_PATIENCE_S):
        self.box, self.min_iou = box, min_iou
        self.patience = int(round(patience_s / dt))
        self.missed = self.ticks = 0

    @property
    def lost(self):
        return self.missed > self.patience

    def step(self, dets):
        """One tick: the detection of the picked car if the tracker can hold it, else None."""
        self.ticks += 1
        best, value = None, 0.0
        for d in dets:
            v = detector.iou(d.box, self.box)
            if v > value:
                best, value = d, v
        if best is None or value < self.min_iou:
            self.missed += 1
            return None
        self.missed = 0
        self.box = best.box
        return best if best.position is not None else None


def load_threshold(path):
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["log_ratio"])


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def summarize(rows, delay_s=PICK_DELAY_S):
    """The picks of one language: `rows` are scored trials {id: row} with row["pick"] (None if no pick was made)."""
    picks = {k: r.get("pick") for k, r in rows.items()}
    made = {k: p for k, p in picks.items() if p}
    good = {k: bool(p["right"] and p.get("track_started", True)) for k, p in made.items()}
    by_signal = {}
    for name in (CLEAR, AMBIGUOUS, CANNOT_VERIFY):
        part = [p for p in made.values() if p["signal"] == name]
        by_signal[name] = {"n": len(part), "right": sum(good[k] for k, p in made.items() if p["signal"] == name)}
    seconds = [p["seconds"] for p in made.values()]
    held = sum(r.get("held_after_lock", 0) for r in rows.values())
    visible = sum(r.get("visible_after_lock", 0) for r in rows.values())
    return {
        "picks": len(made),
        "no_pick": sorted(k for k, p in picks.items() if not p),
        "right": sum(good.values()),
        "wrong": [{"trial": k, "picked": p.get("picked_kind"), "target_was_a_candidate": p.get("target_is_candidate"),
                   "track_started": p.get("track_started", True), "signal": p["signal"], "margin": p["margin"],
                   "log_ratio": p["log_ratio"]} for k, p in sorted(made.items()) if not good[k]],
        "picked_without_position": sorted(k for k, p in made.items() if p.get("picked_has_position") is False),
        "handoff_lost": sorted(k for k, p in made.items() if p.get("handoff_lost")),
        "target_not_a_candidate": sorted(k for k, p in made.items() if not p.get("target_is_candidate")),
        "by_signal": by_signal,
        "seconds_mean": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "seconds_max": round(max(seconds), 3) if seconds else None,
        "over_delay_budget": sum(s > delay_s for s in seconds),
        "hold_after_lock": round(held / visible, 4) if visible else None,
    }
