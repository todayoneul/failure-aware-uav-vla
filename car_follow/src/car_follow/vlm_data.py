"""Training examples for the VLM first pick (docs/demo1.md 7): who can be named in a saved scene, and how.

A training scene (scripts/demo1_vlm_train_scenes.py) records every car of the look-alike group (same blueprint and
colour) with its lane and its position along the road. Any member the drone sees well can be the target when a
sentence names it alone, by the rule the trial lists use (`trials.expressions`), so one scene gives several examples.

Sentences come from `TEMPLATES`; the first template of each expression is the trial lists' own wording. `HELD_OUT`
wordings are never trained on: they measure how far the trained model carries over to commands it has not seen.

`unseen_cell` keeps about a quarter of the dev map cells out of training, so that the part of the VLM dev and dev
scenes in those cells measures the model on roads it was not trained on (the rest share roads with training).
"""
import hashlib

from . import trials

UNSEEN_SALT = "vlm-unseen-3"     # fixed before any training result: 7 of the 24 dev cells, 30 VLM dev + 9 dev scenes

TEMPLATES = {
    "en": {
        "lane": ("follow the {car} in the {lane}", "track the {car} in the {lane}",
                 "follow the {car} driving in the {lane}", "go after the {car} in the {lane}",
                 "keep up with the {car} that is in the {lane}"),
        "front": ("follow the {car} furthest ahead", "follow the frontmost {car}",
                  "track the {car} that is furthest ahead", "follow the {car} at the very front",
                  "go after the {car} leading the group"),
        "back": ("follow the {car} furthest back", "follow the rearmost {car}",
                 "track the {car} that is furthest back", "follow the {car} at the very back",
                 "go after the {car} at the tail of the group"),
    },
    "ko": {
        "lane": ("{lane}의 {car} 따라가", "{lane}에 있는 {car} 따라가", "{lane}으로 달리는 {car} 추적해",
                 "{lane}에서 달리는 {car} 따라가 줘", "{lane}의 {car} 쫓아가"),
        "front": ("가장 앞에 있는 {car} 따라가", "맨 앞의 {car} 따라가", "제일 앞에 있는 {car} 추적해",
                  "가장 앞서가는 {car} 따라가 줘", "맨 앞에서 달리는 {car} 쫓아가"),
        "back": ("가장 뒤에 있는 {car} 따라가", "맨 뒤의 {car} 따라가", "제일 뒤에 있는 {car} 추적해",
                 "가장 뒤처진 {car} 따라가 줘", "맨 뒤에서 달리는 {car} 쫓아가"),
    },
}

HELD_OUT = {
    "en": {"lane": "stay on the {car} travelling in the {lane}",
           "front": "stay on the {car} out in front of the others",
           "back": "stay on the {car} bringing up the rear"},
    "ko": {"lane": "{lane}을 달리는 {car} 계속 쫓아",
           "front": "다른 차들보다 앞선 {car} 계속 쫓아",
           "back": "다른 차들보다 뒤에 처진 {car} 계속 쫓아"},
}


def unseen_cell(cell):
    """Whether a dev map cell is kept out of training."""
    return int(hashlib.sha256(f"{UNSEEN_SALT}:{cell[0]}:{cell[1]}".encode()).hexdigest(), 16) % 4 == 0


def car_name(lang, family, entry):
    """The car as a sentence names it (Korean: with its object particle)."""
    return f"{family} {entry['type']}" if lang == "en" else f"{trials.COLOR_KO[family]} {entry['ko_obj']}"


def render(template, lang, family, entry, index_from_left, lanes):
    lane = trials.lane_name(index_from_left, lanes, lang) if "{lane}" in template else ""
    return template.format(car=car_name(lang, family, entry), lane=lane)


def other_target(cars, target_id, prefer=None):
    """(car, expression) for a look-alike other than `target_id` that a sentence names alone, or None.

    Each car has id, pixels, index_from_left, lanes, along_m and box (its candidate index, None if not detected).
    The expression `prefer` is used when it applies, so the changed command differs from the original only in whom it
    names."""
    seen = [c for c in cars if c["pixels"] >= trials.VISIBLE_MIN_PIXELS]
    options = []
    for car in seen:
        if car["id"] == target_id or car["box"] is None or car["pixels"] < trials.TARGET_MIN_PIXELS:
            continue
        found = trials.expressions(car, [c for c in seen if c["id"] != car["id"]])
        options += [(car, e) for e in found]
    for option in options:
        if option[1] == prefer:
            return option
    return options[0] if options else None


def nameable(cars):
    """[(car, expression)] for the look-alike group `cars` of a scene: members seen well that a sentence names alone.

    Each car has id, pixels, same_road, index_from_left, lanes and along_m."""
    seen = [c for c in cars if c["pixels"] >= trials.VISIBLE_MIN_PIXELS]
    found = []
    for car in seen:
        if car["pixels"] < trials.TARGET_MIN_PIXELS or not car["same_road"]:
            continue
        others = [c for c in seen if c["id"] != car["id"]]
        if others:
            found += [(car, e) for e in trials.expressions(car, others)]
    return found
