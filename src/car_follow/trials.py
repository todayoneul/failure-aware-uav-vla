"""Demo 1 trials: a named car with 2-3 look-alikes (same blueprint, same colour) close by in Town05, and the sentence
that names it (docs/phase2_demo1_plan.md 2.1, 2.5; docs/phase2_demo1.md 1).

A trial is a plain dict, written to configs/phase2/demo1_trials.json (main: demo1_trials_main.json, sealed):
    id, split, seed, map, duration_s
    target       blueprint, color, transform, road, lane, index_from_left, lanes
    lookalikes   [{color, transform, lane_offset, along_m}]  same blueprint and colour as the target
    background   [{blueprint, color, transform}]             other traffic, never the target's type in its colour family
    drone_start  transform 20 m behind the target along its lane and 15 m above, looking at it
    sentence     en, ko, expression (lane, front or back)
    check        what the drone camera saw at the start: pixels of the target and of each look-alike
A transform is [x, y, z, pitch, yaw, roll] (CARLA: metres and degrees).

The sentence must name one car among the matching cars the drone sees at the start (the target and the look-alikes in
view): it gives the target's lane when no other visible match shares it, or says the target is the furthest ahead or the
furthest back when it leads or trails every other visible match by `ORDER_MARGIN_M`. Each trial is built for one of the
three (`INTENTS`: the look-alikes go only where that expression stays true) and is dropped if what the drone sees does
not allow it. Colours are named by family (white, gray, black, red, ...); a target only takes a colour whose family
is clear.
"""
import colorsys
import hashlib
import json
import math
from pathlib import Path

import carla
import numpy as np

from . import sim

ROOT = Path(__file__).resolve().parents[2]
TYPES_PATH = ROOT / "configs" / "phase2" / "vehicle_types.json"
TRIALS_PATH = ROOT / "configs" / "phase2" / "demo1_trials.json"
MAIN_PATH = ROOT / "configs" / "phase2" / "demo1_trials_main.json"

MAP = "Town05_Opt"
DURATION_S = 60.0
FOLLOW_BEHIND_M, FOLLOW_ABOVE_M = 20.0, 15.0
CAMERA = {"width": 960, "height": 540, "fov": 90.0}
SAME_LANE_ALONG_M = (-25.0, -18.0, -12.0, 12.0, 18.0, 25.0)
SIDE_LANE_ALONG_M = (-20.0, -12.0, -6.0, 0.0, 6.0, 12.0, 20.0, 25.0)
MIN_GAP_M = 8.0
FREE_AHEAD_M, FREE_BEHIND_M = 35.0, 30.0
BACKGROUND_COUNT, BACKGROUND_CLEARANCE_M = 40, 30.0
CELL_M = 60.0
# The expressions the trials of a split are built for, in turn: half name the lane, a quarter the front, a quarter
# the back (30 trials: 16, 7, 7).
INTENTS = ("lane", "lane", "front", "back")
TARGET_MIN_PIXELS, VISIBLE_MIN_PIXELS = 300, 30
ORDER_MARGIN_M = 6.0

COLOR_KO = {"white": "흰색", "gray": "회색", "black": "검은색", "red": "빨간색", "orange": "주황색",
            "yellow": "노란색", "green": "초록색", "blue": "파란색", "purple": "보라색"}
HUES = (("red", -20.0, 15.0), ("orange", 15.0, 40.0), ("yellow", 40.0, 70.0), ("green", 70.0, 170.0),
        ("blue", 170.0, 260.0), ("purple", 260.0, 340.0))


# --- names and sentences (no simulator) ---

def load_types(path=TYPES_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))["vehicles"]


def hsv(color):
    """(hue in degrees, saturation, value) of a CARLA colour string 'R,G,B'."""
    r, g, b = (int(c) / 255.0 for c in color.split(","))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s, v


def _hue_family(h):
    h = h - 360.0 if h >= 340.0 else h
    for name, low, high in HUES:
        if low <= h < high:
            return name, min(h - low, high - h)
    raise ValueError(h)


def color_family(color):
    """The colour's family name: white, gray, black, or a hue (red, orange, yellow, green, blue, purple)."""
    h, s, v = hsv(color)
    if v < 0.2:
        return "black"
    if s < 0.2:
        return "white" if v >= 0.78 else "gray" if v >= 0.25 else "black"
    return _hue_family(h)[0]


def clear_color(color):
    """Whether the colour clearly looks like its family (dark, pale and in-between colours are not clear)."""
    h, s, v = hsv(color)
    family = color_family(color)
    if family == "black":
        return v <= 0.15 and (s <= 0.3 or v <= 0.08)
    if family == "white":
        return v >= 0.83 and s <= 0.12
    if family == "gray":
        # In Town05's light grays above 0.5 render nearly white and grays below 0.38 nearly black seen from the drone
        # (docs/phase2_demo1.md 1.2).
        return s <= 0.12 and 0.38 <= v <= 0.5
    margin = _hue_family(h)[1]
    return s >= 0.5 and v >= (0.6 if family in ("orange", "yellow") else 0.35) and margin >= 5.0


def lane_name(index_from_left, lanes, language):
    """Name of a lane counted from the left (0) among `lanes` lanes going the same way."""
    if language == "en":
        if lanes == 2:
            return ("left lane", "right lane")[index_from_left]
        if lanes == 3:
            return ("left lane", "middle lane", "right lane")[index_from_left]
        ordinal = ("", "second", "third", "fourth")
        if index_from_left == 0:
            return "leftmost lane"
        if index_from_left == lanes - 1:
            return "rightmost lane"
        if index_from_left < lanes / 2:
            return f"{ordinal[index_from_left]} lane from the left"
        return f"{ordinal[lanes - 1 - index_from_left]} lane from the right"
    if lanes == 2:
        return ("왼쪽 차선", "오른쪽 차선")[index_from_left]
    if lanes == 3:
        return ("왼쪽 차선", "가운데 차선", "오른쪽 차선")[index_from_left]
    ordinal = ("", "두 번째", "세 번째", "네 번째")
    if index_from_left == 0:
        return "맨 왼쪽 차선"
    if index_from_left == lanes - 1:
        return "맨 오른쪽 차선"
    if index_from_left < lanes / 2:
        return f"왼쪽에서 {ordinal[index_from_left]} 차선"
    return f"오른쪽에서 {ordinal[lanes - 1 - index_from_left]} 차선"


def expressions(target, others, margin=ORDER_MARGIN_M):
    """Ways to name the target among the other visible matches: 'lane', 'front', 'back' (possibly none).

    Each of target and others has index_from_left, lanes and along_m (metres ahead of the target along its lane)."""
    found = []
    if all(o["lanes"] == target["lanes"] and o["index_from_left"] != target["index_from_left"] for o in others):
        found.append("lane")
    if all(target["along_m"] - o["along_m"] >= margin for o in others):
        found.append("front")
    if all(o["along_m"] - target["along_m"] >= margin for o in others):
        found.append("back")
    return found


def sentence(expression, entry, color, index_from_left, lanes):
    """The English and Korean sentences that name the target (`entry` is its vehicle_types.json entry)."""
    family = color_family(color)
    en_car, ko_car = f"{family} {entry['type']}", f"{COLOR_KO[family]} {entry['ko_obj']}"
    if expression == "lane":
        return {"en": f"follow the {en_car} in the {lane_name(index_from_left, lanes, 'en')}",
                "ko": f"{lane_name(index_from_left, lanes, 'ko')}의 {ko_car} 따라가", "expression": expression}
    if expression == "front":
        return {"en": f"follow the {en_car} furthest ahead", "ko": f"가장 앞에 있는 {ko_car} 따라가",
                "expression": expression}
    if expression == "back":
        return {"en": f"follow the {en_car} furthest back", "ko": f"가장 뒤에 있는 {ko_car} 따라가",
                "expression": expression}
    raise ValueError(expression)


def cell_of(x, y):
    return int(math.floor(x / CELL_M)), int(math.floor(y / CELL_M))


def cell_split(cell, map_name=MAP):
    """Which trials may start in a map cell: 'main' (sealed evaluation), 'gate' (pilot and smoke) or 'dev'.

    Cells, not roads: only a dozen Town05 roads can hold a start, and a road split left two roads for the gates."""
    bucket = int(hashlib.sha256(f"{map_name}:{cell[0]}:{cell[1]}".encode()).hexdigest(), 16) % 4
    return ("main", "gate", "dev", "dev")[bucket]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_trials(splits, main_trials, index_path=TRIALS_PATH, main_path=MAIN_PATH, notes=None):
    """Write the open splits to `index_path` and the main trials to `main_path`, sealed by their sha256."""
    main_path.write_text(json.dumps({"split": "main", "sealed": True, "trials": main_trials}, indent=1,
                                    ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    index = {"map": MAP, "notes": notes or {}, "splits": splits,
             "main": {"file": main_path.name, "count": len(main_trials), "sha256": sha256_file(main_path),
                      "sealed": True}}
    index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def load_trials(split, open_sealed=False, index_path=TRIALS_PATH):
    """The trials of one split. The sealed main split needs `open_sealed` and must match its recorded sha256."""
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if split != "main":
        return index["splits"][split]
    if not open_sealed:
        raise PermissionError("the main trials are sealed until the main evaluation (docs/phase2_demo1_plan.md 3.3)")
    path = Path(index_path).parent / index["main"]["file"]
    if sha256_file(path) != index["main"]["sha256"]:
        raise ValueError(f"{path} does not match its sealed sha256")
    return json.loads(path.read_text(encoding="utf-8"))["trials"]


# --- map geometry (CARLA map, no running world needed beyond the map) ---

def as_list(transform):
    l, r = transform.location, transform.rotation
    return [round(v, 3) for v in (l.x, l.y, l.z, r.pitch, r.yaw, r.roll)]


def as_transform(values):
    x, y, z, pitch, yaw, roll = values
    return carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))


def _same_way(wp, other):
    return (other is not None and other.lane_type == carla.LaneType.Driving and other.road_id == wp.road_id
            and other.lane_id * wp.lane_id > 0)


def lane_layout(wp):
    """(index of the waypoint's lane counted from the left, number of lanes going the same way)."""
    left, cur = 0, wp.get_left_lane()
    while _same_way(wp, cur) and left < 8:
        left, cur = left + 1, cur.get_left_lane()
    right, cur = 0, wp.get_right_lane()
    while _same_way(wp, cur) and right < 8:
        right, cur = right + 1, cur.get_right_lane()
    return left, left + right + 1


def free_of_junctions(wp, distance, step=5.0):
    """Whether the lane runs `distance` metres ahead (positive) or behind (negative) without a junction."""
    cur, gone = wp, 0.0
    while gone < abs(distance):
        nxt = cur.next(step) if distance > 0 else cur.previous(step)
        if not nxt or nxt[0].is_junction:
            return False
        cur, gone = nxt[0], gone + step
    return True


def candidate_waypoints(world_map, spacing=8.0):
    """Waypoints a target may start on: not in a junction, at least two lanes going its way, room ahead and behind."""
    found = []
    for wp in world_map.generate_waypoints(spacing):
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction or lane_layout(wp)[1] < 2:
            continue
        if free_of_junctions(wp, FREE_AHEAD_M) and free_of_junctions(wp, -FREE_BEHIND_M):
            found.append(wp)
    return found


def shifted(wp, lane_offset, along):
    """The waypoint `lane_offset` lanes to the right (negative: left) and `along` metres ahead (negative: behind)."""
    cur = wp
    for _ in range(abs(lane_offset)):
        cur = cur.get_right_lane() if lane_offset > 0 else cur.get_left_lane()
        if not _same_way(wp, cur):
            return None
    if along:
        nxt = cur.next(along) if along > 0 else cur.previous(-along)
        if not nxt:
            return None
        cur = nxt[0]
    return None if cur.is_junction else cur


def spawn_transform(wp):
    t = wp.transform
    return carla.Transform(t.location + carla.Location(z=0.5), t.rotation)


def drone_start(wp):
    """20 m behind the target's waypoint along its lane, 15 m above, looking at it."""
    t = wp.transform
    yaw = math.radians(t.rotation.yaw)
    eye = t.location + carla.Location(-FOLLOW_BEHIND_M * math.cos(yaw), -FOLLOW_BEHIND_M * math.sin(yaw),
                                      FOLLOW_ABOVE_M)
    return carla.Transform(eye, sim.look_at(eye, t.location))


# --- planning one trial ---

def slot_fits(intent, lane_offset, along):
    """Whether a look-alike slot fits the expression the trial is built for."""
    if intent == "lane":
        return lane_offset != 0
    if intent == "front":
        return along <= -ORDER_MARGIN_M
    return along >= ORDER_MARGIN_M


def plan(rng, intent, cells, types, colors, spawn_points):
    """Draw one trial (without id, split, sentence or check) built for the expression `intent`.

    `cells` maps a map cell to its candidate waypoints (a cell is drawn first, so starts spread over the map),
    `colors` maps a blueprint id to its recommended colours and `spawn_points` are the map's spawn transforms."""
    wp = rng.choice(cells[rng.choice(sorted(cells))])
    blueprint = rng.choice(sorted(b for b, e in types.items() if e["target"] and any(map(clear_color, colors[b]))))
    color = rng.choice([c for c in colors[blueprint] if clear_color(c)])
    index, lanes = lane_layout(wp)

    slots = [(k, a) for k in range(-index, lanes - index) for a in (SAME_LANE_ALONG_M if k == 0 else SIDE_LANE_ALONG_M)
             if slot_fits(intent, k, a)]
    rng.shuffle(slots)
    wanted, chosen = rng.choice((2, 3)), []
    for k, along in slots:
        if len(chosen) == wanted:
            break
        if any(k == c["lane_offset"] and abs(along - c["along_m"]) < MIN_GAP_M for c in chosen):
            continue
        slot = shifted(wp, k, along)
        if slot is not None:
            chosen.append({"color": color, "transform": as_list(spawn_transform(slot)), "lane_offset": k,
                           "along_m": along})

    group = [wp.transform.location] + [as_transform(c["transform"]).location for c in chosen]
    free = [sp for sp in spawn_points if min(sp.location.distance(g) for g in group) >= BACKGROUND_CLEARANCE_M]
    rng.shuffle(free)
    family, kind = color_family(color), types[blueprint]["type"]
    background = []
    for sp in free[:BACKGROUND_COUNT]:
        for _ in range(20):
            other = rng.choice(sorted(types))
            other_color = rng.choice(colors[other]) if colors[other] else None
            if not (types[other]["type"] == kind and other_color and color_family(other_color) == family):
                break
        background.append({"blueprint": other, "color": other_color, "transform": as_list(sp)})

    trial = {"map": MAP, "duration_s": DURATION_S,
             "target": {"blueprint": blueprint, "color": color, "transform": as_list(spawn_transform(wp)),
                        "road": wp.road_id, "lane": wp.lane_id, "index_from_left": index, "lanes": lanes},
             "lookalikes": chosen, "background": background, "drone_start": as_list(drone_start(wp))}
    return trial


# --- the simulator side ---

def spawn(world, trial, tm=None):
    """Spawn the trial's cars in a fixed order (target, look-alikes, background) and return
    {'target': actor or None, 'lookalikes': [actor or None], 'background': [actor or None]}.
    With a traffic manager every spawned car drives on autopilot."""
    library = world.get_blueprint_library()

    def one(blueprint, color, transform):
        bp = library.find(blueprint)
        if color and bp.has_attribute("color"):
            bp.set_attribute("color", color)
        bp.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(bp, as_transform(transform))
        if actor is not None and tm is not None:
            actor.set_autopilot(True, tm.get_port())
        return actor

    t = trial["target"]
    return {"target": one(t["blueprint"], t["color"], t["transform"]),
            "lookalikes": [one(t["blueprint"], c["color"], c["transform"]) for c in trial["lookalikes"]],
            "background": [one(b["blueprint"], b["color"], b["transform"]) for b in trial["background"]]}


def start_view(world, trial, actors, ticks=5, rgb=False):
    """Instance pixels per actor (and the RGB image if asked) seen from the drone start after `ticks` ticks."""
    pose = as_transform(trial["drone_start"])
    seg, seg_q = sim.camera(world, "sensor.camera.instance_segmentation", CAMERA["width"], CAMERA["height"],
                            CAMERA["fov"], pose)
    cams = [(seg, seg_q)]
    if rgb:
        cams.append(sim.camera(world, "sensor.camera.rgb", CAMERA["width"], CAMERA["height"], CAMERA["fov"], pose))
    try:
        for _ in range(ticks):
            frame = world.tick()
        images = [sim.frame_at(q, frame) for _, q in cams]
    finally:
        for cam, _ in cams:
            cam.stop()
            cam.destroy()
    ids, counts = np.unique(sim.instance_ids(images[0]), return_counts=True)
    pixels = dict(zip(ids.tolist(), counts.tolist()))
    return pixels, (sim.bgr(images[1]) if rgb else None)


def describe(world_map, trial, actors, pixels):
    """Choose the expression for a spawned trial (or None) and the start check, from what the drone sees."""
    target = actors["target"]
    t = target.get_transform()
    forward = t.get_forward_vector()

    def place(actor):
        wp = world_map.get_waypoint(actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        index, lanes = lane_layout(wp)
        d = actor.get_location() - t.location
        same_road = wp.road_id == trial["target"]["road"] and wp.lane_id * trial["target"]["lane"] > 0
        return {"index_from_left": index if same_road else -1, "lanes": lanes if same_road else -1,
                "along_m": d.x * forward.x + d.y * forward.y}

    look_pixels = [pixels.get(a.id, 0) if a is not None else 0 for a in actors["lookalikes"]]
    visible = [place(a) for a, p in zip(actors["lookalikes"], look_pixels) if a is not None and p >= VISIBLE_MIN_PIXELS]
    me = {"index_from_left": trial["target"]["index_from_left"], "lanes": trial["target"]["lanes"], "along_m": 0.0}
    check = {"target_pixels": pixels.get(target.id, 0), "lookalike_pixels": look_pixels,
             "visible_lookalikes": len(visible)}
    return expressions(me, visible), check
