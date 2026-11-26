"""Stage 3 failure trials (docs/failure_plan.md 4): demo 1 trials with an event put into the traffic while it is
recorded, so every configuration, monitor and response later sees the same event.

Batch A (docs/failure_plan.md table 4):
    A0  normal      the demo 1 trial as it is (twins), no event
    A1  cut-in      a distractor escorting the target in the next lane is sped up past it and steered into its
                    lane ahead
    A2  queue       a distractor right ahead of or behind the target in its lane; the light ahead of the pair is
                    held red until both have stood for HOLD_RED_S, then turned green (main-024's pattern)
    A3  occlusion   a start whose straight-on route passes under a structure (the elevated road), twins near
Distractor levels (A1, A2): other (the sentence's words do not fit it: same car type, another colour family),
near_twin (same car-type word and colour family: another blueprint, or the same blueprint in another RGB) and
twin (same blueprint and colour).

A trial is a demo 1 trial (trials.py) plus
    condition, distractor (level), intent, event {kind, window_s}, distractor_car {blueprint, color, transform,
    lane_offset, along_m, near_twin_way} (A1, A2)
The recording keeps the distractor right after the look-alikes, with kind 'distractor'.

Event scripts run inside the recording loop (scripts/make_failure_trials.py). They only steer the Traffic Manager
and the lights. Whether the event really happened is decided afterwards from the recording alone (check_*), as the
Traffic Manager may not do what it is asked.
"""
import math

import carla
import numpy as np

from . import trials

WINDOW_S = (15.0, 45.0)
CONDITIONS = ("A0", "A1", "A2", "A3")
EVENT_KIND = {"A0": None, "A1": "cut_in", "A2": "queue", "A3": "occlusion"}
LEVELS = ("other", "near_twin", "twin")
ROUTE = ["Straight"] * 40            # the pair takes the same turns
DEFAULT_SPEED_DIFF = 30.0            # the Traffic Manager's own default (% under the speed limit)
TARGET_SLOW_DIFF = 50.0              # the target drives 50% under the limit while it is being overtaken
ESCORT_ALONG_M, ESCORT_GAIN = 8.0, 0.5   # before the cut-in the distractor holds 8 m behind (P gain 1/s)
OVERTAKE_MPS = 6.0                   # how much faster than the target it drives to overtake
CUT_BEHIND_M, CUT_UNTIL_M = 25.0, 20.0   # where the distractor may be (along the target) when the cut-in starts
NEXT_LANE_M = (2.5, 6.0)             # lateral offset of a car in the next lane (Town05 lanes reach 5.4 m)
A1_SLOTS = [(k, a) for k in (-1, 1) for a in (-12.0, -6.0, -20.0)]   # the distractor's start: next lane, behind
A1_FREE_M = 180.0                    # A1 starts: no junction this far along the straight-on route (traffic keeps moving)
CLEAR_AHEAD_M = 20.0                 # no junction this far ahead of either car when the cut-in starts
A2_LIGHT_M = (60.0, 250.0)           # A2 starts: a signalled junction this far along the straight-on route
QUEUE_DISTANCE_M = 4.0               # Traffic Manager distance to the car ahead for the pair (6-10 m centre to centre)
CUT_AHEAD_M = 8.0                    # how far ahead the distractor must be before it is forced across
CUT_CHECK_AHEAD_M = 25.0
OVERTAKE_TIMEOUT_S, CHANGE_TIMEOUT_S = 20.0, 6.0
A2_ALONG_M = (12.0, 18.0)            # the A2 distractor's distance ahead of or behind the target in its lane
LIGHT_AHEAD_M = 60.0
PAIR_WITHIN_M = 25.0                 # the pair counts as together in one lane this close (the gap grows on the way)
HOLD_RED_S, MAX_RED_S = 6.0, 25.0
LANE_HALF_M, SAME_WAY_DEG = 1.75, 20.0
QUEUE_GAP_M, QUEUE_STOP_S, STOP_MPS, GO_MPS = 10.0, 3.0, 1.0, 2.0
OCCLUDED_S = 2.0
# Lights slow the targets: in the pilot they had driven 35-117 m by 15 s and 105-372 m by 45 s. The recording decides.
UNDERPASS_FROM_M, UNDERPASS_TO_M, UNDERPASS_MIN_M = 60.0, 140.0, 8.0
STRUCTURES = {carla.CityObjectLabel.Bridge, carla.CityObjectLabel.Roads, carla.CityObjectLabel.Sidewalks,
              carla.CityObjectLabel.Buildings, carla.CityObjectLabel.Static, carla.CityObjectLabel.GuardRail,
              carla.CityObjectLabel.Fences, carla.CityObjectLabel.Walls}


# --- planning (no simulator beyond the map) ---

def distractor_car(rng, level, blueprint, color, types, colors):
    """(blueprint, colour, near-twin way or None) of a distractor at `level`, or None if there is none."""
    kind, family = types[blueprint]["type"], trials.color_family(color)
    clear = {b: [c for c in colors[b] if trials.clear_color(c)] for b in sorted(types) if types[b]["target"]}
    if level == "twin":
        return blueprint, color, None
    if level == "near_twin":
        ways = {"blueprint": [(b, c) for b, cs in clear.items() if b != blueprint and types[b]["type"] == kind
                              for c in cs if trials.color_family(c) == family],
                "rgb": [(blueprint, c) for c in clear.get(blueprint, []) if c != color
                        and trials.color_family(c) == family]}
        order = ["blueprint", "rgb"]
        rng.shuffle(order)
        for way in order:
            if ways[way]:
                b, c = rng.choice(ways[way])
                return b, c, way
        return None
    options = [(b, c) for b, cs in clear.items() if types[b]["type"] == kind for c in cs
               if trials.color_family(c) != family]
    if not options:
        return None
    b, c = rng.choice(options)
    return b, c, None


def plan(rng, condition, level, world_map, cells, types, colors, spawn_points):
    """Draw one trial of `condition` (without id, split, sentence or check), or None if this draw does not fit."""
    if condition == "A2":
        intent = rng.choice(("front", "back"))
    elif condition == "A1" and level != "other":
        intent = rng.choice(("lane", "lane", "front"))   # a word-match car behind the target rules out 'back'
    else:
        intent = rng.choice(trials.INTENTS)
    trial = trials.plan(rng, intent, cells, types, colors, spawn_points)
    trial.update(condition=condition, distractor=level if condition in ("A1", "A2") else "twin", intent=intent,
                 event=None if EVENT_KIND[condition] is None else {"kind": EVENT_KIND[condition],
                                                                   "window_s": list(WINDOW_S)})
    if condition not in ("A1", "A2"):
        return trial
    t = trial["target"]
    car = distractor_car(rng, level, t["blueprint"], t["color"], types, colors)
    if car is None:
        return None
    wp = world_map.get_waypoint(trials.as_transform(t["transform"]).location)
    match = level != "other"
    side = 1.0 if intent == "back" else -1.0      # a word-match car ahead of the target needs 'back', behind 'front'
    slots = list(A1_SLOTS) if condition == "A1" else [(0, side * a) for a in A2_ALONG_M]
    rng.shuffle(slots)
    for k, along in slots:
        if match and not trials.slot_fits(intent, k, along):
            continue
        if any(k == c["lane_offset"] and abs(along - c["along_m"]) < trials.MIN_GAP_M for c in trial["lookalikes"]):
            continue
        if condition == "A2" and any(c["lane_offset"] == 0 and 0 < c["along_m"] / along < 1
                                     for c in trial["lookalikes"]):
            continue   # a look-alike between the pair
        slot = trials.shifted(wp, k, along)
        if slot is None:
            continue
        trial["distractor_car"] = {"blueprint": car[0], "color": car[1], "near_twin_way": car[2],
                                   "transform": trials.as_list(trials.spawn_transform(slot)), "lane_offset": k,
                                   "along_m": along}
        return trial
    return None


def straight_route(wp, length=600.0, step=4.0):
    """Waypoints along the lane, taking at each fork the successor that turns least (a 'Straight' route)."""
    out, cur, gone = [wp], wp, 0.0
    while gone < length:
        nxt = cur.next(step)
        if not nxt:
            break
        yaw = cur.transform.rotation.yaw
        cur = min(nxt, key=lambda w: abs((w.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0))
        out.append(cur)
        gone += step
    return out


def junction_free(route, step=4.0):
    """How far (m) the route runs before its first junction."""
    for i, w in enumerate(route):
        if w.is_junction:
            return i * step
    return len(route) * step


def light_ahead(world, wp, distance=LIGHT_AHEAD_M, step=2.0):
    """(traffic light, metres) of the first signalled junction within `distance` along the lane from `wp`, matched
    by the light's affected lanes; (None, None) if there is none. (World.get_traffic_lights_from_waypoint rarely
    found one in the pilot.)"""
    cur, gone = wp, 0.0
    while gone <= distance:
        nxt = cur.next(step)
        if not nxt:
            return None, None
        if nxt[0].is_junction and not cur.is_junction:
            # A light's stop waypoints lie on the lane before the junction, its affected waypoints inside it.
            for light in world.get_traffic_lights_in_junction(nxt[0].get_junction().id):
                if any(s.road_id == cur.road_id and s.lane_id == cur.lane_id for s in light.get_stop_waypoints()) \
                        or any(a.road_id == nxt[0].road_id and a.lane_id == nxt[0].lane_id
                               for a in light.get_affected_lane_waypoints()):
                    return light, gone
        cur, gone = nxt[0], gone + step
    return None, None


def covered(world, location, height=40.0):
    """Whether a structure is above `location` (a ray straight up hits a bridge, road deck, building ...)."""
    hits = world.cast_ray(location + carla.Location(z=1.0), location + carla.Location(z=height))
    return any(h.label in STRUCTURES for h in hits)


def underpass_ahead(world, wp, step=4.0):
    """The first stretch of the straight-on route under a structure, as (start m, length m, x, y of its start), or
    None."""
    route = straight_route(wp, UNDERPASS_TO_M + 60.0, step)
    flags = [covered(world, w.transform.location) for w in route]
    for i, f in enumerate(flags):
        if f and (i == 0 or not flags[i - 1]):
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            loc = route[i].transform.location
            return i * step, (j - i) * step, round(loc.x, 1), round(loc.y, 1)
    return None


def spawn(world, trial, tm=None):
    """trials.spawn plus the distractor (spawned last; None if it did not fit)."""
    actors = trials.spawn(world, trial, tm)
    actors["distractor"] = None
    d = trial.get("distractor_car")
    if d:
        bp = world.get_blueprint_library().find(d["blueprint"])
        if d["color"] and bp.has_attribute("color"):
            bp.set_attribute("color", d["color"])
        bp.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(bp, trials.as_transform(d["transform"]))
        if actor is not None and tm is not None:
            actor.set_autopilot(True, tm.get_port())
        actors["distractor"] = actor
    return actors


def word_match_actors(trial, actors, types):
    """The look-alikes, and the distractor when the sentence's words fit it (for the start check)."""
    out = list(actors["lookalikes"])
    d = trial.get("distractor_car")
    if d and actors["distractor"] is not None and trial["distractor"] != "other":
        out.append(actors["distractor"])
    return out


# --- event scripts (inside the recording loop) ---

def _speed(actor):
    v = actor.get_velocity()
    return math.hypot(v.x, v.y)


def relation(a, b):
    """(along, lateral, heading difference in degrees) of actor b in actor a's frame (lateral: + right)."""
    ta, tb = a.get_transform(), b.get_transform()
    f, r = ta.get_forward_vector(), ta.get_right_vector()
    d = tb.location - ta.location
    dyaw = abs((tb.rotation.yaw - ta.rotation.yaw + 180.0) % 360.0 - 180.0)
    return d.x * f.x + d.y * f.y, d.x * r.x + d.y * r.y, dyaw


class Event:
    kind = None

    def __init__(self, trial, dt):
        self.trial = trial
        self.window = trial["event"]["window_s"] if trial.get("event") else list(WINDOW_S)
        self.dt = dt
        self.log = {"state": "wait"}

    def setup(self, world, tm, actors):
        pass

    def step(self, k, world, tm, actors):
        pass

    def block(self, reason):
        """Count, per tick in the window, what kept the event from starting (for tuning the script)."""
        blocked = self.log.setdefault("blocked_ticks", {})
        blocked[reason] = blocked.get(reason, 0) + 1


class NoEvent(Event):
    pass


class CutIn(Event):
    """A1: the distractor starts in the next lane behind the target. Until the cut-in starts it escorts the target:
    every tick its desired speed is the target's plus a correction that holds it ESCORT_ALONG_M behind, so the pair
    stays side by side into the window (left to the Traffic Manager, pairs drifted apart in the pilot). In the window,
    once both move with CLEAR_AHEAD_M of road free of junctions, it drives OVERTAKE_MPS faster than the target (which
    is slowed) until it is CUT_AHEAD_M ahead; then a path along the target's lane steers it across while it keeps the
    target's speed. Afterwards it keeps the target's speed at that moment, and the target its default."""
    kind = "cut_in"

    def setup(self, world, tm, actors):
        for a in (actors["target"], actors["distractor"]):
            tm.set_route(a, list(ROUTE))
            tm.auto_lane_change(a, False)

    @staticmethod
    def drive(tm, d, mps):
        tm.set_desired_speed(d, max(0.0, mps) * 3.6)

    def step(self, k, world, tm, actors):
        t, d, s, now = actors["target"], actors["distractor"], self.log["state"], k * self.dt
        m = world.get_map()
        twp, dwp = m.get_waypoint(t.get_location()), m.get_waypoint(d.get_location())
        along, lateral, dyaw = relation(t, d)
        vt = _speed(t)
        if s == "wait":
            self.drive(tm, d, min(vt + ESCORT_GAIN * (-ESCORT_ALONG_M - along), vt + OVERTAKE_MPS))
            if now < self.window[0]:
                return
            if now > self.window[1]:
                self.log["state"] = "missed: never beside the target on a clear stretch in the window"
                return
            beside = NEXT_LANE_M[0] <= abs(lateral) <= NEXT_LANE_M[1] and dyaw < SAME_WAY_DEG
            blocked = ("not in the next lane" if not beside
                       else "not both moving" if vt <= 2.0 or _speed(d) <= 2.0
                       else "too far behind or ahead" if not -CUT_BEHIND_M <= along <= CUT_UNTIL_M
                       else "in a junction" if twp.is_junction or dwp.is_junction
                       else "junction ahead" if not (trials.free_of_junctions(twp, CLEAR_AHEAD_M)
                                                     and trials.free_of_junctions(dwp, CLEAR_AHEAD_M)) else None)
            if blocked:
                self.block(blocked)
                return
            tm.vehicle_percentage_speed_difference(t, TARGET_SLOW_DIFF)
            self.log.update(state="overtake", trigger_tick=k, trigger_along_m=round(along, 2),
                            side="right" if lateral > 0 else "left")
        elif s == "overtake":
            self.drive(tm, d, vt + OVERTAKE_MPS)
            if (k - self.log["trigger_tick"]) * self.dt > OVERTAKE_TIMEOUT_S:
                tm.vehicle_percentage_speed_difference(t, DEFAULT_SPEED_DIFF)
                self.drive(tm, d, vt)
                self.log["state"] = "failed: did not get ahead"
            elif along >= CUT_AHEAD_M and not dwp.is_junction and not twp.is_junction:
                # Steer it by a path along the target's lane ahead (force_lane_change did not move it in the pilot,
                # with automatic lane changes off).
                # Which neighbour of the distractor's lane is the target's (from the map, not the camera-free offset,
                # which bends on curves). A path along the target's lane worked for half the changes in the pilot;
                # force_lane_change (True: to the right, as documented), repeated every tick of the change, with
                # other vehicles ignored for the change only (a cut-in is not a safe lane change).
                right = dwp.get_right_lane()
                to_right = right is not None and right.road_id == twp.road_id and right.lane_id == twp.lane_id
                tm.vehicle_percentage_speed_difference(t, DEFAULT_SPEED_DIFF)
                tm.auto_lane_change(d, True)
                tm.ignore_vehicles_percentage(d, 100.0)
                tm.force_lane_change(d, to_right)
                self.log.update(state="change", change_tick=k, change_along_m=round(along, 2),
                                to="right" if to_right else "left", from_lane=dwp.lane_id, to_lane=twp.lane_id)
        elif s == "change":
            self.drive(tm, d, vt + 1.0)
            tm.force_lane_change(d, self.log["to"] == "right")
            if (dwp.road_id == twp.road_id and dwp.lane_id == twp.lane_id) or abs(lateral) < 1.0:
                tm.auto_lane_change(d, False)
                tm.ignore_vehicles_percentage(d, 0.0)
                self.log.update(state="done", cut_tick=k, cut_along_m=round(along, 2))
            elif (k - self.log["change_tick"]) * self.dt > CHANGE_TIMEOUT_S:
                tm.auto_lane_change(d, False)
                tm.ignore_vehicles_percentage(d, 0.0)
                self.log["state"] = "failed: no lane change"


class Queue(Event):
    """A2: wait for the pair in one lane with the light ahead of the leader, hold that light red until both have
    stood for HOLD_RED_S, then turn it green."""
    kind = "queue"

    def setup(self, world, tm, actors):
        for a in (actors["target"], actors["distractor"]):
            tm.set_route(a, list(ROUTE))
            tm.auto_lane_change(a, False)
            tm.distance_to_leading_vehicle(a, QUEUE_DISTANCE_M)
        self.light = None
        self.stood = 0

    def between(self, world, t, d, along):
        lo, hi = sorted((0.0, along))
        for v in world.get_actors().filter("vehicle.*"):
            if v.id in (t.id, d.id):
                continue
            a, lat, _ = relation(t, v)
            if abs(lat) < LANE_HALF_M and lo < a < hi:
                return True
        return False

    def release(self, world, k, state):
        world.freeze_all_traffic_lights(False)
        self.light.set_state(carla.TrafficLightState.Green)
        for other in self.light.get_group_traffic_lights():
            if other.id != self.light.id:
                other.set_state(carla.TrafficLightState.Red)
        self.log.update(state=state, release_tick=k)

    def step(self, k, world, tm, actors):
        t, d, s, now = actors["target"], actors["distractor"], self.log["state"], k * self.dt
        if s == "wait":
            if now < self.window[0]:
                return
            if now > self.window[1]:
                self.log["state"] = "missed: no light ahead of the pair in the window"
                return
            m = world.get_map()
            twp, dwp = m.get_waypoint(t.get_location()), m.get_waypoint(d.get_location())
            along, lateral, dyaw = relation(t, d)
            # One lane by geometry: a pair 12-18 m apart often sits on two road segments with different map ids.
            same_lane = abs(lateral) < LANE_HALF_M and dyaw < SAME_WAY_DEG
            blocked = ("not in one lane" if not same_lane else "too far apart" if abs(along) > PAIR_WITHIN_M
                       else "in a junction" if twp.is_junction or dwp.is_junction
                       else "a car between" if self.between(world, t, d, along) else None)
            if blocked:
                self.block(blocked)
                return
            leader = d if along > 0 else t
            light, _ = light_ahead(world, m.get_waypoint(leader.get_location()))
            if light is None:
                self.block("no light ahead")
            else:
                self.light = light
                self.light.set_state(carla.TrafficLightState.Red)
                world.freeze_all_traffic_lights(True)
                self.log.update(state="red", trigger_tick=k, light=self.light.id, pair_along_m=round(along, 2),
                                leader="distractor" if leader is d else "target")
        elif s == "red":
            self.stood = self.stood + 1 if max(_speed(t), _speed(d)) < 0.5 else 0
            if self.stood * self.dt >= HOLD_RED_S:
                self.release(world, k, "done")
            elif (k - self.log["trigger_tick"]) * self.dt > MAX_RED_S:
                self.release(world, k, "failed: the pair did not stand together")


class Occlusion(Event):
    """A3: nothing is steered beyond the straight-on route; the occlusion is looked for from the nominal follow pose
    (OcclusionProbe) and checked afterwards."""
    kind = "occlusion"

    def setup(self, world, tm, actors):
        tm.set_route(actors["target"], list(ROUTE))


SCRIPTS = {None: NoEvent, "cut_in": CutIn, "queue": Queue, "occlusion": Occlusion}


def script(trial, dt):
    return SCRIPTS[(trial.get("event") or {}).get("kind")](trial, dt)


class OcclusionProbe:
    """Whether a structure stands between the nominal follow pose (FOLLOW_BEHIND_M behind along the target's heading,
    FOLLOW_ABOVE_M up) and the target's box centre, every `every` ticks (a ray, no camera)."""

    def __init__(self, ticks, every=5):
        self.every = every
        self.flags = np.zeros(ticks, bool)
        self.sampled = np.zeros(ticks, bool)

    def step(self, k, world, target):
        if k % self.every:
            return
        tr = target.get_transform()
        aim = tr.transform(target.bounding_box.location)
        f = tr.get_forward_vector()
        eye = carla.Location(tr.location.x - trials.FOLLOW_BEHIND_M * f.x, tr.location.y - trials.FOLLOW_BEHIND_M * f.y,
                             tr.location.z + trials.FOLLOW_ABOVE_M)
        reach = eye.distance(aim) - 3.0
        hits = world.cast_ray(eye, aim)
        self.sampled[k] = True
        self.flags[k] = any(h.label in STRUCTURES and eye.distance(h.location) < reach for h in hits)

    def per_tick(self):
        """The samples held until the next one."""
        idx = np.flatnonzero(self.sampled)
        out = np.zeros(len(self.flags), bool)
        for a, b in zip(idx, list(idx[1:]) + [len(out)]):
            out[a:b] = self.flags[a]
        return out


# --- checks on the recording alone (pure numpy) ---

def pair_relation(rec, i, j):
    """Per tick: (along, lateral, heading difference) of car j in car i's frame (lateral: + right), and both speeds."""
    pos, rot, vel = (np.asarray(rec[k], dtype=float) for k in ("pos", "rot", "vel"))
    yaw = np.radians(rot[:, i, 1])
    d = pos[:, j, :2] - pos[:, i, :2]
    along = d[:, 0] * np.cos(yaw) + d[:, 1] * np.sin(yaw)
    lateral = -d[:, 0] * np.sin(yaw) + d[:, 1] * np.cos(yaw)
    dyaw = np.abs((rot[:, j, 1] - rot[:, i, 1] + 180.0) % 360.0 - 180.0)
    speed = np.linalg.norm(vel[:, :, :2], axis=2)
    return along, lateral, dyaw, speed[:, i], speed[:, j]


def _runs(mask):
    mask = np.asarray(mask, dtype=int)
    edges = np.flatnonzero(np.diff(np.r_[0, mask, 0]))
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def check_cut_in(rec, d, window=WINDOW_S, dt=0.05):
    """The distractor entered the target's lane ahead (within CUT_CHECK_AHEAD_M) from the next lane and stayed for
    at least 1 s, entering between the window's start and 15 s after its end."""
    along, lateral, dyaw, _, _ = pair_relation(rec, 0, d)
    in_lane = (np.abs(lateral) < LANE_HALF_M) & (dyaw < SAME_WAY_DEG) & (along > 0) & (along <= CUT_CHECK_AHEAD_M)
    beside = (np.abs(lateral) > 2.0) & (np.abs(lateral) < 5.5) & (dyaw < SAME_WAY_DEG)
    lo, hi = int(window[0] / dt), int((window[1] + 15.0) / dt)
    for s, n in _runs(in_lane):
        if n * dt >= 1.0 and lo <= s <= hi and beside[max(0, s - int(5.0 / dt)):s].any():
            return {"happened": True, "tick": s, "t_s": round(s * dt, 2), "ahead_m": round(float(along[s]), 2)}
    return {"happened": False, "reason": "no entry into the target's lane ahead"}


def check_queue(rec, d, window=WINDOW_S, dt=0.05):
    """Target and distractor stood (both under STOP_MPS) in one lane within QUEUE_GAP_M, with no car between, for
    QUEUE_STOP_S or more starting in the window, and both drove off (over GO_MPS) afterwards."""
    along, lateral, dyaw, vt, vd = pair_relation(rec, 0, d)
    q = (np.abs(lateral) < LANE_HALF_M) & (dyaw < SAME_WAY_DEG) & (np.abs(along) <= QUEUE_GAP_M) & \
        (vt < STOP_MPS) & (vd < STOP_MPS)
    others = [i for i in range(len(rec["kind"])) if i not in (0, d)]
    if others:
        pos, rot = np.asarray(rec["pos"], dtype=float), np.asarray(rec["rot"], dtype=float)
        yaw = np.radians(rot[:, 0, 1])
        rel = pos[:, others, :2] - pos[:, :1, :2]
        a = rel[:, :, 0] * np.cos(yaw)[:, None] + rel[:, :, 1] * np.sin(yaw)[:, None]
        lat = -rel[:, :, 0] * np.sin(yaw)[:, None] + rel[:, :, 1] * np.cos(yaw)[:, None]
        lo, hi = np.minimum(0.0, along)[:, None], np.maximum(0.0, along)[:, None]
        q &= ~((np.abs(lat) < LANE_HALF_M) & (a > lo) & (a < hi)).any(axis=1)
    w0, w1 = int(window[0] / dt), int(window[1] / dt)
    for s, n in _runs(q):
        if n * dt >= QUEUE_STOP_S and w0 <= s <= w1:
            after = slice(s + n, min(len(vt), s + n + int(10.0 / dt)))
            if (vt[after] > GO_MPS).any() and (vd[after] > GO_MPS).any():
                return {"happened": True, "tick": s, "t_s": round(s * dt, 2), "stood_s": round(n * dt, 2),
                        "gap_m": round(float(np.abs(along[s])), 2)}
            return {"happened": False, "reason": "stood together but did not both drive off"}
    return {"happened": False, "reason": "never stood together in one lane"}


def check_occlusion(occluded, window=WINDOW_S, dt=0.05):
    """A structure hid the target from the nominal follow pose for OCCLUDED_S or more, starting in the window."""
    w0, w1 = int(window[0] / dt), int(window[1] / dt)
    for s, n in _runs(occluded):
        if n * dt >= OCCLUDED_S and w0 <= s <= w1:
            return {"happened": True, "tick": s, "t_s": round(s * dt, 2), "hidden_s": round(n * dt, 2)}
    return {"happened": False, "reason": "no occlusion of 2 s or more starting in the window"}


def check(trial, rec, occluded=None, dt=0.05):
    kind = (trial.get("event") or {}).get("kind")
    if kind is None:
        return {"happened": None}
    d = int(np.flatnonzero(np.asarray(rec["kind"]) == "distractor")[0]) if kind in ("cut_in", "queue") else None
    if kind == "cut_in":
        return check_cut_in(rec, d, trial["event"]["window_s"], dt)
    if kind == "queue":
        return check_queue(rec, d, trial["event"]["window_s"], dt)
    return check_occlusion(occluded, trial["event"]["window_s"], dt)
