"""Stage 3 failure trials (docs/failure_plan.md 4): demo 1 trials with an event put into the traffic while it is
recorded, so every configuration, monitor and response later sees the same event.

Batch A (docs/failure_plan.md table 4):
    A0  normal      the demo 1 trial as it is (twins), no event
    A1  cut-in      a distractor escorting the target in the next lane is sped up past it and steered into its
                    lane ahead
    A2  queue       a distractor right ahead of or behind the target in its lane; the light ahead of the pair is
                    held red until both have stood for HOLD_RED_S, then turned green (main-024's pattern)
    A3  occlusion   a start whose straight-on route passes under a structure (the elevated road), twins near
Added on the user's request (2026-11-26), outside batch A: the target itself changes lanes (A4)
    A4a lane_single  the target changes lanes once
    A4b lane_weave   the target changes lanes out, back and out again (WEAVE_CHANGES), as soon as the Traffic
                     Manager allows the next change
    A4c lane_swap    a near-twin escorting the target in the next lane, ahead or behind, changes into the target's
                     lane at the same tick as the target changes into its lane: the two cross
Distractor levels (A1, A2): other (the sentence's words do not fit it: same car type, another colour family),
near_twin (same car-type word and colour family: another blueprint, or the same blueprint in another RGB) and
twin (same blueprint and colour). A4c's partner is always a near_twin.

A trial is a demo 1 trial (trials.py) plus
    condition, distractor (level), intent, event {kind, window_s}, distractor_car {blueprint, color, transform,
    lane_offset, along_m, near_twin_way} (A1, A2, A4c); A4 events also carry first_side (left or right)
The recording keeps the distractor right after the look-alikes, with kind 'distractor'. A4 recordings also keep,
per tick, the map lane of the target, the look-alikes and the distractor (LaneProbe), so whether a car changed lanes
is read from the recording alone.

Event scripts run inside the recording loop (scripts/make_failure_trials.py). They only steer the Traffic Manager
and the lights. Whether the event really happened is decided afterwards from the recording alone (check_*), as the
Traffic Manager may not do what it is asked.
"""
import math

import carla
import numpy as np

from . import trials

WINDOW_S = (15.0, 45.0)
CONDITIONS = ("A0", "A1", "A2", "A3", "A4a", "A4b", "A4c")
EVENT_KIND = {"A0": None, "A1": "cut_in", "A2": "queue", "A3": "occlusion",
              "A4a": "lane_single", "A4b": "lane_weave", "A4c": "lane_swap"}
LANE_KINDS = ("lane_single", "lane_weave", "lane_swap")
WITH_DISTRACTOR = ("A1", "A2", "A4c")
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
A2_LIGHT_M = (100.0, 250.0)          # A2 starts: a signalled junction this far along the straight-on route. From
                                     # 60 m, with the tight queue 8 of 25 fresh pairs stood at their first light before
                                     # the window (lights 62-82 m ahead); from 102 m the stand began in the window
QUEUE_DISTANCE_M = 1.0               # Traffic Manager distance to the car ahead for the pair: the tight queue, adopted
                                     # 2026-11-26 (docs/failure.md 8.7). 4.0 stood them 6.8-8.6 m apart centre to
                                     # centre, never overlapping in the image; 1.0 stands them 3.9-5.7 m apart
A2_AHEAD_MATCHED = 2.0 / 3.0         # word-match distractors (twin, near-twin) stand ahead of the target ('furthest
                                     # back') this often, else behind ('furthest ahead'); 'other' ones always ahead:
                                     # the tight queue's switches all came with the distractor ahead (8.4)
CUT_AHEAD_M = 8.0                    # how far ahead the distractor must be before it is forced across
# Tight variant (proposed after step 4, docs/failure.md 8): at the nominal follow pose two cars in one lane overlap in
# the image only within about 6-7 m centre to centre (0-2 m bumper gap); the default events stop at 6.8-11 m. A trial
# carries the tight values in its event (cut_ahead_m, change_speed_extra_mps, target_follow_m, queue_distance_m).
TIGHT = {"A1": {"cut_ahead_m": 4.5, "change_speed_extra_mps": 0.0, "target_follow_m": 1.5}}   # tested, not adopted
# (the tight A2 queue is now the default, QUEUE_DISTANCE_M)
# A cut-in ahead overlaps little even when tight (the camera is behind): the cut-in behind puts the distractor between
# the drone and the target. It escorts the target ESCORT 5 m behind in the next lane, cuts into the target's lane when
# 3.5-7.5 m behind and then tails it 1 m bumper to bumper (Traffic Manager distance), holding 5 m centre to centre.
CUT_BEHIND = {"cut_side": "behind", "escort_along_m": -5.0, "cut_from_m": [-7.5, -3.5], "tail_m": 1.0}
CUT_CHECK_BEHIND_M = (-9.0, -3.0)    # the distractor in the target's lane this far behind (centre to centre)
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
# A4 (the target changes lanes)
A4_FREE_M = {"A4a": A1_FREE_M, "A4b": 300.0, "A4c": A1_FREE_M}   # starts: no junction this far along the
                                     # straight-on route (weave needs about 36 s of driving; with 180 m half the
                                     # weaves stood at lights before their second change in the tuning run)
A4C_SLOTS = [(k, a) for k in (-1, 1) for a in (-12.0, -9.0, 9.0, 12.0)]   # the near-twin: next lane, ahead/behind
LANE_CLEAR_AHEAD_M = 50.0            # no junction this far ahead when a change starts (a change takes 20-30 m; at
                                     # 30 m a target slowing for a light ignored the forced change in the debug run)
STRAIGHT_AHEAD_M, STRAIGHT_BEHIND_M, STRAIGHT_DEG = 60.0, 30.0, 10.0   # single and swap: a change starts only
                                     # where the lane keeps its heading within 10 deg from 30 m behind to 60 m ahead
                                     # (in the pilot flight a curve right after the change made most of the 'after'
                                     # distance error). Most changes then start just out of a curve, which fills part
                                     # of the 'before' phase; 60 m behind cut that (target yaw rate 4 -> 2 deg/s) but
                                     # also the yield (A4a 7/10, A4c 6/10), so the comparison reports the target's yaw
                                     # rate per phase instead (docs/failure.md 6). Not for weave: its 300 m
                                     # junction-free starts lie mostly on the curved ring road, where 80 m / 8 deg let
                                     # 1 weave of 10 happen
STRAIGHT_KINDS = ("lane_single", "lane_swap")
NUDGE_S = 3.0                        # single, weave: a car beside it this long slows the target until a gap opens
LANE_GAP_M = (6.0, 10.0)             # no other car this close behind / ahead in the lane moved into
OWN_AHEAD_M = 15.0                   # nor this close ahead in the own lane (other cars are ignored for the change)
LANE_HOLD_S = 4.0                    # a change not made in time is tried again after this
# The Traffic Manager ignored a forced change 4-5 s after the last one and took it 14 s after (tuning run): it changes
# lanes again only some distance past the last change (by our reading of its code, 10 x speed). Weave waits for it.
TM_GAP_S, TM_GAP_MARGIN_M = 10.0, 15.0   # with a 5 m margin 2 of 10 second changes still timed out
SLOWING_MPS = 1.0                    # no change starts while the target lost this much speed in the last second
                                     # (braking for a light, it ignored the forced change)
WEAVE_CHANGES = 3
LATE_CHANGE_S = 50.0                 # weave: later changes may start until here (the first starts in the window)
LANE_IGNORE_PCT = 100.0              # other cars ignored for the change only (with 0 the Traffic Manager did not
                                     # change lanes in 6 s in the debug run although the lane was free)
SWAP_ALONG_M = (6.0, 16.0)           # swap: the partner's distance along the target when the change starts
SWAP_WITHIN_S = 2.0                  # swap: the two cars cross into each other's lane this close in time
PARTNER_SWITCH_S, SWAP_OTHER_SIDE_M = 4.0, 9.0   # swap: a partner off its place (PARTNER_OFF_M) this long
PARTNER_OFF_M = 3.0                              # holds 9 m on the other side of the target (a partner stuck 3-5 m
                                                 # ahead never switched with a 6 m tolerance in the dev run)
SETTLE_M = 0.4                       # a car this close to its lane's centre is not changing lanes
SETTLE_CAP_S = 4.0                   # a change's start and end are looked for this far around its crossing
MIN_STAY_S = 1.0                     # a lane entered and left again faster than this is a wobble on the line
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
    ahead = None
    if condition == "A2":
        if level == "other":   # its words do not fit: any expression, the distractor always ahead
            ahead, intent = True, rng.choice(trials.INTENTS)
        else:                  # a word-match car ahead of the target needs 'back', behind it 'front'
            ahead = rng.random() < A2_AHEAD_MATCHED
            intent = "back" if ahead else "front"
    elif condition == "A1" and level != "other":
        intent = rng.choice(("lane", "lane", "front"))   # a word-match car behind the target rules out 'back'
    else:
        intent = rng.choice(trials.INTENTS)
    trial = trials.plan(rng, intent, cells, types, colors, spawn_points)
    trial.update(condition=condition, distractor=level if condition in WITH_DISTRACTOR else "twin", intent=intent,
                 event=None if EVENT_KIND[condition] is None else {"kind": EVENT_KIND[condition],
                                                                   "window_s": list(WINDOW_S)})
    if condition in ("A4a", "A4b"):
        trial["event"]["first_side"] = rng.choice(("left", "right"))
    if condition == "A2":
        trial["event"].update(queue_distance_m=QUEUE_DISTANCE_M, distractor_ahead=ahead)
    if condition not in WITH_DISTRACTOR:
        return trial
    t = trial["target"]
    car = distractor_car(rng, level, t["blueprint"], t["color"], types, colors)
    if car is None:
        return None
    wp = world_map.get_waypoint(trials.as_transform(t["transform"]).location)
    match = level != "other"
    side = 1.0 if ahead else -1.0                 # A2: the distractor ahead of the target or behind it
    slots = (list(A1_SLOTS) if condition == "A1" else list(A4C_SLOTS) if condition == "A4c"
             else [(0, side * a) for a in A2_ALONG_M])
    rng.shuffle(slots)
    for k, along in slots:
        if match and not trials.slot_fits(intent, k, along):
            continue
        if any(k == c["lane_offset"] and abs(along - c["along_m"]) < trials.MIN_GAP_M for c in trial["lookalikes"]):
            continue
        if condition == "A2" and any(c["lane_offset"] == 0 and 0 < c["along_m"] / along < 1
                                     for c in trial["lookalikes"]):
            continue   # a look-alike between the pair
        if condition == "A4c" and any(c["lane_offset"] in (0, k) and abs(c["along_m"] - a) < trials.MIN_GAP_M
                                      for c in trial["lookalikes"] for a in (0.0, along)):
            continue   # a look-alike where one of the two cars moves to
        slot = trials.shifted(wp, k, along)
        if slot is None:
            continue
        trial["distractor_car"] = {"blueprint": car[0], "color": car[1], "near_twin_way": car[2],
                                   "transform": trials.as_list(trials.spawn_transform(slot)), "lane_offset": k,
                                   "along_m": along}
        if condition == "A4c":
            trial["event"]["first_side"] = "right" if k > 0 else "left"
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
        ev = self.trial["event"]
        behind = ev.get("cut_side") == "behind"
        hold = ev.get("escort_along_m", -ESCORT_ALONG_M)
        if behind and s in ("change", "done"):   # stay close behind the target in its lane
            self.drive(tm, d, min(vt + ESCORT_GAIN * (hold - along), vt + OVERTAKE_MPS))
        if s == "wait":
            self.drive(tm, d, min(vt + ESCORT_GAIN * (hold - along), vt + OVERTAKE_MPS))
            if now < self.window[0]:
                return
            if now > self.window[1]:
                self.log["state"] = "missed: never beside the target on a clear stretch in the window"
                return
            beside = NEXT_LANE_M[0] <= abs(lateral) <= NEXT_LANE_M[1] and dyaw < SAME_WAY_DEG
            lo, hi = ev["cut_from_m"] if behind else (-CUT_BEHIND_M, CUT_UNTIL_M)
            blocked = ("not in the next lane" if not beside
                       else "not both moving" if vt <= 2.0 or _speed(d) <= 2.0
                       else "too far behind or ahead" if not lo <= along <= hi
                       else "in a junction" if twp.is_junction or dwp.is_junction
                       else "junction ahead" if not (trials.free_of_junctions(twp, CLEAR_AHEAD_M)
                                                     and trials.free_of_junctions(dwp, CLEAR_AHEAD_M)) else None)
            if blocked:
                self.block(blocked)
                return
            if behind:   # no overtaking: straight into the target's lane behind it
                right = dwp.get_right_lane()
                to_right = right is not None and right.road_id == twp.road_id and right.lane_id == twp.lane_id
                tm.distance_to_leading_vehicle(d, ev["tail_m"])
                tm.auto_lane_change(d, True)
                tm.ignore_vehicles_percentage(d, 100.0)
                tm.force_lane_change(d, to_right)
                self.log.update(state="change", trigger_tick=k, change_tick=k, change_along_m=round(along, 2),
                                to="right" if to_right else "left", from_lane=dwp.lane_id, to_lane=twp.lane_id)
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
            elif along >= self.trial["event"].get("cut_ahead_m", CUT_AHEAD_M) and not dwp.is_junction \
                    and not twp.is_junction:
                # Steer it by a path along the target's lane ahead (force_lane_change did not move it in the pilot,
                # with automatic lane changes off).
                # Which neighbour of the distractor's lane is the target's (from the map, not the camera-free offset,
                # which bends on curves). A path along the target's lane worked for half the changes in the pilot;
                # force_lane_change (True: to the right, as documented), repeated every tick of the change, with
                # other vehicles ignored for the change only (a cut-in is not a safe lane change).
                right = dwp.get_right_lane()
                to_right = right is not None and right.road_id == twp.road_id and right.lane_id == twp.lane_id
                tm.vehicle_percentage_speed_difference(t, DEFAULT_SPEED_DIFF)
                if "target_follow_m" in self.trial["event"]:   # tight: the target closes up behind the cut-in
                    tm.distance_to_leading_vehicle(t, self.trial["event"]["target_follow_m"])
                tm.auto_lane_change(d, True)
                tm.ignore_vehicles_percentage(d, 100.0)
                tm.force_lane_change(d, to_right)
                self.log.update(state="change", change_tick=k, change_along_m=round(along, 2),
                                to="right" if to_right else "left", from_lane=dwp.lane_id, to_lane=twp.lane_id)
        elif s == "change":
            if not behind:
                self.drive(tm, d, vt + ev.get("change_speed_extra_mps", 1.0))
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
            tm.distance_to_leading_vehicle(a, self.trial["event"].get("queue_distance_m", QUEUE_DISTANCE_M))
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


def straight_here(wp, ahead=STRAIGHT_AHEAD_M, behind=STRAIGHT_BEHIND_M, step=4.0):
    """The largest heading change (deg) of the lane from `behind` metres back to `ahead` metres on (the
    successor that turns least at forks)."""
    yaw0, worst = wp.transform.rotation.yaw, 0.0
    for walk, dist in ((lambda w: w.next(step), ahead), (lambda w: w.previous(step), behind)):
        cur, gone = wp, 0.0
        while gone < dist:
            nxt = walk(cur)
            if not nxt:
                break
            yaw = cur.transform.rotation.yaw
            cur = min(nxt, key=lambda w: abs((w.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0))
            worst = max(worst, abs((cur.transform.rotation.yaw - yaw0 + 180.0) % 360.0 - 180.0))
            gone += step
    return worst


def lane_key(wp):
    """An integer naming the lane of a waypoint (road, section, lane), or -1 for None."""
    if wp is None:
        return -1
    return (wp.road_id * 100 + wp.section_id) * 100 + wp.lane_id + 50


def neighbour(wp, side):
    """The driving lane next to `wp` on `side` ('left' or 'right') going the same way, or None."""
    nb = wp.get_left_lane() if side == "left" else wp.get_right_lane()
    if nb is None or nb.lane_type != carla.LaneType.Driving or nb.lane_id * wp.lane_id <= 0:
        return None
    return nb


def lane_state(wp, location):
    """(own key, left key, right key, signed offset of `location` from the lane centre (+ right), lane width)."""
    c, r = wp.transform.location, wp.transform.get_right_vector()
    off = (location.x - c.x) * r.x + (location.y - c.y) * r.y
    return (lane_key(wp), lane_key(neighbour(wp, "left")), lane_key(neighbour(wp, "right")), off, wp.lane_width)


class LaneChange(Event):
    """A4: the target changes lanes (lane_single: once; lane_weave: out, back and out again, each once the target
    is TM_GAP_S x its speed plus TM_GAP_MARGIN_M past the last crossing; lane_swap: with the near-twin beside it,
    which changes into the target's lane at the same tick).

    A change starts in the window (weave's later ones until LATE_CHANGE_S) once the target moves without braking
    (SLOWING_MPS), no light holds it, no junction lies within LANE_CLEAR_AHEAD_M, (single, swap) the lane is
    straight from STRAIGHT_BEHIND_M back to STRAIGHT_AHEAD_M on (STRAIGHT_DEG), the lane it moves into has no other
    car within LANE_GAP_M behind or ahead and the own lane none within OWN_AHEAD_M ahead (swap: the partner is
    expected in the other lane, beside it within SWAP_ALONG_M, and in the map's next lane; a partner off its place
    by PARTNER_OFF_M for PARTNER_SWITCH_S holds SWAP_OTHER_SIDE_M on the other side of the target instead; the
    partner too must move without braking and not be held by a light). The first change goes to
    the trial's first_side if that lane exists and may be entered, else to the other side; weave then alternates.
    When a car beside the target keeps the lane shut for NUDGE_S, the target is slowed (TARGET_SLOW_DIFF) until the
    change starts. A change is forced as A1's cut-in (automatic lane changes on, force_lane_change every tick, other
    cars ignored; in a swap each moves into the other's lane) and ends when the map puts the car in the neighbouring
    lane it was sent to; a change not made within CHANGE_TIMEOUT_S is stopped and tried again after LANE_HOLD_S.
    Until the swap the partner escorts the target, holding its
    planned distance along; afterwards it keeps holding it."""

    def __init__(self, trial, dt):
        super().__init__(trial, dt)
        self.kind = trial["event"]["kind"]
        self.wanted = WEAVE_CHANGES if self.kind == "lane_weave" else 1
        self.side = trial["event"].get("first_side", "left")
        self.hold_along = (trial.get("distractor_car") or {}).get("along_m", 0.0)
        self.log.update(changes=[], attempts=0)
        self.prev = {}
        self.shut, self.slowed = 0, False
        self.last_cross = None
        self.off_hold = 0
        self.speeds, self.partner_speeds = [], []

    def setup(self, world, tm, actors):
        cars = [actors["target"]] + ([actors["distractor"]] if self.kind == "lane_swap" else [])
        for a in cars:
            tm.set_route(a, list(ROUTE))
            tm.auto_lane_change(a, False)

    @staticmethod
    def drive(tm, d, mps):
        tm.set_desired_speed(d, max(0.0, mps) * 3.6)

    def crossed(self, name, wp, side):
        """Whether the car's lane is now the `side` neighbour of its lane at the previous tick."""
        prev = self.prev.get(name)
        if prev is None:
            return False
        key = lane_key(wp)
        return key != prev[0] and key == (prev[1] if side == "left" else prev[2])

    def remember(self, name, wp):
        self.prev[name] = (lane_key(wp), lane_key(neighbour(wp, "left")), lane_key(neighbour(wp, "right")))

    def clear(self, world, t, dest_side, partner, width):
        """Why the change to `dest_side` may not start (None if it may): another car within LANE_GAP_M behind or
        ahead in the lane moved into, or within OWN_AHEAD_M ahead in the own lane."""
        sign = 1.0 if dest_side == "right" else -1.0
        for v in world.get_actors().filter("vehicle.*"):
            if v.id == t.id or (partner is not None and v.id == partner.id):
                continue
            a, lat, _ = relation(t, v)
            if -LANE_GAP_M[0] < a < LANE_GAP_M[1] and abs(lat - sign * width) < LANE_HALF_M + 0.5:
                return "a car in the lane moved into"
            if 0.0 < a < OWN_AHEAD_M and abs(lat) < LANE_HALF_M:
                return "a car close ahead"
        return None

    def start(self, k, tm, cars, sides):
        ignore = LANE_IGNORE_PCT
        if self.slowed:
            tm.vehicle_percentage_speed_difference(cars[0], DEFAULT_SPEED_DIFF)
            self.slowed = False
        # Known: after a forced change about half the targets speed up from 8 to 13-14 m/s for a few seconds (dev run).
        # Setting the desired speed exactly did not stop it and made the speed swing (4-7 m/s), so it is left as it
        # is and the comparison splits the drone's motion into along and across the target's heading (lane_phases.py).
        for car, side in zip(cars, sides):
            tm.auto_lane_change(car, True)
            tm.ignore_vehicles_percentage(car, ignore)
            tm.force_lane_change(car, side == "right")
        self.log["attempts"] += 1
        self.log.update(state="change", change_tick=k, sides=list(sides), crossed=[None] * len(cars))

    def stop(self, tm, cars):
        for car in cars:
            tm.auto_lane_change(car, False)
            tm.ignore_vehicles_percentage(car, 0.0)

    def step(self, k, world, tm, actors):
        t, d = actors["target"], actors.get("distractor")
        swap = self.kind == "lane_swap"
        now, s = k * self.dt, self.log["state"]
        m = world.get_map()
        twp = m.get_waypoint(t.get_location())
        vt = _speed(t)
        self.speeds = (self.speeds + [vt])[-int(round(1.0 / self.dt)) - 1:]
        dwp = m.get_waypoint(d.get_location()) if swap else None
        if swap:
            along, lateral, dyaw = relation(t, d)
            self.drive(tm, d, min(vt + ESCORT_GAIN * (self.hold_along - along), vt + OVERTAKE_MPS))
            # A partner that cannot hold its place (a slower car ahead of it, in the tuning run) holds the other side
            vd = _speed(d)
            self.partner_speeds = (self.partner_speeds + [vd])[-int(round(1.0 / self.dt)) - 1:]
            self.off_hold = self.off_hold + 1 if s == "wait" and abs(along - self.hold_along) > PARTNER_OFF_M else 0
            if self.off_hold * self.dt >= PARTNER_SWITCH_S:
                self.hold_along = -math.copysign(SWAP_OTHER_SIDE_M, self.hold_along)
                self.off_hold = 0
                self.log.setdefault("partner_switched_ticks", []).append(k)
        try:
            if s in ("wait", "hold"):
                if now < self.window[0]:
                    return
                if s == "hold" and (k - self.log["hold_tick"]) * self.dt < LANE_HOLD_S:
                    return
                gone = t.get_location().distance(self.last_cross) if self.last_cross is not None else None
                if s == "hold" and gone is not None and gone < TM_GAP_S * vt + TM_GAP_MARGIN_M:
                    self.block("too soon after the last change")
                    return
                last = self.window[1] if not self.log["changes"] else LATE_CHANGE_S
                if now > last:
                    self.log["state"] = ("missed: no change started in the window" if not self.log["changes"]
                                         else f"stopped after {len(self.log['changes'])} changes")
                    if self.slowed:
                        tm.vehicle_percentage_speed_difference(t, DEFAULT_SPEED_DIFF)
                        self.slowed = False
                    return
                side = self.side
                blocked = ("not moving" if vt <= 2.0
                           else "slowing down" if self.speeds[0] - vt > SLOWING_MPS
                           else "at a light" if t.is_at_traffic_light()
                           else "in a junction" if twp.is_junction
                           else "junction ahead" if not trials.free_of_junctions(twp, LANE_CLEAR_AHEAD_M)
                           else "curved road" if self.kind in STRAIGHT_KINDS and straight_here(twp) > STRAIGHT_DEG
                           else None)
                if blocked is None and not swap:
                    # The first change may go either way (first_side first); later ones alternate.
                    sides = [side] + ([("left" if side == "right" else "right")] if not self.log["changes"] else [])
                    why = {}
                    for sd in sides:
                        why[sd] = ("no lane on that side" if neighbour(twp, sd) is None
                                   else self.clear(world, t, sd, None, twp.lane_width))
                        if why[sd] is None:
                            side = sd
                            break
                    else:
                        blocked = why[side]
                    if blocked is None:
                        self.shut = 0
                        self.start(k, tm, [t], [side])
                        return
                    self.shut = self.shut + 1 if "a car in the lane moved into" in why.values() else 0
                    if self.shut * self.dt >= NUDGE_S and not self.slowed:
                        tm.vehicle_percentage_speed_difference(t, TARGET_SLOW_DIFF)
                        self.slowed = True
                        self.log.setdefault("slowed_ticks", []).append(k)
                elif blocked is None and neighbour(twp, side) is None:
                    blocked = "no lane on that side"
                if blocked is None and swap:
                    other = "left" if side == "right" else "right"
                    beside = (NEXT_LANE_M[0] <= abs(lateral) <= NEXT_LANE_M[1] and dyaw < SAME_WAY_DEG
                              and (lateral > 0) == (side == "right"))
                    blocked = ("partner not in the next lane" if not beside
                               else "partner too far along" if not SWAP_ALONG_M[0] <= abs(along) <= SWAP_ALONG_M[1]
                               else "partner in a junction" if dwp.is_junction
                               else "partner not moving" if vd <= 2.0
                               else "partner slowing down" if self.partner_speeds[0] - vd > SLOWING_MPS
                               else "partner at a light" if d.is_at_traffic_light()
                               else "partner not in the map's next lane"
                               if lane_key(neighbour(dwp, other)) != lane_key(twp) else None)
                if blocked is None:
                    blocked = self.clear(world, t, side, d, twp.lane_width)
                if blocked:
                    self.block(blocked)
                    return
                self.start(k, tm, [t, d], [side, "left" if side == "right" else "right"])
                self.log["change_along_m"] = round(along, 2)
            elif s == "change":
                cars = [t, d] if swap else [t]
                wps = [twp, dwp] if swap else [twp]
                for i, (car, wp, side) in enumerate(zip(cars, wps, self.log["sides"])):
                    if self.log["crossed"][i] is None and self.crossed(("t", "d")[i], wp, side):
                        self.log["crossed"][i] = k
                        if i == 0:
                            self.last_cross = car.get_location()
                        tm.auto_lane_change(car, False)
                        tm.ignore_vehicles_percentage(car, 0.0)
                    elif self.log["crossed"][i] is None:
                        tm.force_lane_change(car, side == "right")
                if all(c is not None for c in self.log["crossed"]):
                    self.log["changes"].append({"side": self.log["sides"][0], "start_tick": self.log["change_tick"],
                                                "crossed_ticks": list(self.log["crossed"])})
                    if len(self.log["changes"]) >= self.wanted:
                        self.log["state"] = "done"
                    else:
                        self.side = "left" if self.log["sides"][0] == "right" else "right"
                        self.log.update(state="hold", hold_tick=k)
                elif (k - self.log["change_tick"]) * self.dt > CHANGE_TIMEOUT_S:
                    self.stop(tm, cars)
                    self.block("change timed out")
                    self.log.setdefault("timed_out", []).append(
                        {"start_tick": self.log["change_tick"], "side": self.log["sides"][0],
                         "since_last_s": round((self.log["change_tick"] - self.log["changes"][-1]["crossed_ticks"][0])
                                               * self.dt, 2) if self.log["changes"] else None})
                    if any(c is not None for c in self.log["crossed"]):   # swap: only one of the two crossed
                        self.log["state"] = "failed: only one car changed lanes"
                    else:
                        self.log.update(state="hold", hold_tick=k)   # try again after a hold
        finally:
            self.remember("t", twp)
            if swap:
                self.remember("d", dwp)


SCRIPTS = {None: NoEvent, "cut_in": CutIn, "queue": Queue, "occlusion": Occlusion,
           "lane_single": LaneChange, "lane_weave": LaneChange, "lane_swap": LaneChange}


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


class LaneProbe:
    """Per tick, the map lane of some recorded cars (lane_state: own, left and right lane keys, offset from the lane
    centre, lane width), saved with the recording as lane_cars (their recording indices), lane_key, lane_left,
    lane_right, lane_off, lane_width (ticks x cars). The map is the client's copy: no simulator call."""

    def __init__(self, ticks, cars):
        self.cars = list(cars)          # [(recording index, actor)]
        n = len(self.cars)
        self.key, self.left, self.right = (np.full((ticks, n), -1, np.int64) for _ in range(3))
        self.off, self.width = np.zeros((ticks, n), np.float32), np.zeros((ticks, n), np.float32)

    def step(self, k, world_map):
        for j, (_, actor) in enumerate(self.cars):
            loc = actor.get_location()
            wp = world_map.get_waypoint(loc)
            if wp is not None:
                (self.key[k, j], self.left[k, j], self.right[k, j], self.off[k, j],
                 self.width[k, j]) = lane_state(wp, loc)

    def arrays(self):
        return {"lane_cars": np.array([i for i, _ in self.cars]), "lane_key": self.key, "lane_left": self.left,
                "lane_right": self.right, "lane_off": self.off, "lane_width": self.width}


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


def check_cut_in(rec, d, window=WINDOW_S, dt=0.05, behind=False):
    """The distractor entered the target's lane ahead (within CUT_CHECK_AHEAD_M; behind: CUT_CHECK_BEHIND_M) from
    the next lane and stayed for at least 1 s, entering between the window's start and 15 s after its end."""
    along, lateral, dyaw, _, _ = pair_relation(rec, 0, d)
    span = (CUT_CHECK_BEHIND_M[0] <= along) & (along <= CUT_CHECK_BEHIND_M[1]) if behind else \
        (along > 0) & (along <= CUT_CHECK_AHEAD_M)
    in_lane = (np.abs(lateral) < LANE_HALF_M) & (dyaw < SAME_WAY_DEG) & span
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


def lane_changes(rec, car, dt=0.05):
    """The lane changes of recorded car `car` (a recording index in lane_cars), from the lane record alone:
    [{tick (its first tick in the new lane), t_s, side, start_tick, end_tick, start_s, end_s}]. A change is a tick
    whose lane is the left or right neighbour of the lane at the tick before (a new road segment is not one); a lane
    entered and left again within MIN_STAY_S is a wobble on the line and dropped with the change into it. The change
    lasts from the first tick it leaves the old lane's centre (over SETTLE_M) to the last before it settles in the
    new one, looked for within SETTLE_CAP_S of the crossing."""
    col = list(np.asarray(rec["lane_cars"]).tolist()).index(car)
    key, left, right = (np.asarray(rec[f"lane_{n}"])[:, col] for n in ("key", "left", "right"))
    off = np.abs(np.asarray(rec["lane_off"], dtype=float)[:, col])
    found = []
    for k in range(1, len(key)):
        if key[k] < 0 or key[k] == key[k - 1]:
            continue
        side = "left" if key[k] == left[k - 1] else "right" if key[k] == right[k - 1] else None
        if side is None:
            continue
        if found and found[-1][1] != side and k - found[-1][0] < int(round(MIN_STAY_S / dt)):
            found.pop()
            continue
        found.append((k, side))
    cap, n, out = int(round(SETTLE_CAP_S / dt)), len(key), []
    for k, side in found:
        a = k - 1
        while a - 1 >= 0 and off[a - 1] > SETTLE_M and k - (a - 1) <= cap:
            a -= 1
        b = k
        while b + 1 < n and off[b + 1] > SETTLE_M and b + 1 - k <= cap:
            b += 1
        out.append({"tick": int(k), "t_s": round(k * dt, 2), "side": side, "start_tick": int(a), "end_tick": int(b),
                    "start_s": round(a * dt, 2), "end_s": round(b * dt, 2)})
    return out


def lateral_track(rec, car, dt=0.05):
    """Per tick: car `car`'s lateral position (m, + right) measured from the lane it started in, continuous across
    its lane changes (offset from the current lane's centre plus the lane widths crossed)."""
    col = list(np.asarray(rec["lane_cars"]).tolist()).index(car)
    off = np.asarray(rec["lane_off"], dtype=float)[:, col].copy()
    width = np.asarray(rec["lane_width"], dtype=float)[:, col]
    for c in lane_changes(rec, car, dt):
        k = c["tick"]
        w = 0.5 * (width[k - 1] + width[k])
        off[k:] += -w if c["side"] == "left" else w
    return off


def check_lane(trial, rec, window=WINDOW_S, dt=0.05):
    """A4: the target's lane changes, read from the lane record. lane_single: exactly one; lane_weave: two or more,
    each the other way from the one before; lane_swap: exactly one, and the partner exactly one the other way within
    SWAP_WITHIN_S, the two ending up in each other's lane. The first change must cross between the window's start
    and CHANGE_TIMEOUT_S after its end."""
    kind = trial["event"]["kind"]
    if "lane_key" not in rec:
        return {"happened": False, "reason": "no lane record"}
    ch = lane_changes(rec, 0, dt)
    out = {"changes": ch, "count": len(ch)}
    lo, hi = int(window[0] / dt), int((window[1] + CHANGE_TIMEOUT_S) / dt)
    if not ch:
        return {**out, "happened": False, "reason": "the target never changed lanes"}
    if not lo <= ch[0]["tick"] <= hi:
        return {**out, "happened": False, "reason": f"the first change crossed at {ch[0]['t_s']} s, outside the window"}
    out.update(tick=ch[0]["tick"], t_s=ch[0]["t_s"])
    if kind == "lane_single":
        ok = len(ch) == 1
        return {**out, "happened": ok, **({} if ok else {"reason": f"{len(ch)} changes, not one"})}
    if kind == "lane_weave":
        alternate = all(a["side"] != b["side"] for a, b in zip(ch, ch[1:]))
        ok = len(ch) >= 2 and alternate
        return {**out, "happened": ok, **({} if ok else {"reason": "one change only" if len(ch) < 2
                                                         else "two changes the same way"})}
    d = int(np.flatnonzero(np.asarray(rec["kind"]) == "distractor")[0])
    dch = lane_changes(rec, d, dt)
    out["partner_changes"] = dch
    if len(ch) != 1 or len(dch) != 1:
        return {**out, "happened": False, "reason": f"target {len(ch)} and partner {len(dch)} changes, not one each"}
    a, b = ch[0], dch[0]
    if a["side"] == b["side"] or abs(a["tick"] - b["tick"]) > int(round(SWAP_WITHIN_S / dt)):
        return {**out, "happened": False, "reason": "the partner did not change the other way at the same time"}
    along, lateral, _, _, _ = pair_relation(rec, 0, d)
    pre, post = max(0, min(a["start_tick"], b["start_tick"]) - int(1.0 / dt)), \
        min(len(along) - 1, max(a["end_tick"], b["end_tick"]) + int(1.0 / dt))
    span = slice(min(a["start_tick"], b["start_tick"]), max(a["end_tick"], b["end_tick"]) + 1)
    gap = np.hypot(along[span], lateral[span])
    out.update(lateral_before_m=round(float(lateral[pre]), 2), lateral_after_m=round(float(lateral[post]), 2),
               along_m=round(float(along[a["tick"]]), 2), min_gap_m=round(float(gap.min()), 2))
    if np.sign(lateral[pre]) == np.sign(lateral[post]):
        return {**out, "happened": False, "reason": "the two did not end up in each other's lane"}
    return {**out, "happened": True}


def check(trial, rec, occluded=None, dt=0.05):
    kind = (trial.get("event") or {}).get("kind")
    if kind is None:
        return {"happened": None}
    if kind in LANE_KINDS:
        return check_lane(trial, rec, trial["event"]["window_s"], dt)
    d = int(np.flatnonzero(np.asarray(rec["kind"]) == "distractor")[0]) if kind in ("cut_in", "queue") else None
    if kind == "cut_in":
        return check_cut_in(rec, d, trial["event"]["window_s"], dt, trial["event"].get("cut_side") == "behind")
    if kind == "queue":
        return check_queue(rec, d, trial["event"]["window_s"], dt)
    return check_occlusion(occluded, trial["event"]["window_s"], dt)
