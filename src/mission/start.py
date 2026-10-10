"""Where a flight starts: one representation, the rule a start has to meet, and a planner of starts. No simulator, no model.

A start is four numbers: where the vehicle stands on the ground (x, y), which way it faces (yaw) and how high it climbs
before the first decision (height). The canonical evaluation already takes exactly these from an episode (`start_xy`,
`start_yaw_deg`, `start_height_m`; scripts/visual_search.SearchEnv.reset): the scene is loaded with the vehicle at that
place and heading, the vehicle comes to rest on the ground, takes off and climbs to the height. `StartState` is that, with
a name, for the interactive map, an experiment plan, the planner and the log alike.

    x, y        the map's frame (north, east), metres
    yaw_deg     0 faces +x (north), 90 faces +y (east): the simulator's own heading, clockwise seen from above
    height_m    above where the vehicle rests on the ground at the start

No number of the rule is new. The clearance is the one every planned evaluation start keeps from every obstacle
(configs/visual_search.json, generalization.start_margin_m); the lowest height is the executor's own floor
(configs/flight_limits.json, minimum_clearance_m); the highest is the map's ceiling (altitude.ceiling_m).

None of this is given to a policy. A start decides where the cameras are; the policy sees what they show.
"""
import json
import math
import random
from dataclasses import dataclass,asdict
from pathlib import Path
from src.landing.surface import SurfaceSet,NO_SURFACE
from src.visual_search.maps import MapGeometry,load_landing
from src.visual_search.generalization import others_in_view,sector_of,instruction_for_id

ROOT=Path(__file__).resolve().parents[2]
LIMITS=ROOT/'configs/flight_limits.json'
YAW_MODES=('random','toward','away')


@dataclass(frozen=True)
class StartState:
    x:float
    y:float
    yaw_deg:float
    height_m:float

    def __post_init__(self):
        for name in ('x','y','yaw_deg','height_m'):object.__setattr__(self,name,float(getattr(self,name)))

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,data):return cls(data['x'],data['y'],data['yaw_deg'],data['height_m'])

    @classmethod
    def from_episode(cls,episode,default_height_m):
        """The start of a planned or recorded flight; an episode without a height starts at the cruise height."""
        return cls(episode['start_xy'][0],episode['start_xy'][1],episode['start_yaw_deg'],episode.get('start_height_m',default_height_m))

    def episode_fields(self):
        """The fields the canonical environment reads its start from."""
        return {'start_xy':[self.x,self.y],'start_yaw_deg':self.yaw_deg,'start_height_m':self.height_m}

    def moved(self,**changes):
        return StartState(**{**asdict(self),**changes})


def wrap_deg(angle):
    """An angle in (-180, 180]."""
    value=(angle+180.)%360.-180.
    return 180. if value==-180. else value


def rule(config,map_config,limits=None):
    """What a start has to meet on this map, each number from where it already exists."""
    limits=limits or json.loads(LIMITS.read_text(encoding='utf-8'))
    return {'clearance_m':config['generalization']['start_margin_m'],'clearance_from':'configs/visual_search.json generalization.start_margin_m',
            'height_m':[limits['minimum_clearance_m'],map_config['altitude']['ceiling_m']],
            'height_from':['configs/flight_limits.json minimum_clearance_m','the map file, altitude.ceiling_m'],
            'default_height_m':config['cruise_height_m'],'default_height_from':'configs/visual_search.json cruise_height_m',
            'area':map_config['area'],'ground_z':map_config['ground_z']}


def validate(start,map_config,layout,config,surface_point=None,limits=None):
    """Whether a flight may start here, and every reason why not.

    `surface_point` is the world point under a map click, when the start was chosen that way: what was clicked has to be the
    ground. Returns {'valid', 'reasons', 'facts'}; the reasons are sentences for the display."""
    terms=rule(config,map_config,limits);reasons=[];facts={'rule':{key:terms[key] for key in ('clearance_m','height_m')}}
    values=(start.x,start.y,start.yaw_deg,start.height_m)
    if not all(math.isfinite(value) for value in values):
        return {'valid':False,'reasons':['the start is not four finite numbers'],'facts':facts}
    geometry=MapGeometry(map_config,layout)
    if not geometry.inside_area(start.x,start.y):
        area=terms['area'];reasons.append(f'outside the map area (x {area["x"][0]:.0f}..{area["x"][1]:.0f}, y {area["y"][0]:.0f}..{area["y"][1]:.0f})')
    gap,name=geometry.nearest(start.x,start.y);facts.update(clearance_m=None if name is None else round(gap,2),nearest=name)
    if name is not None and gap<=0.:reasons.append(f'inside {name}: a flight starts on open ground')
    elif gap<terms['clearance_m']:reasons.append(f'obstacle clearance {gap:.1f} m < {terms["clearance_m"]:.1f} m ({name})')
    if surface_point is not None:
        above=terms['ground_z']-surface_point[2];facts['clicked_surface_above_ground_m']=round(above,2)
        # A click on a roof or on an object's top resolves to that top, not to ground a vehicle can stand on.
        if abs(above)>load_landing()['touchdown']['from_above_tolerance_m']:reasons.append(f'not on the ground: the clicked surface is {above:.1f} m above it')
    low,high=terms['height_m']
    if start.height_m<low:reasons.append(f'height {start.height_m:.1f} m is below the minimum flight clearance ({low:.1f} m)')
    if start.height_m>high:reasons.append(f'height {start.height_m:.1f} m is above the ceiling ({high:.1f} m)')
    return {'valid':not reasons,'reasons':reasons,'facts':facts}


def check(start,map_config,layout,config,surface_point=None):
    result=validate(start,map_config,layout,config,surface_point)
    if not result['valid']:raise ValueError('INVALID START. Reason: '+'; '.join(result['reasons']))
    return result


def step_height(start,direction,map_config,config,step_m=1.,limits=None):
    """One step of the height keys: up or down by a step, never past the rule's bounds. Returns (start, message or None)."""
    low,high=rule(config,map_config,limits)['height_m'];wanted=start.height_m+direction*step_m
    if wanted>high+1e-9:return start,f'start height stays at {start.height_m:.1f} m: the ceiling is {high:.1f} m'
    if wanted<low-1e-9:return start,f'start height stays at {start.height_m:.1f} m: the minimum flight clearance is {low:.1f} m'
    return start.moved(height_m=round(wanted,3)),None


def heading(origin,toward):
    """The yaw that faces from one ground point to another, or None when they are the same point."""
    dx,dy=toward[0]-origin[0],toward[1]-origin[1]
    return None if math.hypot(dx,dy)<1e-6 else wrap_deg(math.degrees(math.atan2(dy,dx)))


def spawn_report(rest_z,map_config,landing=None):
    """After the scene is loaded: did the vehicle come to rest on the map's ground?

    The canonical loop counts every height from where the vehicle rests at the start. On a platform or on an object that
    reference is wrong, and a touchdown on a pad can read as a collision. `rest_z` is the resting height the simulator reports."""
    landing=landing or load_landing();expected=map_config['ground_z']-landing['vehicle']['rest_offset_m']
    tolerance=landing['evaluator']['surface_height_tolerance_m'];error=rest_z-expected
    return {'rest_z':rest_z,'expected_z':expected,'error_m':error,'tolerance_m':tolerance,'on_ground':abs(error)<=tolerance}


def reached(state,ground_z):
    """The start a vehicle is actually at: its place, its heading and its height above where it rested."""
    x,y,z,w=state['orientation'];yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    return StartState(state['position'][0],state['position'][1],wrap_deg(yaw),ground_z-state['position'][2])


def reproduction(wanted,got):
    """How far the start a vehicle is at lies from the one that was asked for."""
    return {'xy_error_m':math.hypot(got.x-wanted.x,got.y-wanted.y),'yaw_error_deg':abs(wrap_deg(got.yaw_deg-wanted.yaw_deg)),
            'height_error_m':got.height_m-wanted.height_m}


def view_of(config,geometry,target,start):
    """What a start shows of an object, from the map's boxes: for planning and for the record, never for a policy.

    `initially_visible`: in the first front view. `occluded`: no free line to it from the start, whichever way the vehicle
    turns. `visible_after_climb`: a free line from the ceiling above the start. `requires_translation`: neither turning nor
    climbing on the spot shows it; the vehicle has to move."""
    settings=config['generalization'];centre=geometry.objects[target]['centre'];dx,dy=centre[0]-start.x,centre[1]-start.y
    distance=math.hypot(dx,dy);bearing=wrap_deg(math.degrees(math.atan2(dy,dx))-start.yaw_deg)
    sight=geometry.sees(start.x,start.y,start.height_m,target);ceiling=geometry.config['altitude']['ceiling_m']
    high=geometry.sees(start.x,start.y,ceiling,target)
    blocker=None if sight else geometry.blocking((start.x,start.y,start.height_m),geometry.sight_points(target)[1],ignore=(target,))
    in_view=bool(sight and abs(bearing)<=settings['fov_half_deg'])
    return {'start_distance_m':round(distance,2),'target_bearing_deg':round(bearing,2),'sector':sector_of(bearing),
            'initially_visible':in_view,'initially_invisible':not in_view,'line_of_sight':bool(sight),'occluded':not sight,'occluded_by':blocker,
            'visible_after_climb':bool(high),'requires_translation':not sight and not high,
            'others_in_view':others_in_view(config,geometry,target,start.x,start.y,start.height_m,start.yaw_deg,settings['fov_half_deg'])}


def starts_of(data,map_id=None):
    """Every start position in a plan file, whatever its shape: (map, [x, y]) of each dictionary that has a `start_xy`."""
    found=[]
    def walk(item):
        if isinstance(item,dict):
            if 'start_xy' in item and (map_id is None or item.get('map',map_id)==map_id):found.append(list(item['start_xy']))
            for value in item.values():walk(value)
        elif isinstance(item,list):
            for value in item:walk(value)
    walk(data);return found


def plan_starts(config,map_config,layout,target,task,count,seed,distance_m,height_m=None,yaw='random',avoid=(),separation_m=None,
                instruction_id=None,max_ticks=None,map_name=None,attempts=None):
    """`count` starts for one object, drawn once from a seed: the same arguments give the same starts.

    Each start is accepted only if it meets the rule a start has to meet (`validate`), lies between the two distances from
    the object, and keeps `separation_m` from every position in `avoid` and from the plan's own earlier starts. The yaw is
    drawn from the whole circle (`random`), or faces the object (`toward`) or away from it (`away`). Returns the episodes in
    the form the canonical environment flies, each with what its start shows of the object (`view_of`)."""
    if yaw not in YAW_MODES:raise ValueError(f'Unknown yaw mode: {yaw!r}')
    if target not in map_config['layouts'][layout]:raise ValueError(f'{target} is not an object of layout {layout!r}')
    low,high=distance_m
    if not 0<=low<=high:raise ValueError('The distances must be 0 <= minimum <= maximum')
    if task=='land' and not SurfaceSet(map_config,layout)[target].landable:raise ValueError(NO_SURFACE)
    settings=config['generalization'];geometry=MapGeometry(map_config,layout);centre=geometry.objects[target]['centre']
    heights=tuple(height_m or map_config['altitude']['start_m']);separation=settings['exclusion_radius_m'] if separation_m is None else separation_m
    instruction_id=instruction_id or map_config.get('mission',{}).get('instructions',{}).get(task,task)
    rng=random.Random(f'arbitrary|{map_config["id"]}|{layout}|{target}|{task}|{seed}');taken=[list(point) for point in avoid];episodes=[]
    for attempt in range(attempts or settings['attempts']['default']*max(1,count)):
        if len(episodes)==count:break
        distance=rng.uniform(low,high);direction=rng.uniform(0.,360.);height=rng.uniform(*heights);turn=rng.uniform(-180.,180.)
        x=centre[0]+distance*math.cos(math.radians(direction));y=centre[1]+distance*math.sin(math.radians(direction))
        to_target=math.degrees(math.atan2(centre[1]-y,centre[0]-x))
        facing={'random':turn,'toward':to_target,'away':to_target+180.}[yaw]
        start=StartState(round(x,2),round(y,2),round(wrap_deg(facing),2),round(height,2))
        if not validate(start,map_config,layout,config)['valid']:continue
        if any(math.hypot(start.x-px,start.y-py)<separation for px,py in taken):continue
        taken.append([start.x,start.y]);index=len(episodes)
        episode={'id':f'as-{seed}-{index:03d}-{target}-{task}','kind':'arbitrary_start','split':'arbitrary_start','seed':seed,'index':index,
                 'map':map_name or map_config['id'],'layout':layout,'target':target,'task':task,'strategy':map_config.get('mission',{}).get('strategy','right'),
                 **start.episode_fields(),'start_direction_deg':round(direction,2),'instruction_id':instruction_id,
                 'instruction':instruction_for_id(config,map_config,target,instruction_id),**view_of(config,geometry,target,start)}
        if max_ticks:episode['max_ticks']=int(max_ticks)
        episodes.append(episode)
    if len(episodes)<count:raise RuntimeError(f'Only {len(episodes)} of {count} starts found for {target} between {low} and {high} m')
    return episodes
