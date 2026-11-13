"""Semantic targets of an interactive AeroVLA-OFT mission: what can be named, the sentence, and when a start is refused.

No simulator and no model. A target's place is kept here for the map, the evaluator and the log. The sentence built for
the policy names the object and nothing else, and it is built from the evaluation's own templates
(configs/visual_search.json) and the shared object catalogue, so no wording is written a second time.
"""
import math
from pathlib import Path
from src.landing import evaluator
from src.landing.surface import SurfaceSet,NO_SURFACE
from src.mission.control import TASKS
from src.mission.start import StartState,wrap_deg
from src.visual_search.episodes import policy_inputs,load_config
from src.visual_search.maps import load_map,MapGeometry

ROOT=Path(__file__).resolve().parents[2]
DEMO=ROOT/'configs/mission/grounding_film_demo.json'
NOT_SEMANTIC='grounding_film requires a semantic object target. Select a configured landmark/object.'
# A landing may be asked for on any object with a landing surface (src/landing/surface.py): a pad, a cube, a cylinder.
NOT_LANDABLE=NO_SURFACE


def load_demo(path=None):
    """The interactive map: a map file like those under configs/maps/, with a `mission` section."""
    return load_map(str(path or DEMO))


def semantic_targets(map_config,layout=None):
    """Every object of the layout that a sentence can name, in the layout's order.

    `position` is the middle of the object's top (NED), `minimum`/`maximum` its box, `surface` its landing surface and
    `landable` whether it has one: for the map and the evaluator."""
    layout=layout or map_config['mission']['layout'];geometry=MapGeometry(map_config,layout);ground=map_config['ground_z'];items=[]
    surfaces=SurfaceSet(map_config,layout)
    for name in map_config['layouts'][layout]:
        spec=map_config['objects'][name];item=geometry.objects[name];(x,y),(sx,sy,height)=item['centre'],item['size_m']
        attributes=spec.get('attributes',{});surface=surfaces[name]
        # An object with a landing cap is as high as the cap's top.
        top=max(height,surface.top_m) if surface.landable else height
        items.append({'object_id':name,'noun':spec['noun'],'name':spec['noun'].removeprefix('the ').capitalize(),
                      'color':attributes.get('color'),'shape':attributes.get('shape'),'landable':surface.landable,'surface':surface.describe(),
                      'position':[x,y,ground-top],'size_m':[sx,sy,height],'native':'native' in spec,
                      'minimum':[x-sx/2,y-sy/2,ground-top],'maximum':[x+sx/2,y+sy/2,ground]})
    return items


def target_at(targets,point,margin_m=1.):
    """The semantic object whose footprint holds a clicked world point, or None (bare ground, a building)."""
    inside=[item for item in targets if all(item['minimum'][i]-margin_m<=point[i]<=item['maximum'][i]+margin_m for i in (0,1))]
    return min(inside,key=lambda item:math.hypot(item['position'][0]-point[0],item['position'][1]-point[1])) if inside else None


def sentence(config,map_config,target,task):
    """The sentence for one task: the evaluation's template (configs/visual_search.json) filled with the object's noun."""
    if task not in TASKS:raise ValueError(f'Unknown task: {task!r}')
    text=config['instructions'][map_config['mission']['instructions'][task]].format(noun=target['noun'])
    # The gate every visual-search policy input passes: it refuses a direction, a distance or coordinates in the sentence.
    return policy_inputs(None,None,text)['instruction']


def refusal(target,task):
    """Why a mission may not start, or None."""
    if target is None or not target.get('object_id'):return NOT_SEMANTIC
    if task not in TASKS:return f'Unknown task: {task!r}'
    if task=='land' and not target['landable']:return NOT_LANDABLE
    return None


def check_start(target,task):
    reason=refusal(target,task)
    if reason:raise ValueError(reason)


def launch_view(launch,target):
    """Where a target lies from a launch pose: for the display, never for the policy."""
    dx,dy=target['position'][0]-launch['xy'][0],target['position'][1]-launch['xy'][1]
    bearing=(math.degrees(math.atan2(dy,dx))-launch['yaw_deg']+180)%360-180
    return {'distance_m':math.hypot(dx,dy),'bearing_deg':bearing}


def start_of(launch,default_height_m):
    """A launch (a preset of the map file, or one the user placed) as a start state. A preset without a height of its own
    starts at the evaluation's cruise height."""
    return StartState(launch['xy'][0],launch['xy'][1],launch['yaw_deg'],launch.get('height_m',default_height_m))


def custom_launch(start):
    """A start the user chose, in the form a preset launch has."""
    return {'id':'custom','name':'Custom start','custom':True,'xy':[start.x,start.y],'yaw_deg':wrap_deg(start.yaw_deg),'height_m':start.height_m}


def mission_episode(map_path,map_config,target,task,text,launch,mission_id,max_ticks,default_height_m=None):
    """One flight in the form the canonical evaluation flies (scripts/visual_search.run_episode): the same fields as a planned
    start. `target` is used there to observe and to score; the policy is handed `instruction` and the two frames. The start
    (place, heading, height) is the launch's, a preset or one the user placed; the canonical environment reads it from here."""
    settings=map_config['mission']
    # A launch without a height of its own starts at the evaluation's cruise height, as it always did.
    if default_height_m is None:default_height_m=load_config()['cruise_height_m']
    return {'id':f'mission-{mission_id}-{target["object_id"]}-{task}','kind':'interactive','map':str(map_path),'layout':settings['layout'],
            'target':target['object_id'],'task':task,'instruction':text,'strategy':settings['strategy'],
            **start_of(launch,default_height_m).episode_fields(),'launch':launch['id'],'max_ticks':int(max_ticks),
            'start_distance_m':launch_view(launch,target)['distance_m']}


def outcome(summary,reading=None):
    """Words for the reading of one flight. An approach is read by the canonical evaluator alone
    (scripts/visual_search.summarise). A landing is read by the same steps with the landing region of the surface that was
    touched (src/landing/evaluator.py, `reading`); on a pad that is the canonical reading."""
    task=summary.get('task','approach');reason=summary.get('reason');selected=summary.get('selected')
    if reading is not None and task=='land':
        return {'success':reading['success'],'text':evaluator.words(reading,summary),'reason':reason,'selected':selected,
                'correct_target':summary.get('correct_target'),'wrong_target':summary.get('wrong_target'),
                'landed_on':summary.get('landed_on'),'collision':summary.get('hit'),'stopped':summary.get('stopped'),
                'final_distance_m':summary.get('final_distance_m'),'steps':summary.get('steps'),
                'physical_landing':reading['physical_landing'],'finalizer_latched':reading['finalizer_latched'],
                'inside_region':reading['inside_region'],'surface':reading['target_surface']['surface']}
    if summary.get('success'):
        text='landed on the named pad; latched and disarmed' if task=='land' else 'stopped in the air beside the named object'
    elif reason=='collision' or summary.get('hit'):text=f'COLLISION with {summary.get("hit") or "an obstacle"}'
    elif task=='approach' and summary.get('landed'):text=f'touched down on {summary.get("landed_on")} during an approach'
    elif summary.get('wrong_target'):text=f'ended at {selected}, not at the named object'
    elif task=='land' and summary.get('landed') and not summary.get('system_land_success'):
        text='touched the named pad, but not a stable landing in its landing region'
    elif reason=='max_steps':text='ran out of decisions before ending the mission'
    elif task=='land':text='stopped without landing on the named pad'
    else:text='stopped away from the named object' if summary.get('stopped') else f'ended: {reason}'
    return {'success':bool(summary.get('success')),'text':text,'reason':reason,'selected':selected,
            'correct_target':summary.get('correct_target'),'wrong_target':summary.get('wrong_target'),
            'landed_on':summary.get('landed_on'),'collision':summary.get('hit'),'stopped':summary.get('stopped'),
            'final_distance_m':summary.get('final_distance_m'),'steps':summary.get('steps')}
