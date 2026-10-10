"""Semantic targets of an interactive AeroVLA-OFT mission: what can be named, the sentence, and when a start is refused.

No simulator and no model. A target's place is kept here for the map, the evaluator and the log. The sentence built for
the policy names the object and nothing else, and it is built from the evaluation's own templates
(configs/visual_search.json) and the shared object catalogue, so no wording is written a second time.
"""
import math
from pathlib import Path
from src.mission.control import TASKS
from src.visual_search.episodes import policy_inputs
from src.visual_search.maps import load_map,MapGeometry

ROOT=Path(__file__).resolve().parents[2]
DEMO=ROOT/'configs/mission/grounding_film_demo.json'
NOT_SEMANTIC='grounding_film requires a semantic object target. Select a configured landmark/object.'
NOT_LANDABLE='Selected object is not landable. Use APPROACH or select a landing pad.'


def load_demo(path=None):
    """The interactive map: a map file like those under configs/maps/, with a `mission` section."""
    return load_map(str(path or DEMO))


def semantic_targets(map_config,layout=None):
    """Every object of the layout that a sentence can name, in the layout's order.

    `position` is the middle of the object's top (NED), `minimum`/`maximum` its box: for the map and the evaluator."""
    layout=layout or map_config['mission']['layout'];geometry=MapGeometry(map_config,layout);ground=map_config['ground_z'];items=[]
    for name in map_config['layouts'][layout]:
        spec=map_config['objects'][name];item=geometry.objects[name];(x,y),(sx,sy,height)=item['centre'],item['size_m']
        attributes=spec.get('attributes',{})
        items.append({'object_id':name,'noun':spec['noun'],'name':spec['noun'].removeprefix('the ').capitalize(),
                      'color':attributes.get('color'),'shape':attributes.get('shape'),'landable':item['landable'],
                      'position':[x,y,ground-height],'size_m':[sx,sy,height],'native':'native' in spec,
                      'minimum':[x-sx/2,y-sy/2,ground-height],'maximum':[x+sx/2,y+sy/2,ground]})
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


def mission_episode(map_path,map_config,target,task,text,launch,mission_id,max_ticks):
    """One flight in the form the canonical evaluation flies (scripts/visual_search.run_episode): the same fields as a planned
    start. `target` is used there to observe and to score; the policy is handed `instruction` and the two frames."""
    settings=map_config['mission']
    return {'id':f'mission-{mission_id}-{target["object_id"]}-{task}','kind':'interactive','map':str(map_path),'layout':settings['layout'],
            'target':target['object_id'],'task':task,'instruction':text,'strategy':settings['strategy'],
            'start_xy':list(launch['xy']),'start_yaw_deg':launch['yaw_deg'],'launch':launch['id'],'max_ticks':int(max_ticks),
            'start_distance_m':launch_view(launch,target)['distance_m']}


def outcome(summary):
    """Words for the canonical reading of one flight (scripts/visual_search.summarise). Nothing is judged again here."""
    task=summary.get('task','approach');reason=summary.get('reason');selected=summary.get('selected')
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
