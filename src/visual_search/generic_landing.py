"""Phase B, stage 1: the starts of the generic landing data and of its checks. No simulator, no torch.

  training    landings on pads, cubes and cylinders from arbitrary starts (10-44 m, 3-12 m high, the object in the first
              view or a turn away from it), and some of the same starts flown again with the sentence that asks for an
              approach; recorded in the layouts earlier episodes were recorded in
  pilot       fifteen starts on which the candidate checkpoints are compared
  validation  thirty landing starts and six approach twins behind the gates
  regression  a fresh set with the canonical validation's make-up, for what the baseline could already do

Everything is fixed in configs/generic_landing.json. A start is accepted only if a flight may start there
(src/mission/start.validate), the object is in a free line from it, the straight track to it is clear, and the teacher
(src/landing/teacher.py) lands from it in a dry run on the map's boxes. The checks are flown in layouts no episode was
recorded in, and every start keeps its distance from every earlier start of its scene.
"""
import json
import math
import random
from pathlib import Path
from src.landing.surface import SurfaceSet
from src.landing.teacher import rollout
from src.mission import start as starts
from .generalization import sector_of,instruction_for_id
from .maps import MapGeometry,load_map
from . import gen_v3c,grounding_arch

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/generic_landing.json'
FOLDER=ROOT/'configs/generic_landing'
FILES={name:FOLDER/f'{name}.json' for name in ('pilot','validation','regression')}
TRAIN_PLAN=FOLDER/'train_plan.json'
KINDS=('pad','cube','cylinder')
# The four landings of a kind in the pilot, and the ten of the validation: (band, object in the first view).
PILOT_ROWS=(('near',True),('mid',False),('mid',True),('far',False))
VALIDATION_ROWS=(('near',True),('near',False),('near',True),('mid',False),('mid',True),('mid',False),('mid',True),('far',False),('far',True),('far',False))


def load(path=None):
    return json.loads(Path(path or CONFIG).read_text(encoding='utf-8'))


def earlier_points(*files):
    """Every start a new start keeps away from, by scene: all earlier plans and sets, and the given files."""
    points=grounding_arch.earlier_points(*grounding_arch.FILES.values())
    for path in files:
        if Path(path).exists():
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            for group in ('episodes','train','val'):points+=[(e['map'],e['start_xy']) for e in data.get(group,[])]
    return points


def kind_of(stage,target):
    return next(kind for kind,names in stage['targets'].items() if target in names)


def draw(config,oft,stage,scene,layout,target,band,visible,seed,split,name,excluded,tries=60):
    """One landing start, or None when this seed offers none of this kind here."""
    settings=config['generalization'];geometry=MapGeometry(scene,layout);centre=geometry.objects[target]['centre'];surface=SurfaceSet(scene,layout)[target]
    rng=random.Random(f'generic_landing|{scene["id"]}|{layout}|{split}|{seed}|{target}|{band}|{visible}')
    low,high=stage['bands_m'][band];span=stage['first_view']['visible_deg' if visible else 'invisible_deg']
    for attempt in range(tries):
        distance=rng.uniform(low,high);direction=rng.uniform(0.,360.);height=round(rng.uniform(*stage['height_m']),2)
        bearing=rng.choice((-1,1))*rng.uniform(*span)
        x=centre[0]+distance*math.cos(math.radians(direction));y=centre[1]+distance*math.sin(math.radians(direction))
        to_target=math.degrees(math.atan2(centre[1]-y,centre[0]-x))
        start=starts.StartState(round(x,2),round(y,2),round(starts.wrap_deg(to_target-bearing),2),height)
        if excluded(start.x,start.y,direction) or not starts.validate(start,scene,layout,config)['valid']:continue
        view=starts.view_of(config,geometry,target,start)
        # A turn has to be enough to see it, and the first view has to hold it or not as the row asks.
        if not view['line_of_sight'] or view['initially_visible']!=visible:continue
        if not visible and abs(view['target_bearing_deg'])-math.degrees(math.atan(geometry.objects[target]['size_m'][0]/2/distance))<46.:continue
        if not geometry.corridor_clear((start.x,start.y),centre,start.height_m,settings['corridor_half_width_m'],settings['corridor_clearance_m'],ignore=(target,)):continue
        label=f'{band}_{"visible" if visible else "search"}'
        episode={'id':name.format(seed=seed),'split':split,'seed':seed,'map':scene['id'],'layout':layout,'target':target,'kind':label,'case':label,
                 'band':band,'surface':surface.surface_type,'surface_kind':kind_of(stage,target),'sector':sector_of(view['target_bearing_deg']),'strategy':'right',
                 **start.episode_fields(),'target_bearing_deg':view['target_bearing_deg'],'start_distance_m':view['start_distance_m'],
                 'start_direction_deg':round(direction,2),'instruction_id':stage['instructions']['land'],
                 'instruction':instruction_for_id(config,scene,target,stage['instructions']['land']),'task':'land','max_ticks':stage['max_ticks'],
                 'initially_visible':visible,'others_in_view':view['others_in_view']}
        plan=rollout(config,oft,scene,layout,episode,surface)
        if plan['reason']!='teacher_stop' or not plan['landed']:continue
        episode['plan']={'ticks':plan['ticks'],'clearance_m':round(plan['clearance_m'],2),'climbed_m':round(plan['climbed_m'],2),
                         'acquired_tick':plan['acquired_tick'],'states':plan['states']}
        return episode
    return None


def approach_twin(config,oft,stage,scene,episode):
    """The same start told to approach the object instead of landing on it, or None when the teacher does not finish that."""
    twin={key:value for key,value in episode.items() if key!='plan'}
    twin.update(id=f'{episode["id"]}-approach',task='approach',instruction_id=stage['instructions']['approach'],twin_of=episode['id'],
                instruction=instruction_for_id(config,scene,episode['target'],stage['instructions']['approach']))
    plan=rollout(config,oft,scene,episode['layout'],twin,SurfaceSet(scene,episode['layout'])[episode['target']])
    if plan['reason']!='teacher_stop' or plan['landed']:return None
    twin['plan']={'ticks':plan['ticks'],'clearance_m':round(plan['clearance_m'],2),'climbed_m':round(plan['climbed_m'],2),
                  'acquired_tick':plan['acquired_tick'],'states':plan['states']}
    return twin


def _build(config,oft,stage,split,prefix,layouts,rows,twins,seed,points):
    """Landing starts for every kind of surface by the given rows [(band, visible)], with approach twins of the rows named in
    `twins` (indices). Objects, scenes and layouts take turns. Returns (episodes in flying order, next seed)."""
    scenes={name:load_map(name) for name in stage['scenes']};episodes=[];counter=0
    for kind in KINDS:
        objects=stage['targets'][kind]
        for index,(band,visible) in enumerate(rows):
            target=objects[index%len(objects)]
            for attempt in range(400):
                name=stage['scenes'][(counter+attempt)%len(stage['scenes'])];layout=layouts[((counter+attempt)//len(stage['scenes']))%len(layouts)]
                scene=scenes[name];excluded=gen_v3c.keep_away(points,name,stage['separation_m'])
                episode=draw(config,oft,stage,scene,layout,target,band,visible,seed,split,f'{prefix}-{{seed:05d}}-{target}-{band}_{"visible" if visible else "search"}',excluded)
                seed+=1
                if episode is None:continue
                twin=approach_twin(config,oft,stage,scene,episode) if index in twins else None
                if index in twins and twin is None:continue
                break
            else:raise RuntimeError(f'No {split} start: {kind} {band} visible={visible}')
            counter+=1;points.append((episode['map'],episode['start_xy']));episode['set']=split;episodes.append(episode)
            if twin:twin['set']=split;episodes.append(twin)
    return episodes,seed


def rows_for(count):
    """`count` landings of one kind: bands in turn, and within a band the first view in turn, starting with the other one for
    the second object so that both objects, all bands and both first views are evenly met."""
    rows=[];bands=('near','mid','far')
    for index in range(count):
        band=bands[(index//2)%3];rows.append((band,(index//6+index%2)%2==0))
    return rows


def spread(count,total):
    """`count` indices spread evenly over `total`."""
    return {int(index*total/count) for index in range(count)}


def build_pilot(config,oft,stage,points):
    episodes,_=_build(config,oft,stage,'pilot','glp',stage['layouts']['evaluation'][:2],PILOT_ROWS,{2},stage['seeds']['pilot'][0],points)
    return episodes


def build_validation(config,oft,stage,points):
    episodes,_=_build(config,oft,stage,'validation','glv',stage['layouts']['evaluation'][2:],VALIDATION_ROWS,{1,4},stage['seeds']['validation'][0],points)
    return episodes


def build_regression(config,oft,stage,points):
    """A fresh set with the canonical validation's make-up (src/visual_search/gen_v3c.build_validation), new seeds, new starts.

    Two sets of that make-up already stand in these layouts, and one of its rows (a pad of the other colour close by in the
    first view) has little room left: the set is drawn from the first of the seeds 43000, 43100, 43200, ... that gives one."""
    first,last=stage['seeds']['regression']
    for seed in range(first,last,100):
        try:episodes=gen_v3c.build_validation(config,oft,seed=seed,points=list(points),split='regression',prefix='glr')
        except RuntimeError:continue
        for episode in episodes:episode['set']='regression'
        points+=[(episode['map'],episode['start_xy']) for episode in episodes if not episode.get('twin_of')]
        return episodes
    raise RuntimeError('No regression set in the seed range')


def build_training(config,oft,stage,points):
    """The recorded episodes: {'train': [...], 'val': [...]}."""
    plan={}
    for split,land,twins in (('train',stage['training']['land_per_kind'],stage['training']['approach_twins_per_kind']),
                             ('val',stage['training']['val_land_per_kind'],stage['training']['val_approach_twins_per_kind'])):
        plan[split],_=_build(config,oft,stage,split,'gl' if split=='train' else 'glval',stage['layouts']['train'],rows_for(land),spread(twins,land),stage['seeds'][split][0],points)
    return plan


def describe(stage,which,episodes):
    text={'pilot':'Generic landing, pilot starts: flown once by the frozen baseline and by each candidate checkpoint; the candidate that goes on is chosen on them.',
          'validation':'Generic landing, validation starts: thirty landings (ten on pads, ten on cubes, ten on cylinders) and six approach twins, behind the gates of '
                       'configs/generic_landing.json.',
          'regression':'Generic landing, regression starts: a fresh set with the make-up of the canonical validation (every start at most 44 m from a pad or an '
                       'object the baseline was trained to reach), flown by the frozen baseline and by the checkpoint that passed the validation.'}
    return {'description':text[which]+' Written before any generic landing episode was recorded. Flown in layouts of the two training scenes that no episode was '
                          'recorded in; seeds and starts are disjoint from every earlier plan and set.',
            'seeds':stage['seeds'][which],'bands_m':stage['bands_m'],'episodes':episodes}


def critical_ids(episodes):
    """Starts whose first view holds an object related to the named one (same colour or shape, within 60 m) and not the named one."""
    scenes={}
    def related_first(episode):
        scene=scenes.setdefault(episode['map'],load_map(episode['map']))
        return not episode.get('initially_visible',episode.get('plan',{}).get('acquired_tick')==0) and any(
            gen_v3c.relation_of(scene,episode['target'],item['name']) and item['distance_m']<=60. for item in episode.get('others_in_view',[]))
    return [episode['id'] for episode in episodes if related_first(episode)]
