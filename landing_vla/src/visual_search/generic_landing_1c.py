"""Phase B, stage 1c: the starts of the corrective data and of its checks. No simulator, no torch.

Two things are corrected, and each has its own kind of episode (configs/generic_landing_stage1c.json):

  recovery     a landing that is pushed (src/landing/teacher.RecoveryPushes: flown, never a label) into a state the stage
               1b student failed from and the teacher never flies into: in over the object without the height, over its
               edge beside the middle, coming down before the middle, lifted again during the last metres
  same colour  a landing whose first view holds another object of the named colour and not the named one; a turn shows
               the named one. The teacher keeps searching, as it always does without its target, whatever else is in
               view. Some of these starts are flown again with the sentence that names the other object (a twin): the
               same picture, another noun, another object to land on.

The sets, in the order they are drawn (each keeps away from the ones before it and from every earlier start):

  regression       the canonical validation's make-up, new seeds
  validation       stage 1's validation format: thirty landings and six approach twins (gate A)
  same_color       twenty-four same-colour landings (gate B), and for the record the twins of some of them
  pilot            twenty-four starts on which stage 1b's checkpoint and the candidates are compared
  teacher_pilot    sixteen forced landings and sixteen same-colour starts the teacher flies before anything is recorded;
                   after the gates the two students fly the forced sixteen, for the record
  train_plan       the recorded episodes

A start is accepted as in stage 1: a flight may start there, the track to the object is clear, and the stage's teacher
lands from it in a dry run. A forced landing is accepted only if its push comes to pass in the dry run and a camera shows
the object when the push lets go. A same-colour start is accepted only if the teacher's own track never goes near the
other object.
"""
import json
import math
import random
from pathlib import Path
from src.landing.surface import SurfaceSet
from src.landing.teacher import rollout,abort_radius,FORCED
from src.mission import start as starts
from .generalization import sector_of,instruction_for_id
from .maps import MapGeometry,load_map,load_landing
from . import gen_v3c
from . import generic_landing as stages

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/generic_landing_stage1c.json'
SETS=('regression','validation','same_color','same_color_twins','pilot','teacher_pilot')


def load(path=None):
    stage=stages.load(path or CONFIG);stage.setdefault('file',str(Path(path or CONFIG).relative_to(ROOT).as_posix()) if Path(path or CONFIG).is_absolute() else str(path or CONFIG));return stage


def files_of(stage):return {name:stages.folder_of(stage)/f'{name}.json' for name in SETS}
def train_plan_of(stage):return stages.train_plan_of(stage)


def pushes_of(stage,category,rng,surface,landing):
    """The forced moves of one episode of a category, drawn inside the ranges the stage fixes."""
    rule=stage['recovery']['forced'][category];pick=lambda key:round(rng.uniform(*rule[key]),2)
    if category=='under_climb':
        return [{'when':'under_climb','within_m':rule['within_m'],'rise_m':pick('rise_m'),'band_m':rule['band_m'],'until_m':pick('until_m'),'forward_m':rule['forward_m'],'ticks':rule['ticks']}]
    if category=='edge':
        return [{'when':'edge','from_m':pick('from_m'),'min_above_m':rule['min_above_m'],'miss_m':pick('miss_m'),'side':rng.choice((-1,1)),'forward_m':rule['forward_m'],'ticks':rule['ticks']}]
    if category=='premature':
        return [{'when':'premature','from_m':pick('from_m'),'floor_m':pick('floor_m'),'forward_m':rule['forward_m'],'down_m':rule['down_m'],'ticks':rule['ticks']}]
    # Lifted again twice during the last metres, the second time lower; never out of the part of the top where the descent goes on.
    first=pick('below_m');second=round(rng.uniform(rule['below_m'][0]*.5,first-.3),2)
    return [{'when':'stall','below_m':first,'up_m':pick('up_m'),'forward_m':pick('forward_m'),'ticks':rng.randint(*rule['ticks'])},
            {'when':'stall','below_m':max(.4,second),'up_m':pick('up_m'),'forward_m':0.,'ticks':rng.randint(*rule['ticks'])}]


def recorded(plan,keys=('ticks','clearance_m','climbed_m','acquired_tick','states')):
    out={'ticks':plan['ticks'],'clearance_m':round(plan['clearance_m'],2),'climbed_m':round(plan['climbed_m'],2),'acquired_tick':plan['acquired_tick'],'states':plan['states']}
    if plan.get('forced'):out['forced']=[{'when':item['when'],'distance_m':round(item['distance_m'],2),'above_top_m':round(item['above_top_m'],2)} for item in plan['forced']]
    return out


def draw_recovery(config,oft,stage,scene,layout,target,category,band,visible,seed,split,name,excluded,make=None):
    """A landing start whose teacher flight is pushed into `category` and recovers, or None. `make` draws the forced moves
    of the episode (stage 1c's own by default; a later stage has its own states)."""
    surface=SurfaceSet(scene,layout)[target];landing=load_landing();mine=dict(stage)
    if category=='under_climb':
        # The push comes while the vehicle is still on its way up: the start has to be lower than the place it is due at.
        mine['height_m']=[stage['height_m'][0],min(stage['height_m'][1],surface.top_m+stage['recovery']['forced']['under_climb']['rise_m'][0]-.5)]
    base=stages.draw(config,oft,mine,scene,layout,target,band,visible,seed,split,name,excluded)
    if base is None:return None
    rng=random.Random(f'recovery|{base["id"]}|{category}')
    for attempt in range(6):
        episode={key:value for key,value in base.items() if key!='plan'};episode['pushes']=(make or pushes_of)(stage,category,rng,surface,landing)
        trace=[];plan=rollout(config,oft,scene,layout,episode,surface,kind=stages.teacher_of(stage),trace=trace)
        if plan['reason']!='teacher_stop' or not plan['landed'] or plan['forced_left'] or not plan['forced_ticks']:continue
        # When a push lets go a camera has to show the object: what the teacher does next must be something the pictures can tell.
        let_go=[after for before,after in zip(trace,trace[1:]) if before['perturbed'] and not after['perturbed']]
        if not let_go or not all(row['seen'] for row in let_go):continue
        label=f'{band}_{category}';episode.update(kind=label,case=label,recovery=category,plan=recorded(plan))
        episode['plan']['let_go']=[{'distance_m':round(row['distance_m'],2),'above_top_m':round(row['height_m']-surface.top_m,2),'state':row['teacher_state']} for row in let_go]
        return episode
    return None


def near_other(trace,geometry,target,other,radius):
    """Does a track come within `radius` of `other` while it is nearer to it than to the named object?"""
    centre=geometry.objects[other]['centre']
    return any(math.hypot(centre[0]-row['x'],centre[1]-row['y'])<=radius and math.hypot(centre[0]-row['x'],centre[1]-row['y'])<row['distance_m'] for row in trace)


def draw_same_color(config,oft,stage,scene,layout,target,other,seed,split,name,excluded,tries=4000):
    """A landing start that shows `other` (same colour, another shape) in its first view and not `target`, which a turn
    shows; the teacher lands on `target` and its track never goes near `other`. Or None."""
    settings=config['generalization'];rule=stage['same_color'];geometry=MapGeometry(scene,layout);surface=SurfaceSet(scene,layout)[target]
    centre=geometry.objects[target]['centre'];there=geometry.objects[other]['centre']
    rng=random.Random(f'same_color|{scene["id"]}|{layout}|{split}|{seed}|{target}|{other}')
    for attempt in range(tries):
        distance=rng.uniform(*rule['target_distance_m']);direction=rng.uniform(0.,360.);height=round(rng.uniform(*stage['height_m']),2)
        x=centre[0]+distance*math.cos(math.radians(direction));y=centre[1]+distance*math.sin(math.radians(direction))
        away=math.hypot(there[0]-x,there[1]-y)
        if not rule['other_distance_m'][0]<=away<=rule['other_distance_m'][1]:continue
        # Facing the other object, a little off its middle.
        yaw=math.degrees(math.atan2(there[1]-y,there[0]-x))+rng.uniform(-rule['other_off_centre_deg'],rule['other_off_centre_deg'])
        start=starts.StartState(round(x,2),round(y,2),round(starts.wrap_deg(yaw),2),height)
        if excluded(start.x,start.y,direction) or not starts.validate(start,scene,layout,config)['valid']:continue
        view=starts.view_of(config,geometry,target,start);shown=next((item for item in view['others_in_view'] if item['name']==other),None)
        if shown is None or view['initially_visible'] or not view['line_of_sight']:continue
        if abs(view['target_bearing_deg'])-math.degrees(math.atan(geometry.objects[target]['size_m'][0]/2/distance))<46.:continue
        if not geometry.corridor_clear((start.x,start.y),centre,start.height_m,settings['corridor_half_width_m'],settings['corridor_clearance_m'],ignore=(target,)):continue
        band=next(key for key,(low,high) in stage['bands_m'].items() if low<=distance<=high+1e-9)
        episode={'id':name.format(seed=seed),'split':split,'seed':seed,'map':scene['id'],'layout':layout,'target':target,'kind':f'{band}_same_color','case':f'{band}_same_color',
                 'band':band,'surface':surface.surface_type,'surface_kind':stages.kind_of(stage,target),'sector':sector_of(view['target_bearing_deg']),'strategy':'right',
                 **start.episode_fields(),'target_bearing_deg':view['target_bearing_deg'],'start_distance_m':view['start_distance_m'],'start_direction_deg':round(direction,2),
                 'instruction_id':stage['instructions']['land'],'instruction':instruction_for_id(config,scene,target,stage['instructions']['land']),'task':'land',
                 'max_ticks':stage['max_ticks'],'initially_visible':False,'others_in_view':view['others_in_view'],
                 'distractor':other,'relation':gen_v3c.relation_of(scene,target,other),'distractor_first':True,'distractor_distance_m':shown['distance_m'],
                 'distractor_bearing_deg':shown['bearing_deg'],'distractor_landable':bool(SurfaceSet(scene,layout)[other].landable)}
        trace=[];plan=rollout(config,oft,scene,layout,episode,surface,kind=stages.teacher_of(stage),trace=trace)
        if plan['reason']!='teacher_stop' or not plan['landed'] or near_other(trace,geometry,target,other,config['arrive_distance_m']):continue
        episode['plan']=recorded(plan);return episode
    return None


def same_color_twin(config,oft,stage,scene,episode):
    """The same start told to land on the other object, which its first view shows. None when that object has no landing
    surface or the teacher does not land on it from there."""
    other=episode['distractor'];layout=episode['layout'];surface=SurfaceSet(scene,layout)[other];geometry=MapGeometry(scene,layout)
    if not surface.landable:return None
    start=starts.StartState.from_episode(episode,config['cruise_height_m']);view=starts.view_of(config,geometry,other,start)
    if not view['initially_visible']:return None
    twin={key:value for key,value in episode.items() if key not in ('plan','distractor_first','distractor_distance_m','distractor_bearing_deg','distractor_landable')}
    band=next((key for key,(low,high) in stage['bands_m'].items() if low<=view['start_distance_m']<=high+1e-9),'near' if view['start_distance_m']<stage['bands_m']['near'][0] else 'far')
    twin.update(id=f'{episode["id"]}-twin',target=other,distractor=episode['target'],twin_of=episode['id'],surface=surface.surface_type,surface_kind=stages.kind_of(stage,other),
                kind=f'{band}_same_color_twin',case=f'{band}_same_color_twin',band=band,initially_visible=True,distractor_first=False,
                target_bearing_deg=view['target_bearing_deg'],start_distance_m=view['start_distance_m'],sector=sector_of(view['target_bearing_deg']),others_in_view=view['others_in_view'],
                instruction=instruction_for_id(config,scene,other,stage['instructions']['land']))
    trace=[];plan=rollout(config,oft,scene,layout,twin,surface,kind=stages.teacher_of(stage),trace=trace)
    if plan['reason']!='teacher_stop' or not plan['landed'] or near_other(trace,geometry,other,episode['target'],config['arrive_distance_m']):return None
    twin['plan']=recorded(plan);return twin


def pair_layouts(stage,scenes,target,other,layouts):
    """(scene, layout) of the given layouts in which the two objects stand near enough for one start to show the other first."""
    reach=stage['same_color']['pair_within_m'];found=[]
    for layout in layouts:
        for name in stage['scenes']:
            objects=MapGeometry(scenes[name],layout).objects
            if target in objects and other in objects and math.dist(objects[target]['centre'][:2],objects[other]['centre'][:2])<=reach:found.append((name,layout))
    return found


class Builder:
    """Draws starts one after the other from a running seed, each keeping away from all before it."""
    def __init__(self,config,oft,stage,points,seed):
        self.config,self.oft,self.stage,self.points,self.seed=config,oft,stage,points,seed;self.scenes={name:load_map(name) for name in stage['scenes']};self.counter=0

    def excluded(self,name):return gen_v3c.keep_away(self.points,name,self.stage['separation_m'])
    def keep(self,episode):self.points.append((episode['map'],episode['start_xy']));self.counter+=1;return episode

    def recovery(self,split,prefix,layouts,target,category,band,visible,attempts=400):
        for attempt in range(attempts):
            name=self.stage['scenes'][(self.counter+attempt)%len(self.stage['scenes'])];layout=layouts[((self.counter+attempt)//len(self.stage['scenes']))%len(layouts)]
            episode=draw_recovery(self.config,self.oft,self.stage,self.scenes[name],layout,target,category,band,visible,self.seed,split,
                                  f'{prefix}-{{seed:05d}}-{target}-{band}_{category}',self.excluded(name));self.seed+=1
            if episode:episode['set']=split;return self.keep(episode)
        raise RuntimeError(f'No {split} start: {target} {category} {band}')

    def landing(self,split,prefix,layouts,target,band,visible,twin=False,attempts=400):
        """A plain landing start (stage 1's kind), with its approach twin when asked."""
        for attempt in range(attempts):
            name=self.stage['scenes'][(self.counter+attempt)%len(self.stage['scenes'])];layout=layouts[((self.counter+attempt)//len(self.stage['scenes']))%len(layouts)]
            episode=stages.draw(self.config,self.oft,self.stage,self.scenes[name],layout,target,band,visible,self.seed,split,
                                f'{prefix}-{{seed:05d}}-{target}-{band}_{"visible" if visible else "search"}',self.excluded(name));self.seed+=1
            if episode is None:continue
            other=stages.approach_twin(self.config,self.oft,self.stage,self.scenes[name],episode) if twin else None
            if twin and other is None:continue
            episode['set']=split;self.keep(episode)
            if other:other['set']=split
            return [episode]+([other] if other else [])
        raise RuntimeError(f'No {split} start: {target} {band} visible={visible}')

    def same_color(self,split,prefix,layouts,target,other,twin=False,attempts=120):
        places=pair_layouts(self.stage,self.scenes,target,other,layouts)
        if not places:raise RuntimeError(f'No layout with {target} near {other}')
        for attempt in range(attempts):
            name,layout=places[(self.counter+attempt)%len(places)]
            episode=draw_same_color(self.config,self.oft,self.stage,self.scenes[name],layout,target,other,self.seed,split,f'{prefix}-{{seed:05d}}-{target}-past-{other}',self.excluded(name));self.seed+=1
            if episode is None:continue
            second=same_color_twin(self.config,self.oft,self.stage,self.scenes[name],episode) if twin else None
            if twin and second is None:continue
            episode['set']=split;self.keep(episode)
            if second:second['set']=split
            return [episode]+([second] if second else [])
        raise RuntimeError(f'No {split} start: {target} past {other}')


BANDS=('near','mid','far')


def build_recovery(builder,split,prefix,layouts,counts):
    """`counts`: {kind of surface: {category: episodes}}. Objects, bands and first views take turns."""
    stage=builder.stage;episodes=[]
    for kind,categories in counts.items():
        index=0;bands=stage['training'].get('recovery_bands',{}).get(kind,BANDS) if split in ('train','val') else BANDS
        for category,count in categories.items():
            for number in range(count):
                target=stage['targets'][kind][index%2];band=bands[(index//2)%len(bands)];visible=(index//6+index%2)%2==0;index+=1
                episodes.append(builder.recovery(split,prefix,layouts,target,category,band,visible))
    return episodes


def build_same_color(builder,split,prefix,layouts,rows):
    """`rows`: [[target, other, starts, of them with a twin]]. The twins follow their starts."""
    episodes=[]
    for target,other,count,twins in rows:
        chosen=stages.spread(twins,count) if twins else set()
        for number in range(count):episodes+=builder.same_color(split,prefix,layouts,target,other,twin=number in chosen)
    return episodes


def build_validation(builder):
    """Gate A: stage 1's validation format (ten landings per kind of surface, two of them with an approach twin)."""
    stage=builder.stage;layouts=stage['layouts']['evaluation'][2:];episodes=[]
    for kind in stages.KINDS:
        for index,(band,visible) in enumerate(stages.VALIDATION_ROWS):
            episodes+=builder.landing('validation',stage['prefix']['validation'],layouts,stage['targets'][kind][index%2],band,visible,twin=index in (1,4))
    return episodes


def build_pilot(builder):
    """Twenty-four starts: eight forced cube landings (two per state), eight same-colour landings, four pad and four cylinder landings."""
    stage=builder.stage;layouts=stage['layouts']['evaluation'];prefix=stage['prefix']['pilot'];rule=stage['pilot']
    episodes=build_recovery(builder,'pilot',prefix,layouts,{'cube':{category:rule['recovery_per_state'] for category in FORCED}})
    episodes+=build_same_color(builder,'pilot',prefix,layouts,rule['same_color'])
    for kind in ('pad','cylinder'):
        for index,(band,visible) in enumerate(stages.PILOT_ROWS):episodes+=builder.landing('pilot',prefix,layouts,stage['targets'][kind][index%2],band,visible)
    return episodes


def build_teacher_pilot(builder):
    stage=builder.stage;layouts=stage['layouts']['evaluation'];prefix=stage['prefix']['teacher_pilot'];rule=stage['teacher_pilot']
    episodes=build_recovery(builder,'teacher_pilot',prefix,layouts,{'cube':{category:rule['recovery_per_state'] for category in FORCED}})
    return episodes+build_same_color(builder,'teacher_pilot',prefix,layouts,rule['same_color'])


def build_regression(config,oft,stage,points):
    first,last=stage['seeds']['regression']
    for seed in range(first,last,100):
        try:episodes=gen_v3c.build_validation(config,oft,seed=seed,points=list(points),split='regression',prefix=stage['prefix']['regression'])
        except RuntimeError:continue
        for episode in episodes:episode['set']='regression'
        return episodes
    raise RuntimeError('No regression set in the seed range')


def loose_points(stage,which):
    """Earlier starts a set keeps away from: all but the sets named in its `may_be_near`."""
    near={(ROOT/path).resolve() for path in stage[which].get('may_be_near',[])}
    return stages.earlier_points(*[path for path in stages.previous_files(stage) if Path(path).resolve() not in near])


def check_points(stage):
    """Every start that is flown for a check anywhere in the repository: each plan and set under configs/ but the plans of
    recorded episodes (and this stage's own files, which are added as they are drawn)."""
    points=[];mine=stages.folder_of(stage).resolve()
    def walk(item):
        if isinstance(item,dict):
            if 'start_xy' in item and item.get('map'):points.append((item['map'],list(item['start_xy'])))
            for value in item.values():walk(value)
        elif isinstance(item,list):
            for value in item:walk(value)
    for path in sorted((ROOT/'configs').rglob('*.json')):
        if path.name=='train_plan.json' or path.parent.resolve()==mine:continue
        walk(json.loads(path.read_text(encoding='utf-8')))
    return points


def firsts(episodes):return [(episode['map'],episode['start_xy']) for episode in episodes if not episode.get('twin_of')]


def build_all(config,oft,stage):
    """Every set of the stage. Checks first, then the recorded episodes, which keep away from all of them."""
    sets={};strict=loose_points(stage,'checks');own=[]
    sets['regression']=build_regression(config,oft,stage,loose_points(stage,'regression'));own+=firsts(sets['regression'])
    checks=Builder(config,oft,stage,strict+own,stage['seeds']['validation'][0])
    sets['validation']=build_validation(checks)
    checks.seed=stage['seeds']['same_color'][0];both=build_same_color(checks,'same_color',stage['prefix']['same_color'],stage['layouts']['evaluation'],stage['same_color']['gate_rows'])
    sets['same_color']=[episode for episode in both if not episode.get('twin_of')];sets['same_color_twins']=[episode for episode in both if episode.get('twin_of')]
    checks.seed=stage['seeds']['pilot'][0];sets['pilot']=build_pilot(checks)
    checks.seed=stage['seeds']['teacher_pilot'][0];sets['teacher_pilot']=build_teacher_pilot(checks)
    own=[point for point in checks.points if point not in strict]
    # A recorded start keeps away from every start that is flown for a check, earlier or this stage's, and from this stage's
    # other recorded starts. It need not keep away from an earlier recorded start: both are data.
    plan={};flown=check_points(stage)+own
    for split,recovery,same in (('train',stage['training']['recovery'],stage['training']['same_color']),('val',stage['training']['val_recovery'],stage['training']['val_same_color'])):
        builder=Builder(config,oft,stage,flown+[point for other in plan.values() for point in firsts(other)],stage['seeds'][f'{split}_recovery'][0])
        episodes=build_recovery(builder,split,stage['prefix'][f'{split}_recovery'],stage['layouts']['train'],recovery)
        builder.seed=stage['seeds'][f'{split}_same_color'][0]
        plan[split]=episodes+build_same_color(builder,split,stage['prefix'][f'{split}_same_color'],stage['layouts']['train'],same)
    sets['train_plan']=plan
    return sets


def describe(stage,which,episodes):
    text={'regression':'the canonical validation\'s make-up from new seeds, flown once by the frozen baseline, by stage 1b\'s checkpoint and by the chosen checkpoint of stage 1c',
          'validation':'gate A: thirty landings (ten on pads, ten on cubes, ten on cylinders) and six approach twins, the format of stage 1',
          'same_color':'gate B: twenty-four landings whose first view holds another object of the named colour and not the named one',
          'same_color_twins':'for the record, after the gates: starts of gate B told to land on the other object, which their first view shows',
          'pilot':'twenty-four starts flown once by stage 1b\'s checkpoint and by each candidate: eight forced cube landings, eight same-colour landings, four pad and four cylinder landings',
          'teacher_pilot':'sixteen forced cube landings and sixteen same-colour starts the teacher flies before anything is recorded; after the gates the two students fly the forced sixteen, for the record'}
    return {'description':f'Generic landing, stage 1c, {which}: {text[which]}. Written before any episode of stage 1c was recorded ({stage["file"]}). Flown in layouts of the two '
                          'training scenes that no episode was recorded in; seeds and starts are disjoint from every earlier plan and set.',
            'seeds':stage['seeds'].get(which,stage['seeds']['same_color'] if which=='same_color_twins' else None),'bands_m':stage['bands_m'],'episodes':episodes}


def repeat_ids(stage,episodes):
    """Gate B's second flights: every start whose other object is a cone, and of the rest those whose other object stands nearest."""
    count=stage['same_color']['repeat']['starts'];cones=[e for e in episodes if not e['distractor_landable']];rest=sorted((e for e in episodes if e['distractor_landable']),key=lambda e:e['distractor_distance_m'])
    return [e['id'] for e in (cones+rest)[:count]]
