"""Gen-v3c plans: hard negatives for target selection, a pilot check and a fresh canonical validation set.

No simulator, no torch. The second Gen-v3 checkpoint sometimes takes an object that shares the named
object's colour or shape for the target, mostly when that object is what the first view shows and the
named one is outside it (docs/canonical_clean_baseline.md). The added episodes put exactly that in
front of the model, with the teacher's answer: keep turning until the named object is in view.

  first   the first view holds a related object near its middle; the named object is outside the view
  lost    the named object is in view and the approach has begun; a forced turn (flown, never a label)
          then puts a related object in the middle of the view and the named one outside it
  swap    a `first` pose flown again in the layout that exchanges the two objects
  query   a `first` pose told to go to the object in view instead

Three groups of layouts keep the three uses apart: k-n for the recorded episodes, t and u for the pilot
flights, p-s for the validation set. Every start keeps away from every earlier start of its scene.
"""
import json
import math
import random
from pathlib import Path
from .maps import load_map,object_position,MapGeometry
from .generalization import make_start,variant,rollout,others_in_view,related
from .gen_v3 import plan_one,PADS
from . import canonical

ROOT=Path(__file__).resolve().parents[2]
FILES={'validation':ROOT/'configs/gen_v3c_canonical_validation.json','pilot':ROOT/'configs/gen_v3c_pilot.json'}
# The recorded episodes' plan lives with the dataset, which is not in Git; this copy is.
TRAIN_PLAN=ROOT/'outputs/examples/gen_v3c/hard_negative_plan.json'
SCENES=('field','lot')
# The pilot's twelve flights: kind, scene, layout, named object, object in the first view, band, task.
PILOT=(('color_first','field','t','blue_pad','blue_cube','far','land'),('color_first','field','t','red_pad','red_cube','mid','land'),
       ('color_first','lot','t','blue_pad','blue_cone','far','land'),('color_first','field','t','blue_cube','blue_pad','mid','approach'),
       ('shape_first','lot','t','blue_cube','red_cube','mid','approach'),('shape_first','lot','t','red_cube','blue_cube','far','approach'),
       ('lost','field','t','blue_pad','blue_cube','far','land'),('lost','lot','t','blue_pad','blue_cone','mid','land'),
       ('swap',0),('swap',2),
       ('search','field','t','blue_pad',None,'mid','land'),('search','lot','t','red_pad',None,'far','land'))
# Rows of one band of the validation set: role, object, base kind, side, what the first view must hold, and the place
# (scene, layout) each of the three bands is flown in first. The places are spread so that every scene and layout is used,
# the pad's neighbour is a cone, a cube and the other pad in turn, and the red pad has the red cube beside it twice and the
# blue pad once. If a place offers no such start, the other layouts follow in turn.
ROWS=(('visible','blue_pad','visible',1,None,(('field','p'),('lot','r'),('field','s'))),
      ('search','blue_pad','search',None,None,(('lot','p'),('field','q'),('lot','s'))),
      ('edge','blue_pad','peripheral',1,None,(('field','q'),('lot','q'),('field','r'))),
      ('past_color','blue_pad','search',1,'same_color',(('field','r'),('lot','r'),('field','p'))),
      ('past_shape','blue_pad','search',-1,'same_shape',(('field','p'),('lot','p'),('lot','q'))),
      ('pair','blue_pad','visible',-1,'pair',(('lot','r'),('field','s'),('lot','q'))),
      ('red_past','red_pad','search',1,'related',(('field','s'),('lot','r'),('lot','p'))),
      ('red_visible','red_pad','visible',-1,None,(('lot','s'),('field','r'),('field','p'))))
SEARCH_SIDES=(-1,'behind',1)


def relation_of(scene,target,other):
    """How `other` is related to `target`: 'same_color', 'same_shape' or None."""
    return next((relation for relation in ('same_color','same_shape') if related(scene,target,other,relation)),None)


def neighbours(scene,layout,target,relation,reach_m):
    """Objects of a layout related to `target` (same_color, same_shape, or `related` for either) that stand within reach of it, nearest first."""
    here=object_position(scene,layout,target);found=[]
    for name in scene['layouts'][layout]:
        kind=relation_of(scene,target,name)
        if name==target or kind is None or (relation not in (None,'related') and kind!=relation):continue
        distance=math.dist(here,object_position(scene,layout,name))
        if distance<=reach_m:found.append((distance,name))
    return [name for _,name in sorted(found)]


def earlier_points(*files):
    """Starts a Gen-v3c start keeps away from, by map: every earlier plan, the sealed test, both canonical sets, and the given files."""
    points=canonical.earlier_points('test')
    for path in (canonical.FILES['test'],)+files:
        if Path(path).exists():points+=[(e['map'],e['start_xy']) for e in json.loads(Path(path).read_text(encoding='utf-8'))['episodes']]
    return points


def keep_away(points,map_id,radius):
    """Predicate over (x, y, direction): too near an earlier start of this map. A start is written to the centimetre, so the
    radius is held with a few centimetres to spare."""
    mine=[xy for where,xy in points if where==map_id]
    return lambda x,y,direction:any(math.hypot(x-px,y-py)<radius+.05 for px,py in mine)


def plan_first(config,oft,scene,layout,target,other,label,span,task,instruction_id,seed,split,name,side,excluded,rule,tries=8):
    """A start that shows `other` (and, with `centred_deg`, shows it near the middle) while `target` is outside the view. Returns (episode, next seed)."""
    distractor={'relation':[other],'within_m':rule['within_m']}
    if rule.get('centred_deg') is not None:distractor['centred_deg']=rule['centred_deg']
    episode,seed=plan_one(config,oft,scene,layout,target,label,{'base':'search','distance_m':span,'distractor':distractor},task,instruction_id,seed,split,name,
                          side=side,excluded=excluded,tries=tries)
    if episode:episode.update(distractor=other,relation=relation_of(scene,target,other))
    return episode,seed


def plan_lost(config,oft,scene,layout,target,other,label,span,task,instruction_id,seed,split,name,side,excluded,rule,tries=8):
    """A start with `target` in view whose approach is interrupted by a forced turn that leaves `other` in the middle of the view
    and `target` outside it. The teacher's answer after the turn is the one it always gives without the target: keep turning."""
    geometry=MapGeometry(scene,layout);ceiling=scene['altitude']['ceiling_m'];step=math.degrees(oft['action_bounds']['yaw_rad'][1]);view=config['generalization']['fov_half_deg']
    centre=geometry.objects[other]['centre']
    for attempt in range(tries):
        base,seed=plan_one(config,oft,scene,layout,target,label,{'base':'visible','distance_m':span},task,instruction_id,seed,split,name,side=side,excluded=excluded,tries=1)
        if base is None:continue
        flown=[];rollout(config,oft,geometry,base,ceiling,trace=flown)
        rng=random.Random(f'lost|{base["id"]}');ticks=list(range(rule['after_ticks'][0],min(rule['after_ticks'][1],len(flown)-1)+1));rng.shuffle(ticks)
        for tick in ticks:
            row=flown[tick]
            if row['state']!='approach' or row['distance_m']<rule['min_remaining_m']:continue
            dx,dy=centre[0]-row['x'],centre[1]-row['y'];away=math.hypot(dx,dy)
            bearing=(math.degrees(math.atan2(dy,dx)-row['yaw_rad'])+180.)%360.-180.
            if abs(bearing)<rule['min_turn_deg'] or away>rule['within_m'] or not geometry.sees(row['x'],row['y'],row['height_m'],other):continue
            degrees=round(bearing+rng.uniform(-rule['centred_deg'],rule['centred_deg']),1)
            episode={key:value for key,value in base.items() if key!='plan'};episode['kick']={'after_ticks':tick,'degrees':degrees}
            again=[];plan=rollout(config,oft,geometry,episode,ceiling,trace=again)
            if plan['reason']!='teacher_stop' or (task=='land' and not plan['landed']) or plan['climbed_m']>0:continue
            # The first decision that is the teacher's own again: the named object must be gone from the view and the other in it.
            after=tick+math.ceil(abs(degrees)/step)
            if after+1>=len(again) or again[after]['visible'] or again[after+1]['visible']:continue
            pose=again[after];shown=others_in_view(config,geometry,target,pose['x'],pose['y'],pose['height_m'],math.degrees(pose['yaw_rad']),view)
            seen=next((item for item in shown if item['name']==other),None)
            if seen is None or abs(seen['bearing_deg'])>rule['centred_deg']+1.:continue
            back=next((item['tick'] for item in again[after:] if item['visible']),None)
            if back is None:continue
            episode['plan']={'ticks':plan['ticks'],'clearance_m':round(plan['clearance_m'],2),'climbed_m':round(plan['climbed_m'],2),'acquired_tick':plan['acquired_tick'],'states':plan['states']}
            episode.update(distractor=other,relation=relation_of(scene,target,other),
                           lost={'turn_ends_tick':after,'target_again_tick':back,'distractor_bearing_deg':seen['bearing_deg'],'distractor_distance_m':seen['distance_m']})
            return episode,seed
    return None,seed


def _twin(config,oft,scene,base,kind,settings,turns,phrases):
    """The swap or the query twin of a `first` episode, or None when the map does not offer one inside the range."""
    target,other=base['target'],base['distractor']
    if kind=='swap':
        layout=scene.get('swaps',{}).get(base['layout'])
        # Only a layout that puts the named object exactly where the other stood makes the same view hold the target.
        if layout is None or object_position(scene,layout,target)!=object_position(scene,base['layout'],other):return None
        twin=variant(config,oft,scene,base,f'{base["id"]}-swapped',layout=layout)
    else:
        task='approach'
        if other in PADS:
            task=settings['pad_tasks'][turns.get(('task','query'),0)%len(settings['pad_tasks'])];turns[('task','query')]=turns.get(('task','query'),0)+1
        key=('phrase',task,other);phrase=phrases[task]['train'][turns.get(key,0)%len(phrases[task]['train'])];turns[key]=turns.get(key,0)+1
        twin=variant(config,oft,scene,base,base['id'].replace(target,other,1)+'-query',target=other,task=task,instruction_id=phrase,allow_climb=True)
    if twin is None or twin['start_distance_m']>settings['twin_max_start_m']:return None
    # Either way the object named is now the one the first view holds.
    twin.update(kind=kind,case=kind,twin_of=base['id'],source_kind=base['kind'],base='visible')
    twin.pop('distractor',None);twin.pop('relation',None)
    if kind=='swap':twin['swap_of']=base['id']
    else:twin.update(distractor=target,relation=relation_of(scene,other,target))
    return twin


def build_training(config,oft,split,taken=()):
    """The recorded episodes of one split, in recording order: the three kinds of base episode, then their twins.
    `taken` holds episodes of the other split, whose starts this one keeps away from as well."""
    settings=config['gen_v3c'];hard=settings['hard_negative'];column=0 if split=='train' else 1;seed=settings['seeds'][split][0]
    scenes={name:load_map(name) for name in SCENES};points=earlier_points(*FILES.values())+[(e['map'],e['start_xy']) for e in taken]
    phrases=config['gen_v3']['instructions'];turns={};episodes=[];bases={}
    def turn(key,pool,advance=True):
        value=pool[turns.get(key,0)%len(pool)]
        if advance:turns[key]=turns.get(key,0)+1
        return value
    bands=list(settings['bands_m'])
    for kind in ('color_first','shape_first','lost'):
        entry=hard['kinds'][kind];bases[kind]=[]
        for index in range(entry['count'][column]):
            # The validation rows start further along the list, so that they are not the first rows of the training split again.
            target,other=entry['rows'][(index+(0 if split=='train' else 3))%len(entry['rows'])]
            # Places that have the two within reach of each other, the two scenes in turn.
            found={name:[(name,layout) for layout in settings['train_layouts'][name] if other in neighbours(scenes[name],layout,target,'related',settings['reach_m'])] for name in SCENES}
            choices=[found[name][index] for index in range(max(len(items) for items in found.values())) for name in SCENES if index<len(found[name])]
            if not choices:raise RuntimeError(f'No training layout has {other} within reach of {target}')
            task=turn(('task',kind),hard['pad_tasks']) if target in PADS else 'approach'
            phrase=turn(('phrase',task,target),phrases[task]['train']);wanted=turn(('band',kind),hard['band_cycle'])
            side=turn(('side',kind),(1,-1) if kind=='lost' else (1,-1,'behind',-1,1));episode=None
            for band in dict.fromkeys([wanted]+bands):
                for shift in range(2*len(choices)):
                    name,layout=choices[(turns.get(('place',target,other),0)+shift)%len(choices)];scene=scenes[name]
                    maker,rule=(plan_lost,hard['lost']) if kind=='lost' else (plan_first,hard['first_view'])
                    episode,seed=maker(config,oft,scene,layout,target,other,kind,settings['bands_m'][band],task,phrase,seed,split,
                                       f'{split}-{{seed:05d}}-{target}-{kind}',side,keep_away(points,name,settings['separation_m']),rule)
                    if episode:break
                if episode:break
            if episode is None:raise RuntimeError(f'No {kind} start for {target} with {other} ({split} {index})')
            turns[('place',target,other)]=turns.get(('place',target,other),0)+1
            episode.update(band=band,band_m=settings['bands_m'][band]);episodes.append(episode);bases[kind].append(episode)
            points.append((episode['map'],episode['start_xy']))
    # Twins: every `first` pose is used for at most one twin; swap twins take them from the front, query twins from the back.
    used=set()
    for kind in ('swap','query'):
        entry=hard['kinds'][kind];pattern=[source for source,share in entry['of'].items() for _ in range(share)];made=0;cursor={source:0 for source in entry['of']}
        order={source:(bases[source] if kind=='swap' else bases[source][::-1]) for source in entry['of']}
        for slot in range(4*entry['count'][column]+len(pattern)):
            if made==entry['count'][column]:break
            source=pattern[slot%len(pattern)]
            while cursor[source]<len(order[source]):
                base=order[source][cursor[source]];cursor[source]+=1
                if base['id'] in used:continue
                twin=_twin(config,oft,scenes[base['map']],base,kind,hard,turns,phrases)
                if twin:
                    twin.update(band=base['band'],band_m=base['band_m']);episodes.append(twin);used.add(base['id']);made+=1;break
        if made<entry['count'][column]:raise RuntimeError(f'Only {made} {kind} twins could be made ({split})')
    return episodes


def build_pilot(config,oft):
    """The twelve flights of the pilot check, in two layouts nothing was recorded in."""
    settings=config['gen_v3c'];hard=settings['hard_negative'];seed=settings['seeds']['pilot'][0];scenes={name:load_map(name) for name in SCENES}
    points=earlier_points(FILES['validation']);episodes=[];view=config['gen_v3c']['validation']['past_within_m']
    for index,row in enumerate(PILOT):
        if row[0]=='swap':
            base=episodes[row[1]];scene=scenes[base['map']];episode=_twin(config,oft,scene,base,'swap',hard,{},config['gen_v3']['instructions'])
            if episode is None:raise RuntimeError(f'No swap twin for pilot start {base["id"]}')
            episode.update(band=base['band'],band_m=base['band_m'])
        else:
            kind,name,layout,target,other,band,task=row;scene=scenes[name];span=settings['bands_m'][band];phrase='land' if task=='land' else 'find'
            label=f'pilot-{{seed:05d}}-{target}-{kind}';excluded=keep_away(points,name,settings['separation_m']);episode=None
            for attempt in range(200):
                if kind=='lost':episode,seed=plan_lost(config,oft,scene,layout,target,other,kind,span,task,phrase,seed,'pilot',label,(1,-1)[index%2],excluded,hard['lost'],tries=1)
                elif kind=='search':
                    episode,seed=plan_one(config,oft,scene,layout,target,kind,{'base':'search','distance_m':span},task,phrase,seed,'pilot',label,side=(1,-1)[index%2],excluded=excluded,tries=1)
                    # Nothing related in the first view: this start asks only whether the search still works.
                    if episode and any(relation_of(scene,target,item['name']) and item['distance_m']<=view for item in episode.get('others_in_view',[])):episode=None
                else:episode,seed=plan_first(config,oft,scene,layout,target,other,kind,span,task,phrase,seed,'pilot',label,(1,-1)[index%2],excluded,hard['first_view'],tries=1)
                if episode:break
            if episode is None:raise RuntimeError(f'No pilot start: {row}')
            episode.update(band=band,band_m=span)
        episode.update(role=episode['kind'],set='pilot');episodes.append(episode);points.append((episode['map'],episode['start_xy']))
    return episodes


def build_validation(config,oft):
    """The fresh validation set, in flying order: three bands of twelve starts, in four layouts of each scene that nothing else uses."""
    settings=config['gen_v3c'];seed=settings['seeds']['validation'][0];points=earlier_points();episodes=[];counter=0
    scenes={name:load_map(name) for name in SCENES};within=settings['validation']['past_within_m']
    order=[(name,settings['validation_layouts'][name][index]) for index in range(4) for name in SCENES]
    for number,(band,span) in enumerate(settings['bands_m'].items()):
        for role,target,base,side,first,places in ROWS:
            side=SEARCH_SIDES[number%len(SEARCH_SIDES)] if side is None else side
            choices=[places[number]]*40+[order[(counter+shift)%len(order)] for shift in range(760)]
            for attempt in range(800):
                name,layout=choices[attempt];scene=scenes[name];other=None
                if first=='pair':other=canonical.neighbour(scene,layout,target)
                elif first:other=next(iter(neighbours(scene,layout,target,first,settings['reach_m'])),None)
                if first and other is None:continue
                excluded=keep_away(points,name,settings['separation_m']);label=f'v3c-{{seed:05d}}-{target}-{band}_{role}'
                if first and first!='pair':
                    episode,seed=plan_first(config,oft,scene,layout,target,other,f'{band}_{role}',span,'land','land',seed,'validation',label,side,excluded,{'within_m':within},tries=4)
                else:
                    episode,seed=plan_one(config,oft,scene,layout,target,f'{band}_{role}',{'base':base,'distance_m':span},'land','land',seed,'validation',label,
                                          side=side,excluded=excluded,tries=4)
                if episode is None:continue
                twins=[]
                if first=='pair':
                    # The same pose with the pad and its neighbour exchanged, and the same pose told to find the neighbour.
                    twins=[('swap',variant(config,oft,scene,episode,f'{episode["id"]}-swapped',layout=scene['swaps'][layout],allow_climb=True)),
                           ('query',variant(config,oft,scene,episode,f'{episode["id"]}-find-{other}',target=other,task='approach',instruction_id='find',allow_climb=True))]
                elif role in ('visible','search'):
                    twins=[('approach',variant(config,oft,scene,episode,f'{episode["id"]}-approach',task='approach',instruction_id='approach'))]
                if all(item is not None and item['start_distance_m']<=settings['max_start_m'] for _,item in twins):break
            else:raise RuntimeError(f'No validation start: {band} {role}')
            counter+=1;points.append((episode['map'],episode['start_xy']))
            for item_role,item in [(role,episode)]+twins:
                if item is not episode:item['twin_of']=episode['id']
                item.update(role=item_role,set='validation',band=band,band_m=span,neighbour=other if item_role!='query' else target)
                if other and item_role!='query':item['relation']=relation_of(scene,item['target'],other)
                episodes.append(item)
    return episodes


def describe(config,which,episodes):
    settings=config['gen_v3c']
    text={'validation':'Gen-v3c fresh canonical validation starts: every start is at most {limit:.0f} m from the object named. Flown in layouts that no episode was recorded '
                       'in and no other set uses; used for the smoke and representative gates of the corrected checkpoint and for nothing else. Written before any '
                       'Gen-v3c episode was recorded.',
          'pilot':'Gen-v3c pilot check: twelve starts in two layouts that no episode was recorded in and no other set uses, flown before any flight on the validation set.'}
    return {'description':text[which].format(limit=settings['max_start_m'])+' Seeds and starts are disjoint from every earlier plan and set.',
            'mission':config['canonical']['mission'],'evaluator':settings['evaluator'],'max_start_m':settings['max_start_m'],'bands_m':settings['bands_m'],
            'seeds':settings['seeds'][which],'layouts':settings[f'{which}_layouts'],'episodes':episodes}


def category(episode):
    """How a start bears on target selection: what is related to the named object and where it is at the start."""
    return {'relation':episode.get('relation'),'first_view':episode.get('kind','').endswith('_first') or episode.get('role','').startswith(('past','red_past')),
            'lost':'kick' in episode,'swap':episode.get('role',episode.get('kind'))=='swap','query':episode.get('role',episode.get('kind'))=='query'}
