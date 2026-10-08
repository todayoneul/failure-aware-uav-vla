"""Gen-v3 plans: the fresh held-out test set, the added training episodes and the small pilot set.

No simulator, no torch. Everything here is start states and sentences; every episode is accepted only
after the teacher finishes it in the dry run. Three things are new against the earlier plans:

  task        a sentence asks either to approach an object (stop in the air near it) or to land on a pad
  distractor  a start can be required to show another object: one of the same colour, one of the same
              shape, one in the middle of the view while the target is at its edge, or one in view
              while the target is not
  twins       the same vehicle pose flown again with the layout swapped, with another object named, or
              with the other task, so that only the scene or only the sentence differs

The test set is written once, before any training episode is planned, into configs/gen_v3_test_spawns.json.
"""
import math
import random
from .maps import load_map,object_position
from .generalization import make_start,variant,in_wedge

TEST_SCENE='depot'
TRAIN_MAPS=('field','lot','field','lot','field','blocks','lot','field','lot','field','lot','blocks')
PADS=('blue_pad','red_pad')
# Objects named in approach episodes. The blue cube and the blue cylinder, which exist only in the new scenes, come up twice per
# cycle; the pads are also named in every landing episode.
APPROACH_TARGETS=('blue_pad','blue_cube','blue_cylinder','red_cube','red_pad','blue_cube','green_cylinder','blue_cylinder','orange_ball','blue_cone')
LAND_TARGETS=('blue_pad','blue_pad','red_pad')
RELATIONS=('same_shape','same_color','any')


def spec_of(config,kind):
    """A kind of the Gen-v3 recipe with its ranges filled in from the named distance range."""
    settings=config['gen_v3'];spec=dict(settings['kinds'][kind])
    if 'range' in spec:spec['distance_m']=settings['ranges_m'][spec['range']]
    if 'distractor' in spec:spec['distractor']={'within_m':settings['distractor_within_m'],**spec['distractor']}
    return spec


def plan_one(config,oft,map_config,layout,target,label,spec,task,instruction_id,seed,split,name,side=None,excluded=None,tries=40):
    """One accepted start, trying consecutive seeds. Returns (episode, next seed)."""
    # Only a start with the target out of view can have it behind the vehicle.
    if side=='behind' and spec.get('base')!='search':side=1
    for attempt in range(tries):
        episode=make_start(config,oft,map_config,layout,target,label,'v3',seed,split,side=side,excluded=excluded,instruction_id=instruction_id,
                           spec=spec,task=task,max_ticks=config['gen_v3']['max_ticks'],name=name.format(seed=seed))
        seed+=1
        if episode:return episode,seed
    return None,seed


def site_wedge(held_out,map_config,layout,target):
    """The held-out wedge of the place a target stands on. Wedges belong to places, so a pad put where the cube stood inherits its wedge."""
    place=object_position(map_config,layout,target)
    for other_layout,targets in held_out['wedges'].get(map_config['id'],{}).items():
        for other,wedge in targets.items():
            if other in map_config['layouts'].get(other_layout,{}) and object_position(map_config,other_layout,other)==place:return wedge
    return None


def exclusion(config,points,map_config,layout,target,held_out):
    """Reject a training start near any earlier evaluation start of its map or inside the held-out wedge of its target's place."""
    radius=config['gen_v3']['separation_m'];wedge=site_wedge(held_out,map_config,layout,target);mine=[p for where,p in points if where==map_config['id']]
    def excluded(x,y,direction):
        if wedge and in_wedge(direction,wedge):return True
        return any(math.hypot(x-px,y-py)<radius for px,py in mine)
    return excluded


def _label(episode,**fields):
    episode.update(fields);return episode


def build_test(config,oft):
    """The fresh held-out sets. Seeds count up from the test range's start; a twin keeps the seed of its base start."""
    settings=config['gen_v3'];depot=load_map(TEST_SCENE);seed=settings['seeds']['test'][0];sets={};layouts=list(depot['layouts'])
    approach_id='find';land_id='land'
    def one(set_name,layout,target,kind,task,instruction_id,side=None,spec=None,map_config=depot,**fields):
        nonlocal seed
        episode,seed=plan_one(config,oft,map_config,layout,target,kind,spec or spec_of(config,kind),task,instruction_id,seed,set_name.lower(),
                              f'{set_name.lower()}-{{seed:04d}}-{target}-{kind}',side=side,tries=200)
        if episode is None:raise RuntimeError(f'No {set_name} start: {target} {kind} in {map_config["id"]}/{layout}')
        return _label(episode,set=set_name,scene_seen_in_training=bool(map_config['train_layouts']),**fields)

    # Q1: the canonical object alone as the thing to find.
    kinds=('ground_visible','ground_edge','ground_search','ground_search','ground_visible','ground_search','ground_edge','ground_search')
    sets['Q1']=[one('Q1',('c1','c3','c5','c7')[i%4],'blue_pad',kind,'approach',approach_id,side=(1,-1,'behind')[i%3]) for i,kind in enumerate(kinds)]

    # Q2 colour and Q3 shape distractors, each start flown again in Q4 with the two objects' places exchanged.
    sets['Q2']=[];sets['Q3']=[];sets['Q4']=[]
    cases=[('Q2',layout,'red_pad') for layout in ('c1','c2')]*4+[('Q3',layout,partner) for layout,partner in (('c3','blue_cylinder'),('c4','blue_cylinder'),('c5','blue_cube'),('c6','blue_cube'))]*2
    rules=('pair_side_by_side','pair_in_view','distractor_centred','search_past')
    for index,(set_name,layout,partner) in enumerate(cases):
        kind=rules[(index//2)%len(rules)] if set_name=='Q2' else rules[(index//4+index)%len(rules)]
        spec=spec_of(config,kind);spec['distractor']={**spec['distractor'],'relation':[partner]}
        for attempt in range(200):
            base=one(set_name,layout,'blue_pad',kind,'approach',approach_id,side=(1,-1)[index%2],spec=spec,distractor=partner,distractor_rule=kind)
            swapped=variant(config,oft,depot,base,f'q4-{base["seed"]:04d}-blue_pad-{kind}-swapped',layout=depot['swaps'][layout],split='q4')
            if swapped:break
        else:raise RuntimeError(f'No position-swap twin for {set_name} {layout}')
        sets[set_name].append(base)
        sets['Q4'].append(_label(swapped,set='Q4',distractor=partner,distractor_rule=kind,swap_of=base['id'],scene_seen_in_training=False))

    # Q5: one pose, four sentences: the blue pad, the other pad (same shape), another blue object (same colour) and an object
    # that shares neither. Every object named must be reachable for the teacher from that pose.
    sets['Q5']=[];groups=(('red_pad',),('blue_cylinder','blue_cube'),('red_cube','green_cylinder','orange_ball'))
    for index,layout in enumerate(('c1','c3','c2','c4')):
        for attempt in range(400):
            base=one('Q5',layout,'blue_pad','query_pose','approach',approach_id,side=(1,-1,'behind')[index%3],query_pose=index)
            others=[]
            for group in groups:
                found=(variant(config,oft,depot,base,f'q5-{base["seed"]:04d}-{target}-query_pose',target=target,allow_climb=True) for target in group)
                others.append(next((item for item in found if item and item['start_distance_m']<=settings['query_reach_m']),None))
            if all(others):break
        else:raise RuntimeError(f'No query-swap pose in {layout}')
        sets['Q5']+=[base]+[_label(other,set='Q5',query_pose=index,scene_seen_in_training=False) for other in others]

    # Q6: one pose, the sentence asks to land, to approach (trained words) or to find and approach (unseen words).
    sets['Q6']=[]
    for index,kind in enumerate(('land_visible','land_search','land_edge','land_visible','land_search','land_search')):
        for attempt in range(200):
            base=one('Q6',layouts[index%len(layouts)],'blue_pad',kind,'land',land_id,side=(1,-1,'behind')[index%3],terminal_pose=index)
            twins=[variant(config,oft,depot,base,f'{base["id"]}-{instruction}',task='approach',instruction_id=instruction) for instruction in ('approach','find_and_approach')]
            if all(twins):break
        else:raise RuntimeError('No approach twin for a Q6 start')
        sets['Q6']+=[base]+[_label(item,set='Q6',terminal_pose=index,scene_seen_in_training=False) for item in twins]

    # Q7: far starts, approach. Three bands of start distance as in the long-range set.
    sets['Q7']=[];others=('blue_cube','green_cylinder','orange_ball','blue_cylinder','red_cube','blue_cone')
    for number,(band,span) in enumerate(settings['far_ranges_m'].items()):
        targets=('blue_pad','blue_pad','blue_pad','red_pad',others[2*number],others[2*number+1])
        for index,target in enumerate(targets):
            kind=('far_visible','far_search')[(index+number)%2];spec={**spec_of(config,kind),'distance_m':span}
            sets['Q7'].append(one('Q7',layouts[(index+3*number)%len(layouts)],target,kind,'approach',approach_id,side=(1,-1,'behind')[(index+number)%3],
                                  spec=spec,range=band,range_m=span))

    # Q8: the canonical mission from near, medium and far starts; Q8R: the same sentence about the red pad.
    sets['Q8']=[];sets['Q8R']=[]
    near=('land_visible','land_search','land_edge','land_near','land_visible','land_search','land_edge','land_search')
    far=('far_land_visible','far_land_search')*4
    for band,kinds in (('near',near),('medium',near),('far',far)):
        for index,kind in enumerate(kinds):
            spec=spec_of(config,kind)
            if band!='far' and kind!='land_near':spec['distance_m']=settings['ranges_m'][band]
            if kind=='land_near' and band=='medium':kind='land_visible';spec={**spec_of(config,kind),'distance_m':settings['ranges_m'][band]}
            sets['Q8'].append(one('Q8',layouts[(index+{'near':0,'medium':2,'far':4}[band])%len(layouts)],'blue_pad',kind,'land',land_id,
                                  side=(1,-1,'behind')[index%3],spec=spec,range=band))
    for index,(band,kind) in enumerate((('near','land_visible'),('near','land_search'),('medium','land_search'),('medium','land_edge'),('far','far_land_visible'),('far','far_land_search'))):
        spec=spec_of(config,kind)
        if band!='far':spec['distance_m']=settings['ranges_m'][band]
        sets['Q8R'].append(one('Q8R',layouts[(2*index+1)%len(layouts)],'red_pad',kind,'land',land_id,side=(1,-1)[index%2],spec=spec,range=band))

    # Q10: canonical starts told in words the training data never used.
    sets['Q10']=[]
    for band in ('near','medium','far'):
        chosen=[e for e in sets['Q8'] if e['range']==band];chosen=[next(e for e in chosen if e['plan']['acquired_tick']==0),next(e for e in chosen if e['plan']['acquired_tick'])]
        for base in chosen:
            for phrase in settings['instructions']['land']['held_out']:
                sets['Q10'].append(_label(variant(config,oft,depot,base,f'q10-{base["seed"]:04d}-blue_pad-{phrase}',instruction_id=phrase,split='q10',allow_climb=True),
                                          set='Q10',range=band,base=base['id'],scene_seen_in_training=False))

    # QS: the training scenes in a layout that was never trained on, with the pad on a place it never stood on.
    sets['QS']=[]
    for name in ('field','lot'):
        scene=load_map(name);layout=scene['held_out_layouts'][0]
        for index,kind in enumerate(('ground_visible','ground_search','search_past','ground_edge')):
            sets['QS'].append(one('QS',layout,'blue_pad',kind,'approach',approach_id,side=(1,-1)[index%2],map_config=scene))
        for index,(band,kind) in enumerate((('near','land_visible'),('near','land_search'),('medium','land_search'),('far','far_land_visible'))):
            spec=spec_of(config,kind)
            if band!='far':spec['distance_m']=settings['ranges_m'][band]
            sets['QS'].append(one('QS',layout,'blue_pad',kind,'land',land_id,side=(1,-1)[index%2],spec=spec,map_config=scene,range=band))
    return {'description':'Fresh held-out start states for Gen-v3, written before any Gen-v3 training episode was planned. Seeds '
                          f'{settings["seeds"]["test"][0]}+ were never used before. Sets Q1-Q10 are flown in the Depot (scene C), where no training or '
                          'validation episode exists; QS is flown in the two training scenes in a layout that is never trained on.',
            'levels':{'Q1':'the blue pad as the thing to find','Q2':'a red pad in the first view','Q3':'a blue cylinder or a blue cube in the first view',
                      'Q4':'the Q2 and Q3 poses with the two objects exchanged','Q5':'four poses, each told to find four different objects',
                      'Q6':'six poses, each told to land, to approach, and to find and approach (unseen words)','Q7':'starts 40-90 m away, approach',
                      'Q8':'the canonical mission: find the blue landing pad and land on it','Q8R':'the same mission on the red pad',
                      'Q10':'canonical starts in unseen words','QS':'training scenes, held-out layout'},
            'canonical':config['instructions'][land_id].format(noun='the blue landing pad'),'scene':TEST_SCENE,'sets':sets}


def test_points(test,held_out=None,long_range=None):
    """Every evaluation start a training start must keep away from: (map, xy)."""
    points=[(e['map'],e['start_xy']) for episodes in test['sets'].values() for e in episodes]
    for other in (held_out,long_range):
        if other:points+=[(e['map'],e['start_xy']) for episodes in other['sets'].values() for e in episodes]
    return points


def plan_training(config,oft,split,points,held_out):
    """The added training or validation episodes: every kind of the recipe, cycling scenes, layouts, objects, sides and sentences.

    Each of those cycles has its own counter, so no object is tied to one sentence, one side or one layout."""
    settings=config['gen_v3'];seed=settings['seeds'][split][0];episodes=[];maps={};column=0 if split=='train' else 1;turns={};running=0
    def turn(key,pool):return pool[turns.get(key,0)%len(pool)]
    for kind,entry in settings['kinds'].items():
        if 'count' not in entry:continue
        task=entry['task'];phrases=settings['instructions'][task]['train']
        for index in range(entry['count'][column]):
            episode=None;running+=1
            for shift in range(4*len(TRAIN_MAPS)):
                map_id=TRAIN_MAPS[(running+shift)%len(TRAIN_MAPS)];scene=maps.setdefault(map_id,load_map(map_id))
                layout=turn(('layout',map_id,task),settings['train_layouts'][map_id])
                # Blocks gets the pads only: its own objects were trained there already.
                pool=[t for t in (LAND_TARGETS if task=='land' else APPROACH_TARGETS) if t in scene['layouts'][layout] and (map_id!='blocks' or t in PADS)]
                target=turn(('target',task,map_id=='blocks'),pool);spec=spec_of(config,kind)
                if 'distractor' in spec and 'relation' not in entry['distractor']:spec['distractor']['relation']=RELATIONS[(turns.get(('relation',kind),0)+shift)%len(RELATIONS)]
                episode,seed=plan_one(config,oft,scene,layout,target,kind,spec,task,turn(('phrase',task,target),phrases),seed,split,
                                      f'{split}-{{seed:04d}}-{target}-{kind}',side=turn(('side',kind,target),(1,-1)),
                                      excluded=exclusion(config,points,scene,layout,target,held_out),tries=12)
                if episode:
                    for key in (('layout',map_id,task),('target',task,map_id=='blocks'),('phrase',task,target),('side',kind,target),('relation',kind)):turns[key]=turns.get(key,0)+1
                    break
                # No such start here: the next scene and layout have the turn, and after a few of those the next object.
                turns[('layout',map_id,task)]=turns.get(('layout',map_id,task),0)+1
                if shift%4==3:turns[('target',task,map_id=='blocks')]=turns.get(('target',task,map_id=='blocks'),0)+1
            if episode is None:raise RuntimeError(f'No {kind} start ({split} {index})')
            episodes.append(episode)
            if entry.get('approach_twin') and index%entry['approach_twin']==0:
                # The same pose told to approach instead: only the sentence differs, and with it the end of the flight.
                approach=settings['instructions']['approach']['train']
                other=variant(config,oft,scene,episode,episode['id'].replace(kind,kind+'_approach'),task='approach',instruction_id=turn(('phrase','approach',episode['target']),approach))
                if other:
                    other.update(kind=kind+'_approach',case=kind+'_approach',twin_of=episode['id']);episodes.append(other)
                    turns[('phrase','approach',episode['target'])]=turns.get(('phrase','approach',episode['target']),0)+1
    return episodes


def plan_pilot(config,oft):
    """A dozen episodes in one layout for the canonical check: every pose is flown once to land and once to approach."""
    settings=config['gen_v3'];scene=load_map('field');seed=settings['seeds']['pilot'][0];episodes=[]
    for index,(target,kind) in enumerate((('blue_pad','land_visible'),('blue_pad','land_search'),('blue_pad','land_edge'),('blue_pad','land_near'),
                                           ('blue_pad','land_visible'),('blue_pad','land_search'),('red_pad','land_visible'),('red_pad','land_search'))):
        for attempt in range(100):
            episode,seed=plan_one(config,oft,scene,'a',target,kind,spec_of(config,kind),'land','land',seed,'pilot',f'pilot-{{seed:04d}}-{target}-{kind}',side=(1,-1)[index%2],tries=100)
            other=variant(config,oft,scene,episode,episode['id']+'-approach',task='approach',instruction_id='approach')
            if other:break
        other.update(kind=kind+'_approach',case=kind+'_approach',twin_of=episode['id']);episodes+=[episode,other]
    return episodes
