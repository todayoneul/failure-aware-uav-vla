"""The canonical clean mission and its two start sets: a validation set for the gates and a fresh test.

No simulator, no torch. The mission is "Find the blue landing pad and land on it." from starts no farther
than 44 m from the pad; farther starts are a separate, open problem (docs/long_range_same_color_grounding.md).

Both sets have the same make-up in three bands of start distance. A band holds landings on the blue pad
from starts with the pad in view, out of view, at the edge of the view, and with only the object that
shares its pair of places in view; a start within range of both places, flown again in the layout that
exchanges the two and again told to find the neighbour; landings on the red pad; and the first two poses
told to approach instead. The validation set is flown in layouts that were trained on (from starts that were not), the
test set in layouts that never were.
"""
import json
import math
from pathlib import Path
from .maps import load_map,object_position
from .generalization import variant
from .gen_v3 import plan_one

ROOT=Path(__file__).resolve().parents[2]
FILES={'validation':ROOT/'configs/gen_v3_canonical_44m_validation.json','test':ROOT/'configs/gen_v3_canonical_44m_test.json'}
PLANNED=ROOT/'outputs/examples/gen_v3/planned_starts.json'
SEALED=ROOT/'configs/gen_v3_test_spawns.json'
# Rows of one band: role, object, base kind, side, layouts it may be flown in (None: all of the set's).
# A `pair` start lies within range of the pad and of its neighbour, so that it can be flown again with the two exchanged and
# with the neighbour named; a `past` start shows only the neighbour.
ROWS={'validation':(('visible','blue_pad','visible',1,None),('search','blue_pad','search',-1,None),('edge','blue_pad','peripheral',1,None),
                    ('pair','blue_pad','visible',-1,('a','b')),('past','blue_pad','search',1,None),('behind','blue_pad','search','behind',None),
                    ('red_visible','red_pad','visible',-1,None),('red_search','red_pad','search',1,None)),
      'test':(('visible','blue_pad','visible',1,None),('search','blue_pad','search',-1,None),('edge','blue_pad','peripheral',1,None),
              # g and h exchange the blue pad with the red pad; i and j exchange it with a blue object.
              ('pair','blue_pad','visible',-1,('g','h')),('past','blue_pad','search',1,None),('visible2','blue_pad','visible',-1,None),
              ('behind','blue_pad','search','behind',None),('pair2','blue_pad','visible',1,('i','j')),
              ('red_visible','red_pad','visible',-1,None),('red_search','red_pad','search',1,None))}


def neighbour(scene,layout,target):
    """The object on the other place of the target's pair in this layout, or None."""
    place=object_position(scene,layout,target)
    for a,b in scene.get('pairs',[]):
        for mine,other in ((a,b),(b,a)):
            if scene['sites'][mine]==place:
                return next((name for name in scene['layouts'][layout] if object_position(scene,layout,name)==scene['sites'][other]),None)
    return None


def earlier_points(which):
    """Starts a canonical start keeps away from, by map: every flown Gen-v3 plan, the sealed test, and for the test the validation set."""
    points=[(where,xy) for where,starts in json.loads(PLANNED.read_text(encoding='utf-8'))['starts'].items() for xy in starts]
    points+=[(e['map'],e['start_xy']) for episodes in json.loads(SEALED.read_text(encoding='utf-8'))['sets'].values() for e in episodes]
    if which=='test':points+=[(e['map'],e['start_xy']) for e in json.loads(FILES['validation'].read_text(encoding='utf-8'))['episodes']]
    return points


def build(config,oft,which):
    """The validation or the test set, in flying order."""
    settings=config['canonical'];seed=settings['seeds'][which][0];points=earlier_points(which);episodes=[];counter=0
    scenes={name:load_map(name) for name in ('field','lot')}
    order=[(name,settings[f'{which}_layouts'][name][index]) for index in range(4) for name in ('field','lot')]
    def excluded_in(map_id):
        mine=[xy for where,xy in points if where==map_id]
        return lambda x,y,direction:any(math.hypot(x-px,y-py)<settings['separation_m'] for px,py in mine)
    for band,span in settings['bands_m'].items():
        for role,target,base,side,allowed in ROWS[which]:
            pair=role.startswith('pair');choices=[(s,l) for s,l in order if allowed is None or l in allowed]
            for attempt in range(600):
                scene_id,layout=choices[(counter+attempt)%len(choices)];scene=scenes[scene_id];other=neighbour(scene,layout,target)
                if (pair or role=='past') and other is None:continue
                spec={'base':base,'distance_m':span}
                if role=='past':spec['distractor']={'relation':[other],'within_m':settings['past_within_m']}
                episode,seed=plan_one(config,oft,scene,layout,target,f'{band}_{role}',spec,'land','land',seed,which,f'c{which[0]}-{{seed:04d}}-{target}-{band}_{role}',
                                      side=side,excluded=excluded_in(scene_id),tries=4)
                if episode is None:continue
                twins=[]
                if pair:
                    # The same pose with the pad and its neighbour exchanged, and the same pose told to find the neighbour.
                    # (With a tall neighbour close by the teacher may climb over it first, as it does for any close obstacle.)
                    twins=[('swap',variant(config,oft,scene,episode,f'{episode["id"]}-swapped',layout=scene['swaps'][layout],allow_climb=True)),
                           ('query',variant(config,oft,scene,episode,f'{episode["id"]}-find-{other}',target=other,task='approach',instruction_id='find',allow_climb=True))]
                elif role in ('visible','search'):
                    # The same pose told to approach instead of to land.
                    twins=[('approach',variant(config,oft,scene,episode,f'{episode["id"]}-approach',task='approach',instruction_id='approach'))]
                # Every flight of the set starts inside the range, the twins included.
                if all(item is not None and item['start_distance_m']<=settings['max_start_m'] for _,item in twins):break
            else:raise RuntimeError(f'No {which} start: {band} {role}')
            counter+=1
            for item_role,item in [(role,episode)]+twins:
                if item is not episode:item['twin_of']=episode['id']
                item.update(role=item_role,set=which,band=band,band_m=span,neighbour=neighbour(scene,item['layout'],item['target']))
                episodes.append(item)
    return episodes


def describe(config,which,episodes):
    settings=config['canonical']
    return {'description':f'Canonical clean mission, {which} starts: every start is at most {settings["max_start_m"]:.0f} m from the object named. '
                          +('Flown in layouts that were trained on, from starts that were not; used for the smoke and representative gates only.' if which=='validation' else
                            'Flown in layouts no training or validation episode used; flown once, after the gates have passed, and not used for any gate.')
                          +' Seeds and starts are disjoint from every earlier plan, from the sealed 156-start Gen-v3 test and from each other.',
            'mission':settings['mission'],'max_start_m':settings['max_start_m'],'bands_m':settings['bands_m'],'seeds':settings['seeds'][which],'episodes':episodes}


def smoke_ids(config,episodes):
    """Ten validation starts: in every band a landing with the pad in view, one with it out of view and the first one's
    approach twin; and one landing on the red pad."""
    chosen=[]
    for band in config['canonical']['bands_m']:
        visible=next(e for e in episodes if e['band']==band and e['role']=='visible');search=next(e for e in episodes if e['band']==band and e['role']=='search')
        chosen+=[visible['id'],search['id'],next(e['id'] for e in episodes if e.get('twin_of')==visible['id'] and e['role']=='approach')]
    chosen.append(next(e['id'] for e in episodes if e['band']=='mid' and e['role']=='red_visible'))
    return chosen
