"""Start sets of the language-vision grounding architecture experiments.

No simulator, no torch. Two architectures are tried on the Gen-v3c data as it is: FiLM on the visual
features, and, only if that does not pass, a language-to-vision cross-attention adapter
(docs/language_vision_grounding_architecture.md). Both are judged on one small adversarial set that
every model flies twice from every start, because single flights of such starts were seen to differ.

  pilot       sixteen starts where the Gen-v3c checkpoints went wrong: an object that shares the named
              object's colour or shape is what the view shows, placed where it stays in view longest
              (on the side the search turns to, or straight ahead and close), or the named object is
              lost to a forced turn that leaves such an object in view
  validation  a fresh canonical validation set for an architecture that passed the pilot

The starts lie in the six layouts of each scene that no episode was ever recorded in (t, u, p-s), away
from every earlier start. The 48 canonical test starts and the sealed Depot set are not touched.
"""
import json
from pathlib import Path
from .maps import load_map
from .gen_v3 import plan_one
from . import canonical,gen_v3c

ROOT=Path(__file__).resolve().parents[2]
FILES={'pilot':ROOT/'configs/grounding_architecture_pilot.json','validation':ROOT/'configs/grounding_architecture_validation.json'}
SCENES=gen_v3c.SCENES
# The pilot's sixteen starts: group, scene, layout, named object, related object, band, task, where the related object is.
#   right   on the side the search turns to, so that the turn carries it through the middle of the view
#   centre  straight ahead and close
#   left    on the other side: it leaves the view after a decision or two
#   any     anywhere in the first view
#   +1, -1  (lost) the side the forced turn goes to: +1 is the long way round for a search that turns right
PILOT=(('same_color','field','t','blue_pad','blue_cube','far','land','right'),('same_color','field','r','blue_pad','blue_cube','mid','land','right'),
       ('same_color','lot','t','blue_pad','blue_cone','near','land','right'),('same_color','field','p','blue_pad','blue_cone','far','land','centre'),
       ('same_color','field','s','red_pad','red_cube','mid','land','right'),('same_color','lot','r','red_pad','red_cube','far','land','right'),
       ('same_shape','lot','p','blue_pad','red_pad','mid','land','right'),('same_shape','field','p','blue_pad','red_pad','far','land','centre'),
       ('same_shape','lot','u','blue_cube','red_cube','mid','approach','right'),
       ('first_view','field','u','blue_cube','blue_pad','mid','approach','any'),('first_view','lot','u','blue_cone','blue_pad','far','approach','any'),
       ('first_view','field','s','blue_pad','blue_cube','near','land','left'),('first_view','lot','q','red_pad','blue_pad','far','land','left'),
       ('lost','field','t','blue_pad','blue_cube','far','land',1),('lost','lot','r','blue_pad','blue_cone','mid','land',-1),('lost','lot','q','blue_pad','red_pad','far','land',1))


def earlier_points(*files):
    """Every start an architecture start keeps away from, by map: all earlier plans and sets, the recorded hard negatives included."""
    points=gen_v3c.earlier_points(*gen_v3c.FILES.values(),*files)
    plan=json.loads(gen_v3c.TRAIN_PLAN.read_text(encoding='utf-8'))
    return points+[(episode['map'],episode['start_xy']) for split in ('train','val') for episode in plan[split]]


def first_view_rule(zone,where):
    """What the first view must hold of the related object, for one of the places named in PILOT."""
    if where=='right':return {'within_m':zone['within_m'],'bearing_deg':zone['right_deg']}
    if where=='left':return {'within_m':zone['within_m'],'bearing_deg':[-zone['right_deg'][1],-zone['right_deg'][0]]}
    if where=='centre':return {'within_m':zone['centre_within_m'],'centred_deg':zone['centre_deg']}
    return {'within_m':zone['within_m']}


def build_pilot(config,oft):
    """The sixteen adversarial starts, in flying order."""
    settings=config['grounding_architecture'];zone=settings['pilot']['hard_zone'];bands=config['gen_v3c']['bands_m'];lost=config['gen_v3c']['hard_negative']['lost']
    seed=settings['seeds']['pilot'][0];scenes={name:load_map(name) for name in SCENES};points=earlier_points();episodes=[]
    reach=config['gen_v3c']['reach_m'];order=list(bands)
    for index,(group,name,layout,target,other,band,task,where) in enumerate(PILOT):
        phrase='land' if task=='land' else 'find';label=f'ap-{{seed:05d}}-{target}-{group}';episode=None
        # The place the row names first; where earlier starts leave no room for it there, the other layouts that have the two
        # objects within reach of each other, and after those the neighbouring bands.
        places=[(name,layout)]+[(scene_id,other_layout) for scene_id in SCENES for other_layout in settings['layouts'][scene_id]
                                if (scene_id,other_layout)!=(name,layout) and other in gen_v3c.neighbours(scenes[scene_id],other_layout,target,'related',reach)]
        if other not in gen_v3c.neighbours(scenes[name],layout,target,'related',reach):raise RuntimeError(f'{other} is not within reach of {target} in {name}/{layout}')
        for chosen in sorted(order,key=lambda item:abs(order.index(item)-order.index(band))):
            for scene_id,place in places:
                scene=scenes[scene_id];span=bands[chosen];excluded=gen_v3c.keep_away(points,scene_id,settings['separation_m'])
                for attempt in range(60):
                    if group=='lost':
                        episode,seed=gen_v3c.plan_lost(config,oft,scene,place,target,other,group,span,task,phrase,seed,'pilot',label,(1,-1)[index%2],excluded,lost,tries=1,turn=where)
                    else:
                        # The target on either side of the nose or behind it, in turn; the related object where the row says.
                        episode,seed=gen_v3c.plan_first(config,oft,scene,place,target,other,group,span,task,phrase,seed,'pilot',label,(1,-1,'behind')[(index+attempt)%3],excluded,
                                                        first_view_rule(zone,where),tries=1)
                    if episode:break
                if episode:break
            if episode:break
        if episode is None:raise RuntimeError(f'No architecture pilot start: {PILOT[index]}')
        episode.update(role=group,set='architecture_pilot',band=chosen,band_m=bands[chosen],where=where,planned_place=[name,layout,band])
        episodes.append(episode);points.append((episode['map'],episode['start_xy']))
    return episodes


def build_validation(config,oft):
    """A fresh canonical validation set for the architecture that passed the pilot: the Gen-v3c make-up, new seeds, new starts."""
    settings=config['grounding_architecture'];points=earlier_points(FILES['pilot'])
    return gen_v3c.build_validation(config,oft,seed=settings['seeds']['validation'][0],points=points,split='validation',prefix='av')


def describe(config,which,episodes):
    settings=config['grounding_architecture']
    text={'pilot':'Language-vision grounding architectures, pilot set: sixteen adversarial starts, each flown twice by every model compared (Gen-v3c, FiLM, and the '
                  'cross-attention adapter if FiLM does not pass). Written before any architecture was trained.',
          'validation':'Language-vision grounding architectures, fresh canonical validation starts for an architecture that passed the pilot: every start is at '
                       'most 44 m from the object named. Used for its smoke and representative gates and for nothing else.'}
    return {'description':text[which]+' Flown in layouts no episode was recorded in. Seeds and starts are disjoint from every earlier plan and set.',
            'mission':config['canonical']['mission'],'evaluator':config['gen_v3c']['evaluator'],'max_start_m':config['gen_v3c']['max_start_m'],
            'bands_m':config['gen_v3c']['bands_m'],'seeds':settings['seeds'][which],'layouts':settings['layouts'],
            **({'repeats':settings['pilot']['repeats']} if which=='pilot' else {}),'episodes':episodes}


def critical_ids(episodes):
    """Validation starts that ask about target selection (a related object in the first view, the named one outside it): flown twice."""
    scenes={}
    def related_first(episode):
        scene=scenes.setdefault(episode['map'],load_map(episode['map']))
        return episode['plan']['acquired_tick']!=0 and any(gen_v3c.relation_of(scene,episode['target'],item['name']) and item['distance_m']<=60. for item in episode.get('others_in_view',[]))
    return [episode['id'] for episode in episodes if related_first(episode)]
