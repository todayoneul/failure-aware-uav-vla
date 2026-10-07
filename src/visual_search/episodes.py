"""Visual-search episodes and the privileged teacher; no simulator, no torch.

Everything that knows where the target is lives here and in the runner's scoring. `policy_inputs`
is the single place that builds what a policy may see, and it takes no target geometry.
"""
import json
import math
import random
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'configs/visual_search.json'
DIRECTION_WORDS=('straight ahead','forward-left','forward-right','to your left','to your right','rear','degrees','meters away')


def load_config(path=None):
    return json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding='utf-8'))


def instruction_for(config,case,target):
    return config['instructions'][config['cases'][case]['instruction']].format(noun=config['targets'][target]['noun'])


def policy_inputs(front,down,instruction,proprio=None):
    """What a visual-search policy receives. There is no parameter through which target geometry could pass."""
    if any(word in instruction for word in DIRECTION_WORDS) or re.search(r'-?\d+\.\d+',instruction):
        raise ValueError('A visual-search instruction must not carry a direction, a distance or coordinates')
    return {'front':front,'down':down,'instruction':instruction,'proprio':proprio}


def make_episode(config,case,target,centre,seed):
    """Start pose for one episode: where to stand and which way to face, relative to the target centre."""
    rng=random.Random(f'{case}|{target}|{seed}');spec=config['targets'][target];kind=config['cases'][case]
    if 'start_xy' in kind:
        # A fixed place and heading shared by every target: only the sentence says which object is meant.
        x,y=kind['start_xy'];distance=math.hypot(centre[0]-x,centre[1]-y)
        to_target=math.degrees(math.atan2(centre[1]-y,centre[0]-x))
        offset=to_target-(kind['start_yaw_deg']+rng.uniform(-kind['yaw_jitter_deg'],kind['yaw_jitter_deg']))
    else:
        bearing=math.radians(rng.uniform(*spec['start_bearing_deg']));distance=rng.uniform(*spec['start_distance_m'])
        x=centre[0]+distance*math.cos(bearing);y=centre[1]+distance*math.sin(bearing)
        to_target=math.degrees(math.atan2(centre[1]-y,centre[0]-x))
        # Positive offset: the vehicle faces left of the target, so the target sits to its right.
        side=rng.choice((-1,1));offset=side*rng.uniform(*kind['yaw_offset_deg'])
    episode={'id':f'{case}-{target}-{seed}','case':case,'target':target,'seed':seed,'start_xy':[x,y],
             'start_yaw_deg':to_target-offset,'target_bearing_deg':offset,'start_distance_m':distance,
             'instruction':instruction_for(config,case,target)}
    if 'kick_after_ticks' in kind:
        episode['kick']={'after_ticks':rng.randint(*kind['kick_after_ticks']),'degrees':rng.choice((-1,1))*rng.uniform(*kind['kick_deg'])}
    return episode


def relative_target(state,centre):
    """Body-frame bearing (deg, right positive) and horizontal distance to the target centre."""
    x,y,z,w=state['orientation'];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    dx,dy=centre[0]-state['position'][0],centre[1]-state['position'][1]
    bearing=math.atan2(dy,dx)-yaw
    return math.degrees(math.atan2(math.sin(bearing),math.cos(bearing))),math.hypot(dx,dy),yaw


class Teacher:
    """Deterministic expert that uses the target's true position.

    Seen: turn toward it and fly forward once it is roughly centred. Not seen: sweep in one fixed
    direction, and climb a little after a full turn without a sighting. Close enough: stop.
    """
    def __init__(self,config,oft):
        self.config=config;self.oft=oft;self.swept=0.;self.climbed=0
        (_,self.forward_max),(_,self.down_max),(_,self.yaw_max)=oft['action_bounds'].values()
        self.direction=1. if config['teacher']['search_direction']=='right' else -1.

    def act(self,bearing_deg,distance_m,visible):
        teacher=self.config['teacher']
        if distance_m<=self.config['arrive_distance_m']:
            return [0.,0.,0.],'stop'
        if visible:
            self.swept=0.
            yaw=max(-self.yaw_max,min(self.yaw_max,math.radians(bearing_deg)))
            forward=self.forward_max*max(0.,1-abs(bearing_deg)/teacher['centre_deg'])
            # Ease off over the last metres so the stop is not a step change.
            forward=min(forward,max(.2,(distance_m-self.config['arrive_distance_m'])/4))
            return [forward,0.,yaw],'approach' if forward>.05 else 'align'
        self.swept+=math.degrees(self.yaw_max)
        if self.swept>=teacher['sweep_before_climb_deg'] and self.climbed<2:
            self.swept=0.;self.climbed+=1;self.pending_climb=teacher['climb_m']
        pending=getattr(self,'pending_climb',0.)
        if pending>0:
            step=min(self.down_max,pending);self.pending_climb=pending-step
            return [0.,-step,0.],'climb'
        return [0.,0.,self.direction*self.yaw_max],'search'
