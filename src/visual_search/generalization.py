"""Generalisation episodes for visual search: start states, the teacher, and a dry run of it.

No simulator, no torch. An episode is fully determined by (map, split, seed): which object is named,
what kind of start it is, where the vehicle stands, which way it faces and how high it flies. Before
an episode is accepted the teacher is flown through it on the map's boxes alone, so every planned
episode is one the teacher reaches the target in without coming near an obstacle.

The teacher here differs from the pilot's in one respect that matters for a policy that sees a single
frame: every label is a function of what the cameras show. It climbs when something close fills the
view ahead, not after counting a full turn. The strategies that need memory (left sweep, alternating
scan, turning back to where the target was last seen, climbing after a completed sweep) exist for
comparison and are marked as such.
"""
import math
import random
from .maps import MapGeometry

KINDS=('visible','peripheral','search','altitude','reacquire')
SECTORS=('front','front_right','right','behind_right','behind','behind_left','left','front_left')
BANDS=('near','medium','far')
# Strategies a single-frame policy can reproduce, and the ones that depend on what the teacher remembers.
STATELESS_STRATEGIES=('right',)
MEMORY_STRATEGIES=('left','scan','last_seen','sweep_climb')
STRATEGIES=STATELESS_STRATEGIES+MEMORY_STRATEGIES


def sector_of(bearing_deg):
    """Which eighth of the circle the target is in, seen from the vehicle (right is positive)."""
    return SECTORS[round(bearing_deg/45)%8]


def band_of(config,distance_m):
    for name,(low,high) in config['generalization']['bands_m'].items():
        if low<=distance_m<=high:return name
    return 'outside'


def kind_table(config):
    """One cycle of start kinds in the configured proportions, interleaved rather than grouped."""
    weights=config['generalization']['kinds'];total=sum(weights.values());table=[];credit={kind:0. for kind in weights}
    for _ in range(total):
        for kind in weights:credit[kind]+=weights[kind]/total
        chosen=max(weights,key=lambda kind:credit[kind]);credit[chosen]-=1.;table.append(chosen)
    return table


def spec_for(config,seed,targets):
    """The discrete choices of one episode follow from its seed, so consecutive seeds are balanced."""
    table=kind_table(config);index=seed//len(targets)
    return {'target':targets[seed%len(targets)],'kind':table[index%len(table)],'band':BANDS[index%len(BANDS)]}


class SearchTeacher:
    """Privileged expert. It is told where the target is; what it does with that is tied to the view.

    Close and in view (front or down camera): stop. In view: turn to it and fly forward once roughly
    centred, climbing first if something close stands in the way. Not in view: climb if a close
    obstacle fills the view ahead, otherwise keep turning.
    """
    def __init__(self,config,oft,strategy='right',ceiling_m=14.):
        if strategy not in STRATEGIES:raise ValueError(f'Unknown search strategy: {strategy!r}')
        self.config=config;self.settings=config['generalization']['teacher'];self.strategy=strategy;self.ceiling=ceiling_m
        (_,self.forward_max),(_,self.down_max),(_,self.yaw_max)=oft['action_bounds'].values()
        self.last_side=None;self.swept=0.;self.pending_climb=0.
        self.leg_sign=1.;self.leg_left=self.leg_length=self.settings['scan_deg']

    def _direction(self):
        if self.strategy=='left':return -1.
        if self.strategy=='last_seen' and self.last_side is not None:return self.last_side
        if self.strategy=='scan':
            # Right for one leg, back left for a longer one, and so on, widening each time.
            if self.leg_left<=0:
                self.leg_sign=-self.leg_sign;self.leg_length+=self.settings['scan_deg'];self.leg_left=self.leg_length
            self.leg_left-=math.degrees(self.yaw_max)
            return self.leg_sign
        return 1.

    def act(self,view):
        """view: bearing_deg, distance_m, visible, below, ahead_m, height_m -> (action, state name)."""
        settings=self.settings;bearing=view['bearing_deg'];distance=view['distance_m']
        if distance<=self.config['arrive_distance_m'] and (view['visible'] or view['below']):
            return [0.,0.,0.],'stop'
        can_climb=view['height_m']<self.ceiling-.25
        if view['visible']:
            self.last_side=1. if bearing>=0 else -1.;self.swept=0.
            self.leg_sign=1.;self.leg_left=self.leg_length=settings['scan_deg']
            yaw=max(-self.yaw_max,min(self.yaw_max,math.radians(bearing)))
            if view['ahead_m']<settings['safety_m'] and can_climb:return [0.,-self.down_max,yaw],'climb'
            forward=self.forward_max*max(0.,1-abs(bearing)/settings['centre_deg'])
            # Ease off over the last metres so the stop is not a step change.
            forward=min(forward,max(.2,(distance-self.config['arrive_distance_m'])/4))
            return [forward,0.,yaw],'approach' if forward>.05 else 'align'
        if self.strategy=='sweep_climb':
            # The pilot's rule: climb after a completed sweep. A single frame cannot tell when that is.
            if self.pending_climb>0:
                step=min(self.down_max,self.pending_climb);self.pending_climb-=step
                return [0.,-step,0.],'climb'
            self.swept+=math.degrees(self.yaw_max)
            if self.swept>=settings['sweep_before_climb_deg'] and can_climb:self.swept=0.;self.pending_climb=settings['climb_m']
        elif view['ahead_m']<settings['blocked_m'] and can_climb:
            return [0.,-self.down_max,0.],'climb'
        return [0.,0.,self._direction()*self.yaw_max],'search'


def geometric_view(config,geometry,target,x,y,height_m,yaw_rad):
    """What the teacher is told, computed from the map's boxes instead of the simulator's depth images."""
    settings=config['generalization'];item=geometry.objects[target];centre=item['centre']
    dx,dy=centre[0]-x,centre[1]-y;distance=math.hypot(dx,dy)
    bearing=math.degrees(math.atan2(math.sin(math.atan2(dy,dx)-yaw_rad),math.cos(math.atan2(dy,dx)-yaw_rad)))
    half=settings['fov_half_deg'];visible=False
    if abs(bearing)<=half:
        for point in geometry.sight_points(target):
            if abs(math.degrees(math.atan2(point[2]-height_m,max(distance,1e-6))))<=half and geometry.blocking((x,y,height_m),point,ignore=(target,)) is None:
                visible=True;break
    top=item['size_m'][2]
    below=height_m>top and distance-item['size_m'][0]/2<=(height_m-top)*math.tan(math.radians(half))
    return {'bearing_deg':bearing,'distance_m':distance,'visible':visible,'below':below,'height_m':height_m,
            'ahead_m':geometry.ahead(x,y,yaw_rad,height_m,settings['ahead_reach_m'],settings['corridor_clearance_m'],ignore=(target,))}


def kick_turn(episode,index,left,yaw_max):
    """The forced turn of a reacquire episode: starts at one tick, then lasts until the angle is used up."""
    kick=episode.get('kick')
    if kick and index==kick['after_ticks']:left=math.radians(kick['degrees'])
    if abs(left)<=1e-6:return None,0.
    turn=max(-yaw_max,min(yaw_max,left))
    return turn,left-turn


def rollout(config,oft,geometry,episode,ceiling_m):
    """Fly the teacher through an episode on the map's boxes: kinematics only, no simulator."""
    settings=config['generalization'];teacher=SearchTeacher(config,oft,episode['strategy'],ceiling_m)
    x,y=episode['start_xy'];yaw=math.radians(episode['start_yaw_deg']);height=episode['start_height_m'];target=episode['target']
    kick_left=0.;stops=0;clearance=float('inf');states={};acquired=None;top=height;reason='max_steps';tick=0
    yaw_max=oft['action_bounds']['yaw_rad'][1]
    for tick in range(settings['max_ticks']):
        view=geometric_view(config,geometry,target,x,y,height,yaw)
        action,state=teacher.act(view)
        if view['visible'] and acquired is None:acquired=tick
        turn,kick_left=kick_turn(episode,tick,kick_left,yaw_max)
        if turn is not None:action=[0.,0.,turn];state='kick'
        states[state]=states.get(state,0)+1
        stops=stops+1 if state=='stop' else 0
        if stops>=config['stop_ticks']:reason='teacher_stop';break
        yaw+=action[2];x+=action[0]*math.cos(yaw);y+=action[0]*math.sin(yaw);height-=action[1];top=max(top,height)
        clearance=min(clearance,geometry.nearest(x,y,above_m=height-settings['corridor_clearance_m'],ignore=(target,))[0])
        if clearance<settings['rollout_clearance_m']:reason='too_close';break
    final=math.hypot(geometry.objects[target]['centre'][0]-x,geometry.objects[target]['centre'][1]-y)
    return {'reason':reason,'ticks':tick+1,'clearance_m':clearance,'climbed_m':top-episode['start_height_m'],'states':states,
            'acquired_tick':acquired,'final_distance_m':final}


def _bearing(settings,kind,rng,side=None):
    """Where the target sits relative to the vehicle's nose at the start (deg, right positive)."""
    sign=side if side in (-1,1) else rng.choice((-1,1))
    if kind in ('visible','reacquire'):low,high=settings['bearing_deg']['visible']
    elif kind=='peripheral':low,high=settings['bearing_deg']['peripheral']
    elif kind=='search':low,high=settings['bearing_deg']['search']
    else:low,high=0.,180.
    if side=='behind':return rng.choice((-1,1))*rng.uniform(150.,180.)
    return sign*rng.uniform(low,high)


def make_start(config,oft,map_config,layout,target,kind,band,seed,split,strategy='right',side=None,
               direction_deg=None,excluded=None,instruction_id=None,name=None):
    """One start state, or None when the map offers none of this kind for this object.

    `direction_deg` restricts where the start lies as seen from the target (low, high); `excluded`
    is a predicate over (x, y, direction) that keeps training starts away from held-out ones.
    """
    settings=config['generalization'];geometry=MapGeometry(map_config,layout);centre=geometry.objects[target]['centre']
    rng=random.Random(f'{map_config["id"]}|{split}|{seed}|{target}|{kind}|{band}')
    low,high=settings['bands_m'][band];ceiling=map_config['altitude']['ceiling_m']
    instruction_id=instruction_id or rng.choice(settings['instructions']['train'])
    for attempt in range(settings['attempts'][kind] if kind in settings['attempts'] else settings['attempts']['default']):
        distance=rng.uniform(low,high)
        direction=rng.uniform(*direction_deg) if direction_deg else rng.uniform(0.,360.)
        x=centre[0]+distance*math.cos(math.radians(direction));y=centre[1]+distance*math.sin(math.radians(direction))
        height=rng.uniform(*map_config['altitude']['start_m']);bearing=_bearing(settings,kind,rng,side)
        if not geometry.inside_area(x,y) or geometry.nearest(x,y)[0]<settings['start_margin_m']:continue
        if excluded and excluded(x,y,direction%360.):continue
        sees=geometry.sees(x,y,height,target)
        if kind=='altitude':
            # Hidden from here by something close and low enough to climb over, and in sight from the ceiling.
            if sees or not geometry.sees(x,y,ceiling,target):continue
            blocker=geometry.blocking((x,y,height),geometry.sight_points(target)[-1],ignore=(target,))
            box=next(item for item in geometry.boxes if item['name']==blocker)
            if box['top_m']>ceiling-settings['corridor_clearance_m']-.5 or geometry._gap(box,x,y)>settings['teacher']['blocked_m']-1.:continue
        else:
            # A search start has no part of the object inside the front view, not just its centre outside.
            if kind=='search' and abs(bearing)-math.degrees(math.atan(geometry.objects[target]['size_m'][0]/2/distance))<46.:continue
            # Nothing may hide the target or stand in the straight track to it.
            if not sees:continue
            if not geometry.corridor_clear((x,y),centre,height,settings['corridor_half_width_m'],settings['corridor_clearance_m'],ignore=(target,)):continue
        to_target=math.degrees(math.atan2(centre[1]-y,centre[0]-x))
        episode={'id':name or f'{split}-{seed:04d}-{target}-{kind}','split':split,'seed':seed,'map':map_config['id'],'layout':layout,
                 'target':target,'kind':kind,'case':kind,'band':band,'sector':sector_of(bearing),'strategy':strategy,
                 'start_xy':[round(x,2),round(y,2)],'start_yaw_deg':round(to_target-bearing,2),'start_height_m':round(height,2),
                 'target_bearing_deg':round(bearing,2),'start_distance_m':round(distance,2),'start_direction_deg':round(direction%360.,2),
                 'instruction_id':instruction_id,'instruction':instruction_for_id(config,map_config,target,instruction_id)}
        if kind=='reacquire':
            kick=settings['kick']
            episode['kick']={'after_ticks':rng.randint(*kick['after_ticks']),'degrees':round(rng.choice((-1,1))*rng.uniform(*kick['degrees']),1)}
        plan=rollout(config,oft,geometry,episode,ceiling)
        if plan['reason']!='teacher_stop':continue
        if kind=='altitude' and plan['climbed_m']<settings['altitude_min_climb_m']:continue
        if kind!='altitude' and plan['climbed_m']>0:continue
        episode['plan']={'ticks':plan['ticks'],'clearance_m':round(plan['clearance_m'],2),'climbed_m':round(plan['climbed_m'],2),
                         'acquired_tick':plan['acquired_tick'],'states':plan['states']}
        return episode
    return None


def instruction_for_id(config,map_config,target,instruction_id):
    return config['instructions'][instruction_id].format(noun=map_config['objects'][target]['noun'])


def exclusion(held_out,map_id,layout,target,radius_m):
    """Predicate that rejects a training start near any held-out start or inside the target's held-out wedge."""
    points=[episode['start_xy'] for episodes in held_out['sets'].values() for episode in episodes if episode['map']==map_id]
    wedge=held_out['wedges'].get(map_id,{}).get(layout,{}).get(target)
    def excluded(x,y,direction):
        if wedge and in_wedge(direction,wedge):return True
        return any(math.hypot(x-px,y-py)<radius_m for px,py in points)
    return excluded


def in_wedge(direction_deg,wedge):
    low,high=wedge
    return (direction_deg-low)%360.<=(high-low)%360.


def plan_split(config,oft,map_config,split,seeds,held_out=None,strategy='right'):
    """Episodes for consecutive seeds of one split. A kind the map cannot offer for an object falls back to `search`."""
    settings=config['generalization'];targets=map_config['train_objects'];layouts=map_config['train_layouts'];episodes=[]
    for seed in seeds:
        spec=spec_for(config,seed,targets);layout=layouts[random.Random(f'layout|{map_config["id"]}|{seed}').randrange(len(layouts))]
        # Left and right alternate over the successive episodes of one kind, so neither side is favoured for any kind.
        table=kind_table(config);index=seed//len(targets);position=index%len(table)
        order=table[:position].count(spec['kind'])+index//len(table)*table.count(spec['kind'])
        side=(1,-1)[(order+seed%len(targets))%2]
        excluded=exclusion(held_out,map_config['id'],layout,spec['target'],settings['exclusion_radius_m']) if held_out else None
        episode=None
        for kind in dict.fromkeys((spec['kind'],'search')):
            for band in dict.fromkeys((spec['band'],)+BANDS):
                episode=make_start(config,oft,map_config,layout,spec['target'],kind,band,seed,split,strategy,side=side,excluded=excluded)
                if episode:break
            if episode:break
        if episode is None:raise RuntimeError(f'No start found for seed {seed} ({spec})')
        episode['requested_kind']=spec['kind'];episodes.append(episode)
    return episodes
