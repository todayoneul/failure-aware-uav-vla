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
from .maps import MapGeometry,load_landing

KINDS=('visible','peripheral','search','altitude','reacquire')
SECTORS=('front','front_right','right','behind_right','behind','behind_left','left','front_left')
BANDS=('near','medium','far')
# Strategies a single-frame policy can reproduce, and the ones that depend on what the teacher remembers.
STATELESS_STRATEGIES=('right',)
MEMORY_STRATEGIES=('left','scan','last_seen','sweep_climb')
STRATEGIES=STATELESS_STRATEGIES+MEMORY_STRATEGIES
# What an instruction asks for at the end: stop in the air near the object, or touch down on it.
TASKS=('approach','land')
STOP_STATES=('stop','landed')


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

    When the view says the task is to land (`task`), nothing changes until the target is found; then
    it flies on over the pad instead of stopping short, descends once it is over the middle, and
    gives the zero action only after touchdown. Every label is still a function of the two views.
    """
    def __init__(self,config,oft,strategy='right',ceiling_m=14.,landing=None):
        if strategy not in STRATEGIES:raise ValueError(f'Unknown search strategy: {strategy!r}')
        self.config=config;self.settings=config['generalization']['teacher'];self.strategy=strategy;self.ceiling=ceiling_m
        self.landing=(landing or load_landing())['teacher']
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
        landing=view.get('task')=='land'
        if landing and view.get('landed'):return [0.,0.,0.],'landed'
        # Over a pad the object is below the vehicle whether or not a camera's centre shows it.
        if not landing and distance<=self.config['arrive_distance_m'] and (view['visible'] or view['below'] or (view.get('over') and view.get('surface_m'))):
            return [0.,0.,0.],'stop'
        can_climb=view['height_m']<self.ceiling-.25
        if landing and (view['visible'] or view['below'] or view.get('over')):
            land=self.landing;above=view['height_m']-view['surface_m']
            self.last_side=1. if bearing>=0 else -1.;self.swept=0.
            committed=view.get('over') and above<=land['commit_height_m'] and distance<=land['commit_radius_m']
            if distance<=land['align_radius_m'] or committed:
                # Over the middle of the pad, or already low over it: straight down, slower over the last metres.
                return [0.,self.down_max if above>land['slow_height_m'] else land['final_descent_m'],0.],'descend'
            yaw=max(-self.yaw_max,min(self.yaw_max,math.radians(bearing)))
            if view['visible'] and view['ahead_m']<settings['safety_m'] and can_climb:return [0.,-self.down_max,yaw],'climb'
            forward=self.forward_max*max(0.,1-abs(bearing)/settings['centre_deg'])
            # Ease in over the pad instead of flying across it.
            forward=min(forward,max(land['min_forward_m'],distance/land['ease_m']))
            # Coming in high, lose height on the way so the pad stays in one of the two views.
            down=land['glide_down_m'] if distance<=land['glide_distance_m'] and above>land['glide_height_m'] else 0.
            return [forward,down,yaw],('approach' if distance>self.config['arrive_distance_m'] else 'final') if forward>.05 else 'align'
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


class Pushes:
    """Forced moves of an approach episode, flown but never a label.

    An approach flight stops short of its target, so the sentence 'approach' is otherwise never seen from
    closer than that. Once the teacher has stopped, the vehicle is pushed on toward the target (and
    sometimes down); where that leaves it the teacher's label is again to stop. After the first push the
    teacher holds its stop for `hold` ticks before the next one, so that label chunks without a forced
    move in them exist."""
    def __init__(self,episode):
        self.queue=[dict(item) for item in episode.get('pushes',[])];self.hold=episode.get('push_hold',1);self.left=0;self.current=None;self.stopped=0;self.done=0

    def pending(self):return bool(self.queue) or self.left>0

    def due(self,item,state,view):
        """Has the moment of the next forced move come?"""
        when=item.get('when','stop')
        # A landing pushed sideways during its last metres, so that it touches down away from the middle of the pad.
        if when=='descend':return state=='descend' and view['height_m']-view['surface_m']<=item['below_m']
        # A hop after touchdown: the vehicle is lifted a little and has to come down and stop again.
        if when=='landed':return state=='landed'
        self.stopped=self.stopped+1 if state in STOP_STATES else 0
        return self.stopped>=(1 if self.done==0 else self.hold)

    def step(self,state,view,yaw_max):
        """The forced action for this tick, or None. `state` is the teacher's state at this tick."""
        if self.left>0:self.left-=1
        else:
            if not self.queue or not self.due(self.queue[0],state,view):return None
            self.current=self.queue.pop(0);self.left=self.current['ticks']-1;self.done+=1;self.stopped=0
        when=self.current.get('when','stop')
        if when=='landed':return [0.,-self.current['up_m'],0.]
        if when=='descend':
            # Straight on while still coming down, but not out of the part of the pad where the descent goes on.
            if view['distance_m']>=self.current['max_distance_m']:self.left=0;return None
            return [self.current['forward_m'],self.current['down_m'],0.]
        # Not past the middle of the pad, and not lower than a safe height above its top.
        if view['distance_m']<=self.current['until_m']:self.left=0;return None
        yaw=max(-yaw_max,min(yaw_max,math.radians(view['bearing_deg'])))
        down=min(self.current['down_m'],max(0.,view['height_m']-view['surface_m']-self.current['floor_m']))
        return [self.current['forward_m']*max(0.,1-abs(view['bearing_deg'])/45.),down,yaw]


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
            'over':geometry.over(x,y,target),'surface_m':top if item['landable'] else 0.,
            'ahead_m':geometry.ahead(x,y,yaw_rad,height_m,settings['ahead_reach_m'],settings['corridor_clearance_m'],ignore=(target,))}


def kick_turn(episode,index,left,yaw_max):
    """The forced turn of a reacquire episode: starts at one tick, then lasts until the angle is used up."""
    kick=episode.get('kick')
    if kick and index==kick['after_ticks']:left=math.radians(kick['degrees'])
    if abs(left)<=1e-6:return None,0.
    turn=max(-yaw_max,min(yaw_max,left))
    return turn,left-turn


def rollout(config,oft,geometry,episode,ceiling_m,trace=None):
    """Fly the teacher through an episode on the map's boxes: kinematics only, no simulator.

    `trace`, when a list is given, receives the pose and the teacher's state at every tick (before the tick's move)."""
    settings=config['generalization'];landing=load_landing();teacher=SearchTeacher(config,oft,episode['strategy'],ceiling_m,landing)
    x,y=episode['start_xy'];yaw=math.radians(episode['start_yaw_deg']);height=episode['start_height_m'];target=episode['target']
    kick_left=0.;stops=0;clearance=float('inf');states={};acquired=None;top=height;reason='max_steps';tick=0
    yaw_max=oft['action_bounds']['yaw_rad'][1];task=episode.get('task','approach');landed=False;pushes=Pushes(episode)
    # Heights are counted from where the vehicle rests on the ground, so standing on a pad reads as the pad's height.
    rest=geometry.objects[target]['size_m'][2]
    for tick in range(episode.get('max_ticks',settings['max_ticks'])):
        view=geometric_view(config,geometry,target,x,y,height,yaw)
        # Touchdown: over the pad and down on its top. The simulator holds the vehicle there until it is told to climb.
        landed=task=='land' and view['over'] and height<=rest+1e-6
        view.update(task=task,landed=landed)
        action,state=teacher.act(view)
        if view['visible'] and acquired is None:acquired=tick
        turn,kick_left=kick_turn(episode,tick,kick_left,yaw_max)
        if turn is not None:action=[0.,0.,turn];state='kick'
        forced=pushes.step(state,view,yaw_max)
        if forced:action=forced;state='push'
        states[state]=states.get(state,0)+1
        if trace is not None:trace.append({'tick':tick,'x':x,'y':y,'height_m':height,'yaw_rad':yaw,'visible':view['visible'],'distance_m':view['distance_m'],'state':state})
        stops=stops+1 if state in STOP_STATES else 0
        if stops>=config['stop_ticks'] and not pushes.pending():reason='teacher_stop';break
        if landed and action[1]>=0:continue
        yaw+=action[2];x+=action[0]*math.cos(yaw);y+=action[0]*math.sin(yaw);height-=action[1];top=max(top,height)
        if task=='land' and geometry.over(x,y,target):height=max(height,rest)
        clearance=min(clearance,geometry.nearest(x,y,above_m=height-settings['corridor_clearance_m'],ignore=(target,))[0])
        if clearance<settings['rollout_clearance_m']:reason='too_close';break
    final=math.hypot(geometry.objects[target]['centre'][0]-x,geometry.objects[target]['centre'][1]-y)
    return {'reason':reason,'ticks':tick+1,'clearance_m':clearance,'climbed_m':top-episode['start_height_m'],'states':states,
            'acquired_tick':acquired,'final_distance_m':final,'landed':landed}


def _bearing(settings,kind,rng,side=None,span=None):
    """Where the target sits relative to the vehicle's nose at the start (deg, right positive)."""
    sign=side if side in (-1,1) else rng.choice((-1,1))
    if span:low,high=span
    elif kind in ('visible','reacquire'):low,high=settings['bearing_deg']['visible']
    elif kind=='peripheral':low,high=settings['bearing_deg']['peripheral']
    elif kind=='search':low,high=settings['bearing_deg']['search']
    else:low,high=0.,180.
    if side=='behind':return rng.choice((-1,1))*rng.uniform(150.,180.)
    return sign*rng.uniform(low,high)


def related(map_config,target,other,relation):
    """Whether `other` is a distractor of the given kind for `target`: same_color, same_shape, a list of names, or any."""
    if other==target:return False
    if isinstance(relation,(list,tuple)):return other in relation
    if relation in ('same_color','same_shape'):
        key=relation.split('_')[1];a=map_config['objects'][target].get('attributes',{});b=map_config['objects'][other].get('attributes',{})
        return key in a and a.get(key)==b.get(key)
    return True


def others_in_view(config,geometry,target,x,y,height_m,yaw_deg,view_deg):
    """Every other object of the layout that the front view shows from a pose: name, bearing (right positive), distance."""
    found=[]
    for name,item in geometry.objects.items():
        if name==target:continue
        dx,dy=item['centre'][0]-x,item['centre'][1]-y
        bearing=(math.degrees(math.atan2(dy,dx))-yaw_deg+180.)%360.-180.
        if abs(bearing)<=view_deg and geometry.sees(x,y,height_m,name):
            found.append({'name':name,'bearing_deg':round(bearing,1),'distance_m':round(math.hypot(dx,dy),1)})
    return found


def distractor_ok(rule,map_config,target,distance,shown):
    """Does the start show a distractor the way a rule asks: in view, near the middle of it, or nearer than the target."""
    fits=[item for item in shown if related(map_config,target,item['name'],rule.get('relation','any')) and item['distance_m']<=rule.get('within_m',float('inf'))]
    if rule.get('centred_deg') is not None:fits=[item for item in fits if abs(item['bearing_deg'])<=rule['centred_deg']]
    if rule.get('nearer_by_m') is not None:fits=[item for item in fits if item['distance_m']<=distance-rule['nearer_by_m']]
    # Side by side: about as far away as the target, so neither stands out by its size in the view.
    if rule.get('max_gap_m') is not None:fits=[item for item in fits if abs(item['distance_m']-distance)<=rule['max_gap_m']]
    return bool(fits)


def make_start(config,oft,map_config,layout,target,kind,band,seed,split,strategy='right',side=None,
               direction_deg=None,excluded=None,instruction_id=None,name=None,spec=None,task='approach',max_ticks=None):
    """One start state, or None when the map offers none of this kind for this object.

    `direction_deg` restricts where the start lies as seen from the target (low, high); `excluded`
    is a predicate over (x, y, direction) that keeps training starts away from held-out ones.
    `spec` describes a targeted kind: the base kind whose rules it follows and its own ranges of
    distance, target bearing and start height; `spec['distractor']` asks for another object in the
    first view. `task` is what the instruction asks for at the end (approach or land).
    """
    settings=config['generalization'];geometry=MapGeometry(map_config,layout);centre=geometry.objects[target]['centre']
    rng=random.Random(f'{map_config["id"]}|{split}|{seed}|{target}|{kind}|{band}')
    spec=spec or {};label=kind;kind=spec.get('base',kind)
    low,high=spec.get('distance_m') or settings['bands_m'][band];ceiling=map_config['altitude']['ceiling_m']
    instruction_id=instruction_id or rng.choice(settings['instructions']['train'])
    for attempt in range(settings['attempts'][kind] if kind in settings['attempts'] else settings['attempts']['default']):
        distance=rng.uniform(low,high)
        direction=rng.uniform(*direction_deg) if direction_deg else rng.uniform(0.,360.)
        x=centre[0]+distance*math.cos(math.radians(direction));y=centre[1]+distance*math.sin(math.radians(direction))
        height=rng.uniform(*(spec.get('height_m') or map_config['altitude']['start_m']));bearing=_bearing(settings,kind,rng,side,spec.get('bearing_deg'))
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
            if kind=='search' and not spec.get('partly_in_view') and abs(bearing)-math.degrees(math.atan(geometry.objects[target]['size_m'][0]/2/distance))<46.:continue
            # Nothing may hide the target or stand in the straight track to it.
            if not sees:continue
            if not geometry.corridor_clear((x,y),centre,height,settings['corridor_half_width_m'],settings['corridor_clearance_m'],ignore=(target,)):continue
        to_target=math.degrees(math.atan2(centre[1]-y,centre[0]-x))
        shown=None
        if spec.get('distractor') or task=='land':
            shown=others_in_view(config,geometry,target,x,y,height,to_target-bearing,settings['fov_half_deg'])
            if spec.get('distractor') and not distractor_ok(spec['distractor'],map_config,target,distance,shown):continue
        episode={'id':name or f'{split}-{seed:04d}-{target}-{label}','split':split,'seed':seed,'map':map_config['id'],'layout':layout,
                 'target':target,'kind':label,'case':label,'band':band_of(config,distance) if spec else band,'sector':sector_of(bearing),'strategy':strategy,
                 'start_xy':[round(x,2),round(y,2)],'start_yaw_deg':round(to_target-bearing,2),'start_height_m':round(height,2),
                 'target_bearing_deg':round(bearing,2),'start_distance_m':round(distance,2),'start_direction_deg':round(direction%360.,2),
                 'instruction_id':instruction_id,'instruction':instruction_for_id(config,map_config,target,instruction_id)}
        if spec:episode['base']=kind
        if spec.get('pushes'):
            push=spec['pushes']
            if 'sequence' in push:
                # Forced moves of a landing, each at its own moment (see Pushes.due).
                episode['pushes']=[{**{k:v for k,v in item.items() if k!='ticks'},'ticks':rng.randint(*item['ticks'])} for item in push['sequence']]
            else:
                episode['pushes']=[{'ticks':rng.randint(*push['ticks']),'forward_m':push['forward_m'],'down_m':rng.choice(push['down_m']),
                                    'until_m':push['until_m'],'floor_m':push['floor_m']} for _ in range(rng.randint(*push['count']))]
                episode['push_hold']=push['hold']
        if task!='approach' or max_ticks or shown is not None:
            episode['task']=task;episode['others_in_view']=shown or []
            if max_ticks:episode['max_ticks']=max_ticks
        if kind=='reacquire':
            kick=settings['kick']
            episode['kick']={'after_ticks':rng.randint(*kick['after_ticks']),'degrees':round(rng.choice((-1,1))*rng.uniform(*kick['degrees']),1)}
        plan=rollout(config,oft,geometry,episode,ceiling)
        if plan['reason']!='teacher_stop' or (task=='land' and not plan['landed']):continue
        if kind=='altitude' and plan['climbed_m']<settings['altitude_min_climb_m']:continue
        if kind!='altitude' and plan['climbed_m']>0 and not (spec.get('pushes') and task=='land'):continue
        episode['plan']={'ticks':plan['ticks'],'clearance_m':round(plan['clearance_m'],2),'climbed_m':round(plan['climbed_m'],2),
                         'acquired_tick':plan['acquired_tick'],'states':plan['states']}
        return episode
    return None


def instruction_for_id(config,map_config,target,instruction_id):
    spec=map_config['objects'][target]
    return config['instructions'][instruction_id].format(noun=spec['noun'],noun_long=spec.get('noun_long',spec['noun']))


def variant(config,oft,map_config,episode,name,layout=None,target=None,task=None,instruction_id=None,split=None,allow_climb=False):
    """The same vehicle pose with something else changed: the layout, the object named, or what the sentence asks for.

    Returns None when the teacher does not finish the changed episode in the dry run."""
    layout=layout or episode['layout'];target=target or episode['target'];task=task or episode.get('task','approach')
    geometry=MapGeometry(map_config,layout);centre=geometry.objects[target]['centre'];x,y=episode['start_xy']
    to_target=math.degrees(math.atan2(centre[1]-y,centre[0]-x));bearing=(to_target-episode['start_yaw_deg']+180.)%360.-180.
    instruction_id=instruction_id or episode['instruction_id'];distance=math.hypot(centre[0]-x,centre[1]-y)
    changed={k:v for k,v in episode.items() if k not in ('plan','kick','requested_kind','in_wedge','pushes','push_hold')}
    changed.update(id=name,layout=layout,target=target,task=task,split=split or episode['split'],target_bearing_deg=round(bearing,2),
                   start_distance_m=round(distance,2),start_direction_deg=round(math.degrees(math.atan2(y-centre[1],x-centre[0]))%360.,2),
                   sector=sector_of(bearing),instruction_id=instruction_id,instruction=instruction_for_id(config,map_config,target,instruction_id),
                   others_in_view=others_in_view(config,geometry,target,x,y,episode['start_height_m'],episode['start_yaw_deg'],
                                                 config['generalization']['fov_half_deg']),base_start=episode['id'])
    plan=rollout(config,oft,geometry,changed,map_config['altitude']['ceiling_m'])
    if plan['reason']!='teacher_stop' or (task=='land' and not plan['landed']) or (plan['climbed_m']>0 and not allow_climb):return None
    changed['plan']={'ticks':plan['ticks'],'clearance_m':round(plan['clearance_m'],2),'climbed_m':round(plan['climbed_m'],2),
                     'acquired_tick':plan['acquired_tick'],'states':plan['states']}
    return changed


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


def plan_targeted(config,oft,map_config,split,recipe,held_out=None):
    """Extra episodes of the kinds listed in a recipe: {kind: {base, ranges, count: [train, val]}}.

    Seeds start at the recipe's own number so they never meet an earlier plan's; objects cycle, and
    left and right alternate within each kind."""
    settings=config['generalization'];targets=map_config['train_objects'];layouts=map_config['train_layouts'];episodes=[]
    seed=recipe['seeds'][split]
    for kind,spec in recipe['kinds'].items():
        for index in range(spec['count'][0 if split=='train' else 1]):
            target=targets[index%len(targets)];side=(1,-1)[(index//len(targets)+index)%2];episode=None
            bands=BANDS[:1] if 'distance_m' in spec else tuple(dict.fromkeys((BANDS[index%len(BANDS)],)+BANDS))
            for attempt in range(40):
                layout=layouts[random.Random(f'layout|{map_config["id"]}|{seed}').randrange(len(layouts))]
                excluded=exclusion(held_out,map_config['id'],layout,target,settings['exclusion_radius_m']) if held_out else None
                for band in bands:
                    episode=make_start(config,oft,map_config,layout,target,kind,band,seed,split,side=side,excluded=excluded,spec=spec)
                    if episode:break
                seed+=1
                if episode:break
            if episode is None:raise RuntimeError(f'No {kind} start for {target}')
            episodes.append(episode)
    return episodes


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
