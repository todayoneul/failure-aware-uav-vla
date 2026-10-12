"""The privileged teacher for a landing on any landing surface, and a dry run of it. No simulator, no torch.

The canonical teacher (src/visual_search/generalization.SearchTeacher) lands on a pad: it flies level until it is over the
middle and comes down. A pad's top is 2.5 m above the ground and every recorded flight was above it. The top of a cube is
8 m up, a cylinder's 9 m: a vehicle that flies level at 6 m toward one flies into its side. `SurfaceTeacher` is the
canonical teacher with two things added, both only while the sentence asks for a landing and the named object is in view:

    climb     near the object and lower than its top plus a margin: go up (holding back, and stopping short of it, until
              the vehicle is high enough to fly over the top)
    ease in   over a surface smaller than a pad: slow down in proportion to the distance left, so that the vehicle, which
              follows a velocity command about 2.4 s late, is over the middle when it starts to come down

Everything else is the canonical teacher's: the search (turn right; climb when something close fills the view), the
approach, the descent over the middle, the zero action after touchdown, and every approach mission. On a pad flown from
the heights the earlier data was recorded at, the labels are the canonical ones.

The teacher is told where the target is and what its surface is. What it does with that is still tied to what the two
cameras show: it climbs toward a top that stands above it in the front view, and it descends onto a surface that fills the
down view. None of what it knows is given to the student.

`SurfaceTeacher` is the teacher of stage 1 and is kept as it was flown: its data, its plans and its result stay what they
were. Its slow entry labels a level step of 0.15 m for the last metres over a small top, 1.9 times the size the flight loop
reads as a stop, and it stands still while it climbs; a student whose output is a little smaller than its label stopped in
the air there (docs/generic_landing_active_search_v2.md). `StopSafeTeacher` (stage 1b) replaces those two things and
nothing else:

    entry     over a small top the vehicle slows down early and then flies in at one speed that is far from a stop
              (`entry_forward_m`); the descent starts at the canonical radius, as over a pad, and the vehicle coasts to
              about the middle
    climb     once the vehicle is clear of the top it moves in while it climbs, so that the end of the climb and the
              start of the level flight share the forward motion; below that it climbs where it is

Every label of a landing before the touchdown then keeps the margin from the stop rule that the stage's configuration
fixes (src/landing/stop_safety.py measures it with the student's own `is_stop`).

`RecoveryTeacher` (stage 1c) is the stage 1b teacher with one more rule, and `RecoveryPushes` puts a landing into the
states that rule is for. The stage 1b student no longer stops in the air; what is left of its failures on a cube is how
it comes down: it flies in too low and touches the edge, it starts to descend before it is over the middle, or it hangs a
metre or two over the top and commands up and down in turn. The teacher never flies into such a state by itself, so the
data never held the way out of one.

    descent    over a small top the canonical rule 'low over the top and within 3.5 m of the middle: keep descending' is a
               pad's (a cube's usable top ends 3.6 m from its middle). The descent still starts within the canonical
               align radius; once the vehicle is low it goes on within the abort radius, half-way between the align
               radius and the end of the usable top, and beyond that the vehicle climbs back and comes in again.
    forced     a landing is pushed (flown, never a label, as the canonical pushes are) into one of four states: in over
               the object without the height (`under_climb`), over its edge beside the middle (`edge`), coming down
               before the middle (`premature`), or lifted again during the last metres (`stall`). The frames after the
               push carry the teacher's way out: up, to the middle, down, and down without a pause.

`FinalDescentTeacher` (stage 1d) is the stage 1c teacher with the end of a descent that goes on moved outward, and
`RecoveryPushes` knows three more forced moves for the last two metres over a top. The later checkpoints of stage 1c
stopped in the air 0.2-2.3 m over a cube with the middle abeam of them: the place where the stage 1c teacher's label
turns from 'down' into 'up' (2.675 m from the middle), in a state no recorded frame held
(docs/generic_landing_stage1d_final_descent.md). A student sees one pair of pictures and nothing of what it did before,
so the teacher's rule is one of place, not of memory:

    commit     within the canonical align radius of the middle: down (DESCENT_COMMIT)
    continue   low over a small top (the canonical commit height or less) and within the hard recovery radius, which is
               the end of the usable top less the drift of a descent: down, whatever brought the vehicle there and
               wherever the middle is (DESCENT_CONTINUE). A touchdown there is inside the region.
    recover    low and beyond that radius, or beside the top: no descent. Up - in place only beside the top and lower
               than the clearance over it, otherwise up and in (RECOVER_CLEARANCE) - then at the height it flies in at to
               the middle (RECENTER) and down again.

The band between the two radii is the hysteresis: a vehicle that flies in does not descend before the align radius, a
vehicle that is low does not stop descending before the hard recovery radius.
"""
import json
import math
from pathlib import Path
from src.visual_search.generalization import SearchTeacher,geometric_view,Pushes,STOP_STATES
from src.visual_search.maps import MapGeometry,load_landing

STAGE1B=Path(__file__).resolve().parents[2]/'configs/generic_landing_stage1b.json'
STAGE1D=Path(__file__).resolve().parents[2]/'configs/generic_landing_stage1d.json'

EASE_GAIN=.05        # forward metres per decision for each metre left over a small surface: 0.1 per second, about 1 / (4 x the lag)
LEVEL_BAND_M=.25     # a vehicle this little below the height it needs is high enough
COMMIT_MARGIN_M=.5   # over a small surface the vehicle flies in this much above the height below which a descent is committed


class SurfaceTeacher(SearchTeacher):
    """`surface` is the named object's landing surface (src/landing/surface.LandingSurface) or None."""
    def __init__(self,config,oft,strategy='right',ceiling_m=14.,landing=None,surface=None):
        super().__init__(config,oft,strategy,ceiling_m,landing);self.surface=surface
        rule=landing or load_landing();self.pad_reach=rule['touchdown']['landing_region_half_width_m']
        self.small=surface is not None and surface.landable and surface.usable_reach_m<self.pad_reach
        # How high over the top the vehicle flies in. Over a pad: the height at which the canonical descent slows (2.5 m), so that
        # the heights the earlier pad flights were flown at (5 m and more) need no climb. Over a smaller top: just above the height
        # below which the canonical teacher keeps descending wherever it is (3 m), so that the descent starts over the middle.
        self.over_m=self.landing['commit_height_m']+COMMIT_MARGIN_M if self.small else self.landing['slow_height_m']
        self.climb_within=self.landing['glide_distance_m'];self.hold_within=config['arrive_distance_m']

    def act(self,view):
        action,state=super().act(view)
        landing=view.get('task')=='land' and self.surface is not None and self.surface.landable
        if not landing or view.get('landed') or not (view['visible'] or view['below'] or view.get('over')):return action,state
        land=self.landing;distance=view['distance_m'];above=view['height_m']-view['surface_m'];need=view['surface_m']+self.over_m
        # Something else close ahead: the canonical climb over it comes first.
        if state=='climb':return action,state
        if state!='descend' and distance<=self.climb_within and view['height_m']<need-LEVEL_BAND_M and view['height_m']<self.ceiling-.25:
            # Lower than the top it is to land on: up first. Close to the object the vehicle waits until it can fly over it.
            forward=0. if distance<=self.hold_within else action[0]
            return [forward,-min(self.down_max,need-view['height_m']),action[2]],'climb'
        if not self.small:return action,state
        # A top with less room than a pad. The canonical radii (over the middle: descend; low and near it: keep descending) are
        # a pad's; here they shrink with the surface, and the last metres are flown slowly so that the descent starts over
        # the middle and not where the vehicle's lag carries it.
        scale=self.surface.usable_reach_m/self.pad_reach
        middle=distance<=land['align_radius_m']*scale or (view.get('over') and above<=land['commit_height_m'] and distance<=land['commit_radius_m']*scale)
        if middle:return [0.,self.down_max if above>land['slow_height_m'] else land['final_descent_m'],0.],'descend'
        yaw=max(-self.yaw_max,min(self.yaw_max,math.radians(view['bearing_deg'])))
        forward=self.forward_max*max(0.,1-abs(view['bearing_deg'])/self.settings['centre_deg'])
        forward=min(forward,max(land['min_forward_m'],EASE_GAIN*distance))
        down=action[1] if state in ('approach','final','align') else 0.
        return [forward,down,yaw],('approach' if distance>self.config['arrive_distance_m'] else 'final') if forward>.05 else 'align'


def stop_safe_rule(path=None):
    """The numbers of the stage 1b teacher (configs/generic_landing_stage1b.json, `teacher`)."""
    return json.loads(Path(path or STAGE1B).read_text(encoding='utf-8'))['teacher']


class StopSafeTeacher(SearchTeacher):
    """Stage 1b. `surface` is the named object's landing surface or None; `rule` the teacher's numbers."""
    def __init__(self,config,oft,strategy='right',ceiling_m=14.,landing=None,surface=None,rule=None):
        super().__init__(config,oft,strategy,ceiling_m,landing);self.surface=surface;rule=rule or stop_safe_rule()
        self.pad_reach=(landing or load_landing())['touchdown']['landing_region_half_width_m']
        self.small=surface is not None and surface.landable and surface.usable_reach_m<self.pad_reach
        # How high over the top the vehicle flies in: as in stage 1 (a pad: the canonical slow height; a smaller top: just above
        # the height below which the canonical teacher keeps descending).
        self.over_m=self.landing['commit_height_m']+COMMIT_MARGIN_M if self.small else self.landing['slow_height_m']
        self.climb_within=self.landing['glide_distance_m'];self.hold_within=config['arrive_distance_m']
        self.entry=rule['entry_forward_m'];self.ease=rule['ease_gain_per_m'];self.clear=rule['clear_above_top_m']

    def holds(self,view,distance,above):
        """Does a vehicle that has to climb do so where it is? Near the object and not yet clear of its top."""
        return distance<=self.hold_within and above<self.clear

    def act(self,view):
        action,state=super().act(view)
        landing=view.get('task')=='land' and self.surface is not None and self.surface.landable
        if not landing or view.get('landed') or not (view['visible'] or view['below'] or view.get('over')):return action,state
        # The canonical climb over something close ahead and the canonical descent (over the middle, or low over the top and
        # near it) are a pad's and stay: full rate up, and 0.5 or 0.25 m down.
        if state in ('climb','descend'):return action,state
        distance=view['distance_m'];above=view['height_m']-view['surface_m'];bearing=view['bearing_deg']
        yaw=max(-self.yaw_max,min(self.yaw_max,math.radians(bearing)))
        forward=action[0]
        if self.small:
            # Slower the nearer, from far out, down to the entry speed and no further: the vehicle, which follows a command
            # about 2.4 s late, flies the last metres at the entry speed and is told to descend at the canonical radius.
            forward=min(self.forward_max*max(0.,1-abs(bearing)/self.settings['centre_deg']),max(self.entry,self.ease*distance))
        if distance<=self.climb_within and above<self.over_m-LEVEL_BAND_M and view['height_m']<self.ceiling-.25:
            # Lower than it flies in at. Not yet clear of the top and near the object: up, in place. Clear of it: up and in.
            hold=self.holds(view,distance,above)
            return [0. if hold else forward,-min(self.down_max,self.over_m-above),yaw],'climb'
        if not self.small:return action,state
        return [forward,action[1],yaw],('approach' if distance>self.config['arrive_distance_m'] else 'final') if forward>.05 else 'align'


def abort_radius(rule,surface):
    """How far from the middle of a small top a descent that has begun goes on: half-way between the canonical align
    radius, inside which a descent starts, and the end of the usable top."""
    return (rule['align_radius_m']+surface.usable_reach_m)/2


class RecoveryTeacher(StopSafeTeacher):
    """Stage 1c: the stage 1b teacher, with the radius of a descent that goes on fitted to a small top."""
    def __init__(self,config,oft,strategy='right',ceiling_m=14.,landing=None,surface=None,rule=None):
        super().__init__(config,oft,strategy,ceiling_m,landing,surface,rule)
        if self.small:self.landing=dict(self.landing,commit_radius_m=abort_radius(self.landing,surface))


def final_descent_rule(path=None):
    """The numbers of the stage 1d teacher (configs/generic_landing_stage1d.json, `teacher`)."""
    return json.loads(Path(path or STAGE1D).read_text(encoding='utf-8'))['teacher']


def hard_recovery_radius(surface,margin_m):
    """How far from the middle of a small top a vehicle that is low over it goes on descending: the end of the usable top
    less the horizontal drift of the last metres of a descent."""
    return surface.usable_reach_m-margin_m


class FinalDescentTeacher(RecoveryTeacher):
    """Stage 1d: the stage 1c teacher with the descent that goes on reaching to the hard recovery radius, and with no
    climb in place over the top itself (there is no side to fly into there). `final`: the numbers of stage 1d."""
    def __init__(self,config,oft,strategy='right',ceiling_m=14.,landing=None,surface=None,rule=None,final=None):
        super().__init__(config,oft,strategy,ceiling_m,landing,surface,rule);final=final or final_descent_rule()
        self.align=self.landing['align_radius_m'];self.hard=self.landing['commit_radius_m']
        if self.small:self.hard=hard_recovery_radius(surface,final['drift_margin_m']);self.landing=dict(self.landing,commit_radius_m=self.hard)

    def holds(self,view,distance,above):
        return super().holds(view,distance,above) and not (self.small and view.get('over'))


def descent_phase(state,view,teacher):
    """The name of what a landing teacher's decision is, for records and tests (the label itself is the action). A place
    and a state, never a memory: DESCENT_COMMIT (down, within the align radius), DESCENT_CONTINUE (down, farther out),
    RECOVER_CLEARANCE (up near the object), RECENTER (to the middle near the object), TOUCHDOWN, or the state itself."""
    if state=='landed':return 'TOUCHDOWN'
    near=view['distance_m']<=teacher.config['arrive_distance_m']
    if state=='descend':return 'DESCENT_COMMIT' if view['distance_m']<=teacher.landing['align_radius_m'] else 'DESCENT_CONTINUE'
    if state=='climb' and near:return 'RECOVER_CLEARANCE'
    if state in ('final','align') and near:return 'RECENTER'
    return state.upper()


FORCED=('under_climb','edge','premature','stall')
# Stage 1d: the last two metres over a top. An episode of stage 1d carries an `edge` item (which sets how far from the middle
# the vehicle is) and one of these.
FINAL=('low_hover','descent_decay','edge_drift')


class RecoveryPushes(Pushes):
    """Forced moves of a landing episode of stage 1c, flown by the teacher and by a tested policy alike and never a label.
    Each item of `episode['pushes']` names a state (`when`) and is due at a place, not at something a policy puts out:

        under_climb  within `within_m` of the object and between `rise_m` and `rise_m` + `band_m` over its top, on its way
                     up: straight in, without climbing on, to `until_m`
        edge         within `from_m` and at least `min_above_m` over the top: past the middle at `miss_m` on `side`
                     (-1 left, 1 right), until the middle is abeam
        premature    within `from_m`: down (and slowly on) to `floor_m` over the top
        stall        coming down, `below_m` over the top or lower: up again by `up_m` a decision for `ticks` decisions

    Stage 1d adds three for the last metres, each due at a place too (`within_m` of the middle, over the top or beside it):

        low_hover      down to `hold_m` over the top and held there, nearly still, for `hold_ticks` decisions
        descent_decay  down to `from_m` over the top, then a descent that dies away: `start_m` a decision, each decision
                       `ratio` of the one before, for `decay_ticks` decisions
        edge_drift     down to `from_m` over the top, then level and straight on until the vehicle is `out_m` from the
                       middle with the middle behind it

    Canonical items (`stop`, `descend`, `landed`) are handled as the canonical class handles them. A recovery item that is
    not yet due does not keep a flight going: a policy that stops before the place is reached has stopped."""
    def __init__(self,episode):
        super().__init__(episode);self.forced=bool(self.queue) and all(item.get('when') in FORCED+FINAL for item in self.queue);self.fired=[]

    def pending(self):return self.left>0 if self.forced else super().pending()

    def due(self,item,state,view):
        when=item.get('when','stop');above=view['height_m']-view['surface_m'];distance=view['distance_m']
        if when=='under_climb':return item['until_m']<distance<=item['within_m'] and item['rise_m']<=above<=item['rise_m']+item['band_m']
        if when=='edge':return item['miss_m']+1.<distance<=item['from_m'] and above>=item['min_above_m']
        if when=='premature':return distance<=item['from_m'] and above>item['floor_m']+.25
        if when=='stall':return state=='descend' and 0.<above<=item['below_m']
        if when in FINAL:return distance<=item['within_m'] and above>item.get('floor_m',.1)
        return super().due(item,state,view)

    def move(self,item,view,yaw_max):
        """The forced action of this decision, or None when the state is reached."""
        when=item['when'];above=view['height_m']-view['surface_m'];distance=view['distance_m'];bearing=view['bearing_deg']
        turn=lambda degrees:max(-yaw_max,min(yaw_max,math.radians(degrees)))
        if when=='under_climb':
            return None if distance<=item['until_m'] else [item['forward_m']*max(0.,1-abs(bearing)/45.),0.,turn(bearing)]
        if when=='edge':
            # The middle is kept at the angle that makes the track pass it at `miss_m`; abeam of it the push is over.
            wanted=item['side']*math.degrees(math.asin(min(1.,item['miss_m']/max(distance,1e-6))));off=bearing-wanted
            if abs(bearing)>=75. or distance<=item['miss_m']+.3:return None
            return [item['forward_m']*max(.2,1-abs(off)/30.),0.,turn(off)]
        if when=='premature':
            return None if above<=item['floor_m'] else [item['forward_m'],min(item['down_m'],above-item['floor_m']+.05),0.]
        if when=='stall':return [item.get('forward_m',0.),-item['up_m'],0.]
        if when=='low_hover':
            # Down to the height, slowly enough for a vehicle that follows a command late, then nearly still.
            off=above-item['hold_m']
            if abs(off)<=item['band_m']:
                item['held']=item.get('held',0)+1
                return None if item['held']>item['hold_ticks'] else [0.,0.,0.]
            return [0.,max(-item['down_m'],min(item['down_m'],item['gain']*off)),0.]
        if when=='descent_decay':
            if above<=item.get('floor_m',.1):return None
            if above>item['from_m'] and 'decayed' not in item:return [0.,item['down_m'],0.]
            item['decayed']=item.get('decayed',0)+1
            return None if item['decayed']>item['decay_ticks'] else [0.,item['start_m']*item['ratio']**(item['decayed']-1),0.]
        if when=='edge_drift':
            if above>item['from_m'] and 'level' not in item:return [0.,item['down_m'],0.]
            item['level']=True
            if distance>=item['out_m'] and abs(bearing)>90.:return None
            # Slower over the last metre and a half, so that the vehicle is not carried far past the place.
            left=item['out_m']-distance if abs(bearing)>90. else item['out_m']+distance
            return [item['forward_m']*max(.3,min(1.,left/1.5)),0.,0.]
        return None

    def step(self,state,view,yaw_max):
        if not self.forced:return super().step(state,view,yaw_max)
        if self.left<=0:
            if not self.queue or not self.due(self.queue[0],state,view):return None
            self.current=self.queue.pop(0);self.left=self.current['ticks'];self.done+=1
            self.fired.append({'when':self.current['when'],'distance_m':view['distance_m'],'above_top_m':view['height_m']-view['surface_m']})
        action=self.move(self.current,view,yaw_max)
        if action is None:self.left=0;return None
        self.left-=1;return action


TEACHERS={'surface':SurfaceTeacher,'stop_safe':StopSafeTeacher,'recovery':RecoveryTeacher,'final_descent':FinalDescentTeacher}


def teacher_class(surface_of,kind='surface'):
    """A teacher class for the canonical episode loop, which builds its teacher itself: `surface_of()` gives the landing
    surface of the episode being flown. `kind`: `surface` (stage 1), `stop_safe` (stage 1b), `recovery` (stage 1c) or
    `final_descent` (stage 1d)."""
    class Teacher(TEACHERS[kind]):
        def __init__(self,config,oft,strategy='right',ceiling_m=14.,landing=None):
            super().__init__(config,oft,strategy,ceiling_m,landing,surface_of())
    return Teacher


def rollout(config,oft,map_config,layout,episode,surface,landing=None,kind='surface',trace=None):
    """Fly the teacher through an episode on the map's boxes: kinematics only (no lag, no simulator), as the canonical dry
    run does. Says whether the teacher finishes, how near it comes to an obstacle and how much it climbs. `trace`: a list
    that receives every decision (label, state, distance, whether the vehicle stands on the surface, whether the decision
    was a forced move and whether a camera shows the object). The forced moves of a stage 1c episode are flown."""
    settings=config['generalization'];landing=landing or load_landing();geometry=MapGeometry(map_config,layout)
    teacher=TEACHERS[kind](config,oft,episode.get('strategy','right'),map_config['altitude']['ceiling_m'],landing,surface)
    x,y=episode['start_xy'];yaw=math.radians(episode['start_yaw_deg']);height=episode['start_height_m'];target=episode['target']
    task=episode.get('task','approach');stops=0;clearance=float('inf');states={};acquired=None;top=height;reason='max_steps';landed=False;tick=0
    rest=surface.top_m if surface is not None and surface.landable else 0.;pushes=RecoveryPushes(episode);limit=oft['action_bounds']['yaw_rad'][1];forced_ticks=0
    for tick in range(episode.get('max_ticks',settings['max_ticks'])):
        view=geometric_view(config,geometry,target,x,y,height,yaw)
        over=bool(rest) and surface.on_top(surface.offset_of(x,y));landed=task=='land' and over and height<=rest+1e-6
        view.update(task=task,landed=landed,surface_m=rest)
        action,state=teacher.act(view)
        # A forced move of the episode is flown instead of the teacher's action and is no label (only the recovery items:
        # the canonical pushes of an approach are left to the canonical dry run).
        forced=pushes.step(state,{'bearing_deg':view['bearing_deg'],'distance_m':view['distance_m'],'height_m':height,'surface_m':rest},limit) if pushes.forced else None
        if trace is not None:trace.append({'step':tick,'teacher_action':list(action),'teacher_state':state,'landed':landed,'distance_m':view['distance_m'],'height_m':height,
                                           'perturbed':forced is not None,'seen':bool(view['visible'] or view['below'] or view['over']),'x':x,'y':y,'bearing_deg':view['bearing_deg'],
                                           'over':bool(view.get('over'))})
        if forced is not None:action=forced;forced_ticks+=1
        if view['visible'] and acquired is None:acquired=tick
        states[state]=states.get(state,0)+1
        stops=stops+1 if state in STOP_STATES else 0
        if stops>=config['stop_ticks'] and not pushes.pending():reason='teacher_stop';break
        if landed and action[1]>=0:continue
        yaw+=action[2];x+=action[0]*math.cos(yaw);y+=action[0]*math.sin(yaw);height-=action[1];top=max(top,height)
        if task=='land' and rest and surface.on_top(surface.offset_of(x,y)):height=max(height,rest)
        # The named object is flown over on purpose; everything else is kept clear of, at the height flown.
        clearance=min(clearance,geometry.nearest(x,y,above_m=height-settings['corridor_clearance_m'],ignore=(target,))[0])
        if clearance<settings['rollout_clearance_m']:reason='too_close';break
        if not rest or not surface.on_top(surface.offset_of(x,y)):
            # Into the side of the named object itself: lower than its top and inside its footprint.
            item=geometry.objects[target]
            if geometry.over(x,y,target) and height<item['size_m'][2]:reason='hit_target';break
    final=math.hypot(geometry.objects[target]['centre'][0]-x,geometry.objects[target]['centre'][1]-y)
    return {'reason':reason,'ticks':tick+1,'clearance_m':clearance,'climbed_m':top-episode['start_height_m'],'states':states,
            'acquired_tick':acquired,'final_distance_m':final,'landed':landed,'forced_ticks':forced_ticks,'forced':list(pushes.fired),'forced_left':len(pushes.queue)}
