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
"""
import math
from src.visual_search.generalization import SearchTeacher,geometric_view,Pushes,STOP_STATES
from src.visual_search.maps import MapGeometry,load_landing

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


def teacher_class(surface_of):
    """A teacher class for the canonical episode loop, which builds its teacher itself: `surface_of()` gives the landing
    surface of the episode being flown."""
    class Teacher(SurfaceTeacher):
        def __init__(self,config,oft,strategy='right',ceiling_m=14.,landing=None):
            super().__init__(config,oft,strategy,ceiling_m,landing,surface_of())
    return Teacher


def rollout(config,oft,map_config,layout,episode,surface,landing=None):
    """Fly the teacher through an episode on the map's boxes: kinematics only (no lag, no simulator), as the canonical dry
    run does. Says whether the teacher finishes, how near it comes to an obstacle and how much it climbs."""
    settings=config['generalization'];landing=landing or load_landing();geometry=MapGeometry(map_config,layout)
    teacher=SurfaceTeacher(config,oft,episode.get('strategy','right'),map_config['altitude']['ceiling_m'],landing,surface)
    x,y=episode['start_xy'];yaw=math.radians(episode['start_yaw_deg']);height=episode['start_height_m'];target=episode['target']
    task=episode.get('task','approach');stops=0;clearance=float('inf');states={};acquired=None;top=height;reason='max_steps';landed=False;tick=0
    rest=surface.top_m if surface is not None and surface.landable else 0.;pushes=Pushes(episode)
    for tick in range(episode.get('max_ticks',settings['max_ticks'])):
        view=geometric_view(config,geometry,target,x,y,height,yaw)
        over=bool(rest) and surface.on_top(surface.offset_of(x,y));landed=task=='land' and over and height<=rest+1e-6
        view.update(task=task,landed=landed,surface_m=rest)
        action,state=teacher.act(view)
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
            'acquired_tick':acquired,'final_distance_m':final,'landed':landed}
